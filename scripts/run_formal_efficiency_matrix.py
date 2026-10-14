"""Execute the preregistered formal efficiency matrix with resumable cells."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.preprocessing.frame_loader import load_video_tensor
from scripts.run_efficiency_preflight import (
    CHECKPOINT,
    CHECKPOINT_SHA256,
    DECODE,
    INPUTS,
    PROTOCOL,
    build_model,
    configurations,
    requested_state_violations,
    sha256_file,
    tensor_sha256,
    verify_checkpoint,
    verify_inputs,
)


PROTOCOL_SHA256 = "906118c28d8f11b5873093981db807bc935e38470ab1021ac4a8e5e702ef1200"
INPUTS_SHA256 = "ca870d4a3d09c9988212fce2cbb25c846f8083dd1b2bedd667c0f633ae90a946"
PREFLIGHT = ROOT / "runs/efficiency_preflight_20260909/result.json"
PREFLIGHT_SHA256 = "f41f778c52d4e75babec01d24b3d4e070b4bc282448924db457c7fc3c2f6a0c9"
SCHEDULE = ROOT / "experiments/formal_efficiency_schedule_20260909.json"
OUTPUT = ROOT / "runs/formal_efficiency_matrix_20260909"
SMOKE_OUTPUT = ROOT / "runs/formal_efficiency_matrix_20260909_runner_smoke"
FORMAL_CONFIG_IDS = ["E0", "E1", "E2", "E3", "E4"]
PRIMARY_BATCH_SIZES = [1, 2, 4, 8]
BOUNDARY_BATCH_SIZE = 16
BLOCKS = 3
WARMUP_ITERATIONS = 10
MEASURED_ITERATIONS = 50
ORDER_SEED = 123


def build_schedule() -> dict:
    rng = random.Random(ORDER_SEED)
    blocks = []
    for block_index in range(BLOCKS):
        order = FORMAL_CONFIG_IDS.copy()
        rng.shuffle(order)
        blocks.append({"block": block_index + 1, "configuration_order": order})
    return {
        "protocol": "formal-efficiency-matrix-v1",
        "seed": ORDER_SEED,
        "blocks": blocks,
        "batch_order": PRIMARY_BATCH_SIZES + [BOUNDARY_BATCH_SIZE],
        "primary_batch_sizes": PRIMARY_BATCH_SIZES,
        "boundary_batch_size": BOUNDARY_BATCH_SIZE,
        "warmup_iterations_per_cell": WARMUP_ITERATIONS,
        "measured_iterations_per_cell": MEASURED_ITERATIONS,
        "metric_definitions": {
            "preprocess_host_ms": "JPEG load, resize, ToTensor, normalization, and CPU concatenation",
            "preprocess_cuda_ms": "synchronized CPU-to-CUDA tensor transfer",
            "vit_encoder_ms": "synchronized CUDA-event ViT encoder latency",
            "prefix_alignment_ms": "synchronized CUDA-event projection and visual-prefix mapping latency",
            "gpt2_decoder_ms": "synchronized CUDA-event Hugging Face greedy generate latency",
            "end_to_end_ms": "host wall time from preprocessing start through synchronized decode completion",
            "samples_per_second": "batch_size / end_to_end_seconds",
            "generated_tokens_total": "sum of generated sequence lengths through first EOS, including EOS",
            "decoder_ms_per_step": "gpt2_decoder_ms / maximum generated sequence length in the batch",
            "decoder_ms_per_output_token": "gpt2_decoder_ms / generated_tokens_total",
            "peak_allocated_mib": "torch CUDA maximum allocated memory after per-iteration peak reset",
            "peak_reserved_mib": "torch CUDA maximum reserved memory after per-iteration peak reset",
        },
    }


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot summarize an empty value list")
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def metric_summary(values: list[float]) -> dict:
    q25 = percentile(values, 0.25)
    q75 = percentile(values, 0.75)
    return {
        "count": len(values),
        "median": statistics.median(values),
        "q25": q25,
        "q75": q75,
        "iqr": q75 - q25,
        "mean": statistics.mean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def caption_hash(captions: list[str]) -> str:
    payload = json.dumps(captions, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def git_metadata(require_clean: bool) -> dict:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).splitlines()
    if require_clean and status:
        raise RuntimeError(f"Formal run requires a clean worktree, found: {status}")
    return {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "dirty": bool(status),
        "changed_files": status,
    }


def validate_frozen_state() -> tuple[dict, dict, torch.Tensor, dict]:
    if sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise RuntimeError("Formal protocol changed after preregistration")
    if sha256_file(INPUTS) != INPUTS_SHA256:
        raise RuntimeError("Frozen input manifest changed after preregistration")
    if sha256_file(PREFLIGHT) != PREFLIGHT_SHA256:
        raise RuntimeError("Registered passing preflight result changed")
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if not preflight.get("preflight_pass") or preflight.get("formal_timing_started"):
        raise RuntimeError("A passing step-2 preflight is required")
    manifest, tensor = verify_inputs()
    registered_tensor_hash = json.loads(
        (ROOT / "runs/efficiency_preflight_20260909/preflight.json").read_text(encoding="utf-8")
    )["input_manifest"]["tensor_sha256"]
    if tensor_sha256(tensor) != registered_tensor_hash:
        raise RuntimeError("Preprocessed frozen input tensor changed after preflight")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    verify_checkpoint(checkpoint)
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))
    if schedule != build_schedule():
        raise RuntimeError("Frozen execution schedule does not match the registered generator")
    return manifest, checkpoint, tensor, schedule


def frozen_sources() -> dict[str, str]:
    paths = [
        PROTOCOL,
        INPUTS,
        SCHEDULE,
        PREFLIGHT,
        Path(__file__),
        ROOT / "scripts/run_efficiency_preflight.py",
        ROOT / "src/models/caption_model.py",
        ROOT / "src/models/video_encoder.py",
        ROOT / "src/models/text_decoder.py",
        ROOT / "core/preprocessing/frame_loader.py",
    ]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in paths}


def cuda_timed(function):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    value = function()
    end.record()
    torch.cuda.synchronize()
    return value, float(start.elapsed_time(end))


def decode_sequences(model, prefix: torch.Tensor) -> tuple[list[str], list[int]]:
    tokenizer = model.decoder.tokenizer
    attention_mask = torch.ones(prefix.shape[:2], dtype=torch.long, device=prefix.device)
    sequences = model.decoder.model.generate(
        inputs_embeds=prefix,
        attention_mask=attention_mask,
        max_new_tokens=DECODE["max_new_tokens"],
        min_new_tokens=DECODE["min_new_tokens"],
        num_beams=DECODE["num_beams"],
        do_sample=False,
        temperature=DECODE["temperature"],
        top_p=DECODE["top_p"],
        no_repeat_ngram_size=DECODE["no_repeat_ngram_size"],
        repetition_penalty=DECODE["repetition_penalty"],
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    captions = [value.strip() for value in tokenizer.batch_decode(sequences, skip_special_tokens=True)]
    counts = []
    for sequence in sequences.tolist():
        if tokenizer.eos_token_id in sequence:
            counts.append(sequence.index(tokenizer.eos_token_id) + 1)
        else:
            counts.append(len(sequence))
    return captions, counts


@torch.inference_mode()
def run_iteration(model, frame_dirs: list[Path]) -> dict:
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    e2e_started = time.perf_counter()

    host_started = time.perf_counter()
    cpu_videos = [load_video_tensor(path, num_frames=8, image_size=224, device="cpu") for path in frame_dirs]
    cpu_batch = torch.cat(cpu_videos, dim=0).contiguous()
    preprocess_host_ms = (time.perf_counter() - host_started) * 1000.0
    video, preprocess_cuda_ms = cuda_timed(lambda: cpu_batch.to("cuda", non_blocking=False))
    encoder, vit_ms = cuda_timed(lambda: model.encoder(video))

    def build_prefix():
        projected = model.proj(encoder)
        return model.decoder.build_visual_prefix(projected)

    prefix, prefix_ms = cuda_timed(build_prefix)
    decoded, gpt2_ms = cuda_timed(lambda: decode_sequences(model, prefix))
    captions, token_counts = decoded
    torch.cuda.synchronize()
    end_to_end_ms = (time.perf_counter() - e2e_started) * 1000.0

    if len(captions) != len(frame_dirs) or not all(caption.strip() for caption in captions):
        raise RuntimeError(f"Invalid caption output: {captions}")
    if not torch.isfinite(encoder).all() or not torch.isfinite(prefix).all():
        raise RuntimeError("Non-finite encoder or visual-prefix output")
    total_tokens = sum(token_counts)
    decode_steps = max(token_counts)
    if total_tokens <= 0 or decode_steps <= 0:
        raise RuntimeError(f"Invalid generated-token counts: {token_counts}")
    return {
        "preprocess_host_ms": preprocess_host_ms,
        "preprocess_cuda_ms": preprocess_cuda_ms,
        "vit_encoder_ms": vit_ms,
        "prefix_alignment_ms": prefix_ms,
        "gpt2_decoder_ms": gpt2_ms,
        "end_to_end_ms": end_to_end_ms,
        "samples_per_second": len(frame_dirs) / (end_to_end_ms / 1000.0),
        "generated_tokens_total": total_tokens,
        "generated_tokens_mean": statistics.mean(token_counts),
        "decode_steps": decode_steps,
        "decoder_ms_per_step": gpt2_ms / decode_steps,
        "decoder_ms_per_output_token": gpt2_ms / total_tokens,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "caption_sha256": caption_hash(captions),
        "captions": captions,
        "token_counts": token_counts,
    }


def cell_path(output: Path, block: int, config_id: str, batch_size: int) -> Path:
    return output / "cells" / f"block{block:02d}_{config_id}_bs{batch_size:02d}.json"


def run_cell(
    model,
    block: int,
    config: dict,
    batch_size: int,
    frame_dirs: list[Path],
    output: Path,
    warmups: int,
    measurements: int,
) -> dict:
    path = cell_path(output, block, config["id"], batch_size)
    if path.exists():
        cell = json.loads(path.read_text(encoding="utf-8"))
        if cell.get("status") not in ("complete", "oom_allowed_boundary"):
            raise RuntimeError(f"Refusing malformed resume cell: {path}")
        print(f"[resume] block={block} config={config['id']} bs={batch_size} status={cell['status']}", flush=True)
        return cell

    print(f"[cell] block={block} config={config['id']} bs={batch_size} warmup={warmups}", flush=True)
    try:
        for _ in range(warmups):
            run_iteration(model, frame_dirs)
        status = model.encoder.get_optimization_status()
        violations = requested_state_violations(config, status)
        if violations:
            raise RuntimeError(f"Requested optimization state mismatch: {violations}")

        rows = []
        stable_caption_hash = None
        stable_captions = None
        stable_token_counts = None
        for iteration in range(1, measurements + 1):
            row = run_iteration(model, frame_dirs)
            if stable_caption_hash is None:
                stable_caption_hash = row["caption_sha256"]
                stable_captions = row.pop("captions")
                stable_token_counts = row.pop("token_counts")
            else:
                row_captions = row.pop("captions")
                row_token_counts = row.pop("token_counts")
                if row["caption_sha256"] != stable_caption_hash or row_captions != stable_captions:
                    raise RuntimeError("Caption output changed within a formal timing cell")
                if row_token_counts != stable_token_counts:
                    raise RuntimeError("Generated-token counts changed within a formal timing cell")
            row["iteration"] = iteration
            rows.append(row)
            if iteration % 10 == 0 or iteration == measurements:
                print(
                    f"[measure] block={block} config={config['id']} bs={batch_size} "
                    f"iter={iteration}/{measurements} e2e={row['end_to_end_ms']:.2f}ms",
                    flush=True,
                )
        cell = {
            "status": "complete",
            "block": block,
            "config": config,
            "batch_size": batch_size,
            "warmup_iterations": warmups,
            "measured_iterations": measurements,
            "input_video_ids": [path.name for path in frame_dirs],
            "optimization_status": status,
            "optimization_violations": violations,
            "caption_sha256": stable_caption_hash,
            "captions": stable_captions,
            "token_counts": stable_token_counts,
            "iterations": rows,
        }
    except torch.cuda.OutOfMemoryError as exc:
        if batch_size != BOUNDARY_BATCH_SIZE:
            raise
        cell = {
            "status": "oom_allowed_boundary",
            "block": block,
            "config": config,
            "batch_size": batch_size,
            "error": str(exc),
            "iterations": [],
        }
        torch.cuda.empty_cache()
    write_json_atomic(path, cell)
    return cell


def validate_caption_equivalence(cells: list[dict], smoke: bool) -> dict:
    complete = [cell for cell in cells if cell["status"] == "complete"]
    by_batch = defaultdict(list)
    for cell in complete:
        by_batch[cell["batch_size"]].append(cell)
    results = {}
    for batch_size, group in sorted(by_batch.items()):
        unique_hashes = sorted({cell["caption_sha256"] for cell in group})
        unique_tokens = {tuple(cell["token_counts"]) for cell in group}
        expected_cells = 1 if smoke else BLOCKS * len(FORMAL_CONFIG_IDS)
        passed = len(group) == expected_cells and len(unique_hashes) == 1 and len(unique_tokens) == 1
        results[str(batch_size)] = {
            "pass": passed,
            "complete_cells": len(group),
            "expected_cells": expected_cells,
            "unique_caption_hashes": unique_hashes,
            "unique_token_count_vectors": [list(values) for values in sorted(unique_tokens)],
        }
    required_batches = [1] if smoke else PRIMARY_BATCH_SIZES
    if not all(results.get(str(batch_size), {}).get("pass") for batch_size in required_batches):
        raise RuntimeError(f"Cross-configuration caption equivalence failed: {results}")
    return results


def build_final_result(cells: list[dict], schedule: dict, metadata: dict, smoke: bool) -> dict:
    caption_equivalence = validate_caption_equivalence(cells, smoke)
    grouped = defaultdict(list)
    for cell in cells:
        if cell["status"] == "complete":
            grouped[(cell["config"]["id"], cell["batch_size"])].append(cell)

    metric_names = [
        "preprocess_host_ms", "preprocess_cuda_ms", "vit_encoder_ms", "prefix_alignment_ms",
        "gpt2_decoder_ms", "end_to_end_ms", "samples_per_second", "generated_tokens_total",
        "generated_tokens_mean", "decode_steps", "decoder_ms_per_step", "decoder_ms_per_output_token",
        "peak_allocated_mib", "peak_reserved_mib",
    ]
    summaries = []
    for (config_id, batch_size), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        values = {name: [] for name in metric_names}
        block_summaries = []
        for cell in group:
            for row in cell["iterations"]:
                for name in metric_names:
                    values[name].append(float(row[name]))
            block_summaries.append({
                "block": cell["block"],
                "metrics": {name: metric_summary([float(row[name]) for row in cell["iterations"]]) for name in metric_names},
            })
        summaries.append({
            "config_id": config_id,
            "batch_size": batch_size,
            "blocks": len(group),
            "measurements": len(values["end_to_end_ms"]),
            "metrics": {name: metric_summary(series) for name, series in values.items()},
            "block_summaries": block_summaries,
            "caption_sha256": group[0]["caption_sha256"],
            "captions": group[0]["captions"],
            "token_counts": group[0]["token_counts"],
        })

    lookup = {(row["config_id"], row["batch_size"]): row for row in summaries}
    comparisons = []
    for row in summaries:
        if row["config_id"] == "E0":
            continue
        baseline = lookup.get(("E0", row["batch_size"]))
        if baseline is None:
            continue
        base_latency = baseline["metrics"]["end_to_end_ms"]["median"]
        base_throughput = baseline["metrics"]["samples_per_second"]["median"]
        latency = row["metrics"]["end_to_end_ms"]["median"]
        throughput = row["metrics"]["samples_per_second"]["median"]
        comparisons.append({
            "config_id": row["config_id"],
            "baseline": "E0",
            "batch_size": row["batch_size"],
            "median_end_to_end_latency_reduction_percent": (base_latency - latency) / base_latency * 100.0,
            "median_throughput_improvement_percent": (throughput - base_throughput) / base_throughput * 100.0,
            "median_peak_allocated_reduction_mib": (
                baseline["metrics"]["peak_allocated_mib"]["median"]
                - row["metrics"]["peak_allocated_mib"]["median"]
            ),
            "median_peak_reserved_reduction_mib": (
                baseline["metrics"]["peak_reserved_mib"]["median"]
                - row["metrics"]["peak_reserved_mib"]["median"]
            ),
        })

    required_cells = 1 if smoke else BLOCKS * len(FORMAL_CONFIG_IDS) * len(PRIMARY_BATCH_SIZES)
    completed_required = sum(
        cell["status"] == "complete" and cell["batch_size"] in ([1] if smoke else PRIMARY_BATCH_SIZES)
        for cell in cells
    )
    pass_result = completed_required == required_cells
    return {
        "protocol": "formal-efficiency-matrix-v1",
        "paper_ready": not smoke and pass_result,
        "runner_smoke": smoke,
        "formal_timing_completed": not smoke and pass_result,
        "decision": "formal_matrix_complete" if not smoke and pass_result else "runner_smoke_complete",
        "metadata": metadata,
        "schedule": schedule,
        "caption_equivalence": caption_equivalence,
        "required_cells": required_cells,
        "completed_required_cells": completed_required,
        "cell_status_counts": {
            status: sum(cell["status"] == status for cell in cells)
            for status in sorted({cell["status"] for cell in cells})
        },
        "summaries": summaries,
        "comparisons_to_E0": comparisons,
        "limits": [
            "Results apply to the frozen repaired checkpoint, fixed MSVD validation inputs, and local RTX 3050 4 GB.",
            "Test670 was not read and caption quality was not re-estimated.",
            "E5 torch.compile was excluded because the passing preflight marked Triton unavailable.",
            "Tanh GELU is an approximation, not a fused-kernel claim.",
        ],
    }


def run_experiment(smoke: bool, resume: bool) -> int:
    output = SMOKE_OUTPUT if smoke else OUTPUT
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    git = git_metadata(require_clean=True)
    manifest, checkpoint, frozen_tensor, schedule = validate_frozen_state()
    del frozen_tensor
    gc.collect()

    sources = frozen_sources()
    metadata_path = output / "run_metadata.json"
    metadata = {
        "protocol": "formal-efficiency-matrix-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_timing_started": not smoke,
        "runner_smoke": smoke,
        "git": git,
        "checkpoint": {"path": str(CHECKPOINT.resolve()), "sha256": CHECKPOINT_SHA256},
        "protocol_sha256": PROTOCOL_SHA256,
        "input_manifest_sha256": INPUTS_SHA256,
        "preflight_result_sha256": PREFLIGHT_SHA256,
        "schedule_sha256": sha256_file(SCHEDULE),
        "source_sha256": sources,
        "decode": DECODE,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "total_gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
            "deterministic_algorithms": True,
            "tf32": False,
        },
        "test_split_read": False,
    }
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        for field in ("git", "checkpoint", "protocol_sha256", "input_manifest_sha256",
                      "preflight_result_sha256", "schedule_sha256", "source_sha256", "decode", "runner_smoke"):
            if prior[field] != metadata[field]:
                raise RuntimeError(f"Resume metadata mismatch: {field}")
        metadata = prior
    else:
        write_json_atomic(metadata_path, metadata)

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    by_id = {record["video_id"]: Path(record["frames_dir"]) for record in manifest["records"]}
    ordered_dirs = [by_id[record["video_id"]] for record in manifest["records"]]
    config_lookup = {row["id"]: row for row in configurations() if row["id"] in FORMAL_CONFIG_IDS}
    cells = []
    blocks = schedule["blocks"][:1] if smoke else schedule["blocks"]
    for block_entry in blocks:
        block = block_entry["block"]
        config_order = ["E0"] if smoke else block_entry["configuration_order"]
        for config_id in config_order:
            config = config_lookup[config_id]
            print(f"[model] block={block} config={config_id}", flush=True)
            model = build_model(config, checkpoint)
            try:
                batch_sizes = [1] if smoke else schedule["batch_order"]
                for batch_size in batch_sizes:
                    cell = run_cell(
                        model=model,
                        block=block,
                        config=config,
                        batch_size=batch_size,
                        frame_dirs=ordered_dirs[:batch_size],
                        output=output,
                        warmups=1 if smoke else WARMUP_ITERATIONS,
                        measurements=2 if smoke else MEASURED_ITERATIONS,
                    )
                    cells.append(cell)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()

    if sources != frozen_sources():
        raise RuntimeError("Experiment source changed during the formal run")
    cells = [json.loads(path.read_text(encoding="utf-8")) for path in sorted((output / "cells").glob("*.json"))]
    result = build_final_result(cells, schedule, metadata, smoke)
    write_json_atomic(output / "result.json", result)
    print(json.dumps({
        "output": str(output.resolve()),
        "decision": result["decision"],
        "required_cells": result["required_cells"],
        "completed_required_cells": result["completed_required_cells"],
        "cell_status_counts": result["cell_status_counts"],
    }, ensure_ascii=False, indent=2), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-schedule", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.freeze_schedule:
        if SCHEDULE.exists():
            raise FileExistsError(f"Refusing to overwrite frozen schedule: {SCHEDULE}")
        write_json_atomic(SCHEDULE, build_schedule())
        print(json.dumps({"schedule": str(SCHEDULE.resolve()), "sha256": sha256_file(SCHEDULE),
                          "blocks": build_schedule()["blocks"]}, ensure_ascii=False, indent=2))
        return 0
    if args.smoke and args.resume:
        raise ValueError("Runner smoke is intentionally non-resumable")
    return run_experiment(smoke=args.smoke, resume=args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
