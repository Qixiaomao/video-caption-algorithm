"""Run the frozen, non-paper efficiency preflight without formal timing."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.preprocessing.frame_loader import load_video_tensor
from src.models.caption_model import VideoCaptionModel


PROTOCOL = ROOT / "experiments/formal_efficiency_matrix_protocol_20260909.md"
INPUTS = ROOT / "experiments/formal_efficiency_inputs_20260909.json"
CHECKPOINT = ROOT / "runs/full_msvd_infonce_ablation_20260906/checkpoint_epoch066.pt"
CHECKPOINT_SHA256 = "891cdcceef2b52d09092dde00ef35f9e1ee652e686b30178b5c1a9aee81c7e7c"
ANNOTATIONS_SHA256 = "bcda3ba0583704a54297e4da4cc9acbcc5ec1ebffd6ed436a6ba3ee1b8a03c8b"
OUTPUT = ROOT / "runs/efficiency_preflight_20260909"
MEMORY_LIMIT_MIB = 3800.0
COSINE_GATE = 0.999
DECODE = {
    "prompt": "",
    "max_new_tokens": 24,
    "min_new_tokens": 0,
    "num_beams": 1,
    "temperature": 1.0,
    "top_p": 0.9,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 1.0,
}


def configurations() -> list[dict]:
    return [
        {"id": "E0", "name": "reference_fp32", "fp16": False, "attention": False,
         "channels_last": False, "gelu_tanh": False, "inplace_residual": False, "compile": False,
         "required": True},
        {"id": "E1", "name": "fp16", "fp16": True, "attention": False,
         "channels_last": False, "gelu_tanh": False, "inplace_residual": False, "compile": False,
         "required": True},
        {"id": "E2", "name": "fp16_sdpa", "fp16": True, "attention": True,
         "channels_last": False, "gelu_tanh": False, "inplace_residual": False, "compile": False,
         "required": True},
        {"id": "E3", "name": "fp16_sdpa_channels_last", "fp16": True, "attention": True,
         "channels_last": True, "gelu_tanh": False, "inplace_residual": False, "compile": False,
         "required": True},
        {"id": "E4", "name": "safe_full", "fp16": True, "attention": True,
         "channels_last": True, "gelu_tanh": True, "inplace_residual": True, "compile": False,
         "required": True},
        {"id": "E5", "name": "compile_exploratory", "fp16": True, "attention": True,
         "channels_last": True, "gelu_tanh": True, "inplace_residual": True, "compile": True,
         "required": False},
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def verify_inputs() -> tuple[dict, torch.Tensor]:
    manifest = json.loads(INPUTS.read_text(encoding="utf-8"))
    source = Path(manifest["source"]["annotations"])
    if manifest["source"]["annotations_sha256"] != ANNOTATIONS_SHA256:
        raise RuntimeError("Manifest annotation hash is not the registered hash")
    if sha256_file(source) != ANNOTATIONS_SHA256:
        raise RuntimeError("Validation annotations changed after input freeze")
    if len(manifest["records"]) != 16:
        raise RuntimeError("Frozen input manifest must contain exactly 16 videos")

    videos = []
    for record in manifest["records"]:
        for frame in record["frames"]:
            path = Path(frame["path"])
            if sha256_file(path) != frame["sha256"]:
                raise RuntimeError(f"Frozen frame changed: {path}")
        videos.append(load_video_tensor(record["frames_dir"], 8, 224, "cpu"))
    batch = torch.cat(videos, dim=0).contiguous()
    if tuple(batch.shape) != (16, 8, 3, 224, 224):
        raise RuntimeError(f"Unexpected frozen tensor shape: {tuple(batch.shape)}")
    return manifest, batch


def verify_checkpoint(payload: dict) -> None:
    if sha256_file(CHECKPOINT) != CHECKPOINT_SHA256:
        raise RuntimeError("Registered checkpoint hash changed")
    progress = payload.get("progress", {})
    args = payload.get("args", {})
    contract = payload.get("training_contract", {})
    required = {
        "objective_version": 3,
        "alignment_objective": "infonce",
        "alignment_weight": 0.1,
        "info_nce_temperature": 0.07,
        "frozen_gpt2": True,
        "frozen_vit_backbone": True,
    }
    if progress.get("epoch") != 66 or progress.get("step") != 9900:
        raise RuntimeError(f"Unexpected checkpoint progress: {progress}")
    if args.get("cond_mode") != "prefix" or args.get("prefix_len") != 4:
        raise RuntimeError(f"Unexpected conditioning args: {args}")
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError(f"Checkpoint contract mismatch for {key}: {contract.get(key)!r}")


def build_model(config: dict, payload: dict) -> VideoCaptionModel:
    args = payload["args"]
    model = VideoCaptionModel(
        vit_name=args["vit_name"],
        gpt2_name=args["gpt2_name"],
        cond_mode=args["cond_mode"],
        prefix_len=args["prefix_len"],
        freeze_vit=True,
        vit_enable_fp16=config["fp16"],
        vit_enable_attention_fastpath=config["attention"],
        vit_prefer_channels_last=config["channels_last"],
        vit_enable_torch_compile=config["compile"],
        vit_torch_compile_mode="reduce-overhead",
        vit_enable_mlp_bias_gelu_fusion=config["gelu_tanh"],
        vit_enable_residual_layernorm_fusion=False,
        vit_enable_inplace_residual_add_fusion=config["inplace_residual"],
        vit_enable_cupy_fused_pool=False,
        vit_enable_cuda_graph=False,
        use_cupy_prefix_projector=False,
    )
    incompatible = model.load_state_dict(payload["model_state"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint load returned incompatible keys")
    return model.eval().to("cuda")


def requested_state_violations(config: dict, status: dict) -> list[str]:
    requested = {
        "fp16_autocast": config["fp16"],
        "attention_fastpath": config["attention"],
        "channels_last": config["channels_last"],
        "mlp_gelu_tanh_approximation": config["gelu_tanh"],
        "inplace_residual_add": config["inplace_residual"],
        "torch_compile": config["compile"],
    }
    violations = []
    for name, expected in requested.items():
        actual = bool(status[name].get("active"))
        if actual != expected:
            violations.append(f"{name}: expected active={expected}, got {actual} ({status[name].get('reason')})")
    for excluded in ("residual_layernorm_fusion", "cupy_fused_pool", "cuda_graph"):
        if status[excluded].get("active"):
            violations.append(f"excluded optimization unexpectedly active: {excluded}")
    return violations


@torch.inference_mode()
def evaluate_batch(model: VideoCaptionModel, batch_cpu: torch.Tensor) -> dict:
    batch = batch_cpu.to("cuda", non_blocking=False)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    encoder = model.encoder(batch)
    projected = model.proj(encoder)
    prefix = model.decoder.build_visual_prefix(projected)
    captions = model.decoder.generate(projected, **DECODE)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        "captions": captions,
        "encoder": encoder.detach().float().cpu(),
        "prefix": prefix.detach().float().cpu(),
        "elapsed_seconds_smoke_only": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def compare_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    ref_flat = reference.flatten(1)
    candidate_flat = candidate.flatten(1)
    cosine = F.cosine_similarity(ref_flat, candidate_flat, dim=1)
    return {
        "all_finite": bool(torch.isfinite(candidate).all()),
        "min_cosine": float(cosine.min()),
        "mean_cosine": float(cosine.mean()),
        "max_abs_error": float((reference - candidate).abs().max()),
    }


def run_config(config: dict, payload: dict, inputs: torch.Tensor, baseline: dict | None) -> tuple[dict, dict | None]:
    row = {"config": config, "status": "started", "required": config["required"]}
    model = None
    try:
        model = build_model(config, payload)
        pre_status = model.encoder.get_optimization_status()
        if config["compile"] and pre_status["torch_compile"].get("reason") == "triton_not_installed":
            row.update({
                "status": "unavailable",
                "reason": "torch_compile requested but Triton is not installed; no silent fallback allowed",
                "optimization_status": pre_status,
                "pass": None,
            })
            return row, None

        outputs = {}
        for size in (1, 8):
            outputs[str(size)] = evaluate_batch(model, inputs[:size])
        try:
            outputs["16"] = evaluate_batch(model, inputs[:16])
            boundary = {"status": "complete", "peak_reserved_mib": outputs["16"]["peak_reserved_mib"]}
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            boundary = {"status": "oom_allowed_boundary"}

        status = model.encoder.get_optimization_status()
        violations = requested_state_violations(config, status)
        captions8 = outputs["8"]["captions"]
        gates = {
            "strict_checkpoint_load": True,
            "requested_states_exact": not violations,
            "batch8_all_captions_nonempty": len(captions8) == 8 and all(text.strip() for text in captions8),
            "first_caption_batch_invariant": outputs["1"]["captions"][0] == captions8[0],
            "required_memory_below_3800_mib": max(outputs[str(size)]["peak_reserved_mib"] for size in (1, 8)) < MEMORY_LIMIT_MIB,
        }
        comparisons = None
        if baseline is not None:
            comparisons = {
                "caption_exact_matches": sum(a == b for a, b in zip(baseline["captions"], captions8)),
                "caption_count": 8,
                "encoder": compare_tensors(baseline["encoder"], outputs["8"]["encoder"]),
                "prefix": compare_tensors(baseline["prefix"], outputs["8"]["prefix"]),
            }
            gates.update({
                "captions_exactly_equal_to_E0": comparisons["caption_exact_matches"] == 8,
                "encoder_finite_and_cosine_ge_0_999": comparisons["encoder"]["all_finite"] and comparisons["encoder"]["min_cosine"] >= COSINE_GATE,
                "prefix_finite_and_cosine_ge_0_999": comparisons["prefix"]["all_finite"] and comparisons["prefix"]["min_cosine"] >= COSINE_GATE,
            })
        else:
            gates.update({
                "encoder_finite": bool(torch.isfinite(outputs["8"]["encoder"]).all()),
                "prefix_finite": bool(torch.isfinite(outputs["8"]["prefix"]).all()),
            })
        row.update({
            "status": "complete",
            "pass": all(gates.values()),
            "gates": gates,
            "violations": violations,
            "optimization_status": status,
            "batch_boundary_16": boundary,
            "comparison_to_E0": comparisons,
            "smoke_measurements": {
                size: {
                    "elapsed_seconds_smoke_only": output["elapsed_seconds_smoke_only"],
                    "peak_allocated_mib": output["peak_allocated_mib"],
                    "peak_reserved_mib": output["peak_reserved_mib"],
                    "caption_count": len(output["captions"]),
                }
                for size, output in outputs.items()
            },
            "captions_batch8": captions8,
        })
        baseline_payload = None if baseline is not None else {
            "captions": captions8,
            "encoder": outputs["8"]["encoder"],
            "prefix": outputs["8"]["prefix"],
        }
        return row, baseline_payload
    except Exception as exc:
        row.update({"status": "error", "pass": False, "error_type": type(exc).__name__, "error": str(exc)})
        return row, None
    finally:
        del model
        # ViTFrameEncoder keeps a bound-method reference to itself. Force cyclic
        # garbage collection between configurations so CUDA memory measurements
        # do not accumulate models from earlier cells.
        gc.collect()
        torch.cuda.empty_cache()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite preflight output: {OUTPUT}")
    if not torch.cuda.is_available():
        raise RuntimeError("Frozen preflight requires the local CUDA GPU")
    OUTPUT.mkdir(parents=True)

    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    manifest, inputs = verify_inputs()
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    verify_checkpoint(payload)
    source_hashes = {
        str(path.relative_to(ROOT)): sha256_file(path)
        for path in (PROTOCOL, INPUTS, Path(__file__), ROOT / "src/models/caption_model.py",
                     ROOT / "src/models/video_encoder.py", ROOT / "src/models/text_decoder.py",
                     ROOT / "core/preprocessing/frame_loader.py")
    }
    preflight = {
        "protocol": "formal-efficiency-matrix-v1",
        "paper_ready": False,
        "formal_timing_started": False,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "checkpoint": {"path": str(CHECKPOINT.resolve()), "sha256": CHECKPOINT_SHA256},
        "input_manifest": {"path": str(INPUTS.resolve()), "sha256": sha256_file(INPUTS),
                           "tensor_sha256": tensor_sha256(inputs), "video_ids": [r["video_id"] for r in manifest["records"]]},
        "decode": DECODE,
        "source_sha256": source_hashes,
        "runtime": {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
                    "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
                    "total_gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20,
                    "deterministic_algorithms": True, "tf32": False},
        "configurations": configurations(),
    }
    write_json(OUTPUT / "preflight.json", preflight)

    rows = []
    baseline = None
    for config in configurations():
        print(f"[preflight] {config['id']} {config['name']}", flush=True)
        row, candidate_baseline = run_config(config, payload, inputs, baseline)
        rows.append(row)
        if config["id"] == "E0" and candidate_baseline is not None:
            baseline = candidate_baseline
        write_json(OUTPUT / "partial_results.json", {"rows": rows})
        print(json.dumps({"id": config["id"], "status": row["status"], "pass": row.get("pass"),
                          "error": row.get("error"), "violations": row.get("violations")}, ensure_ascii=False), flush=True)

    required_rows = [row for row in rows if row["required"]]
    required_pass = len(required_rows) == 5 and all(row.get("status") == "complete" and row.get("pass") for row in required_rows)
    result = {
        "protocol": "formal-efficiency-matrix-v1",
        "paper_ready": False,
        "formal_timing_started": False,
        "preflight_pass": required_pass,
        "decision": "ready_for_user_review_before_formal_matrix" if required_pass else "formal_matrix_blocked",
        "required_configurations": ["E0", "E1", "E2", "E3", "E4"],
        "exploratory_configuration": "E5",
        "rows": rows,
        "limits": [
            "All elapsed values are one-shot preflight smoke diagnostics, not paper measurements.",
            "No test670 record was read.",
            "No formal timing block was run.",
            "Batch size 16 is an optional 4 GB capability boundary.",
        ],
    }
    write_json(OUTPUT / "result.json", result)
    print(json.dumps({"output": str(OUTPUT.resolve()), "preflight_pass": required_pass,
                      "decision": result["decision"]}, ensure_ascii=False, indent=2), flush=True)
    return 0 if required_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
