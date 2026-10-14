from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from typing import Callable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

import backend_config
from core.config import InferenceConfig, ViTOptimizeConfig
from core.env import assert_core_runtime_ready
from core.models.model_loader import load_caption_model
from core.preprocessing.frame_loader import list_frames, load_video_tensor

MB = 1024.0 * 1024.0
DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 12, 16]


class NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, exc_type, exc, tb):
        torch.cuda.nvtx.range_pop()
        return False


class BenchmarkProfiler:
    def __init__(self, device: str):
        self.device = torch.device(device)
        self.stage_samples: dict[str, list[float]] = defaultdict(list)
        self.stage_peak_mem_mb: dict[str, list[float]] = defaultdict(list)
        self.token_step_ms: list[float] = []
        self.generated_lengths: list[int] = []
        self.total_latency_ms: list[float] = []
        self.preprocess_host_ms: list[float] = []
        self.throughput_samples: list[float] = []
        self.last_texts: list[str] = []
        self.last_video_ids: list[str] = []
        self.max_memory_allocated_mb: float = 0.0

    def clear_measurements(self):
        self.stage_samples.clear()
        self.stage_peak_mem_mb.clear()
        self.token_step_ms.clear()
        self.generated_lengths.clear()
        self.total_latency_ms.clear()
        self.preprocess_host_ms.clear()
        self.throughput_samples.clear()
        self.last_texts = []
        self.last_video_ids = []
        self.max_memory_allocated_mb = 0.0

    def update_global_peak(self) -> float:
        peak_mb = torch.cuda.max_memory_allocated(self.device) / MB
        self.max_memory_allocated_mb = max(self.max_memory_allocated_mb, peak_mb)
        return peak_mb

    def cuda_stage(self, name: str, fn: Callable[[], object]):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        with NvtxRange(name):
            start_event.record()
            out = fn()
            end_event.record()
        torch.cuda.synchronize(self.device)
        latency_ms = start_event.elapsed_time(end_event)
        peak_mem_mb = self.update_global_peak()
        self.stage_samples[name].append(latency_ms)
        self.stage_peak_mem_mb[name].append(peak_mem_mb)
        return out, latency_ms, peak_mem_mb

    def preprocess_stage(self, frame_dirs: list[Path], num_frames: int, image_size: int):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        host_t0 = time.perf_counter()
        with NvtxRange("Preprocessing"):
            start_event.record()
            videos = [
                load_video_tensor(
                    frame_dir,
                    num_frames=num_frames,
                    image_size=image_size,
                    device=str(self.device),
                )
                for frame_dir in frame_dirs
            ]
            video = torch.cat(videos, dim=0)
            end_event.record()
        torch.cuda.synchronize(self.device)
        host_ms = (time.perf_counter() - host_t0) * 1000.0
        cuda_ms = start_event.elapsed_time(end_event)
        peak_mem_mb = self.update_global_peak()
        self.preprocess_host_ms.append(host_ms)
        self.stage_samples["Preprocessing"].append(cuda_ms)
        self.stage_peak_mem_mb["Preprocessing"].append(peak_mem_mb)
        self.last_video_ids = [frame_dir.name for frame_dir in frame_dirs]
        return video, cuda_ms, host_ms, peak_mem_mb


def resolve_batch_frame_dirs(frames_root: str | Path, batch_size: int, allow_repeated_input: bool = False) -> list[Path]:
    """Resolve one real frame directory per batch item.

    ``frames_root`` may be a single video directory or a split-level directory
    containing one subdirectory per video. Repeating a single input is rejected
    unless the caller explicitly labels the run as synthetic.
    """

    root = Path(frames_root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"frames path is not a directory: {root}")

    if list_frames(root):
        candidates = [root]
    else:
        candidates = sorted(
            child for child in root.iterdir() if child.is_dir() and list_frames(child)
        )

    if not candidates:
        raise FileNotFoundError(f"no video directories containing frame_*.jpg under {root}")
    if len(candidates) >= batch_size:
        return candidates[:batch_size]
    if allow_repeated_input:
        return [candidates[index % len(candidates)] for index in range(batch_size)]
    raise ValueError(
        f"batch size {batch_size} requires {batch_size} distinct videos, but only "
        f"{len(candidates)} were found under {root}. Pass --allow-repeated-input only for a labelled synthetic microbenchmark."
    )


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def stats_dict(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_ms": None,
            "std_ms": None,
            "p50_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "max_ms": None,
            "min_ms": None,
        }
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "p50_ms": percentile(values, 0.50),
        "p95_ms": percentile(values, 0.95),
        "p99_ms": percentile(values, 0.99),
        "max_ms": max(values),
        "min_ms": min(values),
    }


def throughput_stats_dict(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean_samples_per_s": None,
            "std_samples_per_s": None,
            "max_samples_per_s": None,
            "min_samples_per_s": None,
        }
    return {
        "count": len(values),
        "mean_samples_per_s": statistics.mean(values),
        "std_samples_per_s": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "max_samples_per_s": max(values),
        "min_samples_per_s": min(values),
    }


def select_greedy_token(
    logits: torch.Tensor,
    eos_token_id: int | None,
    step_index: int,
    min_new_tokens: int,
) -> torch.Tensor:
    """Select the next token while matching HF ``min_new_tokens`` EOS masking."""

    if eos_token_id is not None and step_index < min_new_tokens:
        logits = logits.clone()
        logits[:, eos_token_id] = -torch.inf
    return torch.argmax(logits, dim=-1)


def validate_decoder_outputs(texts: list[str], token_counts: list[int], min_new_tokens: int) -> None:
    """Reject latency samples that did not produce usable caption output."""

    if len(texts) != len(token_counts):
        raise RuntimeError("decoder output mismatch: texts and token counts have different batch sizes")
    invalid = [
        index
        for index, (text, count) in enumerate(zip(texts, token_counts))
        if not text.strip() or count < min_new_tokens
    ]
    if invalid:
        details = ", ".join(
            f"sample={index}:tokens={token_counts[index]}:empty={not texts[index].strip()}" for index in invalid
        )
        raise RuntimeError(
            "invalid caption benchmark output; latency sample is rejected. "
            f"Required non-empty text and at least {min_new_tokens} generated tokens. {details}"
        )


def build_decoder_context(model, prefix_embeds: torch.Tensor, prompt: str) -> torch.Tensor:
    """Build the inference context without inserting an unseen BOS/EOS token."""

    if not prompt:
        return prefix_embeds

    tokenizer = model.decoder.tokenizer
    gpt2 = model.decoder.model
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(prefix_embeds.device)
    if prompt_ids.shape[0] == 1 and prefix_embeds.shape[0] > 1:
        prompt_ids = prompt_ids.expand(prefix_embeds.shape[0], -1)
    prompt_embeds = gpt2.transformer.wte(prompt_ids)
    return torch.cat([prefix_embeds, prompt_embeds], dim=1)


@torch.inference_mode()
def run_decoder_steps(
    model,
    prefix_embeds: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    profile_token_steps: bool,
    profiler: BenchmarkProfiler,
):
    if min_new_tokens < 0 or max_new_tokens <= 0 or min_new_tokens > max_new_tokens:
        raise ValueError("require 0 <= min_new_tokens <= max_new_tokens and max_new_tokens > 0")
    decoder = model.decoder
    tokenizer = decoder.tokenizer
    gpt2 = decoder.model
    device = prefix_embeds.device
    batch_size = prefix_embeds.shape[0]

    full_inputs = build_decoder_context(model, prefix_embeds, prompt)
    running_attention_mask = torch.ones(full_inputs.shape[:2], dtype=torch.long, device=device)

    generated_tokens = [[] for _ in range(batch_size)]
    step_samples: list[float] = []
    past_key_values = None
    next_input_embeds = full_inputs
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with NvtxRange("GPT2_Decoder_Step"):
        for step_idx in range(max_new_tokens):
            step_name = f"GPT2_Decoder_Step/token_{step_idx:02d}"
            if profile_token_steps:
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                with NvtxRange(step_name):
                    start_event.record()
                    outputs = gpt2(
                        inputs_embeds=next_input_embeds,
                        attention_mask=running_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                    end_event.record()
                torch.cuda.synchronize(device)
                step_ms = start_event.elapsed_time(end_event)
                step_samples.append(step_ms)
                profiler.token_step_ms.append(step_ms)
            else:
                with NvtxRange(step_name):
                    outputs = gpt2(
                        inputs_embeds=next_input_embeds,
                        attention_mask=running_attention_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )

            logits = outputs.logits[:, -1, :]
            eos_token_id = tokenizer.eos_token_id
            next_token = select_greedy_token(logits, eos_token_id, step_idx, min_new_tokens)
            if eos_token_id is not None:
                next_token = torch.where(finished, torch.full_like(next_token, eos_token_id), next_token)

            token_ids = next_token.tolist()
            for idx, token_id in enumerate(token_ids):
                if not finished[idx]:
                    generated_tokens[idx].append(token_id)
                    if eos_token_id is not None and token_id == eos_token_id:
                        finished[idx] = True

            past_key_values = outputs.past_key_values
            if finished.all():
                break

            next_input_embeds = gpt2.transformer.wte(next_token).unsqueeze(1)
            running_attention_mask = torch.cat(
                [running_attention_mask, torch.ones((batch_size, 1), dtype=torch.long, device=device)],
                dim=1,
            )

    profiler.generated_lengths.extend(len(tokens) for tokens in generated_tokens)
    texts = [tokenizer.decode(tokens, skip_special_tokens=True).strip() for tokens in generated_tokens]
    token_counts = [len(tokens) for tokens in generated_tokens]
    validate_decoder_outputs(texts, token_counts, min_new_tokens)
    profiler.last_texts = texts
    return {
        "texts": texts,
        "token_count": token_counts,
        "token_step_ms": step_samples,
    }


@torch.inference_mode()
def run_one_iteration(
    model,
    config: InferenceConfig,
    frame_dirs: list[Path],
    prompt: str,
    max_new_tokens: int,
    min_new_tokens: int,
    profile_token_steps: bool,
    profiler: BenchmarkProfiler,
):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(profiler.device)
    iter_t0 = time.perf_counter()

    video, preprocess_cuda_ms, preprocess_host_ms, preprocess_peak_mb = profiler.preprocess_stage(
        frame_dirs=frame_dirs,
        num_frames=config.num_frames,
        image_size=config.image_size,
    )
    batch_size = len(frame_dirs)

    encoder_out, encoder_ms, encoder_peak_mb = profiler.cuda_stage("ViT_Encoder", lambda: model.encoder(video))

    def align_fn():
        emb = model.proj(encoder_out)
        if emb.dim() == 2:
            emb = emb.unsqueeze(1)
        if config.ln_scale is not None and config.ln_scale > 0:
            emb = torch.nn.functional.layer_norm(emb, emb.shape[-1:]) * config.ln_scale
        if config.in_weight is not None and config.in_weight > 0:
            emb = emb * config.in_weight

        if model.decoder.cond_mode == "prefix":
            hidden = model.decoder.model.config.n_embd
            return model.decoder.mapper(emb).view(emb.size(0), model.decoder.prefix_len, hidden)

        mapped = model.decoder.mapper(emb)
        if mapped.dim() == 2:
            mapped = mapped.unsqueeze(1)
        return mapped

    prefix_embeds, align_ms, align_peak_mb = profiler.cuda_stage("Cross_Modal_Alignment", align_fn)
    decoder_result, decoder_ms, decoder_peak_mb = profiler.cuda_stage(
        "GPT2_Decoder_Step",
        lambda: run_decoder_steps(
            model,
            prefix_embeds,
            prompt,
            max_new_tokens,
            min_new_tokens,
            profile_token_steps,
            profiler,
        ),
    )

    iter_ms = (time.perf_counter() - iter_t0) * 1000.0
    throughput = batch_size / (iter_ms / 1000.0)
    profiler.total_latency_ms.append(iter_ms)
    profiler.throughput_samples.append(throughput)
    profiler.update_global_peak()
    return {
        "batch_size": batch_size,
        "input_video_ids": [frame_dir.name for frame_dir in frame_dirs],
        "iteration_ms": iter_ms,
        "throughput_samples_per_s": throughput,
        "captions": decoder_result["texts"],
        "caption_preview": decoder_result["texts"][0] if decoder_result["texts"] else "",
        "generated_tokens": decoder_result["token_count"],
        "generated_tokens_mean": statistics.mean(decoder_result["token_count"]) if decoder_result["token_count"] else 0.0,
        "preprocess_cuda_ms": preprocess_cuda_ms,
        "preprocess_host_ms": preprocess_host_ms,
        "preprocess_peak_mb": preprocess_peak_mb,
        "vit_encoder_ms": encoder_ms,
        "vit_encoder_peak_mb": encoder_peak_mb,
        "cross_modal_alignment_ms": align_ms,
        "cross_modal_alignment_peak_mb": align_peak_mb,
        "gpt2_decoder_ms": decoder_ms,
        "gpt2_decoder_peak_mb": decoder_peak_mb,
        "gpt2_token_step_mean_ms": statistics.mean(decoder_result["token_step_ms"]) if decoder_result["token_step_ms"] else 0.0,
        "gpt2_token_step_max_ms": max(decoder_result["token_step_ms"]) if decoder_result["token_step_ms"] else 0.0,
        "max_memory_allocated_mb": profiler.max_memory_allocated_mb,
    }


def get_env(device: torch.device) -> dict[str, object]:
    props = torch.cuda.get_device_properties(device)
    def package_version(name: str) -> str | None:
        try:
            return metadata.version(name)
        except metadata.PackageNotFoundError:
            return None

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "timm": package_version("timm"),
        "transformers": package_version("transformers"),
        "cupy": package_version("cupy-cuda12x") or package_version("cupy"),
        "triton": package_version("triton"),
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": props.total_memory / MB,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def get_git_metadata() -> dict[str, object]:
    def run_git(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    try:
        status = run_git("status", "--porcelain")
        return {
            "commit": run_git("rev-parse", "HEAD"),
            "branch": run_git("branch", "--show-current"),
            "dirty": bool(status),
            "changed_files": status.splitlines(),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: unable to read git metadata"}


def get_model_optimization_status(model) -> dict[str, object]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "get_optimization_status"):
        return {"error": "encoder does not expose optimization status"}
    return encoder.get_optimization_status()


def get_model_optimization_issues(model) -> list[str]:
    encoder = getattr(model, "encoder", None)
    if encoder is None or not hasattr(encoder, "get_optimization_issues"):
        return ["encoder does not expose optimization issue reporting"]
    return encoder.get_optimization_issues()


def build_reproducibility_metadata(
    config: InferenceConfig,
    args: argparse.Namespace,
    batch_inputs: dict[int, list[Path]],
) -> dict[str, object]:
    ckpt = Path(config.ckpt)
    checkpoint = {
        "path": str(ckpt.resolve()),
        "exists": ckpt.is_file(),
        "bytes": ckpt.stat().st_size if ckpt.is_file() else None,
        "sha256": sha256_file(ckpt) if ckpt.is_file() else None,
    }
    return {
        "protocol_version": 2,
        "git": get_git_metadata(),
        "checkpoint": checkpoint,
        "inference_config": asdict(config),
        "benchmark_args": vars(args),
        "decode_protocol": {
            "algorithm": "deterministic_greedy_with_kv_cache",
            "prompt": args.prompt,
            "min_new_tokens": args.min_new_tokens,
            "max_new_tokens": args.max_new_tokens,
            "eos_masked_before_minimum": True,
            "per_token_synchronization": bool(args.profile_token_steps),
            "note": "This fixed benchmark protocol is not the multi-preset production beam/sampling policy.",
        },
        "input_policy": "synthetic_repeated" if args.allow_repeated_input else "distinct_videos",
        "input_batches": {
            str(batch_size): [
                {"video_id": path.name, "frames_dir": str(path.resolve())} for path in frame_dirs
            ]
            for batch_size, frame_dirs in batch_inputs.items()
        },
    }


def print_env(env: dict[str, object]):
    print("=== Environment ===")
    print(f"torch: {env['torch']}")
    print(f"torch.version.cuda: {env['torch_cuda']}")
    print(f"device: {env['device']}")
    print(f"compute capability: {env['compute_capability']}")
    print(f"total_vram_mb: {env['total_vram_mb']:.1f}")
    print()


def summarize(name: str, samples: list[float], peaks: list[float] | None = None, unit: str = "ms"):
    if not samples:
        return
    mean_val = statistics.mean(samples)
    std_val = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    p99_val = percentile(samples, 0.99)
    line = f"{name:<24} mean={mean_val:8.3f} {unit}  std={std_val:8.3f} {unit}  p99={p99_val:8.3f} {unit}"
    if peaks:
        line += f"  peak_mem={max(peaks):8.1f} MB"
    print(line)


def build_summary(
    profiler: BenchmarkProfiler,
    iteration_rows: list[dict[str, object]],
    batch_size: int,
    status: str = "ok",
    optimization_status: dict[str, object] | None = None,
    optimization_issues: list[str] | None = None,
) -> dict[str, object]:
    end_to_end = stats_dict(profiler.total_latency_ms)
    throughput = throughput_stats_dict(profiler.throughput_samples)
    latency_mean_ms = end_to_end["mean_ms"]
    throughput_from_mean = None
    if latency_mean_ms:
        throughput_from_mean = batch_size / (latency_mean_ms / 1000.0)

    return {
        "status": status,
        "batch_size": batch_size,
        "Preprocess_Latency": stats_dict(profiler.preprocess_host_ms),
        "Preprocess_CUDA_Latency": stats_dict(profiler.stage_samples["Preprocessing"]),
        "ViT_Latency": stats_dict(profiler.stage_samples["ViT_Encoder"]),
        "Cross_Modal_Alignment": stats_dict(profiler.stage_samples["Cross_Modal_Alignment"]),
        "GPT2_Latency": stats_dict(profiler.stage_samples["GPT2_Decoder_Step"]),
        "GPT2_token_step": stats_dict(profiler.token_step_ms),
        "End_to_end_Latency": end_to_end,
        "Throughput": {
            **throughput,
            "from_mean_latency_samples_per_s": throughput_from_mean,
        },
        "generated_tokens": {
            "count": len(profiler.generated_lengths),
            "mean": statistics.mean(profiler.generated_lengths) if profiler.generated_lengths else None,
            "max": max(profiler.generated_lengths) if profiler.generated_lengths else None,
        },
        "peak_memory_mb": {
            "max_memory_allocated_mb": profiler.max_memory_allocated_mb,
            **{stage: max(values) if values else None for stage, values in profiler.stage_peak_mem_mb.items()},
        },
        "caption_preview": profiler.last_texts[0] if profiler.last_texts else "",
        "input_video_ids": list(profiler.last_video_ids),
        "nonempty_caption_count": sum(bool(text.strip()) for text in profiler.last_texts),
        "output_validation": "passed" if iteration_rows else "not_run",
        "optimization_status": optimization_status or {},
        "optimization_issues": optimization_issues or [],
        "iterations": len(iteration_rows),
    }


def ensure_parent(path_str: str | None):
    if not path_str:
        return
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def export_iteration_csv(path_str: str, iteration_rows: list[dict[str, object]]):
    ensure_parent(path_str)
    fieldnames = [
        "iter_index",
        "batch_size",
        "input_video_ids",
        "iteration_ms",
        "throughput_samples_per_s",
        "caption_preview",
        "generated_tokens_mean",
        "preprocess_cuda_ms",
        "preprocess_host_ms",
        "preprocess_peak_mb",
        "vit_encoder_ms",
        "vit_encoder_peak_mb",
        "cross_modal_alignment_ms",
        "cross_modal_alignment_peak_mb",
        "gpt2_decoder_ms",
        "gpt2_decoder_peak_mb",
        "gpt2_token_step_mean_ms",
        "gpt2_token_step_max_ms",
        "max_memory_allocated_mb",
    ]
    with open(path_str, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(iteration_rows, start=1):
            csv_row = {key: row.get(key) for key in fieldnames if key != "iter_index"}
            if isinstance(csv_row.get("input_video_ids"), list):
                csv_row["input_video_ids"] = "|".join(csv_row["input_video_ids"])
            writer.writerow({"iter_index": index, **csv_row})


def export_bs_comparison_csv(path_str: str, comparison_rows: list[dict[str, object]]):
    ensure_parent(path_str)
    fieldnames = [
        "batch_size",
        "status",
        "warmup",
        "iters",
        "end_to_end_mean_ms",
        "end_to_end_std_ms",
        "preprocess_mean_ms",
        "preprocess_std_ms",
        "vit_mean_ms",
        "vit_std_ms",
        "gpt2_mean_ms",
        "gpt2_std_ms",
        "throughput_mean_samples_per_s",
        "throughput_std_samples_per_s",
        "throughput_from_mean_latency_samples_per_s",
        "max_memory_allocated_mb",
    ]
    with open(path_str, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)


def export_summary_json(path_str: str, payload: dict[str, object]):
    ensure_parent(path_str)
    with open(path_str, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def build_config(args: argparse.Namespace) -> InferenceConfig:
    return InferenceConfig(
        ckpt=args.ckpt,
        vit_name=args.vit_name,
        gpt2_name=args.gpt2_name,
        prefix_len=args.prefix_len,
        num_frames=args.num_frames,
        image_size=args.image_size,
        ln_scale=args.ln_scale,
        in_weight=args.in_weight,
        device=args.device,
        backend="torch",
        vit_opt=ViTOptimizeConfig(
            enable_fp16=args.vit_fp16,
            enable_attention_fastpath=args.vit_attention_fastpath,
            prefer_channels_last=args.vit_channels_last,
            enable_torch_compile=args.vit_compile,
            torch_compile_mode=args.vit_compile_mode,
            enable_mlp_bias_gelu_fusion=args.vit_mlp_gelu_approximation,
            enable_residual_layernorm_fusion=args.vit_residual_layernorm_fusion,
            enable_inplace_residual_add_fusion=args.vit_inplace_residual_add,
            enable_cupy_fused_pool=args.vit_cupy_fused_pool,
            cupy_pool_force_fp16=args.vit_cupy_pool_force_fp16,
            enable_cuda_graph=args.vit_cuda_graph,
        ),
        use_cupy_prefix_projector=args.cupy_prefix_projector,
        cupy_prefix_force_fp16=args.cupy_prefix_force_fp16,
    )


def parse_batch_sizes(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not parts:
        raise ValueError("batch sizes cannot be empty")
    batch_sizes = [int(part) for part in parts]
    if any(bs <= 0 for bs in batch_sizes):
        raise ValueError("batch sizes must be positive integers")
    return batch_sizes


def default_output_path(path_str: str, default_name: str) -> str:
    return path_str if path_str else default_name


def add_boolean_argument(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str):
    parser.add_argument(
        name,
        action=argparse.BooleanOptionalAction,
        default=default,
        help=f"{help_text} (default: {default})",
    )


def parse_args():
    parser = argparse.ArgumentParser("Benchmark ViT + GPT-2 video caption inference across batch sizes")
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="A single video frame directory or a split-level directory containing one directory per video",
    )
    parser.add_argument("--ckpt", default=str(backend_config.CKPT_PATH), help="Path to caption model checkpoint")
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0")
    parser.add_argument("--vit-name", default=backend_config.VIT_NAME)
    parser.add_argument("--gpt2-name", default=backend_config.GPT2_NAME)
    parser.add_argument("--prefix-len", type=int, default=backend_config.PREFIX_LEN)
    parser.add_argument("--num-frames", type=int, default=backend_config.NUM_FRAMES)
    parser.add_argument("--image-size", type=int, default=backend_config.IMAGE_SIZE)
    parser.add_argument("--ln-scale", type=float, default=backend_config.LN_SCALE)
    parser.add_argument("--in-weight", type=float, default=backend_config.IN_WEIGHT)
    parser.add_argument("--prompt", default=backend_config.PROMPT3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument(
        "--profile-token-steps",
        action="store_true",
        help="Synchronize and time every generated token; disabled by default because it perturbs end-to-end latency",
    )
    parser.add_argument("--batch-sizes", default="1", help="Comma-separated batch sizes, e.g. 1,2,4,8,12,16")
    parser.add_argument(
        "--allow-repeated-input",
        action="store_true",
        help="Allow repeated videos when the requested batch size exceeds available distinct videos; output is labelled synthetic",
    )
    parser.add_argument(
        "--allow-inactive-optimizations",
        action="store_true",
        help="Do not fail when a requested optimization is proven inactive; issues remain recorded in JSON",
    )
    add_boolean_argument(parser, "--vit-fp16", backend_config.VIT_ENABLE_FP16, "Enable ViT CUDA autocast FP16")
    add_boolean_argument(
        parser,
        "--vit-attention-fastpath",
        backend_config.VIT_ENABLE_ATTENTION_FASTPATH,
        "Set timm fused attention/SDPA fastpath explicitly",
    )
    add_boolean_argument(
        parser,
        "--vit-channels-last",
        backend_config.VIT_PREFER_CHANNELS_LAST,
        "Use channels-last input memory format",
    )
    add_boolean_argument(parser, "--vit-compile", backend_config.VIT_ENABLE_TORCH_COMPILE, "Enable torch.compile")
    parser.add_argument("--vit-compile-mode", default=backend_config.VIT_TORCH_COMPILE_MODE)
    add_boolean_argument(
        parser,
        "--vit-mlp-gelu-approximation",
        backend_config.VIT_ENABLE_MLP_BIAS_GELU_FUSION,
        "Use tanh GELU approximation (not a guaranteed fused kernel)",
    )
    add_boolean_argument(
        parser,
        "--vit-residual-layernorm-fusion",
        backend_config.VIT_ENABLE_RESIDUAL_LAYERNORM_FUSION,
        "Enable backbone-provided residual/LayerNorm fusion flags when supported",
    )
    add_boolean_argument(
        parser,
        "--vit-inplace-residual-add",
        backend_config.VIT_ENABLE_INPLACE_RESIDUAL_ADD_FUSION,
        "Patch inference blocks to use in-place residual adds",
    )
    add_boolean_argument(
        parser,
        "--vit-cupy-fused-pool",
        backend_config.VIT_ENABLE_CUPY_FUSED_POOL,
        "Enable CuPy fused spatial-temporal pooling",
    )
    add_boolean_argument(
        parser,
        "--vit-cupy-pool-force-fp16",
        backend_config.VIT_CUPY_POOL_FORCE_FP16,
        "Force FP16 inside the CuPy pooling kernel",
    )
    add_boolean_argument(parser, "--vit-cuda-graph", backend_config.VIT_ENABLE_CUDA_GRAPH, "Enable ViT CUDA Graph")
    add_boolean_argument(
        parser,
        "--cupy-prefix-projector",
        backend_config.USE_CUPY_PREFIX_PROJECTOR,
        "Enable the CuPy prefix projector",
    )
    add_boolean_argument(
        parser,
        "--cupy-prefix-force-fp16",
        backend_config.CUPY_PREFIX_FORCE_FP16,
        "Force FP16 inside the CuPy prefix projector",
    )
    parser.add_argument("--export-json", default="", help="Summary JSON output path")
    parser.add_argument("--export-csv", default="", help="CSV output path")
    return parser.parse_args()


def benchmark_one_batch_size(
    model,
    config: InferenceConfig,
    args: argparse.Namespace,
    batch_size: int,
    frame_dirs: list[Path],
):
    profiler = BenchmarkProfiler(args.device)
    iteration_rows: list[dict[str, object]] = []

    print()
    print(f"=== Batch Size {batch_size} ===")
    print(f"inputs: {[path.name for path in frame_dirs]}")
    print("Warm-up:")
    for index in range(args.warmup):
        run_one_iteration(
            model,
            config,
            frame_dirs,
            args.prompt,
            args.max_new_tokens,
            args.min_new_tokens,
            args.profile_token_steps,
            profiler,
        )
        print(f"warmup {index + 1}/{args.warmup} done")

    optimization_issues = get_model_optimization_issues(model)
    if optimization_issues and not args.allow_inactive_optimizations:
        formatted = "; ".join(optimization_issues)
        raise RuntimeError(
            "requested optimizations are inactive; benchmark aborted to prevent an invalid ablation. "
            f"Disable the unsupported switches explicitly or pass --allow-inactive-optimizations. {formatted}"
        )

    profiler.clear_measurements()

    print("Measured Runs:")
    for index in range(args.iters):
        row = run_one_iteration(
            model,
            config,
            frame_dirs,
            args.prompt,
            args.max_new_tokens,
            args.min_new_tokens,
            args.profile_token_steps,
            profiler,
        )
        iteration_rows.append(row)
        print(
            f"iter {index + 1:02d}/{args.iters} bs={batch_size} total={row['iteration_ms']:.3f} ms  "
            f"vit={row['vit_encoder_ms']:.3f} ms  gpt2={row['gpt2_decoder_ms']:.3f} ms  "
            f"throughput={row['throughput_samples_per_s']:.3f} samples/s"
        )

    optimization_status = get_model_optimization_status(model)
    optimization_issues = get_model_optimization_issues(model)
    return iteration_rows, build_summary(
        profiler,
        iteration_rows,
        batch_size=batch_size,
        optimization_status=optimization_status,
        optimization_issues=optimization_issues,
    )


def build_comparison_row(summary: dict[str, object], warmup: int, iters: int) -> dict[str, object]:
    if summary["status"] != "ok":
        return {
            "batch_size": summary["batch_size"],
            "status": summary["status"],
            "warmup": warmup,
            "iters": iters,
            "end_to_end_mean_ms": None,
            "end_to_end_std_ms": None,
            "preprocess_mean_ms": None,
            "preprocess_std_ms": None,
            "vit_mean_ms": None,
            "vit_std_ms": None,
            "gpt2_mean_ms": None,
            "gpt2_std_ms": None,
            "throughput_mean_samples_per_s": None,
            "throughput_std_samples_per_s": None,
            "throughput_from_mean_latency_samples_per_s": None,
            "max_memory_allocated_mb": None,
        }

    return {
        "batch_size": summary["batch_size"],
        "status": summary["status"],
        "warmup": warmup,
        "iters": iters,
        "end_to_end_mean_ms": summary["End_to_end_Latency"]["mean_ms"],
        "end_to_end_std_ms": summary["End_to_end_Latency"]["std_ms"],
        "preprocess_mean_ms": summary["Preprocess_Latency"]["mean_ms"],
        "preprocess_std_ms": summary["Preprocess_Latency"]["std_ms"],
        "vit_mean_ms": summary["ViT_Latency"]["mean_ms"],
        "vit_std_ms": summary["ViT_Latency"]["std_ms"],
        "gpt2_mean_ms": summary["GPT2_Latency"]["mean_ms"],
        "gpt2_std_ms": summary["GPT2_Latency"]["std_ms"],
        "throughput_mean_samples_per_s": summary["Throughput"]["mean_samples_per_s"],
        "throughput_std_samples_per_s": summary["Throughput"]["std_samples_per_s"],
        "throughput_from_mean_latency_samples_per_s": summary["Throughput"]["from_mean_latency_samples_per_s"],
        "max_memory_allocated_mb": summary["peak_memory_mb"]["max_memory_allocated_mb"],
    }


def build_oom_summary(batch_size: int, error: Exception) -> dict[str, object]:
    return {
        "status": "OOM",
        "batch_size": batch_size,
        "error": str(error),
        "Preprocess_Latency": stats_dict([]),
        "Preprocess_CUDA_Latency": stats_dict([]),
        "ViT_Latency": stats_dict([]),
        "Cross_Modal_Alignment": stats_dict([]),
        "GPT2_Latency": stats_dict([]),
        "GPT2_token_step": stats_dict([]),
        "End_to_end_Latency": stats_dict([]),
        "Throughput": throughput_stats_dict([]),
        "generated_tokens": {"count": 0, "mean": None, "max": None},
        "peak_memory_mb": {"max_memory_allocated_mb": None},
        "caption_preview": "",
        "iterations": 0,
    }


def main():
    args = parse_args()
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    if args.warmup < 0 or args.iters <= 0:
        raise ValueError("warmup must be >= 0 and iters must be > 0")
    if args.min_new_tokens < 0 or args.min_new_tokens > args.max_new_tokens:
        raise ValueError("require 0 <= min_new_tokens <= max_new_tokens")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark baseline.")

    assert_core_runtime_ready(device=args.device, require_cupy=False)
    config = build_config(args)
    batch_inputs = {
        batch_size: resolve_batch_frame_dirs(
            args.frames_dir,
            batch_size,
            allow_repeated_input=args.allow_repeated_input,
        )
        for batch_size in batch_sizes
    }
    model = load_caption_model(config)
    initial_optimization_issues = get_model_optimization_issues(model)
    if initial_optimization_issues and not args.allow_inactive_optimizations:
        raise RuntimeError(
            "requested optimizations are known to be inactive before the benchmark starts. "
            "Disable them explicitly or pass --allow-inactive-optimizations. "
            + "; ".join(initial_optimization_issues)
        )
    env = get_env(torch.device(args.device))

    print_env(env)
    print("=== Benchmark Config ===")
    print(f"frames_dir: {args.frames_dir}")
    print(f"ckpt: {args.ckpt}")
    print(f"prompt: {args.prompt!r}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"min_new_tokens: {args.min_new_tokens}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"input_policy: {'synthetic_repeated' if args.allow_repeated_input else 'distinct_videos'}")
    print(f"requested_vit_optimizations: {asdict(config.vit_opt)}")
    print()

    single_batch_mode = len(batch_sizes) == 1
    export_csv = args.export_csv
    export_json = args.export_json
    if not single_batch_mode:
        export_csv = default_output_path(export_csv, "benchmark_bs_comparison.csv")
        export_json = default_output_path(export_json, "benchmark_bs_summary.json")
    reproducibility = build_reproducibility_metadata(config, args, batch_inputs) if export_json else None

    all_iteration_rows: dict[int, list[dict[str, object]]] = {}
    all_summaries: dict[int, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        try:
            iteration_rows, summary = benchmark_one_batch_size(
                model,
                config,
                args,
                batch_size,
                batch_inputs[batch_size],
            )
            all_iteration_rows[batch_size] = iteration_rows
            all_summaries[batch_size] = summary
            comparison_rows.append(build_comparison_row(summary, args.warmup, args.iters))
            print(
                f"[BS={batch_size}] mean_end_to_end={summary['End_to_end_Latency']['mean_ms']:.3f} ms  "
                f"mean_vit={summary['ViT_Latency']['mean_ms']:.3f} ms  "
                f"mean_gpt2={summary['GPT2_Latency']['mean_ms']:.3f} ms  "
                f"throughput={summary['Throughput']['from_mean_latency_samples_per_s']:.3f} samples/s  "
                f"max_mem={summary['peak_memory_mb']['max_memory_allocated_mb']:.1f} MB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            gc.collect()
            summary = build_oom_summary(batch_size, exc)
            all_summaries[batch_size] = summary
            comparison_rows.append(build_comparison_row(summary, args.warmup, args.iters))
            print()
            print(f"[BS={batch_size}] OOM encountered, stopping larger batch sizes.")
            break

    if single_batch_mode:
        batch_size = batch_sizes[0]
        iteration_rows = all_iteration_rows.get(batch_size, [])
        summary = all_summaries.get(batch_size)
        if summary and summary["status"] == "ok":
            print()
            print("=== Summary ===")
            summarize("Preprocessing(cuda)", [row["preprocess_cuda_ms"] for row in iteration_rows])
            summarize("Preprocessing(host)", [row["preprocess_host_ms"] for row in iteration_rows])
            summarize("ViT_Encoder", [row["vit_encoder_ms"] for row in iteration_rows])
            summarize("Cross_Modal_Alignment", [row["cross_modal_alignment_ms"] for row in iteration_rows])
            summarize("GPT2_Decoder_Step", [row["gpt2_decoder_ms"] for row in iteration_rows])
            summarize("End-to-end(iter)", [row["iteration_ms"] for row in iteration_rows])
            summarize("Throughput", [row["throughput_samples_per_s"] for row in iteration_rows], unit="samples/s")

        if export_csv and iteration_rows:
            export_iteration_csv(export_csv, iteration_rows)
            print(f"iteration CSV exported to: {export_csv}")
        if export_json and summary:
            export_summary_json(
                export_json,
                {
                    "env": env,
                    "reproducibility": {
                        **(reproducibility or {}),
                        "effective_optimizations": get_model_optimization_status(model),
                        "optimization_issues": get_model_optimization_issues(model),
                    },
                    "summary": summary,
                    "iterations": iteration_rows,
                },
            )
            print(f"summary JSON exported to: {export_json}")
        return

    if export_csv:
        export_bs_comparison_csv(export_csv, comparison_rows)
        print(f"batch-size comparison CSV exported to: {export_csv}")
    if export_json:
        export_summary_json(
            export_json,
            {
                "env": env,
                "reproducibility": {
                    **(reproducibility or {}),
                    "effective_optimizations": get_model_optimization_status(model),
                    "optimization_issues": get_model_optimization_issues(model),
                },
                "comparison": comparison_rows,
                "per_batch_summary": {str(bs): summary for bs, summary in all_summaries.items()},
                "per_batch_iterations": {str(bs): rows for bs, rows in all_iteration_rows.items()},
            },
        )
        print(f"batch-size summary JSON exported to: {export_json}")


if __name__ == "__main__":
    main()









