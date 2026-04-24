from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import statistics
import sys
import time
from collections import defaultdict
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
from core.preprocessing.frame_loader import load_video_tensor

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

    def preprocess_stage(self, frames_dir: str, num_frames: int, image_size: int, batch_size: int):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(self.device)
        host_t0 = time.perf_counter()
        with NvtxRange("Preprocessing"):
            start_event.record()
            video = load_video_tensor(
                frames_dir,
                num_frames=num_frames,
                image_size=image_size,
                device=str(self.device),
            )
            if batch_size > 1:
                video = video.repeat(batch_size, 1, 1, 1, 1)
            end_event.record()
        torch.cuda.synchronize(self.device)
        host_ms = (time.perf_counter() - host_t0) * 1000.0
        cuda_ms = start_event.elapsed_time(end_event)
        peak_mem_mb = self.update_global_peak()
        self.preprocess_host_ms.append(host_ms)
        self.stage_samples["Preprocessing"].append(cuda_ms)
        self.stage_peak_mem_mb["Preprocessing"].append(peak_mem_mb)
        return video, cuda_ms, host_ms, peak_mem_mb


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
        return {"count": 0, "mean_ms": None, "std_ms": None, "p99_ms": None, "max_ms": None, "min_ms": None}
    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "std_ms": statistics.pstdev(values) if len(values) > 1 else 0.0,
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


@torch.inference_mode()
def run_decoder_steps(model, prefix_embeds: torch.Tensor, prompt: str, max_new_tokens: int, profiler: BenchmarkProfiler):
    decoder = model.decoder
    tokenizer = decoder.tokenizer
    gpt2 = decoder.model
    device = prefix_embeds.device
    batch_size = prefix_embeds.shape[0]

    if prompt:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.eos_token_id
        prompt_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

    if prompt_ids.shape[0] == 1 and batch_size > 1:
        prompt_ids = prompt_ids.expand(batch_size, -1)

    prompt_embeds = gpt2.transformer.wte(prompt_ids)
    full_inputs = torch.cat([prefix_embeds, prompt_embeds], dim=1)
    running_attention_mask = torch.ones(full_inputs.shape[:2], dtype=torch.long, device=device)

    generated_tokens = [[] for _ in range(batch_size)]
    step_samples: list[float] = []
    past_key_values = None
    next_input_embeds = full_inputs
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with NvtxRange("GPT2_Decoder_Step"):
        for step_idx in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            step_name = f"GPT2_Decoder_Step/token_{step_idx:02d}"
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

            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1)
            eos_token_id = tokenizer.eos_token_id
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
    profiler.last_texts = texts
    return {
        "texts": texts,
        "token_count": [len(tokens) for tokens in generated_tokens],
        "token_step_ms": step_samples,
    }


@torch.inference_mode()
def run_one_iteration(
    model,
    config: InferenceConfig,
    frames_dir: str,
    prompt: str,
    max_new_tokens: int,
    profiler: BenchmarkProfiler,
    batch_size: int,
):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(profiler.device)
    iter_t0 = time.perf_counter()

    video, preprocess_cuda_ms, preprocess_host_ms, preprocess_peak_mb = profiler.preprocess_stage(
        frames_dir=frames_dir,
        num_frames=config.num_frames,
        image_size=config.image_size,
        batch_size=batch_size,
    )

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
        lambda: run_decoder_steps(model, prefix_embeds, prompt, max_new_tokens, profiler),
    )

    iter_ms = (time.perf_counter() - iter_t0) * 1000.0
    throughput = batch_size / (iter_ms / 1000.0)
    profiler.total_latency_ms.append(iter_ms)
    profiler.throughput_samples.append(throughput)
    profiler.update_global_peak()
    return {
        "batch_size": batch_size,
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
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(device),
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_mb": props.total_memory / MB,
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


def build_summary(profiler: BenchmarkProfiler, iteration_rows: list[dict[str, object]], batch_size: int, status: str = "ok") -> dict[str, object]:
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
            enable_fp16=backend_config.VIT_ENABLE_FP16,
            enable_attention_fastpath=backend_config.VIT_ENABLE_ATTENTION_FASTPATH,
            prefer_channels_last=backend_config.VIT_PREFER_CHANNELS_LAST,
            enable_torch_compile=backend_config.VIT_ENABLE_TORCH_COMPILE,
            torch_compile_mode=backend_config.VIT_TORCH_COMPILE_MODE,
            enable_mlp_bias_gelu_fusion=backend_config.VIT_ENABLE_MLP_BIAS_GELU_FUSION,
            enable_residual_layernorm_fusion=backend_config.VIT_ENABLE_RESIDUAL_LAYERNORM_FUSION,
            enable_inplace_residual_add_fusion=backend_config.VIT_ENABLE_INPLACE_RESIDUAL_ADD_FUSION,
            enable_cupy_fused_pool=backend_config.VIT_ENABLE_CUPY_FUSED_POOL,
            cupy_pool_force_fp16=backend_config.VIT_CUPY_POOL_FORCE_FP16,
        ),
        use_cupy_prefix_projector=backend_config.USE_CUPY_PREFIX_PROJECTOR,
        cupy_prefix_force_fp16=backend_config.CUPY_PREFIX_FORCE_FP16,
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


def parse_args():
    parser = argparse.ArgumentParser("Benchmark ViT + GPT-2 video caption inference across batch sizes")
    parser.add_argument("--frames-dir", required=True, help="Directory that contains frame_*.jpg")
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
    parser.add_argument("--batch-sizes", default="1", help="Comma-separated batch sizes, e.g. 1,2,4,8,12,16")
    parser.add_argument("--export-json", default="", help="Summary JSON output path")
    parser.add_argument("--export-csv", default="", help="CSV output path")
    return parser.parse_args()


def benchmark_one_batch_size(model, config: InferenceConfig, args: argparse.Namespace, batch_size: int):
    profiler = BenchmarkProfiler(args.device)
    iteration_rows: list[dict[str, object]] = []

    print()
    print(f"=== Batch Size {batch_size} ===")
    print("Warm-up:")
    for index in range(args.warmup):
        run_one_iteration(model, config, args.frames_dir, args.prompt, args.max_new_tokens, profiler, batch_size)
        print(f"warmup {index + 1}/{args.warmup} done")

    profiler.clear_measurements()

    print("Measured Runs:")
    for index in range(args.iters):
        row = run_one_iteration(model, config, args.frames_dir, args.prompt, args.max_new_tokens, profiler, batch_size)
        iteration_rows.append(row)
        print(
            f"iter {index + 1:02d}/{args.iters} bs={batch_size} total={row['iteration_ms']:.3f} ms  "
            f"vit={row['vit_encoder_ms']:.3f} ms  gpt2={row['gpt2_decoder_ms']:.3f} ms  "
            f"throughput={row['throughput_samples_per_s']:.3f} samples/s"
        )

    return iteration_rows, build_summary(profiler, iteration_rows, batch_size=batch_size)


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
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark baseline.")

    assert_core_runtime_ready(device=args.device, require_cupy=False)
    config = build_config(args)
    model = load_caption_model(config)
    env = get_env(torch.device(args.device))

    print_env(env)
    print("=== Benchmark Config ===")
    print(f"frames_dir: {args.frames_dir}")
    print(f"ckpt: {args.ckpt}")
    print(f"prompt: {args.prompt!r}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print()

    single_batch_mode = len(batch_sizes) == 1
    export_csv = args.export_csv
    export_json = args.export_json
    if not single_batch_mode:
        export_csv = default_output_path(export_csv, "benchmark_bs_comparison.csv")
        export_json = default_output_path(export_json, "benchmark_bs_summary.json")

    all_iteration_rows: dict[int, list[dict[str, object]]] = {}
    all_summaries: dict[int, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        try:
            iteration_rows, summary = benchmark_one_batch_size(model, config, args, batch_size)
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
                    "config": {
                        "frames_dir": args.frames_dir,
                        "ckpt": args.ckpt,
                        "device": args.device,
                        "prompt": args.prompt,
                        "warmup": args.warmup,
                        "iters": args.iters,
                        "max_new_tokens": args.max_new_tokens,
                        "num_frames": args.num_frames,
                        "image_size": args.image_size,
                        "prefix_len": args.prefix_len,
                        "ln_scale": args.ln_scale,
                        "in_weight": args.in_weight,
                        "batch_sizes": batch_sizes,
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
                "config": {
                    "frames_dir": args.frames_dir,
                    "ckpt": args.ckpt,
                    "device": args.device,
                    "prompt": args.prompt,
                    "warmup": args.warmup,
                    "iters": args.iters,
                    "max_new_tokens": args.max_new_tokens,
                    "num_frames": args.num_frames,
                    "image_size": args.image_size,
                    "prefix_len": args.prefix_len,
                    "ln_scale": args.ln_scale,
                    "in_weight": args.in_weight,
                    "batch_sizes": batch_sizes,
                },
                "comparison": comparison_rows,
                "per_batch_summary": {str(bs): summary for bs, summary in all_summaries.items()},
                "per_batch_iterations": {str(bs): rows for bs, rows in all_iteration_rows.items()},
            },
        )
        print(f"batch-size summary JSON exported to: {export_json}")


if __name__ == "__main__":
    main()









