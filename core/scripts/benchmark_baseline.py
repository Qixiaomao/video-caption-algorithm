from __future__ import annotations

import argparse
import csv
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
from core.config import InferenceConfig
from core.env import assert_core_runtime_ready
from core.models.model_loader import load_caption_model
from core.preprocessing.frame_loader import load_video_tensor

MB = 1024.0 * 1024.0
STAGES = ["Preprocessing", "ViT_Encoder", "Cross_Modal_Alignment", "GPT2_Decoder_Step"]


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
        self.last_text: str = ""

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
        peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / MB
        self.stage_samples[name].append(latency_ms)
        self.stage_peak_mem_mb[name].append(peak_mem_mb)
        return out, latency_ms, peak_mem_mb

    def preprocess_stage(self, frames_dir: str, num_frames: int, image_size: int):
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
            end_event.record()
        torch.cuda.synchronize(self.device)
        host_ms = (time.perf_counter() - host_t0) * 1000.0
        cuda_ms = start_event.elapsed_time(end_event)
        peak_mem_mb = torch.cuda.max_memory_allocated(self.device) / MB
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


@torch.inference_mode()
def run_decoder_steps(
    model,
    prefix_embeds: torch.Tensor,
    prompt: str,
    max_new_tokens: int,
    profiler: BenchmarkProfiler,
):
    decoder = model.decoder
    tokenizer = decoder.tokenizer
    gpt2 = decoder.model
    device = prefix_embeds.device

    if prompt:
        prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    else:
        bos_token_id = tokenizer.bos_token_id
        if bos_token_id is None:
            bos_token_id = tokenizer.eos_token_id
        prompt_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)

    prompt_embeds = gpt2.transformer.wte(prompt_ids)
    full_inputs = torch.cat([prefix_embeds, prompt_embeds], dim=1)
    running_attention_mask = torch.ones(full_inputs.shape[:2], dtype=torch.long, device=device)

    generated_tokens: list[int] = []
    step_samples: list[float] = []
    past_key_values = None
    next_input_embeds = full_inputs

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
            token_id = int(next_token.item())
            generated_tokens.append(token_id)
            past_key_values = outputs.past_key_values

            if token_id == tokenizer.eos_token_id:
                break

            next_input_embeds = gpt2.transformer.wte(next_token).unsqueeze(1)
            running_attention_mask = torch.cat(
                [
                    running_attention_mask,
                    torch.ones((next_input_embeds.shape[0], 1), dtype=torch.long, device=device),
                ],
                dim=1,
            )

    profiler.generated_lengths.append(len(generated_tokens))
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    profiler.last_text = text
    return {"text": text, "token_count": len(generated_tokens), "token_step_ms": step_samples}


@torch.inference_mode()
def run_one_iteration(
    model,
    config: InferenceConfig,
    frames_dir: str,
    prompt: str,
    max_new_tokens: int,
    profiler: BenchmarkProfiler,
):
    torch.cuda.reset_peak_memory_stats(profiler.device)
    iter_t0 = time.perf_counter()

    video, preprocess_cuda_ms, preprocess_host_ms, preprocess_peak_mb = profiler.preprocess_stage(
        frames_dir=frames_dir,
        num_frames=config.num_frames,
        image_size=config.image_size,
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
    profiler.total_latency_ms.append(iter_ms)
    return {
        "iteration_ms": iter_ms,
        "caption": decoder_result["text"],
        "generated_tokens": decoder_result["token_count"],
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


def summarize(name: str, samples: list[float], peaks: list[float] | None = None):
    if not samples:
        return
    mean_ms = statistics.mean(samples)
    std_ms = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    p99_ms = percentile(samples, 0.99)
    line = f"{name:<24} mean={mean_ms:8.3f} ms  std={std_ms:8.3f} ms  p99={p99_ms:8.3f} ms"
    if peaks:
        line += f"  peak_mem={max(peaks):8.1f} MB"
    print(line)


def build_summary(profiler: BenchmarkProfiler, iteration_rows: list[dict[str, object]]) -> dict[str, object]:
    summary = {
        "Preprocessing_cuda": stats_dict(profiler.stage_samples["Preprocessing"]),
        "Preprocessing_host": stats_dict(profiler.preprocess_host_ms),
        "ViT_Encoder": stats_dict(profiler.stage_samples["ViT_Encoder"]),
        "Cross_Modal_Alignment": stats_dict(profiler.stage_samples["Cross_Modal_Alignment"]),
        "GPT2_Decoder_Step": stats_dict(profiler.stage_samples["GPT2_Decoder_Step"]),
        "GPT2_token_step": stats_dict(profiler.token_step_ms),
        "End_to_end": stats_dict(profiler.total_latency_ms),
        "generated_tokens": {
            "count": len(profiler.generated_lengths),
            "mean": statistics.mean(profiler.generated_lengths) if profiler.generated_lengths else None,
            "max": max(profiler.generated_lengths) if profiler.generated_lengths else None,
        },
        "peak_memory_mb": {stage: max(values) if values else None for stage, values in profiler.stage_peak_mem_mb.items()},
        "last_caption": profiler.last_text,
        "iterations": len(iteration_rows),
    }
    return summary


def ensure_parent(path_str: str | None):
    if not path_str:
        return
    Path(path_str).parent.mkdir(parents=True, exist_ok=True)


def export_iteration_csv(path_str: str, iteration_rows: list[dict[str, object]]):
    ensure_parent(path_str)
    fieldnames = [
        "iter_index",
        "iteration_ms",
        "caption",
        "generated_tokens",
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
    ]
    with open(path_str, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(iteration_rows, start=1):
            writer.writerow({"iter_index": index, **row})


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
    )


def parse_args():
    parser = argparse.ArgumentParser("NVTX/Nsight baseline for ViT + GPT-2 video caption inference")
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
    parser.add_argument("--export-json", default="", help="Optional summary JSON output path")
    parser.add_argument("--export-csv", default="", help="Optional iteration CSV output path")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark baseline.")

    assert_core_runtime_ready(device=args.device, require_cupy=False)
    config = build_config(args)
    model = load_caption_model(config)
    device = torch.device(args.device)
    profiler = BenchmarkProfiler(args.device)
    env = get_env(device)
    iteration_rows: list[dict[str, object]] = []

    print_env(env)
    print("=== Benchmark Config ===")
    print(f"frames_dir: {args.frames_dir}")
    print(f"ckpt: {args.ckpt}")
    print(f"prompt: {args.prompt!r}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    if args.export_json:
        print(f"export_json: {args.export_json}")
    if args.export_csv:
        print(f"export_csv:  {args.export_csv}")
    print()

    print("=== Warm-up ===")
    for index in range(args.warmup):
        run_one_iteration(model, config, args.frames_dir, args.prompt, args.max_new_tokens, profiler)
        print(f"warmup {index + 1}/{args.warmup} done")

    profiler.stage_samples.clear()
    profiler.stage_peak_mem_mb.clear()
    profiler.token_step_ms.clear()
    profiler.generated_lengths.clear()
    profiler.total_latency_ms.clear()
    profiler.preprocess_host_ms.clear()
    profiler.last_text = ""

    print()
    print("=== Measured Runs ===")
    for index in range(args.iters):
        row = run_one_iteration(model, config, args.frames_dir, args.prompt, args.max_new_tokens, profiler)
        iteration_rows.append(row)
        print(
            f"iter {index + 1:02d}/{args.iters} total={row['iteration_ms']:.3f} ms  "
            f"decoder={row['gpt2_decoder_ms']:.3f} ms  text={row['caption']!r}"
        )

    print()
    print("=== Summary ===")
    summarize("Preprocessing(cuda)", profiler.stage_samples["Preprocessing"], profiler.stage_peak_mem_mb["Preprocessing"])
    summarize("Preprocessing(host)", profiler.preprocess_host_ms)
    summarize("ViT_Encoder", profiler.stage_samples["ViT_Encoder"], profiler.stage_peak_mem_mb["ViT_Encoder"])
    summarize(
        "Cross_Modal_Alignment",
        profiler.stage_samples["Cross_Modal_Alignment"],
        profiler.stage_peak_mem_mb["Cross_Modal_Alignment"],
    )
    summarize(
        "GPT2_Decoder_Step",
        profiler.stage_samples["GPT2_Decoder_Step"],
        profiler.stage_peak_mem_mb["GPT2_Decoder_Step"],
    )
    summarize("GPT2 token step", profiler.token_step_ms)
    summarize("End-to-end(iter)", profiler.total_latency_ms)

    summary = build_summary(profiler, iteration_rows)
    if profiler.generated_lengths:
        print(f"generated_tokens_mean: {summary['generated_tokens']['mean']:.2f}")
        print(f"generated_tokens_max:  {summary['generated_tokens']['max']}")
    if profiler.last_text:
        print(f"last_caption: {profiler.last_text!r}")

    if args.export_csv:
        export_iteration_csv(args.export_csv, iteration_rows)
        print(f"iteration CSV exported to: {args.export_csv}")
    if args.export_json:
        export_summary_json(
            args.export_json,
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
                },
                "summary": summary,
                "iterations": iteration_rows,
                "token_step_ms": profiler.token_step_ms,
            },
        )
        print(f"summary JSON exported to: {args.export_json}")


if __name__ == "__main__":
    main()
