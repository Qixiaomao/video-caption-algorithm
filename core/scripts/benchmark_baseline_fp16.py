from __future__ import annotations

import gc
from contextlib import nullcontext
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import core.scripts.benchmark_baseline as base


def autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


@torch.inference_mode()
def run_decoder_steps(model, prefix_embeds: torch.Tensor, prompt: str, max_new_tokens: int, profiler: base.BenchmarkProfiler):
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

    with autocast_context(device):
        prompt_embeds = gpt2.transformer.wte(prompt_ids)
        full_inputs = torch.cat([prefix_embeds, prompt_embeds], dim=1)

    running_attention_mask = torch.ones(full_inputs.shape[:2], dtype=torch.long, device=device)
    generated_tokens = [[] for _ in range(batch_size)]
    step_samples: list[float] = []
    past_key_values = None
    next_input_embeds = full_inputs
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    with base.NvtxRange("GPT2_Decoder_Step"):
        for step_idx in range(max_new_tokens):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            step_name = f"GPT2_Decoder_Step/token_{step_idx:02d}"
            torch.cuda.synchronize(device)
            with base.NvtxRange(step_name):
                start_event.record()
                with autocast_context(device):
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

            with autocast_context(device):
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
    config,
    frames_dir: str,
    prompt: str,
    max_new_tokens: int,
    profiler: base.BenchmarkProfiler,
    batch_size: int,
):
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats(profiler.device)
    iter_t0 = base.time.perf_counter()

    video, preprocess_cuda_ms, preprocess_host_ms, preprocess_peak_mb = profiler.preprocess_stage(
        frames_dir=frames_dir,
        num_frames=config.num_frames,
        image_size=config.image_size,
        batch_size=batch_size,
    )

    def encoder_fn():
        with autocast_context(profiler.device):
            return model.encoder(video)

    encoder_out, encoder_ms, encoder_peak_mb = profiler.cuda_stage("ViT_Encoder", encoder_fn)

    def align_fn():
        with autocast_context(profiler.device):
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

    iter_ms = (base.time.perf_counter() - iter_t0) * 1000.0
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
        "generated_tokens_mean": base.statistics.mean(decoder_result["token_count"]) if decoder_result["token_count"] else 0.0,
        "preprocess_cuda_ms": preprocess_cuda_ms,
        "preprocess_host_ms": preprocess_host_ms,
        "preprocess_peak_mb": preprocess_peak_mb,
        "vit_encoder_ms": encoder_ms,
        "vit_encoder_peak_mb": encoder_peak_mb,
        "cross_modal_alignment_ms": align_ms,
        "cross_modal_alignment_peak_mb": align_peak_mb,
        "gpt2_decoder_ms": decoder_ms,
        "gpt2_decoder_peak_mb": decoder_peak_mb,
        "gpt2_token_step_mean_ms": base.statistics.mean(decoder_result["token_step_ms"]) if decoder_result["token_step_ms"] else 0.0,
        "gpt2_token_step_max_ms": max(decoder_result["token_step_ms"]) if decoder_result["token_step_ms"] else 0.0,
        "max_memory_allocated_mb": profiler.max_memory_allocated_mb,
    }


def main():
    args = base.parse_args()
    batch_sizes = base.parse_batch_sizes(args.batch_sizes)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this FP16 benchmark.")

    base.assert_core_runtime_ready(device=args.device, require_cupy=False)
    config = base.build_config(args)
    model = base.load_caption_model(config)
    env = base.get_env(torch.device(args.device))

    print("[AMP] autocast enabled with dtype=torch.float16")
    base.print_env(env)
    print("=== Benchmark Config ===")
    print(f"frames_dir: {args.frames_dir}")
    print(f"ckpt: {args.ckpt}")
    print(f"prompt: {args.prompt!r}")
    print(f"batch_sizes: {batch_sizes}")
    print(f"warmup: {args.warmup}")
    print(f"iters: {args.iters}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print("precision: fp16_autocast")
    print()

    single_batch_mode = len(batch_sizes) == 1
    if single_batch_mode:
        export_csv = args.export_csv or "baseline_fp16_iterations.csv"
        export_json = args.export_json or "baseline_fp16_summary.json"
    else:
        export_csv = args.export_csv or "benchmark_bs_comparison_fp16.csv"
        export_json = args.export_json or "benchmark_bs_summary_fp16.json"

    all_iteration_rows: dict[int, list[dict[str, object]]] = {}
    all_summaries: dict[int, dict[str, object]] = {}
    comparison_rows: list[dict[str, object]] = []

    for batch_size in batch_sizes:
        try:
            profiler = base.BenchmarkProfiler(args.device)
            iteration_rows: list[dict[str, object]] = []

            print()
            print(f"=== Batch Size {batch_size} / FP16 ===")
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

            summary = base.build_summary(profiler, iteration_rows, batch_size=batch_size)
            summary["precision"] = "fp16_autocast"
            all_iteration_rows[batch_size] = iteration_rows
            all_summaries[batch_size] = summary
            comparison_rows.append(base.build_comparison_row(summary, args.warmup, args.iters))
            print(
                f"[BS={batch_size}][FP16] mean_end_to_end={summary['End_to_end_Latency']['mean_ms']:.3f} ms  "
                f"mean_vit={summary['ViT_Latency']['mean_ms']:.3f} ms  "
                f"mean_gpt2={summary['GPT2_Latency']['mean_ms']:.3f} ms  "
                f"throughput={summary['Throughput']['from_mean_latency_samples_per_s']:.3f} samples/s  "
                f"max_mem={summary['peak_memory_mb']['max_memory_allocated_mb']:.1f} MB"
            )
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            gc.collect()
            summary = base.build_oom_summary(batch_size, exc)
            summary["precision"] = "fp16_autocast"
            all_summaries[batch_size] = summary
            comparison_rows.append(base.build_comparison_row(summary, args.warmup, args.iters))
            print()
            print(f"[BS={batch_size}][FP16] OOM encountered, stopping larger batch sizes.")
            break

    if single_batch_mode:
        batch_size = batch_sizes[0]
        iteration_rows = all_iteration_rows.get(batch_size, [])
        summary = all_summaries.get(batch_size)
        if summary and summary["status"] == "ok":
            print()
            print("=== Summary ===")
            base.summarize("Preprocessing(cuda)", [row["preprocess_cuda_ms"] for row in iteration_rows])
            base.summarize("Preprocessing(host)", [row["preprocess_host_ms"] for row in iteration_rows])
            base.summarize("ViT_Encoder", [row["vit_encoder_ms"] for row in iteration_rows])
            base.summarize("Cross_Modal_Alignment", [row["cross_modal_alignment_ms"] for row in iteration_rows])
            base.summarize("GPT2_Decoder_Step", [row["gpt2_decoder_ms"] for row in iteration_rows])
            base.summarize("End-to-end(iter)", [row["iteration_ms"] for row in iteration_rows])
            base.summarize("Throughput", [row["throughput_samples_per_s"] for row in iteration_rows], unit="samples/s")

        if export_csv and iteration_rows:
            base.export_iteration_csv(export_csv, iteration_rows)
            print(f"iteration CSV exported to: {export_csv}")
        if export_json and summary:
            base.export_summary_json(
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
                        "precision": "fp16_autocast",
                    },
                    "summary": summary,
                    "iterations": iteration_rows,
                },
            )
            print(f"summary JSON exported to: {export_json}")
        return

    if export_csv:
        base.export_bs_comparison_csv(export_csv, comparison_rows)
        print(f"batch-size comparison CSV exported to: {export_csv}")
    if export_json:
        base.export_summary_json(
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
                    "precision": "fp16_autocast",
                },
                "comparison": comparison_rows,
                "per_batch_summary": {str(bs): summary for bs, summary in all_summaries.items()},
                "per_batch_iterations": {str(bs): rows for bs, rows in all_iteration_rows.items()},
            },
        )
        print(f"batch-size summary JSON exported to: {export_json}")


if __name__ == "__main__":
    main()
