"""Benchmark: two-step pool+proj vs fused pool+proj kernel.

Run: python -m core.scripts.bench_fused_pool_proj --device cuda --iters 100
"""

import argparse
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch


def pytorch_two_step(feat, weight, bias, bsz, timesteps, pool="gap"):
    """Baseline: PyTorch pool then PyTorch Linear."""
    bt, tokens, channels = feat.shape
    feat_bt = feat.reshape(bsz, timesteps, tokens, channels)
    if pool == "cls":
        pooled = feat_bt[:, :, 0, :].mean(dim=1)
    else:
        pooled = feat_bt[:, :, 1:, :].mean(dim=(1, 2))
    return torch.nn.functional.linear(pooled, weight, bias)


def run_benchmark(fn, warmup, iters, label, *args):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn(*args)
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean = statistics.mean(times)
    std = statistics.pstdev(times)
    print(f"  {label:<40} mean={mean:8.4f} ms  std={std:8.4f} ms")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=197)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--out-dim", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    bt = args.bsz * args.timesteps

    print(f"Input:  [{bt}, {args.tokens}, {args.channels}]")
    print(f"Weight: [{args.out_dim}, {args.channels}]")
    print(f"Warmup={args.warmup}  Iters={args.iters}")
    print()

    from core.operators.cupy_fused_pool_proj import fused_pool_proj

    # ── FP32 ──
    print("[FP32]")
    x_f32 = torch.randn(bt, args.tokens, args.channels, device=device, dtype=torch.float32)
    W_f32 = torch.randn(args.out_dim, args.channels, device=device, dtype=torch.float32)
    b_f32 = torch.randn(args.out_dim, device=device, dtype=torch.float32)

    t_pt_f32 = run_benchmark(pytorch_two_step, args.warmup, args.iters,
                              "PyTorch two-step (pool+proj)", x_f32, W_f32, b_f32,
                              args.bsz, args.timesteps, "gap")

    def fused_f32(feat, weight, bias, bsz, ts):
        return fused_pool_proj(feat, weight, bias, bsz, ts, "gap")

    t_fused_f32 = run_benchmark(fused_f32, args.warmup, args.iters,
                                 "CuPy fused (pool+proj)", x_f32, W_f32, b_f32,
                                 args.bsz, args.timesteps)

    # Correctness check
    out_pt = pytorch_two_step(x_f32, W_f32, b_f32, args.bsz, args.timesteps, "gap")
    out_fused = fused_pool_proj(x_f32, W_f32, b_f32, args.bsz, args.timesteps, "gap")
    if out_fused is not None:
        max_diff = (out_pt - out_fused.float()).abs().max().item()
        print(f"  Correctness check (max diff): {max_diff:.6f}")
    else:
        print("  Fused kernel returned None!")

    # ── FP16 ──
    print()
    print("[FP16]")
    x_f16 = x_f32.half()
    W_f16 = W_f32.half()
    b_f16 = b_f32.half()

    t_pt_f16 = run_benchmark(pytorch_two_step, args.warmup, args.iters,
                              "PyTorch two-step (pool+proj)", x_f16, W_f16, b_f16,
                              args.bsz, args.timesteps, "gap")

    def fused_fp16(feat, weight, bias, bsz, ts):
        return fused_pool_proj(feat, weight, bias, bsz, ts, "gap")

    t_fused_f16 = run_benchmark(fused_fp16, args.warmup, args.iters,
                                 "CuPy fused (pool+proj)", x_f16, W_f16, b_f16,
                                 args.bsz, args.timesteps)

    # Correctness check
    out_pt_f16 = pytorch_two_step(x_f16, W_f16, b_f16, args.bsz, args.timesteps, "gap")
    out_fused_f16 = fused_pool_proj(x_f16, W_f16, b_f16, args.bsz, args.timesteps, "gap")
    if out_fused_f16 is not None:
        max_diff_f16 = (out_pt_f16.float() - out_fused_f16.float()).abs().max().item()
        print(f"  Correctness check (max diff): {max_diff_f16:.6f}")

    # Summary
    print()
    print("=== Summary ===")
    print(f"  FP32: Two-step={t_pt_f32:.4f}  Fused={t_fused_f32:.4f}  Speedup={t_pt_f32/t_fused_f32:.2f}x")
    print(f"  FP16: Two-step={t_pt_f16:.4f}  Fused={t_fused_f16:.4f}  Speedup={t_pt_f16/t_fused_f16:.2f}x")


if __name__ == "__main__":
    main()
