"""Benchmark: naive CuPy pool vs optimized CuPy pool vs PyTorch native.

Run: python -m core.scripts.bench_pool_opt --device cuda --iters 100
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


def pytorch_pool(feat, bsz, timesteps, pool="gap"):
    """Baseline: PyTorch native spatial + temporal pooling."""
    bt, tokens, channels = feat.shape
    feat_bt = feat.reshape(bsz, timesteps, tokens, channels)
    if pool == "cls":
        return feat_bt[:, :, 0, :].mean(dim=1)
    return feat_bt[:, :, 1:, :].mean(dim=(1, 2))


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
    print(f"  {label:<35} mean={mean:8.4f} ms  std={std:8.4f} ms")
    return mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--bsz", type=int, default=8)
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=197)
    parser.add_argument("--channels", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    bt = args.bsz * args.timesteps
    x = torch.randn(bt, args.tokens, args.channels, device=device, dtype=torch.float32)

    print(f"Shape: [{bt}, {args.tokens}, {args.channels}]  bsz={args.bsz}  T={args.timesteps}")
    print(f"Warmup={args.warmup}  Iters={args.iters}")
    print()

    # PyTorch baseline
    print("[FP32]")
    t_pt = run_benchmark(pytorch_pool, args.warmup, args.iters,
                         "PyTorch native (FP32)", x, args.bsz, args.timesteps, "gap")

    from core.operators.cupy_vit_pool import vit_fused_pool_temporal
    from core.operators.cupy_vit_pool_v2 import vit_fused_pool_temporal_v2

    # Naive CuPy FP32
    def naive_f32(feat, bsz, ts):
        return vit_fused_pool_temporal(feat, bsz, ts, "gap", force_fp16=False)

    t_naive_f32 = run_benchmark(naive_f32, args.warmup, args.iters,
                                "CuPy naive (FP32)", x, args.bsz, args.timesteps)

    # Optimized CuPy FP32 - unrolled
    def opt_unroll_f32(feat, bsz, ts):
        return vit_fused_pool_temporal_v2(feat, bsz, ts, "gap", force_fp16=False, version="unroll")

    t_opt_f32 = run_benchmark(opt_unroll_f32, args.warmup, args.iters,
                               "CuPy v2 unroll (FP32)", x, args.bsz, args.timesteps)

    print()
    print("[FP16]")
    x_f16 = x.half()

    # PyTorch FP16
    t_pt_f16 = run_benchmark(pytorch_pool, args.warmup, args.iters,
                              "PyTorch native (FP16)", x_f16, args.bsz, args.timesteps, "gap")

    # Naive CuPy FP16
    def naive_f16(feat, bsz, ts):
        return vit_fused_pool_temporal(feat, bsz, ts, "gap", force_fp16=True)

    t_naive_f16 = run_benchmark(naive_f16, args.warmup, args.iters,
                                "CuPy naive (FP16)", x_f16, args.bsz, args.timesteps)

    # Optimized CuPy FP16 - unrolled
    def opt_unroll_f16(feat, bsz, ts):
        return vit_fused_pool_temporal_v2(feat, bsz, ts, "gap", force_fp16=True, version="unroll")

    t_opt_f16 = run_benchmark(opt_unroll_f16, args.warmup, args.iters,
                               "CuPy v2 unroll (FP16)", x_f16, args.bsz, args.timesteps)

    # Optimized CuPy FP16 - half2 vectorized
    def opt_vec2_f16(feat, bsz, ts):
        return vit_fused_pool_temporal_v2(feat, bsz, ts, "gap", force_fp16=True, version="vec2")

    t_vec2_f16 = run_benchmark(opt_vec2_f16, args.warmup, args.iters,
                                "CuPy v2 half2 (FP16)", x_f16, args.bsz, args.timesteps)

    # Summary
    print()
    print("=== Summary ===")
    print(f"  FP32: PyTorch={t_pt:.4f}  Naive={t_naive_f32:.4f}  Unroll={t_opt_f32:.4f}")
    print(f"  FP16: PyTorch={t_pt_f16:.4f}  Naive={t_naive_f16:.4f}  Unroll={t_opt_f16:.4f}  Half2={t_vec2_f16:.4f}")
    if t_naive_f32 > 0:
        print(f"  FP32 speedup (unroll vs naive): {t_naive_f32/t_opt_f32:.2f}x")
    if t_naive_f16 > 0:
        print(f"  FP16 speedup (unroll vs naive): {t_naive_f16/t_opt_f16:.2f}x")
        print(f"  FP16 speedup (half2 vs naive):  {t_naive_f16/t_vec2_f16:.2f}x")


if __name__ == "__main__":
    main()
