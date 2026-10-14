"""Benchmark: two-step LN+QKV vs fused LN+QKV kernel.

Run: python -m core.scripts.bench_fused_ln_qkv --device cuda --iters 100
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
import torch.nn as nn


def pytorch_two_step(x, ln_w, ln_b, qkv_w, qkv_b):
    """Baseline: PyTorch LN then PyTorch Linear."""
    ln_out = nn.functional.layer_norm(x, (x.shape[-1],), ln_w, ln_b)
    return nn.functional.linear(ln_out, qkv_w, qkv_b)


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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    device = torch.device(args.device)
    total_tokens = args.bsz * args.timesteps * args.tokens
    C = args.channels
    C3 = 3 * C  # QKV combined

    print(f"Input:  [{total_tokens}, {C}]")
    print(f"QKV Weight: [{C3}, {C}]")
    print(f"Warmup={args.warmup}  Iters={args.iters}")
    print()

    from core.operators.cupy_fused_ln_qkv import fused_ln_qkv

    # ── FP32 ──
    print("[FP32]")
    x_f32 = torch.randn(total_tokens, C, device=device, dtype=torch.float32)
    ln_w_f32 = torch.randn(C, device=device, dtype=torch.float32)
    ln_b_f32 = torch.randn(C, device=device, dtype=torch.float32)
    qkv_w_f32 = torch.randn(C3, C, device=device, dtype=torch.float32)
    qkv_b_f32 = torch.randn(C3, device=device, dtype=torch.float32)

    t_pt_f32 = run_benchmark(pytorch_two_step, args.warmup, args.iters,
                              "PyTorch two-step (LN+QKV)",
                              x_f32, ln_w_f32, ln_b_f32, qkv_w_f32, qkv_b_f32)

    def fused_f32(x, ln_w, ln_b, qkv_w, qkv_b):
        return fused_ln_qkv(x, ln_w, ln_b, qkv_w, qkv_b)

    t_fused_f32 = run_benchmark(fused_f32, args.warmup, args.iters,
                                 "CuPy fused (LN+QKV)",
                                 x_f32, ln_w_f32, ln_b_f32, qkv_w_f32, qkv_b_f32)

    # Correctness check
    out_pt = pytorch_two_step(x_f32, ln_w_f32, ln_b_f32, qkv_w_f32, qkv_b_f32)
    out_fused = fused_ln_qkv(x_f32, ln_w_f32, ln_b_f32, qkv_w_f32, qkv_b_f32)
    if out_fused is not None:
        max_diff = (out_pt - out_fused).abs().max().item()
        mean_diff = (out_pt - out_fused).abs().mean().item()
        print(f"  Correctness: max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}")
    else:
        print("  Fused kernel returned None!")

    # ── FP16 ──
    print()
    print("[FP16]")
    x_f16 = x_f32.half()
    ln_w_f16 = ln_w_f32.half()
    ln_b_f16 = ln_b_f32.half()
    qkv_w_f16 = qkv_w_f32.half()
    qkv_b_f16 = qkv_b_f32.half()

    t_pt_f16 = run_benchmark(pytorch_two_step, args.warmup, args.iters,
                              "PyTorch two-step (LN+QKV)",
                              x_f16, ln_w_f16, ln_b_f16, qkv_w_f16, qkv_b_f16)

    def fused_f16(x, ln_w, ln_b, qkv_w, qkv_b):
        return fused_ln_qkv(x, ln_w, ln_b, qkv_w, qkv_b)

    t_fused_f16 = run_benchmark(fused_f16, args.warmup, args.iters,
                                 "CuPy fused (LN+QKV)",
                                 x_f16, ln_w_f16, ln_b_f16, qkv_w_f16, qkv_b_f16)

    # Correctness check
    out_pt_f16 = pytorch_two_step(x_f16, ln_w_f16, ln_b_f16, qkv_w_f16, qkv_b_f16)
    out_fused_f16 = fused_ln_qkv(x_f16, ln_w_f16, ln_b_f16, qkv_w_f16, qkv_b_f16)
    if out_fused_f16 is not None:
        max_diff_f16 = (out_pt_f16.float() - out_fused_f16.float()).abs().max().item()
        mean_diff_f16 = (out_pt_f16.float() - out_fused_f16.float()).abs().mean().item()
        print(f"  Correctness: max_diff={max_diff_f16:.6f}  mean_diff={mean_diff_f16:.6f}")

    # Summary
    print()
    print("=== Summary ===")
    if t_fused_f32 > 0:
        print(f"  FP32: Two-step={t_pt_f32:.4f}  Fused={t_fused_f32:.4f}  Speedup={t_pt_f32/t_fused_f32:.2f}x")
    if t_fused_f16 > 0:
        print(f"  FP16: Two-step={t_pt_f16:.4f}  Fused={t_fused_f16:.4f}  Speedup={t_pt_f16/t_fused_f16:.2f}x")


if __name__ == "__main__":
    main()
