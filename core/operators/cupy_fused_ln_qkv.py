"""Fused LayerNorm + QKV Projection CUDA kernel (Stage 1 of persistent ViT block).

Instead of:
  LN: read x[B,T,C] → write ln_out[B,T,C]     (38MB write)
  QKV: read ln_out[B,T,C] → write qkv[B,T,3C]  (38MB read + 114MB write)

We fuse into:
  fused: read x[B,T,C] → compute LN → compute QKV → write qkv[B,T,3C]
  (eliminates 38MB intermediate write + 38MB read = 76MB memory traffic)

The LN computation is inlined into the QKV matrix multiplication loop.
"""

from __future__ import annotations
import torch

cp = None
_CUPY_AVAILABLE: bool | None = None


def _ensure_cupy() -> bool:
    global cp, _CUPY_AVAILABLE
    if _CUPY_AVAILABLE is not None:
        return _CUPY_AVAILABLE
    try:
        import cupy as _cp
        cp = _cp
        _CUPY_AVAILABLE = True
    except Exception:
        cp = None
        _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


# ──────────────────────────────────────────────
# Fused LN + QKV (FP16, FP32 accumulation)
# ──────────────────────────────────────────────

_FUSED_LN_QKV_F16 = r'''
#include <cuda_fp16.h>

extern "C" __global__
void fused_ln_qkv_f16(const half* __restrict__ x,       // [B*T, tokens, C]
                       const half* __restrict__ ln_w,     // [C] LayerNorm weight
                       const half* __restrict__ ln_b,     // [C] LayerNorm bias
                       const half* __restrict__ qkv_w,    // [3C, C] QKV weight
                       const half* __restrict__ qkv_b,    // [3C] QKV bias
                       half* __restrict__ out,            // [B*T, tokens, 3C]
                       int total_tokens, int C, int C3) {
    // Each thread computes one output element: out[token, out_idx]
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = total_tokens * C3;
    if (idx >= total) return;

    int token = idx / C3;
    int o = idx % C3;

    const half* x_token = x + token * C;

    // Step 1: Compute LayerNorm statistics (mean, variance) for this token
    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += __half2float(x_token[c]);
    }
    float mean = sum / (float)C;

    float var_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float diff = __half2float(x_token[c]) - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum / (float)C + 1e-5f);

    // Step 2: Compute QKV[o] = sum_c(qkv_w[o,c] * LN(x[c])) + qkv_b[o]
    // where LN(x[c]) = ln_w[c] * (x[c] - mean) * inv_std + ln_b[c]
    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        float ln_val = __half2float(ln_w[c]) * (__half2float(x_token[c]) - mean) * inv_std
                       + __half2float(ln_b[c]);
        acc += __half2float(qkv_w[o * C + c]) * ln_val;
    }
    acc += __half2float(qkv_b[o]);

    out[idx] = __float2half_rn(acc);
}
'''

_FUSED_LN_QKV_F32 = r'''
extern "C" __global__
void fused_ln_qkv_f32(const float* __restrict__ x,
                       const float* __restrict__ ln_w,
                       const float* __restrict__ ln_b,
                       const float* __restrict__ qkv_w,
                       const float* __restrict__ qkv_b,
                       float* __restrict__ out,
                       int total_tokens, int C, int C3) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = total_tokens * C3;
    if (idx >= total) return;

    int token = idx / C3;
    int o = idx % C3;

    const float* x_token = x + token * C;

    float sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        sum += x_token[c];
    }
    float mean = sum / (float)C;

    float var_sum = 0.0f;
    for (int c = 0; c < C; ++c) {
        float diff = x_token[c] - mean;
        var_sum += diff * diff;
    }
    float inv_std = rsqrtf(var_sum / (float)C + 1e-5f);

    float acc = 0.0f;
    for (int c = 0; c < C; ++c) {
        float ln_val = ln_w[c] * (x_token[c] - mean) * inv_std + ln_b[c];
        acc += qkv_w[o * C + c] * ln_val;
    }
    acc += qkv_b[o];

    out[idx] = acc;
}
'''

_KERNELS: dict[str, object] = {}


def _kernel(name: str, src: str):
    if name in _KERNELS:
        return _KERNELS[name]
    k = cp.RawKernel(src, name)
    _KERNELS[name] = k
    return k


def _torch_to_cupy(tensor: torch.Tensor):
    if hasattr(cp, "from_dlpack"):
        return cp.from_dlpack(tensor)
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))


def fused_ln_qkv(
    x: torch.Tensor,
    ln_weight: torch.Tensor,
    ln_bias: torch.Tensor,
    qkv_weight: torch.Tensor,
    qkv_bias: torch.Tensor,
) -> torch.Tensor | None:
    """Fused LayerNorm + QKV Projection.

    Args:
        x: [total_tokens, C] input (already reshaped)
        ln_weight: [C] LayerNorm weight
        ln_bias: [C] LayerNorm bias
        qkv_weight: [3C, C] combined QKV weight
        qkv_bias: [3C] combined QKV bias

    Returns:
        [total_tokens, 3C] output, or None on failure
    """
    if not _ensure_cupy():
        return None
    if x.device.type != "cuda":
        return None

    total_tokens, C = x.shape
    C3 = qkv_weight.shape[0]  # 3C

    use_fp16 = (x.dtype == torch.float16)

    x_c = x.contiguous()
    ln_w = ln_weight.contiguous()
    ln_b = ln_bias.contiguous()
    w = qkv_weight.contiguous()
    b = qkv_bias.contiguous()

    if use_fp16:
        for t in [x_c, ln_w, ln_b, w, b]:
            if t.dtype != torch.float16:
                t = t.half()
    else:
        for t in [x_c, ln_w, ln_b, w, b]:
            if t.dtype != torch.float32:
                t = t.float()

    out = torch.empty((total_tokens, C3), device=x.device, dtype=x.dtype)

    try:
        stream = torch.cuda.current_stream(device=x.device)
        with cp.cuda.ExternalStream(stream.cuda_stream):
            x_cp = _torch_to_cupy(x_c)
            ln_w_cp = _torch_to_cupy(ln_w)
            ln_b_cp = _torch_to_cupy(ln_b)
            w_cp = _torch_to_cupy(w)
            b_cp = _torch_to_cupy(b)
            out_cp = _torch_to_cupy(out)

            threads = 256
            total = total_tokens * C3
            blocks = (total + threads - 1) // threads

            if use_fp16:
                k = _kernel("fused_ln_qkv_f16", _FUSED_LN_QKV_F16)
            else:
                k = _kernel("fused_ln_qkv_f32", _FUSED_LN_QKV_F32)

            k((blocks,), (threads,),
              (x_cp, ln_w_cp, ln_b_cp, w_cp, b_cp, out_cp,
               total_tokens, C, C3))

        return out
    except Exception as e:
        print(f"[fused_ln_qkv] Error: {e}")
        return None


__all__ = ["fused_ln_qkv"]
