"""Fused pool + projection CUDA kernel.

Eliminates the intermediate [B, channels] tensor between pooling and projection.
Instead of:
  pool: [B*T, tokens, C] → [B, C]   (write to global memory)
  proj: [B, C] → [B, D]             (read from global memory)

We fuse into:
  fused: [B*T, tokens, C] × W[C, D] → [B, D]  (single pass, no intermediate)

Trade-off: more computation (B*T * tokens * C * D FLOPs) but eliminates one
global memory round-trip (B*C write + B*C read = 2*B*C memory traffic).
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
# Fused GAP + Linear (FP16 accumulation in FP32)
# ──────────────────────────────────────────────

_FUSED_GAP_PROJ_F16 = r'''
#include <cuda_fp16.h>
extern "C" __global__
void fused_gap_proj_f16(const half* __restrict__ x,    // [B*T, tokens, C]
                         const half* __restrict__ W,    // [D, C]  (row-major)
                         const half* __restrict__ bias, // [D]
                         half* __restrict__ y,          // [B, D]
                         int bsz, int timesteps, int tokens, int channels, int out_dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = bsz * out_dim;
    if (idx >= total) return;

    int b = idx / out_dim;
    int o = idx % out_dim;

    int patch_tokens = tokens - 1;
    int row_stride = tokens * channels;

    float acc = 0.0f;
    float w_bias = __half2float(bias[o]);

    for (int c = 0; c < channels; ++c) {
        float w_val = __half2float(W[o * channels + c]);

        float pool_sum = 0.0f;
        for (int t = 0; t < timesteps; ++t) {
            int t_base = (b * timesteps + t) * row_stride + channels + c;
            for (int p = 0; p < patch_tokens; ++p) {
                pool_sum += __half2float(x[t_base + p * channels]);
            }
        }
        float pool_val = pool_sum / (float)(timesteps * patch_tokens);
        acc += w_val * pool_val;
    }

    y[idx] = __float2half_rn(acc + w_bias);
}
'''

_FUSED_GAP_PROJ_F32 = r'''
extern "C" __global__
void fused_gap_proj_f32(const float* __restrict__ x,    // [B*T, tokens, C]
                         const float* __restrict__ W,    // [D, C]
                         const float* __restrict__ bias, // [D]
                         float* __restrict__ y,          // [B, D]
                         int bsz, int timesteps, int tokens, int channels, int out_dim) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = bsz * out_dim;
    if (idx >= total) return;

    int b = idx / out_dim;
    int o = idx % out_dim;

    int patch_tokens = tokens - 1;
    int row_stride = tokens * channels;

    float acc = 0.0f;
    float w_bias = bias[o];

    for (int c = 0; c < channels; ++c) {
        float w_val = W[o * channels + c];

        float pool_sum = 0.0f;
        for (int t = 0; t < timesteps; ++t) {
            int t_base = (b * timesteps + t) * row_stride + channels + c;
            for (int p = 0; p < patch_tokens; ++p) {
                pool_sum += x[t_base + p * channels];
            }
        }
        float pool_val = pool_sum / (float)(timesteps * patch_tokens);
        acc += w_val * pool_val;
    }

    y[idx] = acc + w_bias;
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


def fused_pool_proj(
    feat: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    bsz: int,
    timesteps: int,
    pool: str = "gap",
) -> torch.Tensor | None:
    """Fused spatial-temporal pooling + linear projection.

    Args:
        feat: [B*T, tokens, C] input features
        weight: [D, C] linear weight (transposed)
        bias: [D] linear bias
        bsz: batch size
        timesteps: number of frames
        pool: "cls" or "gap"

    Returns:
        [B, D] output tensor, or None on failure
    """
    if not _ensure_cupy():
        return None
    if feat.device.type != "cuda":
        return None
    if feat.ndim != 3:
        return None

    bt, tokens, channels = feat.shape
    if bt != bsz * timesteps:
        return None

    out_dim = weight.shape[0]
    if weight.shape[1] != channels:
        return None

    use_fp16 = (feat.dtype == torch.float16)

    x = feat.contiguous()
    W = weight.contiguous()
    b = bias.contiguous()

    if use_fp16:
        if x.dtype != torch.float16:
            x = x.half()
        if W.dtype != torch.float16:
            W = W.half()
        if b.dtype != torch.float16:
            b = b.half()
    else:
        if x.dtype != torch.float32:
            x = x.float()
        if W.dtype != torch.float32:
            W = W.float()
        if b.dtype != torch.float32:
            b = b.float()

    out = torch.empty((bsz, out_dim), device=x.device, dtype=x.dtype)

    try:
        stream = torch.cuda.current_stream(device=x.device)
        with cp.cuda.ExternalStream(stream.cuda_stream):
            x_cp = _torch_to_cupy(x)
            W_cp = _torch_to_cupy(W)
            b_cp = _torch_to_cupy(b)
            out_cp = _torch_to_cupy(out)

            threads = 256
            blocks = (bsz * out_dim + threads - 1) // threads

            if use_fp16:
                k = _kernel("fused_gap_proj_f16", _FUSED_GAP_PROJ_F16)
            else:
                k = _kernel("fused_gap_proj_f32", _FUSED_GAP_PROJ_F32)

            k((blocks,), (threads,),
              (x_cp, W_cp, b_cp, out_cp,
               bsz, timesteps, tokens, channels, out_dim))

        return out
    except Exception as e:
        print(f"[fused_pool_proj] Error: {e}")
        return None


__all__ = ["fused_pool_proj"]
