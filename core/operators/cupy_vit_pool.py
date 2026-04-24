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


_F32_CLS = r'''
extern "C" __global__
void vit_pool_cls_f32(const float* x, float* y, int bsz, int timesteps, int tokens, int channels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = bsz * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float acc = 0.0f;
    for (int t = 0; t < timesteps; ++t) {
        int src = ((b * timesteps + t) * tokens + 0) * channels + c;
        acc += x[src];
    }
    y[idx] = acc / (float)timesteps;
}
'''


_F32_GAP = r'''
extern "C" __global__
void vit_pool_gap_f32(const float* x, float* y, int bsz, int timesteps, int tokens, int channels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = bsz * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float acc = 0.0f;
    int patch_tokens = tokens - 1;
    for (int t = 0; t < timesteps; ++t) {
        for (int p = 1; p < tokens; ++p) {
            int src = ((b * timesteps + t) * tokens + p) * channels + c;
            acc += x[src];
        }
    }
    y[idx] = acc / (float)(timesteps * patch_tokens);
}
'''


_F16_CLS = r'''
#include <cuda_fp16.h>
extern "C" __global__
void vit_pool_cls_f16(const half* x, half* y, int bsz, int timesteps, int tokens, int channels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = bsz * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float acc = 0.0f;
    for (int t = 0; t < timesteps; ++t) {
        int src = ((b * timesteps + t) * tokens + 0) * channels + c;
        acc += __half2float(x[src]);
    }
    y[idx] = __float2half_rn(acc / (float)timesteps);
}
'''


_F16_GAP = r'''
#include <cuda_fp16.h>
extern "C" __global__
void vit_pool_gap_f16(const half* x, half* y, int bsz, int timesteps, int tokens, int channels) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = bsz * channels;
    if (idx >= total) return;
    int b = idx / channels;
    int c = idx % channels;

    float acc = 0.0f;
    int patch_tokens = tokens - 1;
    for (int t = 0; t < timesteps; ++t) {
        for (int p = 1; p < tokens; ++p) {
            int src = ((b * timesteps + t) * tokens + p) * channels + c;
            acc += __half2float(x[src]);
        }
    }
    y[idx] = __float2half_rn(acc / (float)(timesteps * patch_tokens));
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
    if not _ensure_cupy():
        raise RuntimeError("CuPy is not available")
    if hasattr(cp, "from_dlpack"):
        return cp.from_dlpack(tensor)
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))



def vit_fused_pool_temporal(
    feat: torch.Tensor,
    bsz: int,
    timesteps: int,
    pool: str,
    force_fp16: bool = True,
) -> torch.Tensor | None:
    """Fused ViT token pool + temporal mean.

    Returns None on unsupported shape/device/backend so caller can fallback.
    """

    if not _ensure_cupy():
        return None
    if feat.device.type != "cuda":
        return None
    if feat.ndim != 3:
        return None
    if pool not in {"cls", "gap"}:
        return None

    bt, tokens, channels = feat.shape
    if bt != bsz * timesteps:
        return None
    if pool == "gap" and tokens <= 1:
        return None

    x = feat.contiguous()
    target_dtype = torch.float16 if force_fp16 else (torch.float16 if x.dtype == torch.float16 else torch.float32)
    if x.dtype != target_dtype:
        x = x.to(target_dtype)

    out = torch.empty((bsz, channels), device=x.device, dtype=target_dtype)

    try:
        stream = torch.cuda.current_stream(device=x.device)
        with cp.cuda.ExternalStream(stream.cuda_stream):
            x_cp = _torch_to_cupy(x)
            out_cp = _torch_to_cupy(out)

            threads = 256
            blocks = (bsz * channels + threads - 1) // threads

            if target_dtype == torch.float16:
                if pool == "cls":
                    k = _kernel("vit_pool_cls_f16", _F16_CLS)
                else:
                    k = _kernel("vit_pool_gap_f16", _F16_GAP)
            else:
                if pool == "cls":
                    k = _kernel("vit_pool_cls_f32", _F32_CLS)
                else:
                    k = _kernel("vit_pool_gap_f32", _F32_GAP)

            k((blocks,), (threads,), (x_cp, out_cp, bsz, timesteps, tokens, channels))

        # out is already a torch tensor backed by device memory that CuPy wrote into.
        return out
    except Exception:
        return None


__all__ = ["vit_fused_pool_temporal"]





