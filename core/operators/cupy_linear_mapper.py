from __future__ import annotations

import torch
import torch.nn as nn

try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False


_F32_KERNEL = r'''
extern "C" __global__
void linear_bias_f32(
    const float* x,
    const float* w,
    const float* b,
    float* y,
    int rows,
    int in_features,
    int out_features
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * out_features;
    if (idx >= total) return;

    int row = idx / out_features;
    int col = idx % out_features;
    float acc = b ? b[col] : 0.0f;
    int x_offset = row * in_features;
    int w_offset = col * in_features;

    for (int k = 0; k < in_features; ++k) {
        acc += x[x_offset + k] * w[w_offset + k];
    }
    y[idx] = acc;
}
'''


_F16_KERNEL = r'''
#include <cuda_fp16.h>
extern "C" __global__
void linear_bias_f16(
    const half* x,
    const half* w,
    const half* b,
    half* y,
    int rows,
    int in_features,
    int out_features
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * out_features;
    if (idx >= total) return;

    int row = idx / out_features;
    int col = idx % out_features;
    float acc = b ? __half2float(b[col]) : 0.0f;
    int x_offset = row * in_features;
    int w_offset = col * in_features;

    for (int k = 0; k < in_features; ++k) {
        acc += __half2float(x[x_offset + k]) * __half2float(w[w_offset + k]);
    }
    y[idx] = __float2half_rn(acc);
}
'''


_LINEAR_KERNELS: dict[str, object] = {}


def _get_kernel(kind: str):
    if kind in _LINEAR_KERNELS:
        return _LINEAR_KERNELS[kind]
    if kind == "f16":
        kernel = cp.RawKernel(_F16_KERNEL, "linear_bias_f16")
    else:
        kernel = cp.RawKernel(_F32_KERNEL, "linear_bias_f32")
    _LINEAR_KERNELS[kind] = kernel
    return kernel


def _torch_to_cupy(tensor: torch.Tensor):
    if hasattr(cp, "from_dlpack"):
        return cp.from_dlpack(tensor)
    return cp.fromDlpack(torch.utils.dlpack.to_dlpack(tensor))


def _cupy_to_torch(array) -> torch.Tensor:
    return torch.utils.dlpack.from_dlpack(array)


def _cupy_linear_bias(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None, force_fp16: bool) -> torch.Tensor:
    rows = x.shape[0]
    in_features = x.shape[1]
    out_features = weight.shape[0]

    if force_fp16:
        target_dtype = torch.float16
        kind = "f16"
    else:
        target_dtype = torch.float16 if x.dtype == torch.float16 else torch.float32
        kind = "f16" if target_dtype == torch.float16 else "f32"

    x_work = x.contiguous()
    w_work = weight.detach().contiguous()
    b_work = bias.detach().contiguous() if bias is not None else None

    if x_work.dtype != target_dtype:
        x_work = x_work.to(target_dtype)
    if w_work.dtype != target_dtype:
        w_work = w_work.to(target_dtype)
    if b_work is not None and b_work.dtype != target_dtype:
        b_work = b_work.to(target_dtype)

    y_work = torch.empty((rows, out_features), device=x_work.device, dtype=target_dtype)

    stream = torch.cuda.current_stream(device=x_work.device)
    with cp.cuda.ExternalStream(stream.cuda_stream):
        x_cp = _torch_to_cupy(x_work)
        w_cp = _torch_to_cupy(w_work)
        b_cp = _torch_to_cupy(b_work) if b_work is not None else None
        y_cp = _torch_to_cupy(y_work)

        threads = 256
        blocks = (rows * out_features + threads - 1) // threads
        kernel = _get_kernel(kind)
        kernel((blocks,), (threads,), (x_cp, w_cp, b_cp, y_cp, rows, in_features, out_features))

    return _cupy_to_torch(y_cp)


class CuPyLinearCompat(nn.Linear):
    """`nn.Linear` drop-in with optional CuPy kernel in CUDA inference.

    Keeps `weight`/`bias` parameter names unchanged for checkpoint compatibility.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        enabled: bool = False,
        force_fp16: bool = True,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.enabled = enabled
        self.force_fp16 = force_fp16
        self.last_backend = "torch"
        self.last_error = ""

    def _should_use_cupy(self, x: torch.Tensor) -> bool:
        if not self.enabled or not _CUPY_AVAILABLE:
            return False
        if self.training:
            return False
        if x.device.type != "cuda":
            return False
        if x.requires_grad:
            return False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._should_use_cupy(x):
            self.last_backend = "torch"
            self.last_error = ""
            return super().forward(x)

        input_shape = x.shape
        x2d = x.reshape(-1, input_shape[-1])
        try:
            y2d = _cupy_linear_bias(x2d, self.weight, self.bias, self.force_fp16)
            self.last_backend = "cupy"
            self.last_error = ""
            return y2d.reshape(*input_shape[:-1], self.out_features)
        except Exception as exc:
            self.last_backend = "torch_fallback"
            self.last_error = str(exc)
            return super().forward(x)


__all__ = ["CuPyLinearCompat"]

