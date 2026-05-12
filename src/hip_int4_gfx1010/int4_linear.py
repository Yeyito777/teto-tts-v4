from __future__ import annotations

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_ext = None


def load_extension(verbose: bool = False):
    global _ext
    if _ext is None:
        # Arch PyTorch's ROCm extension helper otherwise emits every architecture
        # PyTorch was built for, making local rebuilds much slower. This runtime
        # path is specifically for the user's RX 5700 XT / gfx1010.
        os.environ.setdefault("PYTORCH_ROCM_ARCH", "gfx1010")
        root = Path(__file__).resolve().parent
        _ext = load(
            name="gfx1010_int4_linear_ext",
            sources=[str(root / "int4_linear.cpp"), str(root / "int4_linear.hip")],
            verbose=verbose,
            extra_cflags=["-DGLOG_USE_GLOG_EXPORT"],
            extra_cuda_cflags=["--offload-arch=gfx1010", "-DGLOG_USE_GLOG_EXPORT"],
        )
    return _ext


def quantize_weight_int4(weight: torch.Tensor, groupsize: int = 128):
    """Return qweight uint8 [N,ceil(K/2)], scale/zero bf16 [N,G]."""
    w = weight.detach().float().cpu().contiguous()
    n, k = w.shape
    g = (k + groupsize - 1) // groupsize
    kp = g * groupsize
    if kp != k:
        w = F.pad(w, (0, kp - k))
    wg = w.view(n, g, groupsize)
    w_min = wg.amin(dim=-1, keepdim=True)
    w_max = wg.amax(dim=-1, keepdim=True)
    scale = ((w_max - w_min).clamp(min=1e-6) / 15.0)
    zero = w_min + scale * 8.0
    q = ((wg - (zero - scale * 8.0)) / scale).round().clamp(0, 15).to(torch.uint8)
    q = q.view(n, kp)
    if k % 2:
        q = F.pad(q, (0, 1), value=8)
    lo = q[:, 0::2]
    hi = q[:, 1::2]
    packed = (lo | (hi << 4)).contiguous()
    # Drop extra padded byte if any? Kernel expects ceil(original K/2). Keep based on original k.
    packed = packed[:, : (k + 1) // 2].contiguous()
    return packed, scale.squeeze(-1).to(torch.bfloat16).contiguous(), zero.squeeze(-1).to(torch.bfloat16).contiguous()


class Gfx1010Int4Linear(nn.Module):
    def __init__(self, linear: nn.Linear, groupsize: int = 128, verbose_build: bool = False):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.groupsize = groupsize
        q, s, z = quantize_weight_int4(linear.weight, groupsize=groupsize)
        self.register_buffer("qweight", q, persistent=True)
        self.register_buffer("scales", s, persistent=True)
        self.register_buffer("zeros", z, persistent=True)
        if linear.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", linear.bias.detach().to(torch.bfloat16).cpu().contiguous(), persistent=True)
        self._verbose_build = verbose_build

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ext = load_extension(verbose=self._verbose_build)
        orig_shape = x.shape[:-1]
        x2 = x.contiguous().view(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        y2 = ext.int4_linear_bf16(x2, self.qweight, self.scales, self.zeros, self.bias, self.groupsize)
        return y2.view(*orig_shape, self.out_features)


def replace_linear(module: nn.Module, groupsize: int = 128, verbose_build: bool = False):
    count = 0
    params = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            params += child.weight.numel()
            setattr(module, name, Gfx1010Int4Linear(child, groupsize=groupsize, verbose_build=verbose_build))
            count += 1
        else:
            c, p = replace_linear(child, groupsize=groupsize, verbose_build=verbose_build)
            count += c
            params += p
    return count, params
