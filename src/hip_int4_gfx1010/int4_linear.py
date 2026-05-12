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


def quantize_weight_int4(weight: torch.Tensor, groupsize: int = 128, symmetric: bool = False):
    """Return qweight uint8 [N,ceil(K/2)], scale/zero bf16 [N,G]."""
    w = weight.detach().float().cpu().contiguous()
    n, k = w.shape
    g = (k + groupsize - 1) // groupsize
    kp = g * groupsize
    if kp != k:
        w = F.pad(w, (0, kp - k))
    wg = w.view(n, g, groupsize)
    if symmetric:
        absmax = wg.abs().amax(dim=-1, keepdim=True)
        scale = (absmax.clamp(min=1e-6) / 7.0)
        zero = None
        q = (torch.round(wg / scale).clamp(-8, 7) + 8).to(torch.uint8)
    else:
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
    zero_out = None if zero is None else zero.squeeze(-1).to(torch.bfloat16).contiguous()
    return packed, scale.squeeze(-1).to(torch.bfloat16).contiguous(), zero_out


class Gfx1010Int4Linear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        groupsize: int = 128,
        verbose_build: bool = False,
        symmetric: bool = False,
        torch_dequant_threshold: int = 0,
    ):
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.groupsize = groupsize
        self.symmetric = symmetric
        self.torch_dequant_threshold = torch_dequant_threshold
        q, s, z = quantize_weight_int4(linear.weight, groupsize=groupsize, symmetric=symmetric)
        self.register_buffer("qweight", q, persistent=True)
        self.register_buffer("scales", s, persistent=True)
        if z is not None:
            self.register_buffer("zeros", z, persistent=True)
        else:
            self.zeros = None
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
        if self.torch_dequant_threshold and x2.shape[0] >= self.torch_dequant_threshold:
            # Prefill-only escape hatch: for large M, our decode-optimized GEMV
            # kernel launches one CTA per output element and becomes inefficient.
            # Dequantize this layer transiently and use rocBLAS for the large-M
            # matmul, then discard the temporary weight. Decode M=1 still uses
            # the persistent packed-int4 HIP path.
            packed = self.qweight
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            q = torch.stack((lo, hi), dim=-1).flatten(1)[:, : self.in_features]
            scales = self.scales.repeat_interleave(self.groupsize, dim=1)[:, : self.in_features].float()
            if self.symmetric:
                w = (q.float() - 8.0) * scales
            else:
                zeros = self.zeros.repeat_interleave(self.groupsize, dim=1)[:, : self.in_features].float()
                w = (q.float() - 8.0) * scales + zeros
            y2 = torch.nn.functional.linear(x2, w.to(torch.bfloat16), self.bias)
            return y2.view(*orig_shape, self.out_features)
        if self.symmetric:
            y2 = ext.int4_linear_bf16_sym(x2, self.qweight, self.scales, self.bias, self.groupsize)
        else:
            y2 = ext.int4_linear_bf16(x2, self.qweight, self.scales, self.zeros, self.bias, self.groupsize)
        return y2.view(*orig_shape, self.out_features)


def replace_linear(
    module: nn.Module,
    groupsize: int = 128,
    verbose_build: bool = False,
    skip_prefixes: tuple[str, ...] = (),
    symmetric: bool = False,
    torch_dequant_threshold: int = 0,
    _prefix: str = "",
):
    count = 0
    params = 0
    for name, child in list(module.named_children()):
        full_name = f"{_prefix}.{name}" if _prefix else name
        if skip_prefixes and any(full_name == p or full_name.startswith(p + ".") for p in skip_prefixes):
            continue
        if isinstance(child, nn.Linear):
            params += child.weight.numel()
            setattr(module, name, Gfx1010Int4Linear(
                child,
                groupsize=groupsize,
                verbose_build=verbose_build,
                symmetric=symmetric,
                torch_dequant_threshold=torch_dequant_threshold,
            ))
            count += 1
        else:
            c, p = replace_linear(
                child,
                groupsize=groupsize,
                verbose_build=verbose_build,
                skip_prefixes=skip_prefixes,
                symmetric=symmetric,
                torch_dequant_threshold=torch_dequant_threshold,
                _prefix=full_name,
            )
            count += c
            params += p
    return count, params
