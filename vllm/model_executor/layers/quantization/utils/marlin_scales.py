# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental Marlin scale kernels; select explicitly while benchmarking."""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Literal

import torch

ScaleBackend = Literal["torch", "batched", "fused_per_expert"]
_backend: ContextVar[ScaleBackend] = ContextVar("marlin_scale_backend", default="torch")
_tile_groups: ContextVar[int] = ContextVar("marlin_scale_tile_groups", default=32)


def get_backend() -> ScaleBackend:
    return _backend.get()


@contextmanager
def use_backend(backend: ScaleBackend, *, tile_groups: int = 32):
    """Select scale preparation for the current context without changing defaults."""
    if backend not in ("torch", "batched", "fused_per_expert"):
        raise ValueError(f"Unknown scale backend: {backend}")
    if tile_groups not in (8, 16, 32):
        raise ValueError("tile_groups must be 8, 16, or 32")
    token = _backend.set(backend)
    tile_token = _tile_groups.set(tile_groups)
    try:
        yield
    finally:
        _tile_groups.reset(tile_token)
        _backend.reset(token)


def _check_activation(input_dtype):
    if input_dtype not in (None, torch.float16, torch.bfloat16, torch.float8_e4m3fn):
        raise ValueError("MXFP4 requires FP16/BF16 or FP8 E4M3 activations")
    return input_dtype == torch.float8_e4m3fn


def _check_invalid(invalid):
    if invalid.numel() and bool(invalid.any()):
        raise AssertionError("MXFP4 scales must have E8M0 exponent <= 249 for FP8")


def permute_scales(s, size_k, size_n, group_size, is_a_8bit=False, *, out=None):
    """Permute logical [G, N] or [E, G, N] scales, preserving their dtype."""
    if s.ndim == 1:
        s = s.reshape(-1, size_n)
    if s.ndim not in (2, 3) or s.shape[-1] != size_n:
        raise ValueError("Expected [G, N] or [E, G, N] scales")
    if out is None:
        out = torch.empty(s.shape, dtype=s.dtype, device=s.device)
    single = not (group_size < size_k and group_size != -1 and not is_a_8bit)
    torch.ops._C.marlin_permute_scales_out(s, out, single, _tile_groups.get())
    return out


def prepare_mxfp4_scales(
    scales,
    size_k,
    size_n,
    group_size,
    param_dtype,
    input_dtype=None,
    *,
    padded_k=None,
    padded_n=None,
    out=None,
    invalid=None,
    check=True,
):
    """Fuse raw [N, G] or [E, N, G] E8M0 scales into Marlin layout."""
    a8 = _check_activation(input_dtype)
    if param_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError("Expected FP16, BF16, or FP32 intermediate precision")
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("Expected raw uint8 or E8M0 scales")
    if group_size <= 0 or size_k % group_size:
        raise ValueError("size_k must be divisible by positive group_size")
    if scales.ndim not in (2, 3) or scales.shape[-2:] != (size_n, size_k // group_size):
        raise ValueError("Expected raw [N, K/group_size] or [E, N, K/group_size]")
    pk = size_k if padded_k is None else padded_k
    pn = size_n if padded_n is None else padded_n
    if pk < size_k or pn < size_n or pk % group_size or pn % 64:
        raise ValueError("Invalid padded dimensions; padded_n must be divisible by 64")
    shape = (*scales.shape[:-2], pk // group_size, pn)
    if out is None:
        out = torch.empty(shape, dtype=torch.float8_e8m0fnu, device=scales.device)
    elif out.shape != shape:
        raise ValueError("Output shape mismatch")
    tile = _tile_groups.get()
    experts = scales.shape[0] if scales.ndim == 3 else 1
    blocks = experts * (pn // 64) * ((pk // group_size + tile - 1) // tile)
    if invalid is None:
        if a8 and not check:
            raise ValueError("Unchecked FP8 calls must supply an invalid buffer")
        invalid = torch.empty(
            (blocks if a8 else 0,), dtype=torch.bool, device=scales.device
        )
    torch.ops._C.marlin_prepare_mxfp4_scales_out(
        scales, out, invalid, param_dtype == torch.float16, a8, tile
    )
    if check:
        _check_invalid(invalid)
    return out


def prepare_mxfp4_scales_selected(scales, *args, **kwargs):
    """Use E=1 launches as an ablation of the batched implementation."""
    if get_backend() == "fused_per_expert" and scales.ndim == 3:
        if scales.shape[0] == 0:
            return prepare_mxfp4_scales(scales, *args, **kwargs)
        outputs = [prepare_mxfp4_scales(s, *args, **kwargs) for s in scales]
        return torch.cat([s.unsqueeze(0) for s in outputs], 0)
    return prepare_mxfp4_scales(scales, *args, **kwargs)
