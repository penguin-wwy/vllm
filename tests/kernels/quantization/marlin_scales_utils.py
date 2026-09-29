# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone access to scale kernels and existing preparation functions."""

import ast
import importlib.util
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "vllm/model_executor/layers/quantization/utils"


def load_kernels():
    name = "_standalone_marlin_scales"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, UTILS / "marlin_scales.py")
        if spec is None or spec.loader is None:
            raise ImportError("Cannot load the scale kernel module")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def load_preparation(kernels, input_dtype=None):
    """Load actual preparation bodies without the full vLLM dependency graph."""
    ns = {
        "torch": torch,
        "math": math,
        "scale_kernels": kernels,
        "RoutedExperts": torch.nn.Module,
        "round_up": lambda x, n: (x + n - 1) // n * n,
        "logger": SimpleNamespace(warning_once=lambda *args: None),
        "get_marlin_input_dtype": lambda **kwargs: input_dtype,
        "ops": SimpleNamespace(
            gptq_marlin_repack=getattr(torch.ops._C, "gptq_marlin_repack", None)
        ),
    }
    selections = {
        "marlin_utils.py": {
            "get_scale_perms",
            "marlin_permute_scales",
            "marlin_moe_permute_scales",
            "marlin_permute_bias",
            "marlin_pad_scales",
            "marlin_pad_dim",
            "marlin_padded_nk",
            "marlin_pad_qweight",
        },
        "marlin_utils_fp4.py": {
            "mxfp4_marlin_process_scales",
            "_nvfp4_compute_scale_factor",
            "nvfp4_marlin_process_scales",
            "nvfp4_marlin_process_global_scale",
            "_repack_marlin_experts",
            "prepare_fp4_layer_for_marlin",
            "prepare_moe_fp4_layer_for_marlin",
            "prepare_moe_mxfp4_layer_for_marlin",
            "prepare_nvfp4_moe_layer_for_marlin",
            "rand_marlin_weight_mxfp4_like",
            "rand_marlin_weight_nvfp4_like",
        },
    }
    for filename, names in selections.items():
        path = UTILS / filename
        tree = ast.parse(path.read_text())
        nodes: list[ast.stmt] = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
        ]
        if {n.name for n in nodes if isinstance(n, ast.FunctionDef)} != names:
            raise RuntimeError(f"Preparation functions changed in {path}")
        exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), ns)
    return ns


def register_fake_ops():
    """Reuse production fake implementations in the standalone test process."""
    path = ROOT / "vllm/_custom_ops.py"
    tree = ast.parse(path.read_text())
    names = {
        "_marlin_permute_scales_out_fake": "marlin_permute_scales_out",
        "_mxfp4_marlin_process_scales_out_fake": "mxfp4_marlin_process_scales_out",
        "_marlin_prepare_mxfp4_scales_out_fake": "marlin_prepare_mxfp4_scales_out",
    }
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in names
            and not torch._C._dispatch_has_kernel_for_dispatch_key(
                f"_C::{names[node.name]}", "Meta"
            )
        ):
            exec(
                compile(ast.Module([node], type_ignores=[]), str(path), "exec"),
                {"register_fake": torch.library.register_fake},
            )


def load_opcheck():
    """Use the existing FP8-aware opcheck helper without importing vLLM."""
    path = ROOT / "tests/kernels/utils.py"
    nodes: list[ast.stmt] = [
        node
        for node in ast.parse(path.read_text()).body
        if (
            isinstance(node, ast.FunctionDef)
            and node.name in ("fp8_allclose", "opcheck")
        )
        or (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "ALL_OPCHECK_TEST_UTILS"
        )
    ]
    namespace = {
        "torch": torch,
        "TensorLikeType": torch.Tensor,
        "Any": Any,
        "Sequence": Sequence,
        "patch": patch,
    }
    exec(compile(ast.Module(nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace["opcheck"]


def run_scales(
    raw,
    kernels,
    ns,
    backend,
    group_size,
    param_dtype,
    input_dtype=None,
    *,
    padded_k=None,
    padded_n=None,
):
    """Run the same raw-input boundary for each benchmark variant."""
    n, g = raw.shape[-2:]
    k = g * group_size
    pk = k if padded_k is None else padded_k
    pn = n if padded_n is None else padded_n
    with kernels.use_backend(backend, tile_groups=kernels._tile_groups.get()):
        if backend in ("batched", "fused_per_expert"):
            return kernels.prepare_mxfp4_scales_selected(
                raw,
                k,
                n,
                group_size,
                param_dtype,
                input_dtype,
                padded_k=pk,
                padded_n=pn,
            )
        decoded = raw.view(torch.float8_e8m0fnu).to(param_dtype)
        experts = decoded.unsqueeze(0) if raw.ndim == 2 else decoded
        outputs = []
        for expert in experts:
            s = ns["marlin_pad_scales"](expert.T, n, k, pn, pk, group_size)
            s = ns["marlin_permute_scales"](
                s, pk, pn, group_size, input_dtype == torch.float8_e4m3fn
            )
            outputs.append(ns["mxfp4_marlin_process_scales"](s, input_dtype))
        return (
            outputs[0]
            if raw.ndim == 2
            else torch.cat([s.unsqueeze(0) for s in outputs], 0)
        )
