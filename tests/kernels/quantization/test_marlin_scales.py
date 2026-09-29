# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bit-exact layout/encoding tests shared by native CUDA and standalone HIP.

The contract is immutable scales in, Marlin bytes out. Compare against existing
PyTorch preparation to catch permutation, precision, and expert-boundary errors
at the tensor level before GEMM or model tests.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.kernels.quantization.marlin_scales_utils import (
    load_kernels,
    load_opcheck,
    load_preparation,
    register_fake_ops,
    run_scales,
)


@pytest.fixture(scope="module")
def api():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    if not hasattr(torch.ops._C, "marlin_permute_scales_out"):
        from vllm import _custom_ops  # noqa: F401
    kernels = load_kernels()
    register_fake_ops()
    return kernels, load_preparation(kernels)


def assert_bytes(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.is_contiguous()
    assert torch.equal(
        actual.view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.uint8]
)
@pytest.mark.parametrize(
    "group_size,a8", [(-1, False), (32, False), (128, False), (32, True)]
)
@pytest.mark.parametrize("transpose", [False, True])
def test_permutation_preserves_bits_and_experts(api, dtype, group_size, a8, transpose):
    kernels, ns = api
    g, n = (1 if group_size == -1 else 256 // group_size), 128
    x = torch.arange(3 * g * n, device="cuda").reshape(3, g, n).to(dtype)
    if transpose:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    expected = torch.stack(
        [ns["marlin_permute_scales"](s, 256, n, group_size, a8) for s in x]
    )
    for tile in (8, 16, 32):
        with kernels.use_backend("batched", tile_groups=tile):
            assert_bytes(kernels.permute_scales(x, 256, n, group_size, a8), expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("a8", [False, True])
@pytest.mark.parametrize(
    "batched,padded", [(False, False), (True, False), (True, True)]
)
def test_fused_scales_match_encoding_and_padding(api, dtype, a8, batched, padded):
    kernels, ns = api
    n, g = 128, 5
    high = (143 if dtype == torch.float16 else 250) if a8 else 256
    backing = (torch.arange(3 * n * g * 4, device="cuda") % high).to(torch.uint8)
    raw = backing.reshape(3, n * 2, g * 2)[:, ::2, ::2]
    raw.copy_((torch.arange(raw.numel(), device="cuda") % high).reshape(raw.shape))
    if not batched:
        raw = raw[0]
    before = raw.clone()
    act = torch.float8_e4m3fn if a8 else None
    kwargs = {"padded_k": 192, "padded_n": 192} if padded else {}
    expected = run_scales(raw, kernels, ns, "torch", 32, dtype, act, **kwargs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for backend in ("batched", "fused_per_expert"):
            actual = run_scales(raw, kernels, ns, backend, 32, dtype, act, **kwargs)
            assert_bytes(actual, expected)
    stream.synchronize()
    assert torch.equal(raw, before)


@pytest.mark.parametrize("backend", ["torch", "batched", "fused_per_expert"])
def test_fp8_rejects_overflow_but_keeps_249_boundary(api, backend):
    kernels, ns = api
    raw = torch.full((2, 64, 1), 249, dtype=torch.uint8, device="cuda")
    act = torch.float8_e4m3fn
    good = run_scales(raw, kernels, ns, backend, 32, torch.bfloat16, act)
    assert torch.all(good.view(torch.uint8) == 255)
    raw[1, 63, 0] = 250
    with pytest.raises(AssertionError):
        run_scales(raw, kernels, ns, backend, 32, torch.bfloat16, act)


def test_out_reuse_and_graph_replay_observe_new_scales(api):
    kernels, _ = api
    raw = torch.full((2, 128, 4), 110, dtype=torch.uint8, device="cuda")
    out = kernels.prepare_mxfp4_scales(raw, 128, 128, 32, torch.bfloat16)
    ptr = out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        kernels.prepare_mxfp4_scales(raw, 128, 128, 32, torch.bfloat16, out=out)
    for exponent in (115, 120, 125):
        raw.fill_(exponent)
        graph.replay()
        torch.accelerator.synchronize()
        assert out.data_ptr() == ptr
        assert torch.all(out.view(torch.uint8) == exponent)


def test_empty_batch_and_overlap_rejection(api):
    kernels, _ = api
    raw = torch.empty((0, 64, 1), dtype=torch.uint8, device="cuda")
    assert kernels.prepare_mxfp4_scales(raw, 32, 64, 32, torch.bfloat16).shape == (
        0,
        1,
        64,
    )
    x = torch.ones((2, 64), dtype=torch.float16, device="cuda")
    with pytest.raises(RuntimeError, match="overlap"):
        kernels.permute_scales(x, 64, 64, 32, out=x)


def test_invalid_flags_cannot_overwrite_input(api):
    kernels, _ = api
    raw = torch.full((2, 64, 1), 120, dtype=torch.uint8, device="cuda")
    invalid = raw.view(torch.bool).reshape(-1)[:2]
    with pytest.raises(RuntimeError, match="overlap"):
        kernels.prepare_mxfp4_scales(
            raw, 32, 64, 32, torch.bfloat16, torch.float8_e4m3fn, invalid=invalid
        )


def test_custom_op_mutation_and_fake_contracts(api):
    kernels, _ = api
    opcheck = load_opcheck()
    raw = torch.full((2, 64, 4), 120, dtype=torch.uint8, device="cuda")
    out = torch.empty((2, 4, 64), dtype=torch.float8_e8m0fnu, device="cuda")
    invalid = torch.empty(2, dtype=torch.bool, device="cuda")
    opcheck(
        torch.ops._C.marlin_prepare_mxfp4_scales_out,
        (raw, out, invalid, False, True, 32),
    )
    s = raw.view(torch.float8_e8m0fnu).to(torch.bfloat16).transpose(1, 2)
    permuted = torch.empty(s.shape, dtype=s.dtype, device=s.device)
    opcheck(torch.ops._C.marlin_permute_scales_out, (s, permuted, False, 32))


@pytest.mark.parametrize(
    "nvfp4,variant,input_dtype",
    [
        (False, variant, act)
        for variant in ("dense", "mixed_moe", "pure_moe")
        for act in (None, torch.float8_e4m3fn)
    ]
    + [(True, variant, None) for variant in ("dense", "mixed_moe")],
)
def test_preparation_call_sites_match_reference(api, nvfp4, variant, input_dtype):
    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        pytest.skip("Marlin weight repack is unavailable on this build")
    kernels, _ = api
    ns = load_preparation(kernels, input_dtype)
    torch.manual_seed(12)
    e, n, k = 2, 128, 128
    if variant == "dense":
        n, k = 200, 288
    group = 16 if nvfp4 else 32

    def weight(shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda")

    def scale(shape):
        if nvfp4:
            return torch.ones(shape, dtype=torch.float8_e4m3fn, device="cuda")
        return torch.randint(110, 120, shape, dtype=torch.uint8, device="cuda")

    def run(backend):
        torch.manual_seed(12)
        layer = torch.nn.Module()
        layer.params_dtype = torch.bfloat16
        with kernels.use_backend(backend):
            if variant == "dense":
                layer.output_size_per_partition, layer.input_size_per_partition = n, k
                layer.weight, layer.weight_scale = (
                    weight((n, k // 2)),
                    scale((n, k // group)),
                )
                if nvfp4:
                    layer.weight_global_scale = torch.ones(1, device="cuda")
                ns["prepare_fp4_layer_for_marlin"](layer, input_dtype)
                return layer.weight_scale, getattr(layer, "weight_global_scale", None)
            w13, w2 = weight((e, 2 * n, k // 2)), weight((e, k, n // 2))
            s13, s2 = scale((e, 2 * n, k // group)), scale((e, k, n // group))
            if variant == "pure_moe":
                result = ns["prepare_moe_mxfp4_layer_for_marlin"](
                    layer, w13, w2, s13, s2, None, None
                )
                return result[2], result[3]
            layer.moe_config = SimpleNamespace(
                num_local_experts=e, hidden_dim=k, intermediate_size_per_partition=n
            )
            layer.w13_weight, layer.w2_weight = w13, w2
            layer.w13_weight_scale, layer.w2_weight_scale = s13, s2
            if nvfp4:
                layer.w13_weight_scale_2 = torch.ones(1, device="cuda")
                layer.w2_weight_scale_2 = torch.ones(1, device="cuda")
            ns["prepare_moe_fp4_layer_for_marlin"](layer, input_dtype)
            return layer.w13_weight_scale, layer.w2_weight_scale

    expected = run("torch")
    for backend in ("batched", "fused_per_expert"):
        for actual, reference in zip(run(backend), expected):
            if reference is not None:
                assert_bytes(actual, reference)


@pytest.mark.parametrize("a8", [False, True])
def test_single_group_uses_channelwise_permutation(api, a8):
    kernels, ns = api
    raw = (
        (torch.arange(3 * 128, device="cuda") % 30 + 110)
        .to(torch.uint8)
        .reshape(3, 128, 1)
    )
    act = torch.float8_e4m3fn if a8 else None
    expected = run_scales(raw, kernels, ns, "torch", 32, torch.bfloat16, act)
    for backend in ("batched", "fused_per_expert"):
        assert_bytes(
            run_scales(raw, kernels, ns, backend, 32, torch.bfloat16, act), expected
        )


def test_unchecked_fp8_reports_errors_for_later_validation(api):
    kernels, _ = api
    raw = torch.full((2, 64, 1), 250, dtype=torch.uint8, device="cuda")
    invalid = torch.empty(2, dtype=torch.bool, device="cuda")
    kernels.prepare_mxfp4_scales(
        raw,
        32,
        64,
        32,
        torch.bfloat16,
        torch.float8_e4m3fn,
        invalid=invalid,
        check=False,
    )
    with pytest.raises(AssertionError):
        kernels._check_invalid(invalid)


@pytest.mark.parametrize("nvfp4", [False, True])
def test_weight_test_helpers_keep_independent_dequantized_reference(api, nvfp4):
    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        pytest.skip("Marlin weight repack is unavailable on this build")
    kernels, ns = api
    name = "rand_marlin_weight_nvfp4_like" if nvfp4 else "rand_marlin_weight_mxfp4_like"
    w = torch.ones((64, 128), dtype=torch.bfloat16, device="cuda")
    group = 16 if nvfp4 else 32
    torch.manual_seed(8)
    expected = ns[name](w, group)
    for backend in ("batched", "fused_per_expert"):
        torch.manual_seed(8)
        with kernels.use_backend(backend):
            actual = ns[name](w, group)
        for a, b in zip(actual, expected):
            assert torch.equal(
                a.contiguous().reshape(-1).view(torch.uint8),
                b.contiguous().reshape(-1).view(torch.uint8),
            )


def test_nvfp4_specialized_moe_keeps_global_scale_and_padding(api):
    if not hasattr(torch.ops._C, "gptq_marlin_repack"):
        pytest.skip("Marlin weight repack is unavailable on this build")
    kernels, ns = api
    layer = SimpleNamespace(
        num_experts=2,
        hidden_size=128,
        intermediate_size_per_partition=96,
        params_dtype=torch.bfloat16,
    )
    w13 = torch.randint(0, 256, (2, 192, 64), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (2, 128, 48), dtype=torch.uint8, device="cuda")
    s13 = (
        (torch.arange(2 * 192 * 8, device="cuda") % 8 + 1)
        .reshape(2, 192, 8)
        .to(torch.float8_e4m3fn)
    )
    s2 = (
        (torch.arange(2 * 128 * 6, device="cuda") % 4 + 1)
        .reshape(2, 128, 6)
        .to(torch.float8_e4m3fn)
    )
    global_scale = torch.ones(1, device="cuda")

    def run():
        return ns["prepare_nvfp4_moe_layer_for_marlin"](
            layer, w13, s13, global_scale, w2, s2, global_scale, True
        )

    expected = run()
    for backend in ("batched",):
        with kernels.use_backend(backend):
            for a, b in zip(run(), expected):
                assert_bytes(a, b)
