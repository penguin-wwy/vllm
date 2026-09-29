# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare Marlin scale preparation with a standalone stable extension.

Scales/prepare mode measures synchronized wall time including allocations,
Python loops and validation. Kernel mode replays graphs over rotating buffers;
FP8 error buffers are checked outside that timing. ZLUDA is correctness-only.
"""

import argparse
import hashlib
import json
import math
import random
import statistics
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import get_args

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tests.kernels.quantization.marlin_scales_utils import (  # noqa: E402
    load_kernels,
    load_preparation,
    run_scales,
)

CASES = {
    "e256_n2048_k7168_g32_w4a16": (256, 2048, 7168, 32, False, "mxfp4"),
    "e256_n2048_k7168_g32_w4a8": (256, 2048, 7168, 32, True, "mxfp4"),
    "e8_n4096_k8192_g128_w4a16": (8, 4096, 8192, 128, False, "generic"),
    "smoke": (3, 128, 160, 32, False, "mxfp4"),
}
BACKENDS = ("torch", "cached", "separate", "fused_per_expert", "batched")


def bits_equal(a, b):
    return (
        a.shape == b.shape
        and a.dtype == b.dtype
        and torch.equal(
            a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
        )
    )


def make_cached(ns, group, dtype, act, k):
    single = group == k or act is not None
    perm = ns["get_scale_perms"]()[int(single)]
    p = torch.tensor(perm, device="cuda")
    q = torch.tensor([0, 2, 1, 3], device="cuda")

    def run(raw, fmt, check=True):
        decoded = raw.view(torch.float8_e8m0fnu).to(dtype) if fmt == "mxfp4" else raw
        outputs = []
        generic_out = (
            torch.empty(
                (raw.shape[0], raw.shape[2], raw.shape[1]),
                dtype=raw.dtype,
                device=raw.device,
            )
            if fmt == "generic"
            else None
        )
        for i, expert in enumerate(decoded):
            s = expert.T.reshape(-1, p.numel())[:, p].reshape(-1, raw.shape[1])
            if fmt == "mxfp4":
                if act is None:
                    s = s.reshape(-1, 4)[:, q].reshape(s.shape)
                s = s.to(torch.float8_e8m0fnu)
                if act is not None:
                    if check:
                        assert s.view(torch.uint8).max() <= 249
                    s = (s.view(torch.uint8) + 6).view(torch.float8_e8m0fnu)
            if generic_out is not None:
                generic_out[i] = s
            else:
                outputs.append(s)
        if generic_out is not None:
            return generic_out
        return torch.cat([s.unsqueeze(0) for s in outputs], 0)

    return run


def make_eager(kernels, ns, backend, group, dtype, act, fmt, k):
    cached = make_cached(ns, group, dtype, act, k) if backend == "cached" else None

    def run(raw):
        if cached:
            return cached(raw, fmt)
        if fmt == "mxfp4":
            return run_scales(raw, kernels, ns, backend, group, dtype, act)
        s = raw.transpose(1, 2)
        with kernels.use_backend(backend, tile_groups=kernels._tile_groups.get()):
            return ns["marlin_moe_permute_scales"](s, k, raw.shape[1], group)

    return run


def make_kernel(kernels, ns, backend, raw, group, dtype, act, fmt, tile):
    e, n, g = raw.shape
    k = g * group
    a8 = act is not None
    out_dtype = torch.float8_e8m0fnu if fmt == "mxfp4" else dtype
    out = torch.empty((e, g, n), dtype=out_dtype, device=raw.device)
    flags = []
    if backend == "cached":
        cached = make_cached(ns, group, dtype, act, k)
        return lambda: cached(raw, fmt, check=False), flags
    if backend == "batched":
        if fmt == "generic":

            def batched_generic():
                torch.ops._C.marlin_permute_scales_out(
                    raw.transpose(1, 2), out, g == 1, tile
                )
                return out

            return batched_generic, flags
        invalid = torch.empty(
            (e * (n // 64) * math.ceil(g / tile) if a8 else 0,),
            dtype=torch.bool,
            device=raw.device,
        )
        flags.append(invalid)

        def batched():
            torch.ops._C.marlin_prepare_mxfp4_scales_out(
                raw, out, invalid, dtype == torch.float16, a8, tile
            )
            return out

        return batched, flags
    parts = [torch.empty((g, n), dtype=out_dtype, device=raw.device) for _ in range(e)]
    if fmt == "generic":

        def generic():
            for i, (src, dst) in enumerate(zip(raw, parts)):
                torch.ops._C.marlin_permute_scales_out(src.T, dst, g == 1, tile)
                out[i].copy_(dst)
            return out

        return generic, flags
    if backend == "fused_per_expert":
        flags = [
            torch.empty(
                ((n // 64) * math.ceil(g / tile) if a8 else 0,),
                dtype=torch.bool,
                device=raw.device,
            )
            for _ in range(e)
        ]

        def fused():
            for src, dst, invalid in zip(raw, parts, flags):
                torch.ops._C.marlin_prepare_mxfp4_scales_out(
                    src, dst, invalid, dtype == torch.float16, a8, tile
                )
            torch.cat([s.unsqueeze(0) for s in parts], 0, out=out)
            return out

        return fused, flags
    decoded = torch.empty(raw.shape, dtype=dtype, device=raw.device)
    permuted = [torch.empty((g, n), dtype=dtype, device=raw.device) for _ in range(e)]
    flags = [
        torch.empty(
            (math.ceil(g * n / 256) if a8 else 0,), dtype=torch.bool, device=raw.device
        )
        for _ in range(e)
    ]

    def separate():
        decoded.copy_(raw.view(torch.float8_e8m0fnu))
        for src, perm, dst, invalid in zip(decoded, permuted, parts, flags):
            torch.ops._C.marlin_permute_scales_out(src.T, perm, a8 or g == 1, tile)
            torch.ops._C.mxfp4_marlin_process_scales_out(perm, dst, invalid, a8)
        torch.cat([s.unsqueeze(0) for s in parts], 0, out=out)
        return out

    return separate, flags


def graph_measure(calls, rounds, timer):
    for fn in calls:
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    held = []
    with torch.cuda.graph(graph):
        for fn in calls:
            held.append(fn())
    for _ in range(3):
        graph.replay()
    if timer == "cupti":
        from flashinfer.testing import bench_gpu_time_with_cupti

        samples = bench_gpu_time_with_cupti(graph.replay)
        return [float(t) * 1000 / len(calls) for t in samples]
    samples = []
    for _ in range(rounds):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / len(calls))
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument(
        "--cases", nargs="+", choices=list(CASES), default=list(CASES)[:3]
    )
    parser.add_argument("--backends", nargs="+", choices=BACKENDS, default=None)
    parser.add_argument(
        "--mode", choices=["scales", "kernel", "prepare"], default="scales"
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument(
        "--layout",
        choices=["auto", "ng", "gn"],
        default="auto",
        help="Physical scale layout; auto uses NG for MXFP4, GN for generic",
    )
    parser.add_argument(
        "--format", choices=["auto", "mxfp4", "generic"], default="auto"
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--cold-mib", type=int, default=256)
    parser.add_argument(
        "--tile-groups", nargs="+", type=int, choices=[8, 16, 32], default=[32]
    )
    parser.add_argument("--timer", choices=["auto", "events", "cupti"], default="auto")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile-dir", type=Path, help="Save untimed profiler traces")
    args = parser.parse_args()
    if args.rounds < 1 or args.cold_mib < 1:
        parser.error("rounds and cold-mib must be positive")
    deps = subprocess.check_output(["readelf", "-d", str(args.library)], text=True)
    if torch.version.hip and "libcudart" in deps and not args.verify_only:
        parser.error("CUDA/ZLUDA runs are correctness-only; use --verify-only")
    torch.ops.load_library(str(args.library.resolve()))
    kernels = load_kernels()
    available = [
        "torch",
        "cached",
        *[x for x in get_args(kernels.ScaleBackend) if x != "torch"],
    ]
    if args.backends is None:
        args.backends = available
    elif any(x not in available for x in args.backends):
        parser.error(f"This branch supports only: {available}")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    timer = (
        ("events" if torch.version.hip else "cupti")
        if args.timer == "auto"
        else args.timer
    )
    metadata = {
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "hip": torch.version.hip,
        "cuda": torch.version.cuda,
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "compiled_backend": "cuda" if "libcudart" in deps else "hip",
        "validation": "outside-timing" if args.mode == "kernel" else "inside-timing",
        "mode": args.mode,
        "timer": timer,
        "cold_mib": args.cold_mib,
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "sources": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                ROOT / "csrc/libtorch_stable/quantization/marlin/marlin_scales.cu",
                ROOT / "vllm/model_executor/layers/quantization/utils/marlin_scales.py",
                ROOT / "vllm/model_executor/layers/quantization/utils/marlin_utils.py",
                ROOT
                / "vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py",
                ROOT / "tests/kernels/quantization/marlin_scales_utils.py",
                Path(__file__).resolve(),
            ]
        },
    }
    print(json.dumps(metadata), flush=True)
    rows = []
    for name in args.cases:
        e, n, k, group, a8, preset_format = CASES[name]
        fmt = preset_format if args.format == "auto" else args.format
        if fmt == "generic" and a8:
            parser.error("generic FP8 activation conversion is outside this benchmark")
        act = torch.float8_e4m3fn if a8 else None
        ns = load_preparation(kernels, act)
        torch.manual_seed(42)
        raw = torch.randint(
            110, 120, (e, n, k // group), device="cuda", dtype=torch.uint8
        )
        if fmt == "generic":
            raw = raw.to(dtype)
        layout_name = (
            ("ng" if fmt == "mxfp4" else "gn") if args.layout == "auto" else args.layout
        )
        if layout_name == "gn":
            raw = raw.transpose(1, 2).contiguous().transpose(1, 2)
        reference = make_eager(kernels, ns, "torch", group, dtype, act, fmt, k)(raw)
        variants = {}
        for backend in args.backends:
            for tile in (
                args.tile_groups if backend not in ("torch", "cached") else [32]
            ):
                fn = make_eager(kernels, ns, backend, group, dtype, act, fmt, k)
                with kernels.use_backend(
                    "torch" if backend == "cached" else backend, tile_groups=tile
                ):
                    assert bits_equal(fn(raw), reference), (name, backend, tile)
                variants[(backend, tile)] = fn
        print(f"PASS {name}: all variants bit-exact ({fmt}, {dtype})", flush=True)
        if args.verify_only:
            continue
        slots = max(
            2, math.ceil(args.cold_mib * 2**20 / (raw.numel() * raw.element_size())) + 1
        )
        buffers = [raw] + [raw.clone() for _ in range(slots - 1)]
        weights = None
        if args.mode == "prepare":
            if k % 64:
                parser.error("prepare mode requires repack-aligned K")
            weights = torch.randint(
                0, 256, (e, n, k // 2), device="cuda", dtype=torch.uint8
            )
        samples = {key: [] for key in variants}
        peaks = {key: 0 for key in variants}
        if args.mode == "kernel":
            for backend, tile in variants:
                if backend == "torch":
                    continue  # CPU indexing/validation is not graph-capturable.
                pairs = [
                    make_kernel(kernels, ns, backend, b, group, dtype, act, fmt, tile)
                    for b in buffers
                ]
                for fn, flags in pairs:
                    assert bits_equal(fn(), reference), (
                        name,
                        backend,
                        "kernel boundary",
                    )
                    for flag in flags:
                        kernels._check_invalid(flag)
                samples[(backend, tile)] = graph_measure(
                    [p[0] for p in pairs], args.rounds, timer
                )
                for _, flags in pairs:
                    for flag in flags:
                        kernels._check_invalid(flag)
                del pairs
        else:
            rng = random.Random(42)
            for iteration in range(args.rounds):
                order = list(variants)
                rng.shuffle(order)
                for backend, tile in order:
                    fn = variants[(backend, tile)]
                    # Rotate inputs per invocation, including adjacent variants.
                    b = buffers[
                        (iteration * len(order) + order.index((backend, tile))) % slots
                    ]
                    with kernels.use_backend(
                        "torch" if backend == "cached" else backend, tile_groups=tile
                    ):
                        torch.accelerator.synchronize()
                        torch.accelerator.reset_peak_memory_stats()
                        allocated = torch.accelerator.memory_allocated()
                        start = time.perf_counter()
                        packed = (
                            ns["_repack_marlin_experts"](weights, n, k, a8)
                            if weights is not None
                            else None
                        )
                        result = fn(b)
                        torch.accelerator.synchronize()
                        samples[(backend, tile)].append(
                            (time.perf_counter() - start) * 1e6
                        )
                        peaks[(backend, tile)] = max(
                            peaks[(backend, tile)],
                            torch.accelerator.max_memory_allocated() - allocated,
                        )
                        del result, packed
        for (backend, tile), values in samples.items():
            if not values:
                continue
            us = statistics.median(values)
            minimum_bytes = raw.numel() * (
                raw.element_size() + (1 if fmt == "mxfp4" else raw.element_size())
            )
            row = dict(
                case=name,
                format=fmt,
                layout=layout_name,
                input_shape=list(raw.shape),
                input_stride=list(raw.stride()),
                dtype=args.dtype,
                backend=backend,
                tile_groups=tile,
                median_us=us,
                p95_us=sorted(values)[math.ceil(len(values) * 0.95) - 1],
                min_us=min(values),
                max_us=max(values),
                effective_scale_GBps=minimum_bytes / us / 1000,
                peak_extra_MiB=(
                    peaks[(backend, tile)] / 2**20 if args.mode != "kernel" else None
                ),
                samples_us=values,
            )
            rows.append(row)
            print(json.dumps(row), flush=True)
        if args.profile_dir:
            args.profile_dir.mkdir(parents=True, exist_ok=True)
            for (backend, tile), fn in variants.items():
                with kernels.use_backend(
                    "torch" if backend == "cached" else backend, tile_groups=tile
                ):
                    torch.accelerator.synchronize()
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        result = fn(raw)
                        torch.accelerator.synchronize()
                    del result
                label = f"{name}-{backend}-{tile}"
                prof.export_chrome_trace(str(args.profile_dir / f"{label}.json"))
                counts = Counter(event.name for event in prof.events())
                (args.profile_dir / f"{label}-counts.json").write_text(
                    json.dumps(dict(counts), indent=2) + "\n"
                )
        del buffers, raw, weights, variants, reference
        torch.accelerator.empty_cache()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
