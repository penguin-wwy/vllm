// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/torch_utils.h"

namespace {
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
constexpr int threads = 256;

struct Layout {
  int64_t e, g, n, se, sg, sn;
};

Layout layout(const Tensor& s) {
  STD_TORCH_CHECK(s.dim() == 2 || s.dim() == 3, "expected a 2D or 3D tensor");
  int d = s.dim();
  return {d == 3 ? s.size(0) : 1,   s.size(d - 2),   s.size(d - 1),
          d == 3 ? s.stride(0) : 0, s.stride(d - 2), s.stride(d - 1)};
}

void check_no_overlap(const Tensor& s, const Tensor& out) {
  if (s.numel() && out.numel()) {
    int64_t span = 1;
    for (int i = 0; i < s.dim(); ++i) span += (s.size(i) - 1) * s.stride(i);
    auto begin = reinterpret_cast<uintptr_t>(s.const_data_ptr());
    auto end = begin + span * s.element_size();
    auto ob = reinterpret_cast<uintptr_t>(out.const_data_ptr());
    auto oe = ob + out.numel() * out.element_size();
    STD_TORCH_CHECK(oe <= begin || ob >= end,
                    "input and output must not overlap");
  }
}

void check_output(const Tensor& s, const Tensor& out) {
  STD_TORCH_CHECK(s.is_cuda() && out.is_cuda(), "expected GPU tensors");
  STD_TORCH_CHECK(s.get_device_index() == out.get_device_index(),
                  "input and output must be on the same device");
  STD_TORCH_CHECK(out.is_contiguous(), "output must be contiguous");
  STD_TORCH_CHECK(s.dim() == out.dim(), "input and output ranks differ");
  check_no_overlap(s, out);
}

void check_invalid(const Tensor& out, const Tensor& invalid, int64_t count) {
  STD_TORCH_CHECK(
      invalid.is_cuda() && invalid.is_contiguous() &&
          invalid.get_device_index() == out.get_device_index() &&
          invalid.scalar_type() == ScalarType::Bool && invalid.numel() == count,
      "invalid buffer must be a contiguous GPU bool tensor of size ", count);
  check_no_overlap(out, invalid);
}

__device__ int permute_index(int i, bool single) {
  if (single) {
    int j = i % 32;
    return i / 32 * 32 + 2 * (j / 8) + j % 2 + 8 * ((j % 8) / 2);
  }
  return i / 64 * 64 + (i % 8) * 8 + (i % 64) / 8;
}

__device__ int64_t offset(int64_t i, Layout s) {
  return i / (s.g * s.n) * s.se + (i / s.n % s.g) * s.sg + i % s.n * s.sn;
}

template <typename T>
__global__ void permute_flat(const T* src, T* dst, Layout s, bool single,
                             bool contiguous) {
  int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < s.e * s.g * s.n) {
    int width = single ? 32 : 64;
    int64_t j = i / width * width + permute_index(i % width, single);
    dst[i] = src[contiguous ? j : offset(j, s)];
  }
}

// Both paths read a logical [E, G, N] view. Raw MXFP4 uses a transposed view
// of [E, N, G], allowing coalesced loads followed by shared-memory
// transposition.
template <typename T, int TileG, bool MX, bool A8>
__global__ void tiled_scales(const T* src, T* dst, bool* invalid, Layout s,
                             int64_t groups, int64_t columns, bool single,
                             bool round_half) {
  __shared__ uint32_t tile[64][TileG + 1];
  int64_t nt = columns / 64;
  int64_t gt = (groups + TileG - 1) / TileG;
  int64_t expert = blockIdx.x / (nt * gt);
  int64_t first_g = (blockIdx.x / nt % gt) * TileG;
  int64_t first_n = (blockIdx.x % nt) * 64;
  for (int t = threadIdx.x; t < 64 * TileG; t += blockDim.x) {
    int n = t / TileG, g = t % TileG;
    tile[n][g] =
        first_n + n < s.n && first_g + g < s.g
            ? src[expert * s.se + (first_g + g) * s.sg + (first_n + n) * s.sn]
            : 0;
  }
  __syncthreads();
  bool bad = false;
  for (int t = threadIdx.x; t < 64 * TileG; t += blockDim.x) {
    int g = t / 64, n = t % 64;
    if (first_g + g < groups) {
      int j = n;
      if constexpr (MX && !A8) j = (n & ~3) + (n % 2) * 2 + (n % 4) / 2;
      T value = tile[permute_index(j, single)][g];
      if constexpr (MX) {
        if (round_half) {
          value = value < 103 ? 0 : (value > 142 ? 255 : value);
        }
        if constexpr (A8) {
          bad |= value > 249;
          value += 6;
        }
      }
      dst[(expert * groups + first_g + g) * columns + first_n + n] = value;
    }
  }
  if constexpr (MX && A8) {
    int any_bad = __syncthreads_or(bad);
    if (threadIdx.x == 0) invalid[blockIdx.x] = any_bad;
  }
}

void check_launch() {
  auto error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));
}

template <typename T, bool MX, bool A8>
void launch_tiled(const Tensor& s, Tensor& out, bool* invalid, Layout shape,
                  int64_t groups, int64_t columns, bool single, bool round_half,
                  int tile_groups, cudaStream_t stream) {
  int64_t blocks =
      shape.e * (columns / 64) * ((groups + tile_groups - 1) / tile_groups);
  STD_TORCH_CHECK(blocks <= INT32_MAX, "launch grid too large");
  if (!blocks) return;
#define LAUNCH(G)                                                           \
  tiled_scales<T, G, MX, A8><<<blocks, threads, 0, stream>>>(               \
      reinterpret_cast<const T*>(s.const_data_ptr()),                       \
      reinterpret_cast<T*>(out.mutable_data_ptr()), invalid, shape, groups, \
      columns, single, round_half)
  if (tile_groups == 8) {
    LAUNCH(8);
  } else if (tile_groups == 16) {
    LAUNCH(16);
  } else {
    LAUNCH(32);
  }
#undef LAUNCH
}

void check_tile(int64_t tile_groups) {
  STD_TORCH_CHECK(tile_groups == 8 || tile_groups == 16 || tile_groups == 32,
                  "tile_groups must be 8, 16, or 32");
}

void marlin_permute_scales_out(const Tensor& s, Tensor out, bool single,
                               int64_t tile_groups) {
  check_output(s, out);
  check_tile(tile_groups);
  auto a = layout(s), b = layout(out);
  STD_TORCH_CHECK(a.e == b.e && a.g == b.g && a.n == b.n &&
                      s.scalar_type() == out.scalar_type(),
                  "output shape/dtype mismatch");
  STD_TORCH_CHECK((a.g * a.n) % (single ? 32 : 64) == 0,
                  "incomplete permutation tile");
  torch::stable::accelerator::DeviceGuard guard(s.get_device_index());
  auto stream = get_current_cuda_stream(s.get_device_index());
  if (!s.numel()) return;
  bool tiled = a.n % 64 == 0 && a.sg == 1;
  auto launch = [&]<typename T>() {
    if (tiled) {
      launch_tiled<T, false, false>(s, out, nullptr, a, a.g, a.n, single, false,
                                    tile_groups, stream);
    } else {
      int64_t blocks = (s.numel() + threads - 1) / threads;
      STD_TORCH_CHECK(blocks <= INT32_MAX, "launch grid too large");
      permute_flat<<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const T*>(s.const_data_ptr()),
          reinterpret_cast<T*>(out.mutable_data_ptr()), a, single,
          s.is_contiguous());
    }
  };
  switch (s.element_size()) {
    case 1:
      launch.template operator()<uint8_t>();
      break;
    case 2:
      launch.template operator()<uint16_t>();
      break;
    case 4:
      launch.template operator()<uint32_t>();
      break;
    default:
      STD_TORCH_CHECK(false, "expected 1-, 2-, or 4-byte scales");
  }
  check_launch();
}

void marlin_prepare_mxfp4_scales_out(const Tensor& s, Tensor out,
                                     Tensor invalid, bool round_half, bool a8,
                                     int64_t tile_groups) {
  check_output(s, out);
  check_tile(tile_groups);
  auto raw = layout(s), b = layout(out);
  Layout a{raw.e, raw.n, raw.g, raw.se, raw.sn, raw.sg};
  STD_TORCH_CHECK((s.scalar_type() == ScalarType::Byte ||
                   s.scalar_type() == ScalarType::Float8_e8m0fnu) &&
                      out.scalar_type() == ScalarType::Float8_e8m0fnu,
                  "expected encoded E8M0 input and E8M0 output");
  STD_TORCH_CHECK(a.e == b.e && a.g <= b.g && a.n <= b.n && b.n % 64 == 0,
                  "invalid padded output shape");
  int64_t blocks = b.e * (b.n / 64) * ((b.g + tile_groups - 1) / tile_groups);
  check_invalid(out, invalid, a8 ? blocks : 0);
  check_no_overlap(s, invalid);
  torch::stable::accelerator::DeviceGuard guard(s.get_device_index());
  auto stream = get_current_cuda_stream(s.get_device_index());
  auto bad = reinterpret_cast<bool*>(invalid.mutable_data_ptr());
  if (a8) {
    launch_tiled<uint8_t, true, true>(s, out, bad, a, b.g, b.n, true,
                                      round_half, tile_groups, stream);
  } else {
    launch_tiled<uint8_t, true, false>(s, out, bad, a, b.g, b.n, b.g == 1,
                                       round_half, tile_groups, stream);
  }
  check_launch();
}
}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "marlin_permute_scales_out(Tensor s, Tensor! out, bool single, int "
      "tile_groups=32) -> ()");
  m.def(
      "marlin_prepare_mxfp4_scales_out(Tensor s, Tensor! out, Tensor! invalid, "
      "bool round_half, bool a8, int tile_groups=32) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("marlin_permute_scales_out", TORCH_BOX(&marlin_permute_scales_out));
  m.impl("marlin_prepare_mxfp4_scales_out",
         TORCH_BOX(&marlin_prepare_mxfp4_scales_out));
}
