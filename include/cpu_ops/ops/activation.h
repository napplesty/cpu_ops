#pragma once

// Elementwise activation kernels: gelu (tanh approximation) and silu.
//
// Compute is f32 on the SIMD lanes; storage may be f32, float16_t or
// bfloat16_t (narrow inputs widen at load; narrow outputs round to
// nearest-even scalar-wise — a vector f32->bf16 RNE convert is future
// work). Buffers may alias (y == x).
//
// gelu is the tanh approximation  0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3)))
// — PyTorch's approximate="tanh" — matching the epilogue::Gelu op; the
// exact-erf variant is deliberately not provided (see epilogue/fusion.h).
//
// Determinism: fixed lane formulas and a fixed elementwise mapping —
// bit-identical for any thread count (task split covers disjoint ranges).

#include <algorithm>
#include <cstddef>
#include <type_traits>

#include "cpu_ops/detail/simd.h"
#include "cpu_ops/detail/thread_pool.h"
#include "cpu_ops/element_types.h"
#include "cpu_ops/ops/detail/parallel.h"
#include "cpu_ops/status.h"

namespace cpu_ops {
namespace ops {
namespace activation_detail {

// Applies a generic lane functor (valid for any Vec<float, N>) to
// storage-typed buffers.
template <typename T, typename F>
void elementwise(const T* x, T* y, std::size_t n, F&& lane_fn, int num_threads) {
  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads()
                           : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;

  const auto widen = [](const T* q) {
    if constexpr (std::is_same<T, float>::value) {
      return FV::load(q);
    } else if constexpr (std::is_same<T, float16_t>::value) {
      return simd::widen_f16<W>(reinterpret_cast<const uint16_t*>(q));
    } else {
      return simd::widen_bf16<W>(reinterpret_cast<const uint16_t*>(q));
    }
  };
  const auto narrow = [](const FV& v, T* q) {
    if constexpr (std::is_same<T, float>::value) {
      v.store(q);
    } else if constexpr (std::is_same<T, float16_t>::value) {
      simd::narrow_f16<W>(v, reinterpret_cast<uint16_t*>(q));
    } else {
      simd::narrow_bf16<W>(v, reinterpret_cast<uint16_t*>(q));
    }
  };

  // Coarse contiguous slices (a parallel_for wave per p tasks costs tens of
  // microseconds; fine-grained waves would swamp millisecond-scale rows).
  const int max_tasks = std::max(1, p * 8);
  const std::size_t per = (n + max_tasks - 1) / max_tasks;
  const int n_tasks = static_cast<int>((n + per - 1) / per);
  const std::size_t chunk = per;
  detail::run_batched(pool, n_tasks, p, [&](int task, int /*tid*/) {
    const std::size_t i0 = static_cast<std::size_t>(task) * chunk;
    const std::size_t i1 = std::min(n, i0 + chunk);
    std::size_t i = i0;
    for (; i + W <= i1; i += W) {
      narrow(lane_fn(widen(x + i)), y + i);
    }
    for (; i < i1; ++i) {
      const simd::Vec<float, 1> r = lane_fn(
          simd::Vec<float, 1>{static_cast<float>(x[i])});
      y[i] = T(r.v[0]);
    }
  });
}

}  // namespace activation_detail

// y[i] = gelu(x[i])  (tanh approximation)
template <typename T>
Status gelu(const T* x, T* y, std::size_t n, int num_threads = 0) {
  if (!x || !y) return Status::kErrorInvalidArguments;
  activation_detail::elementwise(
      x, y, n, [](const auto& v) { return simd::gelu(v); }, num_threads);
  return Status::kSuccess;
}

// y[i] = silu(x[i]) = x[i] * sigmoid(x[i])
template <typename T>
Status silu(const T* x, T* y, std::size_t n, int num_threads = 0) {
  if (!x || !y) return Status::kErrorInvalidArguments;
  activation_detail::elementwise(
      x, y, n, [](const auto& v) { return simd::silu(v); }, num_threads);
  return Status::kSuccess;
}

}  // namespace ops
}  // namespace cpu_ops
