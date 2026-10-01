#pragma once

// Normalization kernels over rows of a [rows x cols] matrix (cols
// contiguous): rmsnorm, fused add+rmsnorm (the transformer-block
// workhorse: residual sum and its normalized form in one pass over the
// inputs), and layernorm.
//
//   rmsnorm:     y[i,j]  = x[i,j] / rms_i * w[j],  rms_i = sqrt(mean(x_i²)+eps)
//   fused_add:   r_out[i,j] = x[i,j] + r[i,j]
//                y[i,j]     = r_out[i,j] / rms(r_out_i) * w[j]
//   layernorm:   y[i,j]  = (x[i,j] − mu_i) / sqrt(var_i + eps) * w[j]
//
// Compute is f32: row statistics accumulate through two independent W-lane
// accumulators with a fixed reduction tree, the scale factors use scalar
// IEEE 1/sqrtf. Storage may be f32 / float16_t / bfloat16_t (narrow
// outputs round to nearest-even scalar-wise). fused_add stashes the summed
// residual in a per-thread f32 row scratch so the inputs are read exactly
// once from memory.
//
// Determinism: per-row fixed evaluation order — bit-identical for any
// thread count.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <type_traits>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/thread/thread_pool.h"
#include "cpu_ops/numeric_types.h"
#include "cpu_ops/ops/detail/parallel.h"
#include "cpu_ops/status.h"

namespace cpu_ops {
namespace ops {
namespace norm_detail {

inline float* new_fbuf(std::size_t elems) {
  return static_cast<float*>(::operator new(elems * sizeof(float), std::align_val_t(64)));
}
inline void del_fbuf(void* p) { ::operator delete(p, std::align_val_t(64)); }

template <typename T>
inline simd::Vec<float, simd::native_width<float>()> widen4(const T* q) {
  constexpr int W = simd::native_width<float>();
  if constexpr (std::is_same<T, float>::value) {
    return simd::Vec<float, W>::load(q);
  } else if constexpr (std::is_same<T, float16_t>::value) {
    return simd::widen_f16<W>(reinterpret_cast<const uint16_t*>(q));
  } else {
    return simd::widen_bf16<W>(reinterpret_cast<const uint16_t*>(q));
  }
}

// Σ float(x)² and Σ float(x) over [0, n) in one pass; fixed reduction.
template <typename T>
inline void sum_sumsq_row(const T* x, int n, float* s_out, float* ss_out) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  FV s0 = FV::set1(0.0f), s1 = FV::set1(0.0f);
  FV q0 = FV::set1(0.0f), q1 = FV::set1(0.0f);
  int i = 0;
  for (; i + 2 * W <= n; i += 2 * W) {
    const FV a = widen4(x + i);
    const FV b = widen4(x + i + W);
    s0 = simd::add(s0, a);
    s1 = simd::add(s1, b);
    q0 = simd::fmadd(a, a, q0);
    q1 = simd::fmadd(b, b, q1);
  }
  for (; i + W <= n; i += W) {
    const FV a = widen4(x + i);
    s0 = simd::add(s0, a);
    q0 = simd::fmadd(a, a, q0);
  }
  *s_out = simd::hsum(simd::add(s0, s1));
  *ss_out = simd::hsum(simd::add(q0, q1));
  for (; i < n; ++i) {
    const float f = static_cast<float>(x[i]);
    *s_out += f;
    *ss_out += f * f;
  }
}

template <typename T>
inline float sumsq_row(const T* x, int n) {
  float s, ss;
  sum_sumsq_row(x, n, &s, &ss);
  (void)s;
  return ss;
}

// y[i] = T(src[i] * s * float(w[i])) with f32 lane math (src is f32).
template <typename T>
inline void scale_row(const float* src, const T* w, T* y, int n, float s) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  int i = 0;
  for (; i + W <= n; i += W) {
    const FV r = simd::mul(FV::load(src + i), FV::set1(s));
    const FV o = simd::mul(r, widen4(w + i));
    if constexpr (std::is_same<T, float>::value) {
      o.store(y + i);
    } else {
      for (int l = 0; l < W; ++l) y[i + l] = T(o.v[l]);
    }
  }
  for (; i < n; ++i) y[i] = T(src[i] * s * static_cast<float>(w[i]));
}

// Store W f32 lanes to narrow storage with hardware/vector RNE converts.
template <typename T>
inline void narrow_store(const simd::Vec<float, simd::native_width<float>()>& v,
                         T* q);

template <typename T, int W>
inline void narrow_store_w(const simd::Vec<float, W>& v, T* q) {
  if constexpr (std::is_same<T, float>::value) {
    v.store(q);
  } else if constexpr (std::is_same<T, float16_t>::value) {
    simd::narrow_f16<W>(v, reinterpret_cast<uint16_t*>(q));
  } else {
    simd::narrow_bf16<W>(v, reinterpret_cast<uint16_t*>(q));
  }
}

// y[i] = T(float(x[i]) * s * float(w[i])) with f32 lane math (T input).
template <typename T>
inline void scale_row_t(const T* x, const T* w, T* y, int n, float s) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  int i = 0;
  for (; i + W <= n; i += W) {
    const FV o = simd::mul(simd::mul(widen4(x + i), FV::set1(s)), widen4(w + i));
    narrow_store_w<T, W>(o, y + i);
  }
  for (; i < n; ++i) {
    y[i] = T(static_cast<float>(x[i]) * s * static_cast<float>(w[i]));
  }
}

// y[i] = T((float(x[i]) − b) * s * float(w[i])) with f32 lane math.
template <typename T>
inline void center_scale_row(const T* x, const T* w, T* y, int n, float b,
                             float s) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  int i = 0;
  for (; i + W <= n; i += W) {
    const FV c = simd::sub(widen4(x + i), FV::set1(b));
    const FV o = simd::mul(simd::mul(c, FV::set1(s)), widen4(w + i));
    narrow_store_w<T, W>(o, y + i);
  }
  for (; i < n; ++i) {
    y[i] = T((static_cast<float>(x[i]) - b) * s * static_cast<float>(w[i]));
  }
}

// dst[i] = float(x[i]) + float(r[i]); returns nothing (f32 row scratch).
template <typename T>
inline void add_row_f32(const T* x, const T* r, float* dst, int n) {
  constexpr int W = simd::native_width<float>();
  int i = 0;
  for (; i + W <= n; i += W) {
    simd::add(widen4(x + i), widen4(r + i)).store(dst + i);
  }
  for (; i < n; ++i) {
    dst[i] = static_cast<float>(x[i]) + static_cast<float>(r[i]);
  }
}

}  // namespace norm_detail

// y[i,j] = x[i,j] / rms_i * w[j]
template <typename T>
Status rmsnorm(const T* x, const T* w, T* y, int rows, int cols, float eps,
               int num_threads = 0) {
  if (rows < 0 || cols < 1) return Status::kErrorInvalidProblem;
  if (!x || !w || !y) return Status::kErrorInvalidArguments;
  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads()
                           : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;
  const float inv_cols = 1.0f / static_cast<float>(cols);
  const int rows_per = std::max(1, (rows + p * 8 - 1) / (p * 8));
  const int n_tasks = (rows + rows_per - 1) / rows_per;
  ops::detail::run_batched(pool, n_tasks, p, [&](int task, int /*tid*/) {
    const int r0 = task * rows_per;
    const int r1 = std::min(rows, r0 + rows_per);
    for (int r = r0; r < r1; ++r) {
      const T* xr = x + static_cast<std::size_t>(r) * cols;
      T* yr = y + static_cast<std::size_t>(r) * cols;
      const float inv =
          1.0f / std::sqrtf(norm_detail::sumsq_row(xr, cols) * inv_cols + eps);
      // Second pass re-reads xr — the row is L1-resident by then.
      norm_detail::scale_row_t(xr, w, yr, cols, inv);
    }
  });
  return Status::kSuccess;
}

// res_out[i,j] = x[i,j] + res[i,j];  y[i,j] = res_out[i,j] / rms_i * w[j].
// One read pass over x and res; the sum lives in a per-thread f32 scratch.
template <typename T>
Status fused_add_rmsnorm(const T* x, const T* res, const T* w, T* y, T* res_out,
                         int rows, int cols, float eps, int num_threads = 0) {
  if (rows < 0 || cols < 1) return Status::kErrorInvalidProblem;
  if (!x || !res || !w || !y || !res_out) return Status::kErrorInvalidArguments;
  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads()
                           : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;
  const float inv_cols = 1.0f / static_cast<float>(cols);
  // Per-thread scratch (parallel_for hands out pool worker ids, not [0, p)).
  const int nslots = pool.num_threads();
  float* ws = norm_detail::new_fbuf(static_cast<std::size_t>(nslots) * cols);
  const int rows_per = std::max(1, (rows + p * 8 - 1) / (p * 8));
  const int n_tasks = (rows + rows_per - 1) / rows_per;
  ops::detail::run_batched(
      pool, n_tasks, p, [&](int task, int tid) {
        const int r0 = task * rows_per;
        const int r1 = std::min(rows, r0 + rows_per);
        for (int r = r0; r < r1; ++r) {
          const T* xr = x + static_cast<std::size_t>(r) * cols;
          const T* rr = res + static_cast<std::size_t>(r) * cols;
          T* yr = y + static_cast<std::size_t>(r) * cols;
          T* orow = res_out + static_cast<std::size_t>(r) * cols;
          float* scratch = ws + static_cast<std::size_t>(tid) * cols;
          norm_detail::add_row_f32(xr, rr, scratch, cols);
          const float ss = norm_detail::sumsq_row(scratch, cols);
          const float inv = 1.0f / std::sqrtf(ss * inv_cols + eps);
          norm_detail::scale_row(scratch, w, yr, cols, inv);
          constexpr int Wn = simd::native_width<float>();
          int i = 0;
          for (; i + Wn <= cols; i += Wn) {
            norm_detail::narrow_store_w<T, Wn>(
                simd::Vec<float, Wn>::load(scratch + i), orow + i);
          }
          for (; i < cols; ++i) orow[i] = T(scratch[i]);
        }
      });
  norm_detail::del_fbuf(ws);
  return Status::kSuccess;
}

// y[i,j] = (x[i,j] − mu_i) / sqrt(var_i + eps) * w[j]
template <typename T>
Status layernorm(const T* x, const T* w, T* y, int rows, int cols, float eps,
                 int num_threads = 0) {
  if (rows < 0 || cols < 1) return Status::kErrorInvalidProblem;
  if (!x || !w || !y) return Status::kErrorInvalidArguments;
  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads()
                           : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;
  const float inv_cols = 1.0f / static_cast<float>(cols);
  const int rows_per = std::max(1, (rows + p * 8 - 1) / (p * 8));
  const int n_tasks = (rows + rows_per - 1) / rows_per;
  ops::detail::run_batched(pool, n_tasks, p, [&](int task, int /*tid*/) {
    const int r0 = task * rows_per;
    const int r1 = std::min(rows, r0 + rows_per);
    for (int r = r0; r < r1; ++r) {
      const T* xr = x + static_cast<std::size_t>(r) * cols;
      T* yr = y + static_cast<std::size_t>(r) * cols;
      float s, ss;
      norm_detail::sum_sumsq_row(xr, cols, &s, &ss);
      const float mu = s * inv_cols;
      float var = ss * inv_cols - mu * mu;
      if (var < 0.0f) var = 0.0f;  // one-pass formula can go slightly negative
      const float inv = 1.0f / std::sqrtf(var + eps);
      norm_detail::center_scale_row(xr, w, yr, cols, mu, inv);
    }
  });
  return Status::kSuccess;
}

}  // namespace ops
}  // namespace cpu_ops
