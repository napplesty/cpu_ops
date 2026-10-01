#pragma once

// Composable epilogue fusion. Ops are small position-aware functors applied
// in order after the linear combination (alpha * acc + beta * C) on the final
// k-block, while the output tile is still in registers:
//
//   D = chain( alpha * acc + beta * C )
//
// Every op implements a scalar and a lane-wise form (both are required — the
// mainloop picks per tile):
//
//   T operator()(T x, int row, int col) const;
//   template <typename T, int N>
//   simd::Vec<T, N> apply_vec(simd::Vec<T, N> x, int row, int col0) const;
//
// col0 is the column of lane 0. Ops carrying per-column data (bias, scales)
// do a wide load at col0; per-row data is broadcast.

#include <cmath>
#include <tuple>
#include <type_traits>
#include <utility>

#include "cpu_ops/arch/simd.h"

namespace cpu_ops {
namespace epilogue {

// x -> max(x, 0)
struct Relu {
  template <typename T>
  T operator()(T x, int /*row*/, int /*col*/) const {
    return x < T(0) ? T(0) : x;
  }
  template <typename T, int N>
  simd::Vec<T, N> apply_vec(simd::Vec<T, N> x, int /*row*/, int /*col*/) const {
    return simd::max(x, simd::Vec<T, N>::set1(T(0)));
  }
};

// x -> gelu(x), tanh approximation (matches ops::gelu):
//   0.5x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))
struct Gelu {
  template <typename T>
  T operator()(T x, int /*row*/, int /*col*/) const {
    const double inner = 0.7978845608028654 *
                         ((double)x + 0.044715 * (double)x * (double)x * (double)x);
    return T(0.5 * ((double)x * (1.0 + std::tanh(inner))));
  }
  template <typename U, int N>
  simd::Vec<U, N> apply_vec(simd::Vec<U, N> x, int /*row*/, int /*col*/) const {
    if constexpr (std::is_same<U, float>::value) {
      return simd::gelu(x);
    } else {
      simd::Vec<U, N> r;
      for (int i = 0; i < N; ++i) r.v[i] = (*this)(x.v[i], 0, 0);
      return r;
    }
  }
};

// x -> silu(x) = x * sigmoid(x)
struct Silu {
  template <typename T>
  T operator()(T x, int /*row*/, int /*col*/) const {
    return T((double)x / (1.0 + std::exp(-(double)x)));
  }
  template <typename U, int N>
  simd::Vec<U, N> apply_vec(simd::Vec<U, N> x, int /*row*/, int /*col*/) const {
    if constexpr (std::is_same<U, float>::value) {
      return simd::silu(x);
    } else {
      simd::Vec<U, N> r;
      for (int i = 0; i < N; ++i) r.v[i] = (*this)(x.v[i], 0, 0);
      return r;
    }
  }
};

// x -> min(max(x, lo), hi)
template <typename T>
struct Clamp {
  T lo;
  T hi;
  T operator()(T x, int /*row*/, int /*col*/) const {
    return x < lo ? lo : (hi < x ? hi : x);
  }
  template <typename U, int N>
  simd::Vec<U, N> apply_vec(simd::Vec<U, N> x, int /*row*/, int /*col*/) const {
    return simd::min(simd::max(x, simd::Vec<U, N>::set1(static_cast<U>(lo))),
                     simd::Vec<U, N>::set1(static_cast<U>(hi)));
  }
};

// x(i,j) -> x + bias[j]   (bias broadcast along rows, e.g. a linear layer's)
template <typename T>
struct BiasAdd {
  const T* bias = nullptr;
  T operator()(T x, int /*row*/, int col) const { return x + bias[col]; }
  template <typename U, int N>
  simd::Vec<U, N> apply_vec(simd::Vec<U, N> x, int /*row*/, int col) const {
    return simd::add(x, simd::Vec<U, N>::load(bias + col));
  }
};

// x(i,j) -> x * scale[i]   (e.g. per-row dequantization scale)
template <typename T>
struct ScalePerRow {
  const T* scale = nullptr;
  T operator()(T x, int row, int /*col*/) const { return x * scale[row]; }
  template <typename U, int N>
  simd::Vec<U, N> apply_vec(simd::Vec<U, N> x, int row, int /*col*/) const {
    return simd::mul(x, simd::Vec<U, N>::set1(static_cast<U>(scale[row])));
  }
};

// x(i,j) -> x * scale[j]   (e.g. per-output-channel weight scale)
template <typename T>
struct ScalePerCol {
  const T* scale = nullptr;
  T operator()(T x, int /*row*/, int col) const { return x * scale[col]; }
  template <typename U, int N>
  simd::Vec<U, N> apply_vec(simd::Vec<U, N> x, int /*row*/, int col) const {
    return simd::mul(x, simd::Vec<U, N>::load(scale + col));
  }
};

// Ordered composition of ops; Chain<> with no ops is the identity.
template <typename... Ops>
struct Chain {
  std::tuple<Ops...> ops;

  Chain(Ops... o) : ops(std::move(o)...) {}

  template <typename T>
  T apply(T x, int row, int col) const {
    return apply_impl(x, row, col, std::index_sequence_for<Ops...>{});
  }
  template <typename T, int N>
  simd::Vec<T, N> apply_vec(simd::Vec<T, N> x, int row, int col) const {
    return apply_vec_impl(x, row, col, std::index_sequence_for<Ops...>{});
  }

 private:
  template <typename T, std::size_t... I>
  T apply_impl(T x, int row, int col, std::index_sequence<I...>) const {
    (void)row;
    (void)col;
    T v = x;
    ((v = std::get<I>(ops)(v, row, col)), ...);
    return v;
  }
  template <typename T, int N, std::size_t... I>
  simd::Vec<T, N> apply_vec_impl(simd::Vec<T, N> x, int row, int col,
                                 std::index_sequence<I...>) const {
    (void)row;
    (void)col;
    simd::Vec<T, N> v = x;
    ((v = std::get<I>(ops).apply_vec(v, row, col)), ...);
    return v;
  }
};

// GEMM epilogue with a fused op chain:
//   D(i,j) = chain(alpha * acc + beta * C(i,j))   on the final k-block.
//
// Drop-in replacement for LinearCombination (same epilogue concept); use it as
// the Epilogue template argument of gemm::device::Gemm, see
// gemm/device/gemm_fused.h.
//
// Like LinearCombination, the linear-combination arithmetic is pinned to
// explicit fma forms (see linear_combination.h) so that every instantiation
// context — inlined or type-erased — rounds identically; the chain then runs
// on top unchanged.
template <typename T, typename... Ops>
struct LinearCombinationFused {
  struct Params {
    T alpha = T(1);
    T beta = T(0);
    Chain<Ops...> chain{};
  };

  Params params;

  LinearCombinationFused() = default;
  explicit LinearCombinationFused(const Params& p) : params(p) {}

  T operator()(T acc, T source, bool first, bool last, int row, int col) const {
    T value = combine(acc, source, first);
    if (last) value = params.chain.apply(value, row, col);
    return value;
  }

  // Lane-wise form of operator(); mirrors the scalar form fma-for-fma.
  template <int N>
  simd::Vec<T, N> apply_vec(simd::Vec<T, N> acc, simd::Vec<T, N> source, bool first,
                            bool last, int row, int col) const {
    simd::Vec<T, N> value = combine_vec(acc, source, first);
    if (last) value = params.chain.apply_vec(value, row, col);
    return value;
  }

 private:
  T combine(T acc, T source, bool first) const {
    if constexpr (std::is_floating_point<T>::value) {
      if (first) {
        return params.beta != T(0)
                   ? std::fma(params.beta, source, params.alpha * acc)
                   : params.alpha * acc;
      }
      return std::fma(params.alpha, acc, source);
    } else {
      if (first) {
        return params.beta != T(0) ? params.alpha * acc + params.beta * source
                                   : params.alpha * acc;
      }
      return source + params.alpha * acc;
    }
  }

  template <int N>
  simd::Vec<T, N> combine_vec(simd::Vec<T, N> acc, simd::Vec<T, N> source,
                              bool first) const {
    using V = simd::Vec<T, N>;
    const V alpha = V::set1(params.alpha);
    if constexpr (std::is_floating_point<T>::value) {
      if (first) {
        return params.beta != T(0)
                   ? simd::fmadd(V::set1(params.beta), source, simd::mul(alpha, acc))
                   : simd::mul(alpha, acc);
      }
      return simd::fmadd(alpha, acc, source);
    } else {
      if (first) {
        return params.beta != T(0)
                   ? simd::add(simd::mul(alpha, acc),
                               simd::mul(V::set1(params.beta), source))
                   : simd::mul(alpha, acc);
      }
      return simd::add(source, simd::mul(alpha, acc));
    }
  }
};

}  // namespace epilogue
}  // namespace cpu_ops
