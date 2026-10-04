#pragma once

#include <cmath>
#include <type_traits>

#include "cpu_ops/arch/simd.h"

namespace cpu_ops {
namespace epilogue {

// Optional elementwise activation applied on the final write-back.
enum class Activation {
  kNone,
  kRelu,
};

namespace detail {

// alpha * acc + beta * C, with the arithmetic pinned to explicit fma forms so
// every instantiation (vector or scalar, inlined or type-erased) rounds
// identically — fp-contract=fast would otherwise fuse mul+add opportunistically:
//   first && beta != 0:  fma(beta, source, alpha * acc)
//   first && beta == 0:  alpha * acc
//   !first:              fma(alpha, acc, source)
// Integer accumulators keep plain mul/add (exact under any fusion).
template <typename T>
inline T linear_combine(T alpha, T beta, T acc, T source, bool first) {
  if constexpr (std::is_floating_point<T>::value) {
    if (first) {
      return beta != T(0) ? std::fma(beta, source, alpha * acc) : alpha * acc;
    }
    return std::fma(alpha, acc, source);
  } else {
    if (first) {
      return beta != T(0) ? alpha * acc + beta * source : alpha * acc;
    }
    return source + alpha * acc;
  }
}

template <typename T, int N>
inline simd::Vec<T, N> linear_combine_vec(T alpha, T beta, simd::Vec<T, N> acc,
                                          simd::Vec<T, N> source, bool first) {
  using V = simd::Vec<T, N>;
  const V va = V::set1(alpha);
  if constexpr (std::is_floating_point<T>::value) {
    if (first) {
      return beta != T(0) ? simd::fmadd(V::set1(beta), source, simd::mul(va, acc))
                          : simd::mul(va, acc);
    }
    return simd::fmadd(va, acc, source);
  } else {
    if (first) {
      return beta != T(0) ? simd::add(simd::mul(va, acc), simd::mul(V::set1(beta), source))
                          : simd::mul(va, acc);
    }
    return simd::add(source, simd::mul(va, acc));
  }
}

}  // namespace detail

// D = alpha * accumulator + beta * C at the end of the mainloop.
//
// The functor is invoked once per output element per k-block. `first` marks
// the first k-block (beta * C is folded in; C is never read when beta == 0),
// `last` marks the final k-block (the activation is applied). row/col are the
// output coordinates of the element (of the vector's lane 0 for apply_vec);
// this epilogue is position-independent and ignores them.
template <typename T>
struct LinearCombination {
  struct Params {
    T alpha = T(1);
    T beta = T(0);
    Activation activation = Activation::kNone;
  };

  Params params;

  LinearCombination() = default;
  explicit LinearCombination(const Params& p) : params(p) {}

  T operator()(T acc, T source, bool first, bool last, int /*row*/, int /*col*/) const {
    T value = detail::linear_combine(params.alpha, params.beta, acc, source, first);
    if (last && params.activation == Activation::kRelu) {
      value = value < T(0) ? T(0) : value;
    }
    return value;
  }

  // `source` is only consumed when !first or beta != 0.
  template <int N>
  simd::Vec<T, N> apply_vec(simd::Vec<T, N> acc, simd::Vec<T, N> source, bool first,
                            bool last, int /*row*/, int /*col*/) const {
    simd::Vec<T, N> value =
        detail::linear_combine_vec(params.alpha, params.beta, acc, source, first);
    if (last && params.activation == Activation::kRelu) {
      value = simd::max(value, simd::Vec<T, N>::set1(T(0)));
    }
    return value;
  }
};

}  // namespace epilogue
}  // namespace cpu_ops
