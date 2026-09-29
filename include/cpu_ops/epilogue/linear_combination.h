#pragma once

#include <cmath>
#include <type_traits>

#include "cpu_ops/detail/simd.h"

namespace cpu_ops {
namespace epilogue {

// Optional elementwise activation applied on the final write-back.
enum class Activation {
  kNone,
  kRelu,
};

// Computes D = alpha * accumulator + beta * C at the end of the mainloop.
//
// The functor is invoked once per output element per k-block. `first` marks
// the first k-block (beta * C is folded in; C is never read when beta == 0),
// `last` marks the final k-block (the activation is applied). row/col are the
// output coordinates of the element (of the vector's lane 0 for apply_vec);
// this epilogue is position-independent and ignores them.
//
// The arithmetic is pinned down so that every instantiation — vector or
// scalar, inlined or type-erased through the thread pool — rounds identically
// (the compiler's default fp-contract=fast would otherwise fuse mul+add into
// fma opportunistically and produce 1-ulp differences between call sites):
//
//   first && beta != 0:  fma(beta, source, alpha * acc)
//   first && beta == 0:  alpha * acc
//   !first:              fma(alpha, acc, source)
//
// Integer accumulators keep plain mul/add (exact under any fusion).
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
    T value = combine(acc, source, first);
    if (last && params.activation == Activation::kRelu) {
      value = value < T(0) ? T(0) : value;
    }
    return value;
  }

  // Lane-wise form of operator() for the mainloop's full-tile fast path;
  // mirrors the scalar form fma-for-fma so both paths agree bit for bit.
  // `source` is only consumed when !first or beta != 0.
  template <int N>
  simd::Vec<T, N> apply_vec(simd::Vec<T, N> acc, simd::Vec<T, N> source, bool first,
                            bool last, int /*row*/, int /*col*/) const {
    simd::Vec<T, N> value = combine_vec(acc, source, first);
    if (last && params.activation == Activation::kRelu) {
      value = simd::max(value, simd::Vec<T, N>::set1(T(0)));
    }
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
