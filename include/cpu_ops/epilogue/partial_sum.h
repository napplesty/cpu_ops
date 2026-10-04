#pragma once

// Internal epilogue for split-k partial products: the first k-block stores acc
// into the workspace slab, later blocks accumulate onto the slab value
// (`source`); the reduction pass applies the real epilogue exactly once.

#include "cpu_ops/arch/simd.h"

namespace cpu_ops {
namespace epilogue {

template <typename T>
struct PartialSum {
  struct Params {};

  PartialSum() = default;
  explicit PartialSum(const Params&) {}

  T operator()(T acc, T source, bool first, bool /*last*/, int /*row*/,
               int /*col*/) const {
    return first ? acc : source + acc;
  }

  template <int N>
  simd::Vec<T, N> apply_vec(simd::Vec<T, N> acc, simd::Vec<T, N> source, bool first,
                            bool /*last*/, int /*row*/, int /*col*/) const {
    return first ? acc : simd::add(source, acc);
  }
};

}  // namespace epilogue
}  // namespace cpu_ops
