#pragma once

// Internal epilogue for split-k partial products: writes raw accumulator sums
// (no alpha/beta, no activation, no fusion chain) into a workspace slab. The
// reduction pass afterwards applies the user's real epilogue exactly once.
//
// Semantics per k-block: first block stores acc, later blocks accumulate onto
// the slab value. `source` is the slab's current content.

#include "cpu_ops/detail/simd.h"

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
