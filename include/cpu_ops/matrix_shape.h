#pragma once

namespace cpu_ops {

// Statically sized shape with M/N/K extents.
template <int M_, int N_ = 1, int K_ = 1>
struct Shape {
  static constexpr int kM = M_;
  static constexpr int kN = N_;
  static constexpr int kK = K_;
  static constexpr int kCount = M_ * N_ * K_;
};

}  // namespace cpu_ops
