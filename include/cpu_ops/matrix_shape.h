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

// Runtime GEMM problem size.
struct GemmCoord {
  int m = 0;
  int n = 0;
  int k = 0;

  GemmCoord() = default;
  constexpr GemmCoord(int m_, int n_, int k_) : m(m_), n(n_), k(k_) {}
};

}  // namespace cpu_ops
