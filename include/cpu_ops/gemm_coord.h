#pragma once

namespace cpu_ops {

// Runtime GEMM problem size.
struct GemmCoord {
  int m = 0;
  int n = 0;
  int k = 0;

  GemmCoord() = default;
  constexpr GemmCoord(int m_, int n_, int k_) : m(m_), n(n_), k(k_) {}
};

}  // namespace cpu_ops
