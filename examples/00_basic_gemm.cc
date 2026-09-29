#include <cstdio>
#include <vector>

#include "cpu_ops/cpu_ops.h"

int main() {
  using cpu_ops::Status;
  namespace layout = cpu_ops::layout;

  const int m = 3, n = 4, k = 5;
  std::vector<float> A(m * k), B(k * n), C(m * n, 0.0f);
  for (int i = 0; i < m; ++i)
    for (int p = 0; p < k; ++p) A[i * k + p] = static_cast<float>(i + 1);
  for (int p = 0; p < k; ++p)
    for (int j = 0; j < n; ++j) B[p * n + j] = static_cast<float>(j + 1);

  using Gemm = cpu_ops::gemm::device::Gemm<float, layout::RowMajor, float, layout::RowMajor,
                                           float, layout::RowMajor>;
  Gemm::Arguments args{{m, n, k},
                       {A.data(), k},
                       {B.data(), n},
                       {C.data(), n},
                       {C.data(), n},
                       {1.0f, 0.0f}};  // alpha, beta

  Gemm gemm;
  if (gemm(args) != Status::kSuccess) {
    std::printf("gemm failed\n");
    return 1;
  }

  // A is all-1 rows scaled by (i+1), B columns are (j+1), so
  // C(i, j) = k * (i + 1) * (j + 1).
  std::printf("C = A * B  (%dx%dx%d)\n", m, n, k);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) std::printf("%6.1f", C[i * n + j]);
    std::printf("\n");
  }
  return 0;
}
