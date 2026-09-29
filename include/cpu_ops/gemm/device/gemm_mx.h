#pragma once

// GEMM for OCP MX block-scaled operands:
//   D = alpha * (A .* scalesA) * (B .* scalesB) + beta * C
//
// A (m x k) and B (k x n) carry packed low-precision elements plus one E8M0
// scale per 32 elements along k; C/D/alpha/beta are f32. k must be the
// contiguous dimension of both operands (A row-major, B column-major, the
// usual "TN" arrangement for MX): scale blocks are contiguous 32-element
// runs. Decoding happens at pack time; compute and accumulation are f32.

#include "cpu_ops/detail/mma_policy_mx.h"
#include "cpu_ops/epilogue/linear_combination.h"
#include "cpu_ops/gemm/device/gemm.h"
#include "cpu_ops/layout.h"
#include "cpu_ops/matrix_shape.h"
#include "cpu_ops/mx_formats.h"
#include "cpu_ops/status.h"

namespace cpu_ops {
namespace gemm {
namespace device {

template <typename StorageA_, typename StorageB_, typename LayoutC_ = layout::RowMajor,
          typename Epilogue_ = epilogue::LinearCombination<float>,
          typename Config_ = GemmConfig<float>>
class GemmMx {
 public:
  using StorageA = StorageA_;
  using StorageB = StorageB_;
  using LayoutC = LayoutC_;
  using Epilogue = Epilogue_;
  using Config = Config_;

  struct Arguments {
    GemmCoord problem_size;
    MxTensorRef<StorageA> ref_A;  // m x k, row-major (k contiguous)
    MxTensorRef<StorageB> ref_B;  // k x n, column-major (k contiguous)
    TensorRef<const float, LayoutC> ref_C;
    TensorRef<float, LayoutC> ref_D;
    typename Epilogue::Params epilogue;
    // Same semantics as Gemm::Arguments::split_k_slices: 0 automatic,
    // 1 never, >1 forced (clamped so no slice is empty).
    int split_k_slices = 0;
  };

  GemmMx() = default;

  Status operator()(const Arguments& args, int num_threads = 0) const;
};

// Convenience aliases for same-format operands.
template <typename LayoutC = layout::RowMajor>
using GemmMxE4M3 = GemmMx<fp8e4m3_t, fp8e4m3_t, LayoutC>;
template <typename LayoutC = layout::RowMajor>
using GemmMxE5M2 = GemmMx<fp8e5m2_t, fp8e5m2_t, LayoutC>;
template <typename LayoutC = layout::RowMajor>
using GemmMxE2M1 = GemmMx<fp4e2m1_t, fp4e2m1_t, LayoutC>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
