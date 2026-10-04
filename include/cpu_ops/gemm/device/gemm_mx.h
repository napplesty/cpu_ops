#pragma once

// GEMM for OCP MX block-scaled operands:
//   D = alpha * (A .* scalesA) * (B .* scalesB) + beta * C
//
// A (m x k) and B (k x n) carry packed low-precision elements plus one E8M0
// scale per 32 elements along k; C/D/alpha/beta are f32. k must be the
// contiguous dimension of both operands (A row-major, B column-major, the
// usual "TN" arrangement for MX): scale blocks are contiguous 32-element
// runs. Decoding happens at pack time; compute and accumulation are f32.
// On AVX-512F hosts the default config widens the f32 mainloop to 512-bit
// vectors with the Fma512 tile shape (see detail::GemmMxDefaults).

#include "cpu_ops/epilogue/linear_combination.h"
#include "cpu_ops/gemm/device/gemm.h"
#include "cpu_ops/gemm/kernel/run_blocked.h"
#include "cpu_ops/gemm/threadblock/mma_policy_fma.h"
#include "cpu_ops/gemm/threadblock/mma_policy_mx.h"
#include "cpu_ops/gemm_coord.h"
#include "cpu_ops/layout/matrix.h"
#include "cpu_ops/mx_formats.h"
#include "cpu_ops/status.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace device {

namespace detail {

// ISA-dependent defaults of GemmMx. Decode at pack time is already 16-wide on
// AVX-512F hosts; there the f32 mainloop additionally switches to 512-bit
// vectors (VLEN = 16) and the Zen-4-tuned Fma512 tile shape, matching the
// f32 Gemm default. Explicit Config_/MmaPolicy_ arguments always override.
template <typename StorageA, typename StorageB>
struct GemmMxDefaults {
#if defined(CPU_OPS_SIMD_AVX512F)
  using Config = mma::Fma512GemmConfig;
  using Policy = mma::MxPolicy<StorageA, StorageB, 16>;
#else
  using Config = GemmConfig<float>;
  using Policy = mma::MxPolicy<StorageA, StorageB>;
#endif
};

}  // namespace detail

template <typename StorageA_, typename StorageB_, typename LayoutC_ = layout::RowMajor,
          typename Epilogue_ = epilogue::LinearCombination<float>,
          typename Config_ = typename detail::GemmMxDefaults<StorageA_, StorageB_>::Config,
          typename MmaPolicy_ = typename detail::GemmMxDefaults<StorageA_, StorageB_>::Policy>
class GemmMx {
 public:
  using StorageA = StorageA_;
  using StorageB = StorageB_;
  using LayoutC = LayoutC_;
  using Epilogue = Epilogue_;
  using Config = Config_;
  using MmaPolicy = MmaPolicy_;

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

  Status operator()(const Arguments& args, int num_threads = 0) const {
    const int m = args.problem_size.m;
    const int n = args.problem_size.n;
    const int k = args.problem_size.k;
    if (m < 0 || n < 0 || k < 0) return Status::kErrorInvalidProblem;
    if (m == 0 || n == 0) return Status::kSuccess;

    if (k == 0) {
      // Degenerate product: fold beta * C into D without touching A or B.
      const Epilogue epilogue(args.epilogue);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          args.ref_D.at(i, j) = epilogue(0.0f, args.ref_C.at(i, j), true, true, i, j);
        }
      }
      return Status::kSuccess;
    }

    const MxTensorRef<StorageA>& a = args.ref_A;
    const MxTensorRef<StorageB>& b = args.ref_B;
    if (!a.data || !a.scales || !b.data || !b.scales || !args.ref_C.data() ||
        !args.ref_D.data()) {
      return Status::kErrorInvalidArguments;
    }
    // Both operands are contiguous along k: ld counts elements along k for A
    // (m x k row-major) and along k for B (k x n column-major). Scale arrays
    // follow the same layout with ceil(k / 32) entries per row/column.
    const int scale_ld_min = (k + 31) / 32;
    if (a.ld < k || b.ld < k || a.ld_scales < scale_ld_min ||
        b.ld_scales < scale_ld_min) {
      return Status::kErrorInvalidArguments;
    }
    // fp4 packs two elements per byte; a row must start on a byte boundary.
    if ((mma::mx_detail2::traits<StorageA>::kLog2ElemsPerByte > 0 && (a.ld & 1)) ||
        (mma::mx_detail2::traits<StorageB>::kLog2ElemsPerByte > 0 && (b.ld & 1))) {
      return Status::kErrorInvalidArguments;
    }
    const int ldc_min = LayoutC::kIsRowMajor ? n : m;
    if (args.ref_C.ld() < ldc_min || args.ref_D.ld() < ldc_min) {
      return Status::kErrorInvalidArguments;
    }

    using Policy = MmaPolicy;
    return kernel::run_blocked<Policy, LayoutC, Epilogue, Config>(
        a, b, args.ref_C, args.ref_D, m, n, k, args.epilogue, args.split_k_slices,
        num_threads);
  }
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
