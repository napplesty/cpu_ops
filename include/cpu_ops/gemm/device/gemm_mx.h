#pragma once

// GEMM for OCP MX block-scaled operands:
//   D = alpha * (A .* scalesA) * (B .* scalesB) + beta * C
// A (m x k) and B (k x n) carry packed low-precision elements plus one E8M0
// scale per 32 elements along k; k must be the contiguous dimension of both
// ("TN": A row-major, B column-major). C/D/alpha/beta are f32.

#include <type_traits>

#include "cpu_ops/epilogue/linear_combination.h"
#include "cpu_ops/gemm/device/gemm.h"
#include "cpu_ops/gemm/kernel/run_blocked.h"
#include "cpu_ops/gemm/threadblock/mma_policy_fma.h"
#include "cpu_ops/gemm/threadblock/mma_policy_mx.h"
#include "cpu_ops/gemm/threadblock/mma_policy_mx_vnni.h"
#include "cpu_ops/gemm_coord.h"
#include "cpu_ops/layout/matrix.h"
#include "cpu_ops/mx_formats.h"
#include "cpu_ops/status.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace device {

namespace detail {

// ISA-dependent defaults of GemmMx. Explicit Config_/MmaPolicy_ arguments
// always override.
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

  static_assert(std::is_same<StorageA, typename MmaPolicy::ElemA>::value &&
                    std::is_same<StorageB, typename MmaPolicy::ElemB>::value &&
                    std::is_same<float, typename MmaPolicy::AccT>::value,
                "storage types must match the mma policy's ElemA/ElemB, and the "
                "policy's AccT must be float");
  static_assert(threadblock::epilogue_params<Epilogue>::value,
                "Epilogue must provide a Params type and be constructible from it");

  struct Arguments {
    GemmCoord problem_size;
    MxTensorRef<StorageA> ref_A;  // m x k, row-major (k contiguous)
    MxTensorRef<StorageB> ref_B;  // k x n, column-major (k contiguous)
    TensorRef<const float, LayoutC> ref_C;
    TensorRef<float, LayoutC> ref_D;
    typename Epilogue::Params epilogue;
    // Same semantics as Gemm::Arguments::split_k_slices.
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
      kernel::detail::write_zero_k(args.ref_C, args.ref_D, m, n,
                                   Epilogue(args.epilogue));
      return Status::kSuccess;
    }

    const MxTensorRef<StorageA>& a = args.ref_A;
    const MxTensorRef<StorageB>& b = args.ref_B;
    if (!a.data || !a.scales || !b.data || !b.scales || !args.ref_C.data() ||
        !args.ref_D.data()) {
      return Status::kErrorInvalidArguments;
    }
    // ld counts elements along k for both operands; scale arrays follow with
    // ceil(k / 32) entries per row/column.
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

// Opt-in VNNI variants: per-block int8 requantization + vpdpbusd instead of
// the exact f32 decode. Numerics: e2m1 exact; e4m3/e5m2 per-block int8
// accuracy; no inf/nan propagation (see mma_policy_mx_vnni.h).
#if defined(CPU_OPS_SIMD_AVX512VNNI)
template <typename LayoutC = layout::RowMajor>
using GemmMxE4M3Vnni =
    GemmMx<fp8e4m3_t, fp8e4m3_t, LayoutC, epilogue::LinearCombination<float>,
           mma::Vnni512GemmConfig, mma::MxVnniPolicy<fp8e4m3_t, fp8e4m3_t>>;
template <typename LayoutC = layout::RowMajor>
using GemmMxE5M2Vnni =
    GemmMx<fp8e5m2_t, fp8e5m2_t, LayoutC, epilogue::LinearCombination<float>,
           mma::Vnni512GemmConfig, mma::MxVnniPolicy<fp8e5m2_t, fp8e5m2_t>>;
template <typename LayoutC = layout::RowMajor>
using GemmMxE2M1Vnni =
    GemmMx<fp4e2m1_t, fp4e2m1_t, LayoutC, epilogue::LinearCombination<float>,
           mma::Vnni512GemmConfig, mma::MxVnniPolicy<fp4e2m1_t, fp4e2m1_t>>;
#else
template <typename LayoutC = layout::RowMajor>
using GemmMxE4M3Vnni =
    GemmMx<fp8e4m3_t, fp8e4m3_t, LayoutC, epilogue::LinearCombination<float>,
           GemmConfig<int32_t>, mma::MxVnniPolicy<fp8e4m3_t, fp8e4m3_t>>;
template <typename LayoutC = layout::RowMajor>
using GemmMxE5M2Vnni =
    GemmMx<fp8e5m2_t, fp8e5m2_t, LayoutC, epilogue::LinearCombination<float>,
           GemmConfig<int32_t>, mma::MxVnniPolicy<fp8e5m2_t, fp8e5m2_t>>;
template <typename LayoutC = layout::RowMajor>
using GemmMxE2M1Vnni =
    GemmMx<fp4e2m1_t, fp4e2m1_t, LayoutC, epilogue::LinearCombination<float>,
           GemmConfig<int32_t>, mma::MxVnniPolicy<fp4e2m1_t, fp4e2m1_t>>;
#endif

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
