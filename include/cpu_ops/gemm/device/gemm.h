#pragma once

#include <type_traits>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/epilogue/linear_combination.h"
#include "cpu_ops/gemm/kernel/run_blocked.h"
#include "cpu_ops/gemm/threadblock/mma_policy_bf16.h"
#include "cpu_ops/gemm/threadblock/mma_policy_fma.h"
#include "cpu_ops/gemm/threadblock/mma_policy_vnni.h"
#include "cpu_ops/gemm/threadblock/mma_policy_widen.h"
#include "cpu_ops/gemm_coord.h"
#include "cpu_ops/layout/matrix.h"
#include "cpu_ops/numeric_types.h"
#include "cpu_ops/status.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace device {

// Tile sizes steering the blocked mainloop. MR x NR is the register-level
// micro-tile; MC/NC/KC are the cache-level block sizes. Packing buffers use
// roughly MC * KC + KC * NC elements per thread.
template <typename T>
struct GemmConfig {
  static constexpr int kMR = 6;
  static constexpr int kNR = 2 * simd::native_width<T>();
  static constexpr int kMC = 126;  // multiple of kMR
  static constexpr int kNC = 256;  // multiple of kNR for float (16) and double (8)
  static constexpr int kKC = 256;
};

// Multithreaded GEMM:
//   D = alpha * A * B + beta * C   (optionally fused with an activation)
//
// A is m x k, B is k x n, C and D are m x n. C and D may alias each other.
// The library is header-only: including this header makes the full kernel
// definition available, so any combination of element types, layouts,
// epilogue, tile config, and mma policy can be instantiated in the including
// translation unit.
template <typename ElementA_, typename LayoutA_, typename ElementB_, typename LayoutB_,
          typename ElementC_, typename LayoutC_, typename ElementAccumulator_ = ElementC_,
          typename Epilogue_ = epilogue::LinearCombination<ElementC_>,
          typename Config_ = GemmConfig<ElementC_>,
          typename MmaPolicy_ = mma::FmaPolicy<ElementC_>>
class Gemm {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using ElementAccumulator = ElementAccumulator_;
  using Epilogue = Epilogue_;
  using Config = Config_;
  using MmaPolicy = MmaPolicy_;

  struct Arguments {
    GemmCoord problem_size;
    TensorRef<const ElementA, LayoutA> ref_A;
    TensorRef<const ElementB, LayoutB> ref_B;
    TensorRef<const ElementC, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename Epilogue::Params epilogue;
    // 0: automatic — deep reductions (k >= 8 * KC) are split into enough
    //    k slices to give roughly two tasks per thread, bounded by k depth
    //    (>= 1 KC block per slice) and a 4 MiB workspace budget;
    // 1: never split k;
    // >1: force this many k slices (still clamped so no slice is empty).
    int split_k_slices = 0;
  };

  Gemm() = default;

  // num_threads <= 0 selects the size of the global thread pool.
  Status operator()(const Arguments& args, int num_threads = 0) const {
    static_assert(std::is_same<ElementA, typename MmaPolicy::ElemA>::value &&
                      std::is_same<ElementB, typename MmaPolicy::ElemB>::value &&
                      std::is_same<ElementAccumulator, typename MmaPolicy::AccT>::value &&
                      std::is_same<ElementC, typename MmaPolicy::AccT>::value,
                  "element types must match the mma policy's ElemA/ElemB, and both the "
                  "accumulator and C must use the policy's AccT");

    const int m = args.problem_size.m;
    const int n = args.problem_size.n;
    const int k = args.problem_size.k;
    if (m < 0 || n < 0 || k < 0) return Status::kErrorInvalidProblem;
    if (m == 0 || n == 0) return Status::kSuccess;

    if (k == 0) {
      // Degenerate product: fold beta * C into D without touching A or B
      // (which are allowed to be empty in this case).
      const Epilogue epilogue(args.epilogue);
      for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          args.ref_D.at(i, j) = epilogue(ElementC(0), args.ref_C.at(i, j), true, true, i, j);
        }
      }
      return Status::kSuccess;
    }

    if (!args.ref_A.data() || !args.ref_B.data() || !args.ref_C.data() || !args.ref_D.data()) {
      return Status::kErrorInvalidArguments;
    }
    const int lda_min = LayoutA::kIsRowMajor ? k : m;
    const int ldb_min = LayoutB::kIsRowMajor ? n : k;
    const int ldc_min = LayoutC::kIsRowMajor ? n : m;
    const int ldd_min = ldc_min;
    if (args.ref_A.ld() < lda_min || args.ref_B.ld() < ldb_min ||
        args.ref_C.ld() < ldc_min || args.ref_D.ld() < ldd_min) {
      return Status::kErrorInvalidArguments;
    }

    return kernel::run_blocked<MmaPolicy, LayoutC, Epilogue, Config>(
        args.ref_A, args.ref_B, args.ref_C, args.ref_D, m, n, k, args.epilogue,
        args.split_k_slices, num_threads);
  }
};

// uint8 x int8 -> int32 quantized GEMM. Uses 4-way byte dot-product
// accumulation (AVX-VNNI / AVX512-VNNI when available, portable scalar
// fallback otherwise). alpha/beta are int32.
template <typename LayoutA, typename LayoutB, typename LayoutC>
using GemmU8S8S32 = Gemm<uint8_t, LayoutA, int8_t, LayoutB, int32_t, LayoutC, int32_t,
                         epilogue::LinearCombination<int32_t>, GemmConfig<int32_t>,
                         mma::VnniPolicy>;

// Half-precision input GEMM with f32 accumulation and f32 output
// (C/D, alpha/beta are float). A/B are widened to f32 at pack time, so the
// compute itself runs on the f32 FMA path; what halves is operand memory
// traffic, not arithmetic width.
template <typename LayoutA, typename LayoutB, typename LayoutC>
using GemmF16F32 = Gemm<float16_t, LayoutA, float16_t, LayoutB, float, LayoutC, float,
                        epilogue::LinearCombination<float>, GemmConfig<float>,
                        mma::WidenPolicy<float16_t>>;

// Same, with bfloat16 operands. With AVX512-BF16 the default computes
// natively on packed bf16 pairs (vdpbf16ps, f32 accumulation): operand
// traffic halves versus widening, and numerics differ slightly from the
// widen path — the bf16 products skip one rounding, so results are if
// anything more accurate. Pass mma::WidenPolicy<bfloat16_t> and
// GemmConfig<float> explicitly to force the widening path.
#if defined(CPU_OPS_SIMD_AVX512BF16)
template <typename LayoutA, typename LayoutB, typename LayoutC>
using GemmBF16F32 = Gemm<bfloat16_t, LayoutA, bfloat16_t, LayoutB, float, LayoutC, float,
                         epilogue::LinearCombination<float>, mma::Bf16GemmConfig,
                         mma::Bf16Policy>;
#else
template <typename LayoutA, typename LayoutB, typename LayoutC>
using GemmBF16F32 = Gemm<bfloat16_t, LayoutA, bfloat16_t, LayoutB, float, LayoutC, float,
                         epilogue::LinearCombination<float>, GemmConfig<float>,
                         mma::WidenPolicy<bfloat16_t>>;
#endif

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
