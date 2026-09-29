#pragma once

#include "cpu_ops/detail/mma_policy_fma.h"
#include "cpu_ops/detail/mma_policy_vnni.h"
#include "cpu_ops/detail/mma_policy_widen.h"
#include "cpu_ops/detail/simd.h"
#include "cpu_ops/element_types.h"
#include "cpu_ops/epilogue/linear_combination.h"
#include "cpu_ops/layout.h"
#include "cpu_ops/matrix_shape.h"
#include "cpu_ops/status.h"

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
// Only the declaration lives in this header; the library ships explicit
// instantiations for float/double with every combination of RowMajor /
// ColumnMajor operands, the default epilogue, and the default config. Custom
// epilogues or tile configs can be instantiated by including
// cpu_ops/detail/gemm_device_defn.h.
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
  Status operator()(const Arguments& args, int num_threads = 0) const;
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

// Same, with bfloat16 operands.
template <typename LayoutA, typename LayoutB, typename LayoutC>
using GemmBF16F32 = Gemm<bfloat16_t, LayoutA, bfloat16_t, LayoutB, float, LayoutC, float,
                         epilogue::LinearCombination<float>, GemmConfig<float>,
                         mma::WidenPolicy<bfloat16_t>>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
