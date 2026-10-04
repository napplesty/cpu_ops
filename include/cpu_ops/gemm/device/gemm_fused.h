#pragma once

// Convenience aliases for GEMM with a fused epilogue chain, e.g.
//
//   using LinearRelu = GemmFusedF32<RowMajor, RowMajor, RowMajor,
//                                   epilogue::BiasAdd<float>, epilogue::Relu>;

#include "cpu_ops/epilogue/fusion.h"
#include "cpu_ops/gemm/device/gemm.h"

namespace cpu_ops {
namespace gemm {
namespace device {

template <typename LayoutA, typename LayoutB, typename LayoutC, typename... Ops>
using GemmFusedF32 =
    Gemm<float, LayoutA, float, LayoutB, float, LayoutC, float,
         epilogue::LinearCombinationFused<float, Ops...>>;

template <typename LayoutA, typename LayoutB, typename LayoutC, typename... Ops>
using GemmFusedF64 =
    Gemm<double, LayoutA, double, LayoutB, double, LayoutC, double,
         epilogue::LinearCombinationFused<double, Ops...>>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
