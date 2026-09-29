// ARM SME f32 GEMM instantiations. Empty translation unit unless compiled for
// aarch64 with SME and a fixed SVE vector width, e.g.
//   -march=armv9.2-a+sme -msve-vector-bits=512

#include "cpu_ops/detail/mma_policy_sme.h"

#if defined(CPU_OPS_HAS_SME_POLICY)

#include "cpu_ops/detail/gemm_device_defn.h"
#include "cpu_ops/epilogue/linear_combination.h"

namespace cpu_ops {
namespace gemm {
namespace device {

using layout::ColumnMajor;
using layout::RowMajor;

template class Gemm<float, RowMajor, float, RowMajor, float, RowMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, RowMajor, float, RowMajor, float, ColumnMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, RowMajor, float, ColumnMajor, float, RowMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, RowMajor, float, ColumnMajor, float, ColumnMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, ColumnMajor, float, RowMajor, float, RowMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, ColumnMajor, float, RowMajor, float, ColumnMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, ColumnMajor, float, ColumnMajor, float, RowMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;
template class Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor, float,
                    epilogue::LinearCombination<float>, mma::SmeGemmConfigF32,
                    mma::SmePolicyF32>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops

#endif  // CPU_OPS_HAS_SME_POLICY
