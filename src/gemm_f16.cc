#include "cpu_ops/detail/gemm_device_defn.h"

namespace cpu_ops {
namespace gemm {
namespace device {

using layout::ColumnMajor;
using layout::RowMajor;

#define CPU_OPS_INSTANTIATE_WIDEN(LA, LB, LC)                                     \
  template class Gemm<float16_t, LA, float16_t, LB, float, LC, float,             \
                      epilogue::LinearCombination<float>, GemmConfig<float>,      \
                      mma::WidenPolicy<float16_t>>;

CPU_OPS_INSTANTIATE_WIDEN(RowMajor, RowMajor, RowMajor)
CPU_OPS_INSTANTIATE_WIDEN(RowMajor, RowMajor, ColumnMajor)
CPU_OPS_INSTANTIATE_WIDEN(RowMajor, ColumnMajor, RowMajor)
CPU_OPS_INSTANTIATE_WIDEN(RowMajor, ColumnMajor, ColumnMajor)
CPU_OPS_INSTANTIATE_WIDEN(ColumnMajor, RowMajor, RowMajor)
CPU_OPS_INSTANTIATE_WIDEN(ColumnMajor, RowMajor, ColumnMajor)
CPU_OPS_INSTANTIATE_WIDEN(ColumnMajor, ColumnMajor, RowMajor)
CPU_OPS_INSTANTIATE_WIDEN(ColumnMajor, ColumnMajor, ColumnMajor)

#undef CPU_OPS_INSTANTIATE_WIDEN

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
