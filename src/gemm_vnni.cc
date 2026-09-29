#include "cpu_ops/detail/gemm_device_defn.h"

namespace cpu_ops {
namespace gemm {
namespace device {

using layout::ColumnMajor;
using layout::RowMajor;

#define CPU_OPS_INSTANTIATE_VNNI(LA, LB, LC)                                     \
  template class Gemm<uint8_t, LA, int8_t, LB, int32_t, LC, int32_t,             \
                      epilogue::LinearCombination<int32_t>, GemmConfig<int32_t>, \
                      mma::VnniPolicy>;

CPU_OPS_INSTANTIATE_VNNI(RowMajor, RowMajor, RowMajor)
CPU_OPS_INSTANTIATE_VNNI(RowMajor, RowMajor, ColumnMajor)
CPU_OPS_INSTANTIATE_VNNI(RowMajor, ColumnMajor, RowMajor)
CPU_OPS_INSTANTIATE_VNNI(RowMajor, ColumnMajor, ColumnMajor)
CPU_OPS_INSTANTIATE_VNNI(ColumnMajor, RowMajor, RowMajor)
CPU_OPS_INSTANTIATE_VNNI(ColumnMajor, RowMajor, ColumnMajor)
CPU_OPS_INSTANTIATE_VNNI(ColumnMajor, ColumnMajor, RowMajor)
CPU_OPS_INSTANTIATE_VNNI(ColumnMajor, ColumnMajor, ColumnMajor)

#undef CPU_OPS_INSTANTIATE_VNNI

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
