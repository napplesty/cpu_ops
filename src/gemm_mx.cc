#include "cpu_ops/detail/gemm_device_defn.h"

namespace cpu_ops {
namespace gemm {
namespace device {

using layout::ColumnMajor;
using layout::RowMajor;

template class GemmMx<fp8e4m3_t, fp8e4m3_t, RowMajor>;
template class GemmMx<fp8e4m3_t, fp8e4m3_t, ColumnMajor>;
template class GemmMx<fp8e5m2_t, fp8e5m2_t, RowMajor>;
template class GemmMx<fp8e5m2_t, fp8e5m2_t, ColumnMajor>;
template class GemmMx<fp4e2m1_t, fp4e2m1_t, RowMajor>;
template class GemmMx<fp4e2m1_t, fp4e2m1_t, ColumnMajor>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
