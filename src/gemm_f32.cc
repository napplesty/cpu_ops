#include "cpu_ops/detail/gemm_device_defn.h"

namespace cpu_ops {
namespace gemm {
namespace device {

using layout::ColumnMajor;
using layout::RowMajor;

template class Gemm<float, RowMajor, float, RowMajor, float, RowMajor>;
template class Gemm<float, RowMajor, float, RowMajor, float, ColumnMajor>;
template class Gemm<float, RowMajor, float, ColumnMajor, float, RowMajor>;
template class Gemm<float, RowMajor, float, ColumnMajor, float, ColumnMajor>;
template class Gemm<float, ColumnMajor, float, RowMajor, float, RowMajor>;
template class Gemm<float, ColumnMajor, float, RowMajor, float, ColumnMajor>;
template class Gemm<float, ColumnMajor, float, ColumnMajor, float, RowMajor>;
template class Gemm<float, ColumnMajor, float, ColumnMajor, float, ColumnMajor>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
