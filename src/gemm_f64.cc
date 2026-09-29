#include "cpu_ops/detail/gemm_device_defn.h"

namespace cpu_ops {
namespace gemm {
namespace device {

using layout::ColumnMajor;
using layout::RowMajor;

template class Gemm<double, RowMajor, double, RowMajor, double, RowMajor>;
template class Gemm<double, RowMajor, double, RowMajor, double, ColumnMajor>;
template class Gemm<double, RowMajor, double, ColumnMajor, double, RowMajor>;
template class Gemm<double, RowMajor, double, ColumnMajor, double, ColumnMajor>;
template class Gemm<double, ColumnMajor, double, RowMajor, double, RowMajor>;
template class Gemm<double, ColumnMajor, double, RowMajor, double, ColumnMajor>;
template class Gemm<double, ColumnMajor, double, ColumnMajor, double, RowMajor>;
template class Gemm<double, ColumnMajor, double, ColumnMajor, double, ColumnMajor>;

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
