#pragma once

#include <cstdint>

namespace cpu_ops {
namespace layout {

// C-style row-major layout: element (row, col) lives at ptr[row * ld + col].
struct RowMajor {
  static constexpr bool kIsRowMajor = true;
  static constexpr int64_t offset(int row, int col, int ld) {
    return static_cast<int64_t>(row) * ld + col;
  }
};

// Fortran-style column-major layout: element (row, col) lives at ptr[col * ld + row].
struct ColumnMajor {
  static constexpr bool kIsRowMajor = false;
  static constexpr int64_t offset(int row, int col, int ld) {
    return static_cast<int64_t>(col) * ld + row;
  }
};

}  // namespace layout
}  // namespace cpu_ops
