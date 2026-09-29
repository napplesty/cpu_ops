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

// Non-owning 2-D view over a matrix with a leading dimension.
template <typename T, typename Layout>
class TensorRef {
 public:
  using Element = T;
  using LayoutType = Layout;

  TensorRef() : ptr_(nullptr), ld_(0) {}
  TensorRef(T* ptr, int ld) : ptr_(ptr), ld_(ld) {}

  T* data() const { return ptr_; }
  int ld() const { return ld_; }

  T& at(int row, int col) const { return ptr_[Layout::offset(row, col, ld_)]; }

  TensorRef subrect(int row, int col) const {
    return TensorRef(ptr_ + Layout::offset(row, col, ld_), ld_);
  }

 private:
  T* ptr_;
  int ld_;
};

}  // namespace cpu_ops
