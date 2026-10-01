#pragma once

namespace cpu_ops {

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
