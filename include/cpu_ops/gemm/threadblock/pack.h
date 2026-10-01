#pragma once

#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace threadblock {

// Packs the kc x nc panel of B whose top-left corner is (k0, j0) into the
// strip-interleaved layout consumed by MmaAtom:
//
//   strip s covers columns [j0 + s*nr, j0 + s*nr + nr)
//   dst[(s * kc + k) * nr + j] = B(k0 + k, j0 + s*nr + j)
//
// Columns past nc (edge strips) are zero-padded so the micro-kernel can run
// branch-free. DstT may differ from the operand type (widening pack, e.g.
// float16_t -> float); the conversion must be implicit.
template <typename T, typename LayoutB, typename DstT>
void pack_b(const TensorRef<const T, LayoutB>& b, int k0, int j0, int kc, int nc, int nr,
            DstT* dst) {
  for (int jb = 0; jb < nc; jb += nr) {
    const int jmax = (nc - jb < nr) ? (nc - jb) : nr;
    DstT* strip = dst + (jb / nr) * kc * nr;
    for (int k = 0; k < kc; ++k) {
      DstT* row = strip + k * nr;
      int j = 0;
      for (; j < jmax; ++j) row[j] = b.at(k0 + k, j0 + jb + j);
      for (; j < nr; ++j) row[j] = DstT(0);
    }
  }
}

// Packs the mc x kc panel of A whose top-left corner is (i0, k0) into the
// strip-interleaved layout consumed by MmaAtom:
//
//   strip s covers rows [i0 + s*mr, i0 + s*mr + mr)
//   dst[(s * kc + k) * mr + i] = A(i0 + s*mr + i, k0 + k)
//
// Rows past mc (edge strips) are zero-padded. DstT may differ from the
// operand type, see pack_b.
template <typename T, typename LayoutA, typename DstT>
void pack_a(const TensorRef<const T, LayoutA>& a, int i0, int k0, int mc, int kc, int mr,
            DstT* dst) {
  for (int ib = 0; ib < mc; ib += mr) {
    const int imax = (mc - ib < mr) ? (mc - ib) : mr;
    DstT* strip = dst + (ib / mr) * kc * mr;
    for (int k = 0; k < kc; ++k) {
      DstT* col = strip + k * mr;
      int i = 0;
      for (; i < imax; ++i) col[i] = a.at(i0 + ib + i, k0 + k);
      for (; i < mr; ++i) col[i] = DstT(0);
    }
  }
}

}  // namespace threadblock
}  // namespace gemm
}  // namespace cpu_ops
