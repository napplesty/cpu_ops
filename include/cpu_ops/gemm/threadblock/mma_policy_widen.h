#pragma once

// Mma policy for narrow floating-point operands (float16_t, bfloat16_t, and
// other storage types implicitly convertible to float): panels are widened
// to f32 at pack time, so the mainloop reuses the f32 FMA atom and
// accumulates in f32. The packed-panel traffic doubles relative to the
// source operands, which is the cost of not having a native half-precision
// compute path; the win over a full f32 GEMM is halved operand memory
// traffic on the way in.

#include "cpu_ops/arch/mma_atom.h"
#include "cpu_ops/gemm/threadblock/pack.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace mma {

template <typename StorageT, int VLEN = simd::native_width<float>()>
struct WidenPolicy {
  using ElemA = StorageT;
  using ElemB = StorageT;
  using AccT = float;
  using PackedA = float;
  using PackedB = float;

  static constexpr int kKStep = 1;
  static constexpr int pad_kc(int kc) { return kc; }

  template <int MR, int NR>
  using Atom = MmaAtom<float, MR, NR, VLEN>;

  template <typename LayoutA>
  static void pack_a(TensorRef<const StorageT, LayoutA> a, int i0, int k0, int mc, int kc,
                     int /*kc_pad*/, int mr, float* dst) {
    gemm::threadblock::pack_a(a, i0, k0, mc, kc, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const StorageT, LayoutB> b, int k0, int j0, int kc,
                     int /*kc_pad*/, int nc, int nr, float* dst) {
    gemm::threadblock::pack_b(b, k0, j0, kc, nc, nr, dst);
  }
};

}  // namespace mma
}  // namespace cpu_ops
