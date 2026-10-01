#pragma once

// Policy describing how BlockGemm computes one micro-tile with FMA-based
// SIMD vectors. A policy provides:
//
//   using ElemA / ElemB / AccT;                    // operand and accumulator types
//   using PackedA / PackedB;                       // packed panel element types (== ElemA/ElemB
//                                                  // unless packing converts, e.g. widening)
//   static constexpr int kKStep;                   // k granularity (1 here, 4 for dot-product paths)
//   static constexpr int pad_kc(int kc);           // k extent padded to kKStep
//   template <int MR, int NR> using Atom = ...;    // micro-kernel with run(a, b, kc, tile)
//   pack_a / pack_b;                               // panel packing into the atom's layout
//
// Packed panels are laid out per micro-tile strip: strip s covers
// kc_pad * MR (A) resp. kc_pad * NR (B) contiguous elements.

#include "cpu_ops/arch/mma_atom.h"
#include "cpu_ops/gemm/threadblock/pack.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace mma {

template <typename T>
struct FmaPolicy {
  using ElemA = T;
  using ElemB = T;
  using AccT = T;
  // Element types of the packed panels; differ from ElemA/ElemB for policies
  // that convert during packing (see mma_policy_widen.h).
  using PackedA = T;
  using PackedB = T;

  static constexpr int kKStep = 1;
  static constexpr int pad_kc(int kc) { return kc; }

  template <int MR, int NR>
  using Atom = MmaAtom<T, MR, NR>;

  template <typename LayoutA>
  static void pack_a(TensorRef<const T, LayoutA> a, int i0, int k0, int mc, int kc,
                     int /*kc_pad*/, int mr, T* dst) {
    gemm::threadblock::pack_a(a, i0, k0, mc, kc, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const T, LayoutB> b, int k0, int j0, int kc, int /*kc_pad*/,
                     int nc, int nr, T* dst) {
    gemm::threadblock::pack_b(b, k0, j0, kc, nc, nr, dst);
  }
};

}  // namespace mma
}  // namespace cpu_ops
