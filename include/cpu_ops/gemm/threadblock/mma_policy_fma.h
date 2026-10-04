#pragma once

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
  // PackedA/PackedB differ from ElemA/ElemB for policies that convert during packing.
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

// f32 policy pinned to 512-bit vectors (VLEN = 16): the opt-in AVX-512F path.
struct Fma512Policy {
  using ElemA = float;
  using ElemB = float;
  using AccT = float;
  using PackedA = float;
  using PackedB = float;

  static constexpr int kKStep = 1;
  static constexpr int pad_kc(int kc) { return kc; }

  template <int MR, int NR>
  using Atom = MmaAtom<float, MR, NR, 16>;

  template <typename LayoutA>
  static void pack_a(TensorRef<const float, LayoutA> a, int i0, int k0, int mc, int kc,
                     int /*kc_pad*/, int mr, float* dst) {
    gemm::threadblock::pack_a(a, i0, k0, mc, kc, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const float, LayoutB> b, int k0, int j0, int kc,
                     int /*kc_pad*/, int nc, int nr, float* dst) {
    gemm::threadblock::pack_b(b, k0, j0, kc, nc, nr, dst);
  }
};

struct Fma512GemmConfig {
  static constexpr int kMR = 8;
  static constexpr int kNR = 32;
  static constexpr int kMC = 128;
  static constexpr int kNC = 512;
  static constexpr int kKC = 512;
};

}  // namespace mma
}  // namespace cpu_ops
