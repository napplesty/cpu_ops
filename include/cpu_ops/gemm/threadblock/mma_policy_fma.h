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

// f32 policy pinned to 512-bit vectors (VLEN = 16) regardless of
// native_width<float>() — the float width stays at 8 for ISA-tuning reasons
// (see simd.h), so this policy is how the mainloop exploits AVX-512F: twice
// the lanes per FMA plus 32 architectural registers allow a deeper
// accumulator tile. On non-AVX-512 hosts it still compiles and runs through
// the portable Vec<float, 16> fallback, which is how the tests cover it.
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

// Tile config for Fma512Policy: MR x NR = 8 x 32 keeps 16 of 32 ZMM as
// accumulators. NC/KC = 512 measured best on Zen 4 (B panel = 1 MiB f32,
// one L2); see the ISA notes in README.
struct Fma512GemmConfig {
  static constexpr int kMR = 8;
  static constexpr int kNR = 32;  // 2 x 16 f32 lanes
  static constexpr int kMC = 128;  // multiple of kMR
  static constexpr int kNC = 512;  // multiple of kNR
  static constexpr int kKC = 512;
};

}  // namespace mma
}  // namespace cpu_ops
