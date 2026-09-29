#pragma once

// EXPERIMENTAL, NOT TESTED ON HARDWARE OR EMULATOR.
//
// Mma policy for ARM SME (Scalable Matrix Extension): the micro-tile lives in
// the ZA array and is accumulated by FMOPA f32 outer products. The atom is a
// 2x2 grid of ZA 32-bit tiles, i.e. it computes a (2*SVL/32) x (2*SVL/32)
// f32 tile (32x32 at SVL=512).
//
// Build requirements: aarch64, GCC 14+ or Clang 18+,
//   -march=armv9.2-a+sme -msve-vector-bits=<128|256|512|...>
// The whole header is inert unless __ARM_FEATURE_SME is defined together with
// a fixed SVE vector width.

#include "cpu_ops/detail/pack.h"
#include "cpu_ops/detail/simd.h"
#include "cpu_ops/layout.h"

#if defined(CPU_OPS_SIMD_SVE) && defined(__ARM_FEATURE_SME)
#define CPU_OPS_HAS_SME_POLICY 1

namespace cpu_ops {
namespace mma {

// Micro-kernel holding a 2x2 grid of ZA 32-bit tiles. All ZA traffic happens
// inside run(), which executes in streaming SVE mode with a fresh ZA state;
// the accumulated tile is read out to a row-major MR x NR buffer before the
// mode is exited on return.
template <int MR_, int NR_>
struct MmaAtomSmeF32 {
  static constexpr int kTileRows = __ARM_FEATURE_SVE_BITS / 32;
  static constexpr int kMR = MR_;
  static constexpr int kNR = NR_;
  static_assert(kMR == 2 * kTileRows && kNR == 2 * kTileRows,
                "the SME atom is exactly a 2x2 grid of ZA 32-bit tiles");

  __attribute__((arm_streaming)) __attribute__((arm_new_za)) void run(
      const float* a, const float* b, int kc, float* tile) const {
    const svbool_t p = svptrue_b32();
    const svfloat32_t zero = svdup_n_f32(0.0f);
    svzero_za();
    for (int k = 0; k < kc; ++k) {
      // a strip: kc steps of MR contiguous scalars; b strip: kc steps of NR.
      const svfloat32_t av0 = svld1_f32(p, a + k * kMR);
      const svfloat32_t av1 = svld1_f32(p, a + k * kMR + kTileRows);
      const svfloat32_t bv0 = svld1_f32(p, b + k * kNR);
      const svfloat32_t bv1 = svld1_f32(p, b + k * kNR + kTileRows);
      svfmopa_za32_f32_m(0, p, p, av0, bv0);
      svfmopa_za32_f32_m(1, p, p, av0, bv1);
      svfmopa_za32_f32_m(2, p, p, av1, bv0);
      svfmopa_za32_f32_m(3, p, p, av1, bv1);
    }
    for (int r = 0; r < kTileRows; ++r) {
      svst1_f32(p, tile + r * kNR, svreadz_hor_za32_f32_m(zero, p, 0, r));
      svst1_f32(p, tile + r * kNR + kTileRows, svreadz_hor_za32_f32_m(zero, p, 1, r));
      svst1_f32(p, tile + (kTileRows + r) * kNR, svreadz_hor_za32_f32_m(zero, p, 2, r));
      svst1_f32(p, tile + (kTileRows + r) * kNR + kTileRows,
                svreadz_hor_za32_f32_m(zero, p, 3, r));
    }
  }
};

struct SmePolicyF32 {
  using ElemA = float;
  using ElemB = float;
  using AccT = float;
  using PackedA = float;
  using PackedB = float;

  static constexpr int kKStep = 1;
  static constexpr int pad_kc(int kc) { return kc; }

  template <int MR, int NR>
  using Atom = MmaAtomSmeF32<MR, NR>;

  template <typename LayoutA>
  static void pack_a(TensorRef<const float, LayoutA> a, int i0, int k0, int mc, int kc,
                     int /*kc_pad*/, int mr, float* dst) {
    detail::pack_a(a, i0, k0, mc, kc, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const float, LayoutB> b, int k0, int j0, int kc,
                     int /*kc_pad*/, int nc, int nr, float* dst) {
    detail::pack_b(b, k0, j0, kc, nc, nr, dst);
  }
};

// Tile config matching the SME atom: MR = NR = 2 ZA32 tile rows.
struct SmeGemmConfigF32 {
  static constexpr int kMR = 2 * (__ARM_FEATURE_SVE_BITS / 32);
  static constexpr int kNR = kMR;
  static constexpr int kMC = 128;  // a multiple of kMR for every legal SVE width
  static constexpr int kNC = 256;  // a multiple of kNR for every legal SVE width
  static constexpr int kKC = 256;
};

}  // namespace mma
}  // namespace cpu_ops

#endif  // CPU_OPS_SIMD_SVE && __ARM_FEATURE_SME
