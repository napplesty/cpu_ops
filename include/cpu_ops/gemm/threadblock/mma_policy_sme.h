#pragma once

// EXPERIMENTAL, NOT TESTED ON HARDWARE OR EMULATOR.
// Mma policy for ARM SME: f32 FMOPA outer products into a 2x2 grid of ZA
// 32-bit tiles. Requires -march=armv9.2-a+sme -msve-vector-bits=N; the whole
// header is inert unless __ARM_FEATURE_SME and a fixed SVE width are defined.

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/gemm/threadblock/pack.h"
#include "cpu_ops/tensor_ref.h"

#if defined(CPU_OPS_SIMD_SVE) && defined(__ARM_FEATURE_SME)
#define CPU_OPS_HAS_SME_POLICY 1

namespace cpu_ops {
namespace mma {

// run() executes in streaming SVE mode with a fresh ZA state and reads the
// accumulated tile out to a row-major buffer before the mode is exited.
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
    gemm::threadblock::pack_a(a, i0, k0, mc, kc, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const float, LayoutB> b, int k0, int j0, int kc,
                     int /*kc_pad*/, int nc, int nr, float* dst) {
    gemm::threadblock::pack_b(b, k0, j0, kc, nc, nr, dst);
  }
};

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
