#pragma once

// Mma policy for bfloat16 GEMM with native bf16 dot-product accumulation
// (AVX512-BF16 `vdpbf16ps`): packed panels hold PAIRS of bf16 k-slices per
// 32-bit lane, so operand memory traffic halves relative to the widening
// policy (mma_policy_widen.h) and every instruction covers 2 k-steps x 16
// f32 accumulator lanes.
//
// Packed layouts mirror the VNNI policy with kKStep = 2 (kc is padded up to
// even; padded k-slices are 0):
//   A panel, per strip of MR rows:  dst[(g * MR + i) * 2 + t] = A(i, 2g + t)
//   B panel, per strip of NR cols:  dst[(g * NR + j) * 2 + t] = B(2g + t, j)
// i.e. two consecutive bf16 k-slices sit in one 32-bit lane, low half first,
// matching vdpbf16ps.
//
// The packing helpers are plain C++ and compile everywhere (unit-tested
// natively). The atom runs the AVX512-BF16 fast path when
// CPU_OPS_SIMD_AVX512BF16 is set and the portable simd::dpbf16ps fallback
// otherwise (correct but slow) — the device-level GemmBF16F32 alias selects
// this policy only when the ISA is present.

#include <cstdint>
#include <cstring>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/numeric_types.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace threadblock {

template <typename LayoutA>
void pack_a_bf16(const TensorRef<const bfloat16_t, LayoutA>& a, int i0, int k0, int mc,
                 int kc, int kc_pad, int mr, uint16_t* dst) {
  for (int ib = 0; ib < mc; ib += mr) {
    const int imax = (mc - ib < mr) ? (mc - ib) : mr;
    uint16_t* strip = dst + (ib / mr) * kc_pad * mr;
    for (int g = 0; g < kc_pad / 2; ++g) {
      for (int i = 0; i < mr; ++i) {
        for (int t = 0; t < 2; ++t) {
          const int k = 2 * g + t;
          const bool valid = (i < imax) && (k < kc);
          strip[(g * mr + i) * 2 + t] = valid ? a.at(i0 + ib + i, k0 + k).bits : 0;
        }
      }
    }
  }
}

template <typename LayoutB>
void pack_b_bf16(const TensorRef<const bfloat16_t, LayoutB>& b, int k0, int j0, int kc,
                 int kc_pad, int nc, int nr, uint16_t* dst) {
  for (int jb = 0; jb < nc; jb += nr) {
    const int jmax = (nc - jb < nr) ? (nc - jb) : nr;
    uint16_t* strip = dst + (jb / nr) * kc_pad * nr;
    for (int g = 0; g < kc_pad / 2; ++g) {
      for (int j = 0; j < nr; ++j) {
        for (int t = 0; t < 2; ++t) {
          const int k = 2 * g + t;
          const bool valid = (j < jmax) && (k < kc);
          strip[(g * nr + j) * 2 + t] = valid ? b.at(k0 + k, j0 + jb + j).bits : 0;
        }
      }
    }
  }
}

}  // namespace threadblock
}  // namespace gemm

namespace mma {

// Micro-kernel: MR x NR f32 accumulators; per k-pair it broadcasts one packed
// A word per row and dot-products it with NR/16 packed B vectors. The atom
// always exposes its registers (VecT acc[kMR][kVecN]) so the mainloop's
// full-tile vector-store path applies.
template <int MR_, int NR_>
struct MmaAtomBf16 {
  static constexpr int kMR = MR_;
  static constexpr int kNR = NR_;
  static constexpr int kVLEN = 16;  // f32 lanes per packed-B vector
  static constexpr int kVecN = NR_ / kVLEN;
  static_assert(NR_ > 0 && NR_ % kVLEN == 0, "NR must be a multiple of 16");

  using VecT = simd::Vec<float, kVLEN>;

  VecT acc[kMR][kVecN];

  void clear() {
    const VecT zero = VecT::set1(0.0f);
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w] = zero;
  }

  void mma(const uint16_t* a, const uint16_t* b, int kc_pad) {
    for (int g = 0; g < kc_pad / 2; ++g) {
      VecT bv[kVecN];
      for (int w = 0; w < kVecN; ++w) {
        bv[w] = VecT::load(reinterpret_cast<const float*>(b + (g * kNR + w * kVLEN) * 2));
      }
      for (int i = 0; i < kMR; ++i) {
        uint32_t a32;
        std::memcpy(&a32, a + (g * kMR + i) * 2, 4);
        const VecT av = VecT::set1_u32(a32);
        for (int w = 0; w < kVecN; ++w) acc[i][w] = simd::dpbf16ps(acc[i][w], av, bv[w]);
      }
    }
  }

  void store_tile(float* tile) const {
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w].store(tile + i * kNR + w * kVLEN);
  }

  // Single-entry form used by the macro kernel: clear, accumulate, spill.
  void run(const uint16_t* a, const uint16_t* b, int kc_pad, float* tile) {
    clear();
    mma(a, b, kc_pad);
    store_tile(tile);
  }
};

struct Bf16Policy {
  using ElemA = bfloat16_t;
  using ElemB = bfloat16_t;
  using AccT = float;
  using PackedA = uint16_t;
  using PackedB = uint16_t;

  static constexpr int kKStep = 2;
  static constexpr int pad_kc(int kc) { return (kc + 1) & ~1; }

  template <int MR, int NR>
  using Atom = MmaAtomBf16<MR, NR>;

  template <typename LayoutA>
  static void pack_a(TensorRef<const bfloat16_t, LayoutA> a, int i0, int k0, int mc,
                     int kc, int kc_pad, int mr, uint16_t* dst) {
    gemm::threadblock::pack_a_bf16(a, i0, k0, mc, kc, kc_pad, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const bfloat16_t, LayoutB> b, int k0, int j0, int kc,
                     int kc_pad, int nc, int nr, uint16_t* dst) {
    gemm::threadblock::pack_b_bf16(b, k0, j0, kc, kc_pad, nc, nr, dst);
  }
};

// Tile config for the bf16 atom: MR x NR = 8 x 32 f32 lanes (16 of 32 ZMM
// hold accumulators). NC/KC tuned on Zen 4; the narrower operand (2 bytes)
// lets the B panel stay resident in L2 at NC = 1024.
struct Bf16GemmConfig {
  static constexpr int kMR = 8;
  static constexpr int kNR = 32;
  static constexpr int kMC = 128;   // multiple of kMR
  static constexpr int kNC = 1024;  // multiple of kNR
  static constexpr int kKC = 512;
};

}  // namespace mma
}  // namespace cpu_ops
