#pragma once

#include "cpu_ops/arch/simd.h"

namespace cpu_ops {
namespace mma {

// Register-resident micro-kernel: keeps a MR x NR accumulator tile in SIMD
// registers and streams packed panels over k.
//
// Packed layouts (produced by pack_a / pack_b):
//   a: kc steps of MR contiguous scalars, row i of step k at a[k * MR + i]
//   b: kc steps of NR contiguous scalars, col j of step k at b[k * NR + j]
//
// Per k step: broadcast MR scalars of A, load NR/VLEN vectors of B, and issue
// MR * (NR/VLEN) fused multiply-adds.
template <typename T, int MR_, int NR_, int VLEN = simd::native_width<T>()>
struct MmaAtom {
  static constexpr int kMR = MR_;
  static constexpr int kNR = NR_;
  static constexpr int kVLEN = VLEN;
  static constexpr int kVecN = NR_ / VLEN;
  static_assert(NR_ > 0 && NR_ % VLEN == 0, "NR must be a multiple of VLEN");

  using VecT = simd::Vec<T, VLEN>;

  VecT acc[kMR][kVecN];

  void clear() {
    const VecT zero = VecT::set1(T(0));
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w] = zero;
  }

  void mma(const T* a, const T* b, int kc) {
    for (int k = 0; k < kc; ++k) {
      VecT bv[kVecN];
      for (int w = 0; w < kVecN; ++w) bv[w] = VecT::load(b + k * kNR + w * VLEN);
      for (int i = 0; i < kMR; ++i) {
        const VecT av = VecT::set1(a[k * kMR + i]);
        for (int w = 0; w < kVecN; ++w) acc[i][w] = simd::fmadd(av, bv[w], acc[i][w]);
      }
    }
  }

  // Spills the accumulator tile to a row-major MR x NR scalar buffer.
  void store_tile(T* tile) const {
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w].store(tile + i * kNR + w * VLEN);
  }

  // Single-entry form used by the macro kernel: clear, accumulate, spill.
  void run(const T* a, const T* b, int kc, T* tile) {
    clear();
    mma(a, b, kc);
    store_tile(tile);
  }
};

}  // namespace mma
}  // namespace cpu_ops
