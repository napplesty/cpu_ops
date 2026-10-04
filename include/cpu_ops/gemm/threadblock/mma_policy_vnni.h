#pragma once

// Mma policy for quantized GEMM: uint8 x int8 -> int32, accumulating 4 k
// slices per 32-bit lane with a byte dot-product instruction (AVX-VNNI /
// AVX512-VNNI via simd::dpbusd; portable scalar fallback otherwise).
//
// Packed layouts (kc is padded up to a multiple of 4; padded k-slices are 0):
//   A panel, per strip of MR rows:  dst[(g * MR + i) * 4 + t] = A(i, 4g + t)
//   B panel, per strip of NR cols:  dst[(g * NR + j) * 4 + t] = B(4g + t, j)
// i.e. 4 consecutive k bytes sit in one 32-bit lane, matching dpbusd.

#include <cstdint>
#include <cstring>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace threadblock {

template <typename LayoutA>
void pack_a_vnni(const TensorRef<const uint8_t, LayoutA>& a, int i0, int k0, int mc,
                 int kc, int kc_pad, int mr, uint8_t* dst) {
  for (int ib = 0; ib < mc; ib += mr) {
    const int imax = (mc - ib < mr) ? (mc - ib) : mr;
    uint8_t* strip = dst + (ib / mr) * kc_pad * mr;
    for (int g = 0; g < kc_pad / 4; ++g) {
      for (int i = 0; i < mr; ++i) {
        for (int t = 0; t < 4; ++t) {
          const int k = 4 * g + t;
          const bool valid = (i < imax) && (k < kc);
          strip[(g * mr + i) * 4 + t] = valid ? a.at(i0 + ib + i, k0 + k) : uint8_t(0);
        }
      }
    }
  }
}

template <typename LayoutB>
void pack_b_vnni(const TensorRef<const int8_t, LayoutB>& b, int k0, int j0, int kc,
                 int kc_pad, int nc, int nr, uint8_t* dst) {
  for (int jb = 0; jb < nc; jb += nr) {
    const int jmax = (nc - jb < nr) ? (nc - jb) : nr;
    uint8_t* strip = dst + (jb / nr) * kc_pad * nr;
    for (int g = 0; g < kc_pad / 4; ++g) {
      for (int j = 0; j < nr; ++j) {
        for (int t = 0; t < 4; ++t) {
          const int k = 4 * g + t;
          const bool valid = (j < jmax) && (k < kc);
          const int8_t v = valid ? b.at(k0 + k, j0 + jb + j) : int8_t(0);
          strip[(g * nr + j) * 4 + t] = static_cast<uint8_t>(v);
        }
      }
    }
  }
}

}  // namespace threadblock
}  // namespace gemm

namespace mma {

// Micro-kernel: MR x NR int32 accumulators; per k-group of 4 it broadcasts one
// packed A word per row and dot-products it with NR/8 packed B vectors.
template <int MR_, int NR_, int VLEN = simd::native_width<int32_t>()>
struct MmaAtomVnni {
  static constexpr int kMR = MR_;
  static constexpr int kNR = NR_;
  static constexpr int kVLEN = VLEN;
  static constexpr int kVecN = NR_ / VLEN;
  static_assert(NR_ > 0 && NR_ % VLEN == 0, "NR must be a multiple of VLEN");

  using VecT = simd::Vec<int32_t, VLEN>;

  VecT acc[kMR][kVecN];

  void clear() {
    const VecT zero = VecT::set1(0);
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w] = zero;
  }

  void mma(const uint8_t* a, const int8_t* b, int kc_pad) {
    const uint8_t* bb = reinterpret_cast<const uint8_t*>(b);
    for (int g = 0; g < kc_pad / 4; ++g) {
      VecT bv[kVecN];
      for (int w = 0; w < kVecN; ++w) {
        bv[w] =
            VecT::load(reinterpret_cast<const int32_t*>(bb + (g * kNR + w * VLEN) * 4));
      }
      for (int i = 0; i < kMR; ++i) {
        uint32_t a32;
        std::memcpy(&a32, a + (g * kMR + i) * 4, 4);
        const VecT av = VecT::set1_u32(a32);
        for (int w = 0; w < kVecN; ++w) acc[i][w] = simd::dpbusd(acc[i][w], av, bv[w]);
      }
    }
  }

  void store_tile(int32_t* tile) const {
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w].store(tile + i * kNR + w * VLEN);
  }

  // Single-entry form used by the macro kernel: clear, accumulate, spill.
  void run(const uint8_t* a, const int8_t* b, int kc_pad, int32_t* tile) {
    clear();
    mma(a, b, kc_pad);
    store_tile(tile);
  }
};

template <int VLEN = simd::native_width<int32_t>()>
struct VnniPolicy {
  using ElemA = uint8_t;
  using ElemB = int8_t;
  using AccT = int32_t;
  using PackedA = uint8_t;
  using PackedB = int8_t;

  static constexpr int kKStep = 4;
  static constexpr int pad_kc(int kc) { return (kc + 3) & ~3; }

  template <int MR, int NR>
  using Atom = MmaAtomVnni<MR, NR, VLEN>;

  template <typename LayoutA>
  static void pack_a(TensorRef<const uint8_t, LayoutA> a, int i0, int k0, int mc, int kc,
                     int kc_pad, int mr, uint8_t* dst) {
    gemm::threadblock::pack_a_vnni(a, i0, k0, mc, kc, kc_pad, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const int8_t, LayoutB> b, int k0, int j0, int kc, int kc_pad,
                     int nc, int nr, int8_t* dst) {
    gemm::threadblock::pack_b_vnni(b, k0, j0, kc, kc_pad, nc, nr,
                                   reinterpret_cast<uint8_t*>(dst));
  }
};

// Tile config for the 512-bit VNNI path (VLEN = 16): MR x NR = 8 x 32 with
// 16 of 32 ZMM as accumulators. NC/KC tuned on Zen 4 (B panel = 512 KiB).
// Used by GemmU8S8S32 only when CPU_OPS_SIMD_AVX512VNNI is set.
struct Vnni512GemmConfig {
  static constexpr int kMR = 8;
  static constexpr int kNR = 32;
  static constexpr int kMC = 128;   // multiple of kMR
  static constexpr int kNC = 1024;  // multiple of kNR
  static constexpr int kKC = 512;
};

}  // namespace mma
}  // namespace cpu_ops
