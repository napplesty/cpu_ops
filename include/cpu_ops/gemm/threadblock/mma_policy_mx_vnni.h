#pragma once

// Mma policy for OCP MX operands running the k reduction on integer 4-way
// byte dot products (vpdpbusd via simd::dpbusd). Opt-in via GemmMx's
// MmaPolicy_ argument; the exact-decode MxPolicy remains the default.
//
//   sum_p a_p*b_p = wA*wB * ( IDP(au, qb) - 128*sum_p qb_p )
//                 = wA * ( wB*IDP + negcB ),   negcB = -128*wB*sum(qb)
//
// Numerics: e2m1 is quantized exactly; e4m3/e5m2 blocks round their smallest
// elements (per-element error <= amax/128); Inf/NaN inputs are not supported.
//
// Packed strip layout (tile = MR resp. NR): [int8 data: groups x (32 x tile)
// bytes, quad-interleaved like the VNNI policy] [int32 group count] [f32
// metadata: w[count x tile], then negc[count x tile] on the B side].

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/gemm/threadblock/mma_policy_mx.h"
#include "cpu_ops/mx_formats.h"

#if defined(__AVX2__) && defined(__F16C__) && !defined(CPU_OPS_FORCE_SCALAR)
#include <immintrin.h>
#define CPU_OPS_MXVNNI_SIMD 1
#if defined(__AVX512F__)
#define CPU_OPS_MXVNNI_SIMD_512 1
#endif
#endif

namespace cpu_ops {
namespace mma {

namespace mxvnni_detail {

// Strip strides in bytes (PackedA/PackedB are uint8_t buffers).
inline constexpr int panel_stride_a_bytes(int kc_pad, int tile) {
  return kc_pad * tile + 4 + (kc_pad / 32) * tile * 4;
}
inline constexpr int panel_stride_b_bytes(int kc_pad, int tile) {
  return kc_pad * tile + 4 + 2 * (kc_pad / 32) * tile * 4;
}

// Round to nearest-even as an integral f32 via the 1.5*2^23 magic add (valid
// for |x| < 2^22; our scaled values stay within |x| < 128).
inline float rne_f32(float x) {
  const float big = 12582912.0f;
  return (x + big) - big;
}

// Quantizes len (<= 32) raw values (WITHOUT the E8M0 scale) into int8 codes
// with quantizer 2^qexp, q[len..32) zero-filled; returns qexp (-127 if the
// block is all zero). Backs both the packers and the tests' exact reference.
inline int quantize_block_i8(const float* v, int len, int8_t* q) {
  float amax = 0.0f;
  for (int t = 0; t < len; ++t) amax = std::max(amax, std::fabs(v[t]));
  if (amax == 0.0f) {
    for (int t = 0; t < 32; ++t) q[t] = 0;
    return -127;
  }
  uint32_t abits;
  std::memcpy(&abits, &amax, 4);
  const int qexp = static_cast<int>((abits >> 23) & 0xFF) - 127 - 6;
  const uint32_t ibits = static_cast<uint32_t>(127 - qexp) << 23;
  float inv;
  std::memcpy(&inv, &ibits, 4);
  for (int t = 0; t < len; ++t) {
    int32_t iq = static_cast<int32_t>(rne_f32(v[t] * inv));
    iq = iq > 127 ? 127 : (iq < -128 ? -128 : iq);
    q[t] = static_cast<int8_t>(iq);
  }
  for (int t = len; t < 32; ++t) q[t] = 0;
  return qexp;
}

// Decodes one block segment (len <= 32 at absolute offset kk0) and quantizes
// in the TRUE value domain (the SIMD e4m3 decode bias is multiplied back out
// via kDecodeBias, so SIMD and scalar lanes agree). Returns the quantizer
// exponent; callers scale by e8m0_to_float(scale) * 2^qexp.
template <typename T>
inline int quantize_segment(const uint8_t* row, int kk0, int len, int8_t* q) {
  using Tr = mx_detail2::traits<T>;
  alignas(64) float v[32];
  int t = 0;
#ifdef CPU_OPS_MXVNNI_SIMD
  const float s8 = Tr::kDecodeBias;
  if (Tr::kLog2ElemsPerByte > 0) {
    // fp4: step to an even element index so decode8 sees aligned nibbles.
    for (; t < len && ((kk0 + t) & 1); ++t) v[t] = Tr::decode(row, kk0 + t);
  }
#ifdef CPU_OPS_MXVNNI_SIMD_512
  if (len == 32) {  // a full block is 32-aligned, hence nibble-aligned too
    const __m512 vs16 = _mm512_set1_ps(s8);
    _mm512_store_ps(v, _mm512_mul_ps(Tr::decode16(row, kk0), vs16));
    _mm512_store_ps(v + 16, _mm512_mul_ps(Tr::decode16(row, kk0 + 16), vs16));
    t = 32;
  }
#endif
  const __m256 vs = _mm256_set1_ps(s8);
  for (; t + 8 <= len; t += 8)
    _mm256_storeu_ps(v + t, _mm256_mul_ps(Tr::decode8(row, kk0 + t), vs));
#endif
  for (; t < len; ++t) v[t] = Tr::decode(row, kk0 + t);
  for (t = len; t < 32; ++t) v[t] = 0.0f;
  return quantize_block_i8(v, 32, q);
}

// Number of 32-slot groups a [k0, k0 + kc) range splits into at absolute
// E8M0 block boundaries.
inline int segment_count(int k0, int kc) {
  int count = 0;
  for (int k = 0; k < kc;) {
    const int kk = k0 + k;
    k += std::min(kc - k, 32 - (kk & 31));
    ++count;
  }
  return count;
}

// Writes group g's 32 int8 codes for strip row/col i, quad-interleaved like
// the VNNI policy (4 consecutive k per 32-bit lane).
inline void store_group(uint8_t* strip, int g, int i, int tile, const int8_t* q,
                        bool offset_to_u8) {
  uint8_t* gd = strip + g * 32 * tile + i * 4;
  for (int t = 0; t < 32; ++t) {
    const uint8_t byte = static_cast<uint8_t>(q[t]);
    gd[(t >> 2) * tile * 4 + (t & 3)] = offset_to_u8 ? (byte ^ 0x80) : byte;
  }
}

inline void zero_group(uint8_t* strip, int g, int i, int tile) {
  uint8_t* gd = strip + g * 32 * tile + i * 4;
  for (int qd = 0; qd < 8; ++qd) std::memset(gd + qd * tile * 4, 0, 4);
}

// A panel pack: decode + requantize to (u8)(q + 128), per-group f32 scale
// wA = e8m0 * 2^qexp appended after the data (and the int32 group count).
template <typename T>
void mx_vnni_pack_a(const MxTensorRef<T>& a, int i0, int k0, int mc, int kc, int kc_pad,
                    int mr, uint8_t* dst) {
  using Tr = mx_detail2::traits<T>;
  const int stride = panel_stride_a_bytes(kc_pad, mr);
  const int count = segment_count(k0, kc);
  for (int ib = 0; ib < mc; ib += mr) {
    const int imax = (mc - ib < mr) ? (mc - ib) : mr;
    uint8_t* strip = dst + (ib / mr) * stride;
    std::memcpy(strip + kc_pad * mr, &count, 4);
    float* wa = reinterpret_cast<float*>(strip + kc_pad * mr + 4);
    for (int i = 0; i < imax; ++i) {
      const int row = i0 + ib + i;
      const uint8_t* arow =
          a.data + (static_cast<size_t>(row) * a.ld >> Tr::kLog2ElemsPerByte);
      const uint8_t* srow = a.scales + static_cast<size_t>(row) * a.ld_scales;
      int k = 0, g = 0;
      while (k < kc) {
        const int kk = k0 + k;
        const int seg = std::min(kc - k, 32 - (kk & 31));
        int8_t q[32];
        const int qexp = quantize_segment<T>(arow, kk, seg, q);
        wa[g * mr + i] = e8m0_to_float(srow[kk >> 5]) * std::ldexp(1.0f, qexp);
        store_group(strip, g, i, mr, q, true);
        ++g;
        k += seg;
      }
    }
    for (int i = imax; i < mr; ++i) {
      for (int g = 0; g < count; ++g) {
        wa[g * mr + i] = 0.0f;
        zero_group(strip, g, i, mr);
      }
    }
  }
}

// B panel pack: int8 codes, per-group wB and negcB = -128*wB*sum(qb).
template <typename T>
void mx_vnni_pack_b(const MxTensorRef<T>& b, int k0, int j0, int kc, int kc_pad, int nc,
                    int nr, uint8_t* dst) {
  using Tr = mx_detail2::traits<T>;
  const int stride = panel_stride_b_bytes(kc_pad, nr);
  const int count = segment_count(k0, kc);
  for (int jb = 0; jb < nc; jb += nr) {
    const int jmax = (nc - jb < nr) ? (nc - jb) : nr;
    uint8_t* strip = dst + (jb / nr) * stride;
    std::memcpy(strip + kc_pad * nr, &count, 4);
    float* wb = reinterpret_cast<float*>(strip + kc_pad * nr + 4);
    float* ncb = wb + count * nr;
    for (int j = 0; j < jmax; ++j) {
      const int col = j0 + jb + j;
      const uint8_t* bcol =
          b.data + (static_cast<size_t>(col) * b.ld >> Tr::kLog2ElemsPerByte);
      const uint8_t* scol = b.scales + static_cast<size_t>(col) * b.ld_scales;
      int k = 0, g = 0;
      while (k < kc) {
        const int kk = k0 + k;
        const int seg = std::min(kc - k, 32 - (kk & 31));
        int8_t q[32];
        const int qexp = quantize_segment<T>(bcol, kk, seg, q);
        int32_t qsum = 0;
        for (int t = 0; t < 32; ++t) qsum += q[t];
        const float w = e8m0_to_float(scol[kk >> 5]) * std::ldexp(1.0f, qexp);
        wb[g * nr + j] = w;
        ncb[g * nr + j] = -128.0f * (w * static_cast<float>(qsum));
        store_group(strip, g, j, nr, q, false);
        ++g;
        k += seg;
      }
    }
    for (int j = jmax; j < nr; ++j) {
      for (int g = 0; g < count; ++g) {
        wb[g * nr + j] = 0.0f;
        ncb[g * nr + j] = 0.0f;
        zero_group(strip, g, j, nr);
      }
    }
  }
}

}  // namespace mxvnni_detail

// Per 32-element block the int32 dot products are folded into the f32
// accumulators with the per-block scales; rows run in two halves so both
// accumulator sets fit the 32-register file at once.
template <int MR_, int NR_, int VLEN = simd::native_width<int32_t>()>
struct MmaAtomMxVnni {
  static constexpr int kMR = MR_;
  static constexpr int kNR = NR_;
  static constexpr int kVLEN = VLEN;
  static constexpr int kVecN = NR_ / VLEN;
  static constexpr int kIHalf = MR_ / 2;
  static_assert(NR_ > 0 && NR_ % VLEN == 0, "NR must be a multiple of VLEN");
  static_assert(MR_ > 0 && MR_ % 2 == 0, "MR must be even");

  using VecI = simd::Vec<int32_t, VLEN>;
  using VecF = simd::Vec<float, VLEN>;
  using VecT = VecF;  // f32 accumulators, exposed for the vector epilogue

  VecF acc[kMR][kVecN];

  void clear() {
    const VecF zero = VecF::set1(0.0f);
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w] = zero;
  }

  void mma(const uint8_t* a, const uint8_t* b, int kc_pad) {
    int32_t count;
    std::memcpy(&count, a + kc_pad * kMR, 4);
    const float* wa = reinterpret_cast<const float*>(a + kc_pad * kMR + 4);
    const float* wb = reinterpret_cast<const float*>(b + kc_pad * kNR + 4);
    const float* ncb = wb + count * kNR;
    for (int g = 0; g < count; ++g) {
      const uint8_t* ag = a + g * 32 * kMR;
      const uint8_t* bg = b + g * 32 * kNR;
      VecF wbv[kVecN], ncbv[kVecN];
      for (int w = 0; w < kVecN; ++w) {
        wbv[w] = VecF::load(wb + g * kNR + w * VLEN);
        ncbv[w] = VecF::load(ncb + g * kNR + w * VLEN);
      }
      for (int i0 = 0; i0 < kMR; i0 += kIHalf) {
        VecI idp[kIHalf][kVecN];
        for (int i = 0; i < kIHalf; ++i)
          for (int w = 0; w < kVecN; ++w) idp[i][w] = VecI::set1(0);
        for (int q = 0; q < 8; ++q) {
          VecI bv[kVecN];
          for (int w = 0; w < kVecN; ++w) {
            bv[w] = VecI::load(reinterpret_cast<const int32_t*>(
                bg + (q * kNR + w * VLEN) * 4));
          }
          for (int i = 0; i < kIHalf; ++i) {
            uint32_t a32;
            std::memcpy(&a32, ag + (q * kMR + i0 + i) * 4, 4);
            const VecI av = VecI::set1_u32(a32);
            for (int w = 0; w < kVecN; ++w) idp[i][w] = simd::dpbusd(idp[i][w], av, bv[w]);
          }
        }
        for (int i = 0; i < kIHalf; ++i) {
          const VecF wav = VecF::set1(wa[g * kMR + i0 + i]);
          for (int w = 0; w < kVecN; ++w) {
            const VecF t =
                simd::fmadd(wbv[w], simd::cvtepi32_ps(idp[i][w]), ncbv[w]);
            acc[i0 + i][w] = simd::fmadd(wav, t, acc[i0 + i][w]);
          }
        }
      }
    }
  }

  void store_tile(float* tile) const {
    for (int i = 0; i < kMR; ++i)
      for (int w = 0; w < kVecN; ++w) acc[i][w].store(tile + i * kNR + w * VLEN);
  }

  void run(const uint8_t* a, const uint8_t* b, int kc_pad, float* tile) {
    clear();
    mma(a, b, kc_pad);
    store_tile(tile);
  }
};

template <typename StorageA, typename StorageB, int VLEN = simd::native_width<int32_t>()>
struct MxVnniPolicy {
  static_assert(mx_detail2::is_mx_element<StorageA>::value &&
                    mx_detail2::is_mx_element<StorageB>::value,
                "MxVnniPolicy operands must be fp8e4m3_t, fp8e5m2_t, or fp4e2m1_t");

  using ElemA = StorageA;
  using ElemB = StorageB;
  using AccT = float;
  using PackedA = uint8_t;
  using PackedB = uint8_t;

  static constexpr int kKStep = 32;  // E8M0 block granularity
  // Round up to whole 32-slot groups plus one spare: a segment split at an
  // arbitrary k offset (split-k / KC boundary) needs at most one extra slot.
  static constexpr int pad_kc(int kc) { return ((kc + 31) & ~31) + 32; }

  static constexpr int panel_stride_a(int kc_pad, int tile) {
    return mxvnni_detail::panel_stride_a_bytes(kc_pad, tile);
  }
  static constexpr int panel_stride_b(int kc_pad, int tile) {
    return mxvnni_detail::panel_stride_b_bytes(kc_pad, tile);
  }

  template <int MR, int NR>
  using Atom = MmaAtomMxVnni<MR, NR, VLEN>;

  static void pack_a(const MxTensorRef<StorageA>& a, int i0, int k0, int mc, int kc,
                     int kc_pad, int mr, uint8_t* dst) {
    mxvnni_detail::mx_vnni_pack_a(a, i0, k0, mc, kc, kc_pad, mr, dst);
  }

  static void pack_b(const MxTensorRef<StorageB>& b, int k0, int j0, int kc, int kc_pad,
                     int nc, int nr, uint8_t* dst) {
    mxvnni_detail::mx_vnni_pack_b(b, k0, j0, kc, kc_pad, nc, nr, dst);
  }
};

}  // namespace mma
}  // namespace cpu_ops

#if defined(CPU_OPS_MXVNNI_SIMD_512)
#undef CPU_OPS_MXVNNI_SIMD_512
#endif
#if defined(CPU_OPS_MXVNNI_SIMD)
#undef CPU_OPS_MXVNNI_SIMD
#endif
