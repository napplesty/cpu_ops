#pragma once

// Mma policy for OCP MX block-scaled operands (fp8 e4m3/e5m2, fp4 e2m1 with
// E8M0 scales every 32 elements along k). Packing decodes elements and folds
// in the block scale, producing f32 panels; the mainloop is plain f32 FMA.
// A and B may use different storage types.
//
// Layout constraint ("TN"): k must be the contiguous dimension of both
// operands (A m x k row-major, B k x n column-major).

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "cpu_ops/arch/mma_atom.h"
#include "cpu_ops/mx_formats.h"

#if defined(__AVX2__) && defined(__F16C__) && !defined(CPU_OPS_FORCE_SCALAR)
#include <immintrin.h>
#define CPU_OPS_MX_SIMD_DECODE 1
#if defined(__AVX512F__)
#define CPU_OPS_MX_SIMD_DECODE_512 1
#endif
#endif

namespace cpu_ops {
namespace mma {

namespace mx_detail2 {

template <typename T>
struct traits;

template <>
struct traits<fp8e4m3_t> {
  // log2 of elements per byte; ld is divided by this to get row byte strides.
  static constexpr int kLog2ElemsPerByte = 0;
  static float decode(const uint8_t* row, int k) { return fp8e4m3_t::to_float(row[k]); }
#ifdef CPU_OPS_MX_SIMD_DECODE
  // decode8 results are biased by 2^-8 (see below); folded into the scale.
  static constexpr float kDecodeBias = 256.0f;
  // 8 consecutive elements at byte offset k (k+8 must not pass the row end).
  static __m256 decode8(const uint8_t* row, int k) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + k));
    const __m128i w = _mm_cvtepu8_epi16(b);
    // f16 = sign<<15 | (v & 0x7F) << 7 places e4m3's fields so the result is
    // exactly the e4m3 value scaled by 2^-8 — for normals AND denormals alike
    // (hence kDecodeBias).
    const __m128i sign = _mm_slli_epi16(_mm_and_si128(w, _mm_set1_epi16(0x80)), 8);
    const __m128i body = _mm_and_si128(_mm_slli_epi16(w, 7), _mm_set1_epi16(0x3F80));
    __m128i h = _mm_or_si128(sign, body);
    // |v| == 0x7F is NaN in e4m3 but decodes to a normal f16 above; remap.
    const __m128i nan8 = _mm_cmpeq_epi8(_mm_and_si128(b, _mm_set1_epi8(0x7F)),
                                        _mm_set1_epi8(0x7F));
    const __m128i nan16 = _mm_cvtepi8_epi16(nan8);
    h = _mm_or_si128(h, _mm_and_si128(nan16, _mm_set1_epi16(0x7E00)));
    return _mm256_cvtph_ps(h);
  }
#ifdef CPU_OPS_MX_SIMD_DECODE_512
  // 16 consecutive elements at byte offset k (k+16 must not pass the row end).
  static __m512 decode16(const uint8_t* row, int k) {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
    const __m256i w = _mm256_cvtepu8_epi16(b);
    const __m256i sign =
        _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x80)), 8);
    const __m256i body =
        _mm256_and_si256(_mm256_slli_epi16(w, 7), _mm256_set1_epi16(0x3F80));
    __m256i h = _mm256_or_si256(sign, body);
    const __m128i nan8 = _mm_cmpeq_epi8(_mm_and_si128(b, _mm_set1_epi8(0x7F)),
                                        _mm_set1_epi8(0x7F));
    const __m256i nan16 = _mm256_cvtepi8_epi16(nan8);
    h = _mm256_or_si256(h, _mm256_and_si256(nan16, _mm256_set1_epi16(0x7E00)));
    return _mm512_cvtph_ps(h);
  }
#endif
#endif
};

template <>
struct traits<fp8e5m2_t> {
  static constexpr int kLog2ElemsPerByte = 0;
  static float decode(const uint8_t* row, int k) { return fp8e5m2_t::to_float(row[k]); }
#ifdef CPU_OPS_MX_SIMD_DECODE
  static constexpr float kDecodeBias = 1.0f;
  static __m256 decode8(const uint8_t* row, int k) {
    const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(row + k));
    const __m128i w = _mm_cvtepu8_epi16(b);
    // e5m2 and f16 share the exponent bias (15): pure field placement, exact
    // for normals, denormals, inf and nan.
    const __m128i sign = _mm_slli_epi16(_mm_and_si128(w, _mm_set1_epi16(0x80)), 8);
    const __m128i body = _mm_and_si128(_mm_slli_epi16(w, 8), _mm_set1_epi16(0x7F00));
    return _mm256_cvtph_ps(_mm_or_si128(sign, body));
  }
#ifdef CPU_OPS_MX_SIMD_DECODE_512
  static __m512 decode16(const uint8_t* row, int k) {
    const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(row + k));
    const __m256i w = _mm256_cvtepu8_epi16(b);
    const __m256i sign =
        _mm256_slli_epi16(_mm256_and_si256(w, _mm256_set1_epi16(0x80)), 8);
    const __m256i body =
        _mm256_and_si256(_mm256_slli_epi16(w, 8), _mm256_set1_epi16(0x7F00));
    return _mm512_cvtph_ps(_mm256_or_si256(sign, body));
  }
#endif
#endif
};

template <>
struct traits<fp4e2m1_t> {
  static constexpr int kLog2ElemsPerByte = 1;
  static float decode(const uint8_t* row, int k) {
    const uint8_t byte = row[k >> 1];
    return fp4e2m1_t::to_float((k & 1) ? (byte >> 4) : (byte & 0xF));
  }
#ifdef CPU_OPS_MX_SIMD_DECODE
  static constexpr float kDecodeBias = 1.0f;
  // 8 consecutive elements; k must be even (reads the 4 bytes at k/2).
  static __m256 decode8(const uint8_t* row, int k) {
    uint32_t packed4;
    std::memcpy(&packed4, row + (k >> 1), 4);
    const __m128i v = _mm_cvtsi32_si128(static_cast<int>(packed4));
    const __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
    const __m128i idx = _mm_unpacklo_epi8(lo, hi);  // element order in low 8 bytes
    // e2m1 f16 values all have a zero low byte; look up high bytes.
    const __m128i table =
        _mm_setr_epi8(0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46,
                      static_cast<char>(0x80), static_cast<char>(0xB8),
                      static_cast<char>(0xBC), static_cast<char>(0xBE),
                      static_cast<char>(0xC0), static_cast<char>(0xC2),
                      static_cast<char>(0xC4), static_cast<char>(0xC6));
    const __m128i hib = _mm_shuffle_epi8(table, idx);
    return _mm256_cvtph_ps(_mm_unpacklo_epi8(_mm_setzero_si128(), hib));
  }
#ifdef CPU_OPS_MX_SIMD_DECODE_512
  // 16 consecutive elements; k must be even (reads the 8 bytes at k/2).
  static __m512 decode16(const uint8_t* row, int k) {
    uint64_t packed8;
    std::memcpy(&packed8, row + (k >> 1), 8);
    const __m128i v = _mm_cvtsi64_si128(static_cast<long long>(packed8));
    const __m128i lo = _mm_and_si128(v, _mm_set1_epi8(0x0F));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(v, 4), _mm_set1_epi8(0x0F));
    const __m128i idx = _mm_unpacklo_epi8(lo, hi);  // all 16 elements in order
    const __m128i table =
        _mm_setr_epi8(0x00, 0x38, 0x3C, 0x3E, 0x40, 0x42, 0x44, 0x46,
                      static_cast<char>(0x80), static_cast<char>(0xB8),
                      static_cast<char>(0xBC), static_cast<char>(0xBE),
                      static_cast<char>(0xC0), static_cast<char>(0xC2),
                      static_cast<char>(0xC4), static_cast<char>(0xC6));
    const __m128i hib = _mm_shuffle_epi8(table, idx);
    const __m256i h = _mm256_slli_epi16(_mm256_cvtepu8_epi16(hib), 8);
    return _mm512_cvtph_ps(h);
  }
#endif
#endif
};

template <typename T>
struct is_mx_element : std::false_type {};
template <>
struct is_mx_element<fp8e4m3_t> : std::true_type {};
template <>
struct is_mx_element<fp8e5m2_t> : std::true_type {};
template <>
struct is_mx_element<fp4e2m1_t> : std::true_type {};

// Decodes `len` consecutive elements starting at absolute index kk0 of a row,
// multiplies by `scale`, and stores strided (the packed panel's k-major
// layout) into out[t * stride].
template <typename T>
inline void decode_segment(const uint8_t* row, int kk0, int len, float scale, float* out,
                           int stride) {
  using Tr = traits<T>;
  int t = 0;
#ifdef CPU_OPS_MX_SIMD_DECODE
  const float s8 = scale * Tr::kDecodeBias;
  if (Tr::kLog2ElemsPerByte > 0) {
    // fp4: step to an even element index so decode8 sees aligned nibble pairs.
    for (; t < len && ((kk0 + t) & 1); ++t) out[t * stride] = Tr::decode(row, kk0 + t) * scale;
  }
#ifdef CPU_OPS_MX_SIMD_DECODE_512
  const __m512 vs16 = _mm512_set1_ps(s8);
  for (; t + 16 <= len; t += 16) {
    alignas(64) float tmp[16];
    _mm512_store_ps(tmp, _mm512_mul_ps(Tr::decode16(row, kk0 + t), vs16));
    for (int u = 0; u < 16; ++u) out[(t + u) * stride] = tmp[u];
  }
#endif
  const __m256 vs = _mm256_set1_ps(s8);
  for (; t + 8 <= len; t += 8) {
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, _mm256_mul_ps(Tr::decode8(row, kk0 + t), vs));
    for (int u = 0; u < 8; ++u) out[(t + u) * stride] = tmp[u];
  }
#endif
  for (; t < len; ++t) out[t * stride] = Tr::decode(row, kk0 + t) * scale;
}

// A panel pack: (i0+i, k0+k) -> dst[(strip*kc + k) * mr + i], block scale
// folded in; out-of-range rows are zero-filled.
template <typename T>
void mx_pack_a(const MxTensorRef<T>& a, int i0, int k0, int mc, int kc, int mr,
               float* dst) {
  using Tr = traits<T>;
  for (int ib = 0; ib < mc; ib += mr) {
    const int imax = (mc - ib < mr) ? (mc - ib) : mr;
    float* strip = dst + (ib / mr) * kc * mr;
    for (int i = 0; i < imax; ++i) {
      const int row = i0 + ib + i;
      const uint8_t* arow =
          a.data + (static_cast<size_t>(row) * a.ld >> Tr::kLog2ElemsPerByte);
      const uint8_t* srow = a.scales + static_cast<size_t>(row) * a.ld_scales;
      int k = 0;
      while (k < kc) {
        const int kk = k0 + k;
        const int to_boundary = 32 - (kk & 31);
        const int seg = (kc - k < to_boundary) ? (kc - k) : to_boundary;
        decode_segment<T>(arow, kk, seg, e8m0_to_float(srow[kk >> 5]),
                          strip + k * mr + i, mr);
        k += seg;
      }
    }
    for (int k = 0; k < kc; ++k) {
      for (int i2 = imax; i2 < mr; ++i2) strip[k * mr + i2] = 0.0f;
    }
  }
}

// B panel pack: elements (k0+k, j0+j) -> dst[(strip*kc + k) * nr + j].
template <typename T>
void mx_pack_b(const MxTensorRef<T>& b, int k0, int j0, int kc, int nc, int nr,
               float* dst) {
  using Tr = traits<T>;
  for (int jb = 0; jb < nc; jb += nr) {
    const int jmax = (nc - jb < nr) ? (nc - jb) : nr;
    float* strip = dst + (jb / nr) * kc * nr;
    for (int j = 0; j < jmax; ++j) {
      const int col = j0 + jb + j;
      const uint8_t* bcol =
          b.data + (static_cast<size_t>(col) * b.ld >> Tr::kLog2ElemsPerByte);
      const uint8_t* scol = b.scales + static_cast<size_t>(col) * b.ld_scales;
      int k = 0;
      while (k < kc) {
        const int kk = k0 + k;
        const int to_boundary = 32 - (kk & 31);
        const int seg = (kc - k < to_boundary) ? (kc - k) : to_boundary;
        decode_segment<T>(bcol, kk, seg, e8m0_to_float(scol[kk >> 5]),
                          strip + k * nr + j, nr);
        k += seg;
      }
    }
    for (int k = 0; k < kc; ++k) {
      for (int j = jmax; j < nr; ++j) strip[k * nr + j] = 0.0f;
    }
  }
}

}  // namespace mx_detail2

// VLEN selects the f32 mainloop vector width (device-level defaults give 16 on AVX-512F).
template <typename StorageA, typename StorageB, int VLEN = simd::native_width<float>()>
struct MxPolicy {
  static_assert(mx_detail2::is_mx_element<StorageA>::value &&
                    mx_detail2::is_mx_element<StorageB>::value,
                "MxPolicy operands must be fp8e4m3_t, fp8e5m2_t, or fp4e2m1_t");

  using ElemA = StorageA;
  using ElemB = StorageB;
  using AccT = float;
  using PackedA = float;
  using PackedB = float;

  static constexpr int kKStep = 1;
  static constexpr int pad_kc(int kc) { return kc; }

  template <int MR, int NR>
  using Atom = MmaAtom<float, MR, NR, VLEN>;

  static void pack_a(const MxTensorRef<StorageA>& a, int i0, int k0, int mc, int kc,
                     int /*kc_pad*/, int mr, float* dst) {
    mx_detail2::mx_pack_a(a, i0, k0, mc, kc, mr, dst);
  }

  static void pack_b(const MxTensorRef<StorageB>& b, int k0, int j0, int kc,
                     int /*kc_pad*/, int nc, int nr, float* dst) {
    mx_detail2::mx_pack_b(b, k0, j0, kc, nc, nr, dst);
  }
};

}  // namespace mma
}  // namespace cpu_ops

#if defined(CPU_OPS_MX_SIMD_DECODE_512)
#undef CPU_OPS_MX_SIMD_DECODE_512
#endif
#if defined(CPU_OPS_MX_SIMD_DECODE)
#undef CPU_OPS_MX_SIMD_DECODE
#endif
