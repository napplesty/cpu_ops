#pragma once

// Compile-time SIMD ISA detection. Define CPU_OPS_FORCE_SCALAR to disable all
// intrinsics and exercise the portable fallback (used by tests).
//
// Priority order: x86 AVX2 > ARM SVE (fixed-width build) > ARM NEON > scalar.
#if defined(CPU_OPS_FORCE_SCALAR)
// no SIMD ISA selected
#elif defined(__AVX2__) && defined(__FMA__)
#define CPU_OPS_SIMD_AVX2 1
#include <immintrin.h>
#if defined(__AVXVNNI__) || (defined(__AVX512VNNI__) && defined(__AVX512VL__))
#define CPU_OPS_SIMD_VNNI 1
#endif
#elif defined(__ARM_FEATURE_SVE) && defined(__ARM_FEATURE_SVE_BITS)
// Fixed-width SVE/SVE2 build (-msve-vector-bits=N): SVE types get a static
// size and can live in structs, so the Vec<T, N> template maps onto them.
#define CPU_OPS_SIMD_SVE 1
#include <arm_sve.h>
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
#define CPU_OPS_SIMD_NEON 1
#include <arm_neon.h>
#endif

#include <cmath>
#include <cstdint>
#include <cstring>

namespace cpu_ops {
namespace simd {

// Portable scalar vector. This is the fallback for ISAs without a
// specialization below; the compiler is free to auto-vectorize the loops.
// Loads and stores go through memcpy so the same code is safe for packed
// byte buffers reinterpreted as wider lanes.
template <typename T, int N>
struct Vec {
  T v[N];

  static Vec load(const T* p) {
    Vec r;
    for (int i = 0; i < N; ++i) std::memcpy(&r.v[i], p + i, sizeof(T));
    return r;
  }
  static Vec set1(T x) {
    Vec r;
    for (int i = 0; i < N; ++i) r.v[i] = x;
    return r;
  }
  // Broadcasts a raw 32-bit pattern into every lane (int8 dot-product path).
  static Vec set1_u32(uint32_t x) {
    static_assert(sizeof(T) == sizeof(uint32_t), "set1_u32 needs 32-bit lanes");
    T val;
    std::memcpy(&val, &x, sizeof(T));
    Vec r;
    for (int i = 0; i < N; ++i) r.v[i] = val;
    return r;
  }
  void store(T* p) const {
    for (int i = 0; i < N; ++i) std::memcpy(p + i, &v[i], sizeof(T));
  }
};

template <typename T, int N>
inline Vec<T, N> fmadd(const Vec<T, N>& a, const Vec<T, N>& b, const Vec<T, N>& c) {
  Vec<T, N> r;
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] * b.v[i] + c.v[i];
  return r;
}

template <typename T, int N>
inline Vec<T, N> mul(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r;
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] * b.v[i];
  return r;
}

template <typename T, int N>
inline Vec<T, N> add(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r;
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] + b.v[i];
  return r;
}

template <typename T, int N>
inline Vec<T, N> sub(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r;
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] - b.v[i];
  return r;
}

template <typename T, int N>
inline Vec<T, N> max(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r;
  for (int i = 0; i < N; ++i) r.v[i] = a.v[i] < b.v[i] ? b.v[i] : a.v[i];
  return r;
}

template <typename T, int N>
inline Vec<T, N> min(const Vec<T, N>& a, const Vec<T, N>& b) {
  Vec<T, N> r;
  for (int i = 0; i < N; ++i) r.v[i] = b.v[i] < a.v[i] ? b.v[i] : a.v[i];
  return r;
}

// 4-way unsigned x signed byte dot product accumulated per 32-bit lane:
// each lane holds 4 packed k-slices; a is uint8, b is int8.
template <int N>
inline Vec<int32_t, N> dpbusd(Vec<int32_t, N> acc, Vec<int32_t, N> a, Vec<int32_t, N> b) {
  for (int l = 0; l < N; ++l) {
    uint32_t av, bv;
    std::memcpy(&av, &a.v[l], 4);
    std::memcpy(&bv, &b.v[l], 4);
    int32_t s = 0;
    for (int t = 0; t < 4; ++t) {
      s += static_cast<int32_t>(static_cast<uint8_t>(av >> (8 * t))) *
           static_cast<int32_t>(static_cast<int8_t>(bv >> (8 * t)));
    }
    acc.v[l] += s;
  }
  return acc;
}

// ---- Narrow → f32 widening loads -----------------------------------------
// Raw 16-bit layouts only (IEEE binary16; bfloat16 is the high half of a
// binary32), so these stay independent of element_types.h. ISA sections
// below override the generic versions with single-instruction converts.

inline float f16_bits_to_f32(uint16_t h) {
  const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  const uint32_t mant = h & 0x3FF;
  const uint32_t exp = (h >> 10) & 0x1F;
  uint32_t f;
  if (exp == 0) {
    if (mant == 0) {
      f = sign;  // ±0
    } else {  // denormal: normalize into the f32 exponent range
      uint32_t e = 127 - 15 + 1;
      uint32_t m = mant;
      while (!(m & 0x400)) {
        m <<= 1;
        --e;
      }
      f = sign | (e << 23) | ((m & 0x3FF) << 13);
    }
  } else if (exp == 31) {
    f = sign | 0x7F800000u | (mant << 13);  // inf / nan
  } else {
    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
  }
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

template <int W>
inline Vec<float, W> widen_f16(const uint16_t* p) {
  Vec<float, W> r;
  for (int i = 0; i < W; ++i) r.v[i] = f16_bits_to_f32(p[i]);
  return r;
}

template <int W>
inline Vec<float, W> widen_bf16(const uint16_t* p) {
  Vec<float, W> r;
  for (int i = 0; i < W; ++i) {
    const uint32_t f = static_cast<uint32_t>(p[i]) << 16;
    std::memcpy(&r.v[i], &f, 4);
  }
  return r;
}

// ---- Softmax helpers (f32 lanes) -----------------------------------------
// The generic implementations round-trip through a stack array so they work
// for every Vec flavor (scalar fallback and SVE included); AVX2 and NEON
// override exp2/hmax/hsum below. hsum uses a fixed lane order so summation
// results are bit-stable for a given build.

template <int N>
inline Vec<float, N> exp2(Vec<float, N> x) {
  float t[N];
  x.store(t);
  for (int i = 0; i < N; ++i) t[i] = std::exp2f(t[i]);
  return Vec<float, N>::load(t);
}

template <int N>
inline float hmax(const Vec<float, N>& x) {
  float t[N];
  x.store(t);
  float m = t[0];
  for (int i = 1; i < N; ++i) m = m < t[i] ? t[i] : m;
  return m;
}

template <int N>
inline float hsum(const Vec<float, N>& x) {
  float t[N];
  x.store(t);
  float s = t[0];
  for (int i = 1; i < N; ++i) s += t[i];
  return s;
}

#if defined(CPU_OPS_SIMD_AVX2)

template <>
struct Vec<float, 8> {
  __m256 v;
  Vec() = default;
  Vec(__m256 x) : v(x) {}  // implicit on purpose
  static Vec load(const float* p) { return _mm256_loadu_ps(p); }
  static Vec set1(float x) { return _mm256_set1_ps(x); }
  void store(float* p) const { _mm256_storeu_ps(p, v); }
};

inline Vec<float, 8> fmadd(const Vec<float, 8>& a, const Vec<float, 8>& b,
                           const Vec<float, 8>& c) {
  return _mm256_fmadd_ps(a.v, b.v, c.v);
}

inline Vec<float, 8> mul(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_mul_ps(a.v, b.v);
}
inline Vec<float, 8> add(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_add_ps(a.v, b.v);
}
inline Vec<float, 8> sub(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_sub_ps(a.v, b.v);
}

// Widening loads: bf16 is a zero-extend + shift; f16 wants F16C (present on
// effectively every AVX2 part) and otherwise stays on the generic path.
template <>
inline Vec<float, 8> widen_bf16<8>(const uint16_t* p) {
  const __m256i u = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  Vec<float, 8> r;
  r.v = _mm256_castsi256_ps(_mm256_slli_epi32(u, 16));
  return r;
}
#if defined(__F16C__)
template <>
inline Vec<float, 8> widen_f16<8>(const uint16_t* p) {
  Vec<float, 8> r;
  r.v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  return r;
}
#endif
inline Vec<float, 8> max(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_max_ps(a.v, b.v);
}
inline Vec<float, 8> min(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_min_ps(a.v, b.v);
}

// exp2: round-to-nearest integer exponent (magic-add trick) + degree-6
// minimax polynomial for 2^r on [-0.5, 0.5], relative error ~1e-7. Inputs are
// clamped so extreme arguments flush to ~0 / +inf instead of corrupting the
// exponent-bit reconstruction.
template <>
inline Vec<float, 8> exp2(Vec<float, 8> x) {
  const __m256 xc = _mm256_min_ps(_mm256_max_ps(x.v, _mm256_set1_ps(-126.0f)),
                                  _mm256_set1_ps(128.0f));
  const __m256 magic = _mm256_set1_ps(12582912.0f);  // 1.5 * 2^23
  const __m256 n = _mm256_sub_ps(_mm256_add_ps(xc, magic), magic);
  const __m256 r = _mm256_sub_ps(xc, n);  // [-0.5, 0.5]
  __m256 p = _mm256_set1_ps(0.00015309011f);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0013397580f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0096181291f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.0555041100f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.2402265100f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(0.6931471800f));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(1.0f));
  const __m256i e =
      _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(n), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(p, _mm256_castsi256_ps(e));
}

// Fixed reduction trees: identical order on every call, so hsum results are
// bit-stable across threads and runs.
template <>
inline float hmax(const Vec<float, 8>& x) {
  __m128 m = _mm_max_ps(_mm256_castps256_ps128(x.v), _mm256_extractf128_ps(x.v, 1));
  m = _mm_max_ps(m, _mm_movehl_ps(m, m));
  m = _mm_max_ps(m, _mm_shuffle_ps(m, m, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(m);
}

template <>
inline float hsum(const Vec<float, 8>& x) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(x.v), _mm256_extractf128_ps(x.v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ps(s, _mm_shuffle_ps(s, s, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(s);
}

template <>
struct Vec<double, 4> {
  __m256d v;
  Vec() = default;
  Vec(__m256d x) : v(x) {}  // implicit on purpose
  static Vec load(const double* p) { return _mm256_loadu_pd(p); }
  static Vec set1(double x) { return _mm256_set1_pd(x); }
  void store(double* p) const { _mm256_storeu_pd(p, v); }
};

inline Vec<double, 4> fmadd(const Vec<double, 4>& a, const Vec<double, 4>& b,
                            const Vec<double, 4>& c) {
  return _mm256_fmadd_pd(a.v, b.v, c.v);
}

inline Vec<double, 4> mul(const Vec<double, 4>& a, const Vec<double, 4>& b) {
  return _mm256_mul_pd(a.v, b.v);
}
inline Vec<double, 4> add(const Vec<double, 4>& a, const Vec<double, 4>& b) {
  return _mm256_add_pd(a.v, b.v);
}
inline Vec<double, 4> max(const Vec<double, 4>& a, const Vec<double, 4>& b) {
  return _mm256_max_pd(a.v, b.v);
}
inline Vec<double, 4> min(const Vec<double, 4>& a, const Vec<double, 4>& b) {
  return _mm256_min_pd(a.v, b.v);
}

#if defined(CPU_OPS_SIMD_VNNI)

template <>
struct Vec<int32_t, 8> {
  __m256i v;
  Vec() = default;
  Vec(__m256i x) : v(x) {}  // implicit on purpose
  static Vec load(const int32_t* p) {
    return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p));
  }
  static Vec set1(int32_t x) { return _mm256_set1_epi32(x); }
  static Vec set1_u32(uint32_t x) { return _mm256_set1_epi32(static_cast<int32_t>(x)); }
  void store(int32_t* p) const {
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(p), v);
  }
};

inline Vec<int32_t, 8> dpbusd(Vec<int32_t, 8> acc, Vec<int32_t, 8> a, Vec<int32_t, 8> b) {
  acc.v = _mm256_dpbusd_epi32(acc.v, a.v, b.v);
  return acc;
}

inline Vec<int32_t, 8> mul(const Vec<int32_t, 8>& a, const Vec<int32_t, 8>& b) {
  return _mm256_mullo_epi32(a.v, b.v);
}
inline Vec<int32_t, 8> add(const Vec<int32_t, 8>& a, const Vec<int32_t, 8>& b) {
  return _mm256_add_epi32(a.v, b.v);
}
inline Vec<int32_t, 8> max(const Vec<int32_t, 8>& a, const Vec<int32_t, 8>& b) {
  return _mm256_max_epi32(a.v, b.v);
}
inline Vec<int32_t, 8> min(const Vec<int32_t, 8>& a, const Vec<int32_t, 8>& b) {
  return _mm256_min_epi32(a.v, b.v);
}

#endif  // CPU_OPS_SIMD_VNNI
#endif  // CPU_OPS_SIMD_AVX2

#if defined(CPU_OPS_SIMD_SVE)

#define CPU_OPS_DETAIL_SVE_VEC(BITS)                                              \
  template <>                                                                     \
  struct Vec<float, (BITS) / 32> {                                                \
    svfloat32_t v;                                                                \
    Vec() = default;                                                              \
    Vec(svfloat32_t x) : v(x) {}                                                  \
    static Vec load(const float* p) { return svld1_f32(svptrue_b32(), p); }       \
    static Vec set1(float x) { return svdup_n_f32(x); }                           \
    void store(float* p) const { svst1_f32(svptrue_b32(), p, v); }                \
  };                                                                              \
  inline Vec<float, (BITS) / 32> fmadd(const Vec<float, (BITS) / 32>& a,          \
                                       const Vec<float, (BITS) / 32>& b,          \
                                       const Vec<float, (BITS) / 32>& c) {        \
    return svmad_f32_x(svptrue_b32(), a.v, b.v, c.v);                             \
  }                                                                               \
  inline Vec<float, (BITS) / 32> mul(const Vec<float, (BITS) / 32>& a,            \
                                     const Vec<float, (BITS) / 32>& b) {          \
    return svmul_f32_x(svptrue_b32(), a.v, b.v);                                  \
  }                                                                               \
  inline Vec<float, (BITS) / 32> add(const Vec<float, (BITS) / 32>& a,            \
                                     const Vec<float, (BITS) / 32>& b) {          \
    return svadd_f32_x(svptrue_b32(), a.v, b.v);                                  \
  }                                                                               \
  inline Vec<float, (BITS) / 32> max(const Vec<float, (BITS) / 32>& a,            \
                                     const Vec<float, (BITS) / 32>& b) {          \
    return svmax_f32_x(svptrue_b32(), a.v, b.v);                                  \
  }                                                                               \
  inline Vec<float, (BITS) / 32> min(const Vec<float, (BITS) / 32>& a,            \
                                     const Vec<float, (BITS) / 32>& b) {          \
    return svmin_f32_x(svptrue_b32(), a.v, b.v);                                  \
  }                                                                               \
  template <>                                                                     \
  struct Vec<double, (BITS) / 64> {                                               \
    svfloat64_t v;                                                                \
    Vec() = default;                                                              \
    Vec(svfloat64_t x) : v(x) {}                                                  \
    static Vec load(const double* p) { return svld1_f64(svptrue_b64(), p); }      \
    static Vec set1(double x) { return svdup_n_f64(x); }                          \
    void store(double* p) const { svst1_f64(svptrue_b64(), p, v); }               \
  };                                                                              \
  inline Vec<double, (BITS) / 64> fmadd(const Vec<double, (BITS) / 64>& a,        \
                                        const Vec<double, (BITS) / 64>& b,        \
                                        const Vec<double, (BITS) / 64>& c) {      \
    return svmad_f64_x(svptrue_b64(), a.v, b.v, c.v);                             \
  }                                                                               \
  inline Vec<double, (BITS) / 64> mul(const Vec<double, (BITS) / 64>& a,          \
                                      const Vec<double, (BITS) / 64>& b) {        \
    return svmul_f64_x(svptrue_b64(), a.v, b.v);                                  \
  }                                                                               \
  inline Vec<double, (BITS) / 64> add(const Vec<double, (BITS) / 64>& a,          \
                                      const Vec<double, (BITS) / 64>& b) {        \
    return svadd_f64_x(svptrue_b64(), a.v, b.v);                                  \
  }                                                                               \
  inline Vec<double, (BITS) / 64> max(const Vec<double, (BITS) / 64>& a,          \
                                      const Vec<double, (BITS) / 64>& b) {        \
    return svmax_f64_x(svptrue_b64(), a.v, b.v);                                  \
  }                                                                               \
  inline Vec<double, (BITS) / 64> min(const Vec<double, (BITS) / 64>& a,          \
                                      const Vec<double, (BITS) / 64>& b) {        \
    return svmin_f64_x(svptrue_b64(), a.v, b.v);                                  \
  }

#if __ARM_FEATURE_SVE_BITS == 128
CPU_OPS_DETAIL_SVE_VEC(128)
#elif __ARM_FEATURE_SVE_BITS == 256
CPU_OPS_DETAIL_SVE_VEC(256)
#elif __ARM_FEATURE_SVE_BITS == 512
CPU_OPS_DETAIL_SVE_VEC(512)
#elif __ARM_FEATURE_SVE_BITS == 1024
CPU_OPS_DETAIL_SVE_VEC(1024)
#elif __ARM_FEATURE_SVE_BITS == 2048
CPU_OPS_DETAIL_SVE_VEC(2048)
#endif

#undef CPU_OPS_DETAIL_SVE_VEC
#endif  // CPU_OPS_SIMD_SVE

#if defined(CPU_OPS_SIMD_NEON)

template <>
struct Vec<float, 4> {
  float32x4_t v;
  Vec() = default;
  Vec(float32x4_t x) : v(x) {}  // implicit on purpose
  static Vec load(const float* p) { return vld1q_f32(p); }
  static Vec set1(float x) { return vdupq_n_f32(x); }
  void store(float* p) const { vst1q_f32(p, v); }
};

inline Vec<float, 4> fmadd(const Vec<float, 4>& a, const Vec<float, 4>& b,
                           const Vec<float, 4>& c) {
  return vfmaq_f32(c.v, a.v, b.v);
}

inline Vec<float, 4> mul(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vmulq_f32(a.v, b.v);
}
inline Vec<float, 4> add(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vaddq_f32(a.v, b.v);
}
inline Vec<float, 4> sub(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vsubq_f32(a.v, b.v);
}

// Widening loads: f16 is one vcvt per 4 lanes; bf16 is a 16-bit left shift
// into the f32 lanes.
template <>
inline Vec<float, 4> widen_f16<4>(const uint16_t* p) {
  Vec<float, 4> r;
  r.v = vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(p)));
  return r;
}
template <>
inline Vec<float, 4> widen_bf16<4>(const uint16_t* p) {
  Vec<float, 4> r;
  r.v = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16));
  return r;
}
inline Vec<float, 4> max(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vmaxq_f32(a.v, b.v);
}
inline Vec<float, 4> min(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vminq_f32(a.v, b.v);
}

// exp2, same scheme as the AVX2 version (round-to-nearest-even exponent via
// vcvtnq + degree-6 polynomial on [-0.5, 0.5], inputs clamped).
template <>
inline Vec<float, 4> exp2(Vec<float, 4> x) {
  const float32x4_t xc =
      vminq_f32(vmaxq_f32(x.v, vdupq_n_f32(-126.0f)), vdupq_n_f32(128.0f));
  const int32x4_t ni = vcvtnq_s32_f32(xc);
  const float32x4_t n = vcvtq_f32_s32(ni);
  const float32x4_t r = vsubq_f32(xc, n);  // [-0.5, 0.5]
  float32x4_t p = vdupq_n_f32(0.00015309011f);
  p = vfmaq_f32(vdupq_n_f32(0.0013397580f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.0096181291f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.0555041100f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.2402265100f), p, r);
  p = vfmaq_f32(vdupq_n_f32(0.6931471800f), p, r);
  p = vfmaq_f32(vdupq_n_f32(1.0f), p, r);
  const int32x4_t e = vshlq_n_s32(vaddq_s32(ni, vdupq_n_s32(127)), 23);
  return vmulq_f32(p, vreinterpretq_f32_s32(e));
}

#if defined(__aarch64__)
template <>
inline float hmax(const Vec<float, 4>& x) {
  return vmaxvq_f32(x.v);
}
template <>
inline float hsum(const Vec<float, 4>& x) {
  return vaddvq_f32(x.v);
}
#endif  // __aarch64__

#endif  // CPU_OPS_SIMD_NEON

// Number of elements in the widest efficient native vector for T.
template <typename T>
constexpr int native_width() {
#if defined(CPU_OPS_SIMD_AVX2)
  return sizeof(T) == 4 ? 8 : 4;
#elif defined(CPU_OPS_SIMD_SVE)
  return sizeof(T) == 4 ? __ARM_FEATURE_SVE_BITS / 32 : __ARM_FEATURE_SVE_BITS / 64;
#elif defined(CPU_OPS_SIMD_NEON)
  return sizeof(T) == 4 ? 4 : 2;
#else
  return 1;
#endif
}

}  // namespace simd
}  // namespace cpu_ops
