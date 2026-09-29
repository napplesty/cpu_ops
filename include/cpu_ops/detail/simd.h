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
inline Vec<float, 8> max(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_max_ps(a.v, b.v);
}
inline Vec<float, 8> min(const Vec<float, 8>& a, const Vec<float, 8>& b) {
  return _mm256_min_ps(a.v, b.v);
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
inline Vec<float, 4> max(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vmaxq_f32(a.v, b.v);
}
inline Vec<float, 4> min(const Vec<float, 4>& a, const Vec<float, 4>& b) {
  return vminq_f32(a.v, b.v);
}

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
