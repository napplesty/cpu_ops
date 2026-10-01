#pragma once

// Narrow floating-point storage types. They are pure containers: arithmetic
// is performed by widening to float (the GEMM mainloop converts panels to
// f32 at pack time, see mma_policy_widen.h). Conversions to/from float are
// IEEE round-to-nearest-even and preserve infinities and NaNs.

#include <cstdint>
#include <cstring>

#if defined(__F16C__) && !defined(CPU_OPS_FORCE_SCALAR)
#include <immintrin.h>
#define CPU_OPS_HAS_F16C 1
#endif

namespace cpu_ops {

// IEEE 754 binary16, stored as raw bits.
struct float16_t {
  uint16_t bits = 0;

  float16_t() = default;
  float16_t(float f) : bits(from_float(f)) {}  // implicit: storage type
  float16_t(double f) : bits(from_float(static_cast<float>(f))) {}

  // Tag type for constructing from raw bits without conversion.
  struct Raw {};
  constexpr float16_t(uint16_t raw_bits, Raw) : bits(raw_bits) {}

  operator float() const { return to_float(bits); }

  static float to_float(uint16_t h) {
#if defined(CPU_OPS_HAS_F16C)
    return _cvtsh_ss(h);
#else
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    const uint32_t mant = h & 0x3FF;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t f;
    if (exp == 0) {
      if (mant == 0) {
        f = sign;
      } else {
        // Denormal: normalize into the f32 exponent range.
        exp = 127 - 15 + 1;
        uint32_t m = mant;
        while (!(m & 0x400)) {
          m <<= 1;
          --exp;
        }
        f = sign | (exp << 23) | ((m & 0x3FF) << 13);
      }
    } else if (exp == 31) {
      f = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
      f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, 4);
    return out;
#endif
  }

  static uint16_t from_float(float x) {
#if defined(CPU_OPS_HAS_F16C)
    return static_cast<uint16_t>(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
#else
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 16) & 0x8000;
    const uint32_t fexp = (f >> 23) & 0xFF;
    const uint32_t mant = f & 0x7FFFFF;
    if (fexp == 0xFF) {  // inf / nan
      return static_cast<uint16_t>(sign | 0x7C00 | (mant >> 13) | (mant ? 0x200 : 0));
    }
    const int exp = static_cast<int>(fexp) - 127 + 15;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00);  // overflow -> inf
    if (exp <= 0) {
      if (exp < -10) return static_cast<uint16_t>(sign);  // underflow -> zero
      // Denormal result: round mantissa (with implicit leading 1) to the
      // remaining precision.
      const uint32_t mant24 = mant | 0x800000;
      const int shift = 14 - exp;  // in [14, 24]
      uint32_t half = mant24 >> shift;
      const uint32_t rem = mant24 & ((1u << shift) - 1);
      const uint32_t halfway = 1u << (shift - 1);
      if (rem > halfway || (rem == halfway && (half & 1))) ++half;
      return static_cast<uint16_t>(sign | half);
    }
    uint32_t half = (static_cast<uint32_t>(exp) << 10) | (mant >> 13);
    const uint32_t rem = mant & 0x1FFF;
    if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
#endif
  }
};

// bfloat16 (top 16 bits of binary32), stored as raw bits.
struct bfloat16_t {
  uint16_t bits = 0;

  bfloat16_t() = default;
  bfloat16_t(float f) : bits(from_float(f)) {}  // implicit: storage type
  bfloat16_t(double f) : bits(from_float(static_cast<float>(f))) {}

  struct Raw {};
  constexpr bfloat16_t(uint16_t raw_bits, Raw) : bits(raw_bits) {}

  operator float() const { return to_float(bits); }

  static float to_float(uint16_t h) {
    const uint32_t f = static_cast<uint32_t>(h) << 16;
    float out;
    std::memcpy(&out, &f, 4);
    return out;
  }

  static uint16_t from_float(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    if ((f & 0x7FFFFFFF) > 0x7F800000) {  // nan -> quiet nan
      return 0x7FC0;
    }
    const uint32_t rounding_bias = 0x7FFF + ((f >> 16) & 1);
    return static_cast<uint16_t>((f + rounding_bias) >> 16);
  }
};

}  // namespace cpu_ops

#if defined(CPU_OPS_HAS_F16C)
#undef CPU_OPS_HAS_F16C
#endif
