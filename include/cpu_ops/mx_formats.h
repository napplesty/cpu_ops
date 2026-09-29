#pragma once

// OCP MX (Microscaling) element formats: fp8 e4m3 / e5m2 and fp4 e2m1, with
// an E8M0 power-of-two scale shared by every block of 32 elements along k.
//
// These are storage formats only. Decoding to float goes through constexpr
// lookup tables (exact by construction); encoding from float is
// round-to-nearest-even with saturation at the format's maximum magnitude.
//
// fp4 packs two elements per byte; following the OCP MX convention the
// element with the even flat index sits in the byte's low nibble.

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace cpu_ops {

namespace mx_detail {

inline float bits_to_float(uint32_t f) {
  float out;
  std::memcpy(&out, &f, 4);
  return out;
}

constexpr uint32_t f32_bits(uint32_t sign, uint32_t exp, uint32_t mant) {
  return (sign << 31) | (exp << 23) | mant;
}

// e4m3: 1-4-3, bias 7, no infinity; (exp=15, mant=7) is NaN. Denormals are
// mant * 2^-9.
constexpr std::array<uint32_t, 256> build_e4m3_lut() {
  std::array<uint32_t, 256> t{};
  for (int v = 0; v < 256; ++v) {
    const uint32_t sign = (v >> 7) & 1;
    const uint32_t exp = (v >> 3) & 0xF;
    const uint32_t mant = v & 0x7;
    uint32_t f = 0;
    if (exp == 0) {
      if (mant == 0) {
        f = sign << 31;
      } else {
        // Denormal mant * 2^-9: p = position of the top set bit gives the
        // exponent, the remaining bits the fraction.
        const uint32_t p = (mant & 0x4) ? 2 : ((mant & 0x2) ? 1 : 0);
        f = f32_bits(sign, p + 127 - 9, (mant & ((1u << p) - 1)) << (23 - p));
      }
    } else if (exp == 15 && mant == 7) {
      f = f32_bits(sign, 0xFF, 0x400000);  // NaN
    } else {
      f = f32_bits(sign, exp - 7 + 127, mant << 20);
    }
    t[v] = f;
  }
  return t;
}

// e5m2: 1-5-2, bias 15, with infinities (exp=31, mant=0) and NaNs.
// Denormals are mant * 2^-16.
constexpr std::array<uint32_t, 256> build_e5m2_lut() {
  std::array<uint32_t, 256> t{};
  for (int v = 0; v < 256; ++v) {
    const uint32_t sign = (v >> 7) & 1;
    const uint32_t exp = (v >> 2) & 0x1F;
    const uint32_t mant = v & 0x3;
    uint32_t f = 0;
    if (exp == 0) {
      if (mant == 0) {
        f = sign << 31;
      } else {
        uint32_t e = 127 - 16;
        uint32_t m = mant;
        while (!(m & 0x2)) {
          m <<= 1;
          --e;
        }
        f = f32_bits(sign, e + 1, (m & 0x1) << 22);
      }
    } else if (exp == 31) {
      f = f32_bits(sign, 0xFF, mant << 21);  // inf (mant=0) / nan
    } else {
      f = f32_bits(sign, exp - 15 + 127, mant << 21);
    }
    t[v] = f;
  }
  return t;
}

// e2m1: 1-2-1, bias 1, no inf/nan. Values: +-{0, .5, 1, 1.5, 2, 3, 4, 6}.
constexpr std::array<uint32_t, 16> build_e2m1_lut() {
  std::array<uint32_t, 16> t{};
  for (int v = 0; v < 16; ++v) {
    const uint32_t sign = (v >> 3) & 1;
    const uint32_t exp = (v >> 1) & 0x3;
    const uint32_t mant = v & 0x1;
    uint32_t f = 0;
    if (exp == 0) {
      f = mant ? f32_bits(sign, 126, 0) : (sign << 31);  // 0.5 / 0
    } else {
      f = f32_bits(sign, exp - 1 + 127, mant << 22);
    }
    t[v] = f;
  }
  return t;
}

}  // namespace mx_detail

// IEEE-ish 8-bit float, 1-4-3 (OCP float8_e4m3fn semantics: no inf).
struct fp8e4m3_t {
  static constexpr std::array<uint32_t, 256> kLut = mx_detail::build_e4m3_lut();

  uint8_t bits = 0;

  fp8e4m3_t() = default;
  fp8e4m3_t(float f) : bits(from_float(f)) {}  // implicit: storage type

  struct Raw {};
  constexpr fp8e4m3_t(uint8_t raw_bits, Raw) : bits(raw_bits) {}

  operator float() const { return to_float(bits); }

  static float to_float(uint8_t b) { return mx_detail::bits_to_float(kLut[b]); }

  // RNE, saturating: magnitudes past 448 (and inf) map to +-448, NaN to NaN.
  static uint8_t from_float(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 24) & 0x80;
    const uint32_t fexp = (f >> 23) & 0xFF;
    const uint32_t mant = f & 0x7FFFFF;
    if (fexp == 0xFF) {  // inf -> max, nan -> nan
      return static_cast<uint8_t>(sign | (mant ? 0x7F : 0x7E));
    }
    const int e = static_cast<int>(fexp) - 127 + 7;
    if (e > 15) return static_cast<uint8_t>(sign | 0x7E);  // saturate to 448
    if (e <= 0) {
      // fp8 denormal: value = mant24 * 2^(e_unb - 23); quantize to units of
      // 2^-9, i.e. shift right by 14 - e_unb.
      const int e_unb = static_cast<int>(fexp) - 127;
      if (e_unb < -10) return static_cast<uint8_t>(sign);  // underflow -> zero
      const uint32_t mant24 = mant | 0x800000;
      const int shift = 14 - e_unb;  // in [14, 24]
      uint32_t m3 = mant24 >> shift;
      const uint32_t rem = mant24 & ((1u << shift) - 1);
      const uint32_t halfway = 1u << (shift - 1);
      if (rem > halfway || (rem == halfway && (m3 & 1))) ++m3;  // may carry to 0x08
      return static_cast<uint8_t>(sign | m3);
    }
    uint32_t half = (static_cast<uint32_t>(e) << 3) | (mant >> 20);
    const uint32_t rem = mant & 0xFFFFF;
    if (rem > 0x80000 || (rem == 0x80000 && (half & 1))) ++half;
    // Rounding may carry into the NaN encoding (0x7F); saturate to 448.
    if (half > 0x7E) half = 0x7E;
    return static_cast<uint8_t>(sign | half);
  }
};

// 8-bit float, 1-5-2 (with inf and nan, fp16-style exponent).
struct fp8e5m2_t {
  static constexpr std::array<uint32_t, 256> kLut = mx_detail::build_e5m2_lut();

  uint8_t bits = 0;

  fp8e5m2_t() = default;
  fp8e5m2_t(float f) : bits(from_float(f)) {}  // implicit: storage type

  struct Raw {};
  constexpr fp8e5m2_t(uint8_t raw_bits, Raw) : bits(raw_bits) {}

  operator float() const { return to_float(bits); }

  static float to_float(uint8_t b) { return mx_detail::bits_to_float(kLut[b]); }

  // RNE; overflow rounds to infinity, NaN stays NaN.
  static uint8_t from_float(float x) {
    uint32_t f;
    std::memcpy(&f, &x, 4);
    const uint32_t sign = (f >> 24) & 0x80;
    const uint32_t fexp = (f >> 23) & 0xFF;
    const uint32_t mant = f & 0x7FFFFF;
    if (fexp == 0xFF) {
      return static_cast<uint8_t>(sign | 0x7C | (mant ? 0x2 : 0));  // inf / nan
    }
    const int e = static_cast<int>(fexp) - 127 + 15;
    if (e >= 31) return static_cast<uint8_t>(sign | 0x7C);  // overflow -> inf
    if (e <= 0) {
      const int e_unb = static_cast<int>(fexp) - 127;
      if (e_unb < -17) return static_cast<uint8_t>(sign);  // underflow -> zero
      const uint32_t mant24 = mant | 0x800000;
      const int shift = 7 - e_unb;  // in [17, 24]
      uint32_t m2 = mant24 >> shift;
      const uint32_t rem = mant24 & ((1u << shift) - 1);
      const uint32_t halfway = 1u << (shift - 1);
      if (rem > halfway || (rem == halfway && (m2 & 1))) ++m2;  // may carry to 0x04
      return static_cast<uint8_t>(sign | m2);
    }
    uint32_t half = (static_cast<uint32_t>(e) << 2) | (mant >> 21);
    const uint32_t rem = mant & 0x1FFFFF;
    if (rem > 0x100000 || (rem == 0x100000 && (half & 1))) ++half;  // may carry to inf
    return static_cast<uint8_t>(sign | half);
  }
};

// 4-bit float, 1-2-1. Two elements share a byte, so this type only provides
// nibble-level conversion helpers; storage addressing lives in the GEMM
// packing code (mx_traits in mma_policy_mx.h).
struct fp4e2m1_t {
  static constexpr std::array<uint32_t, 16> kLut = mx_detail::build_e2m1_lut();

  static float to_float(uint8_t nibble) {
    return mx_detail::bits_to_float(kLut[nibble & 0xF]);
  }

  // RNE with ties to the even code; NaN is not representable and maps to 0.
  static uint8_t from_float(float x) {
    if (x != x) return 0;  // NaN
    const uint8_t sign = std::signbit(x) ? 0x8 : 0x0;
    const float a = std::fabs(x);
    uint8_t code;
    if (a < 0.25f) {
      code = 0;
    } else if (a == 0.25f) {
      code = 0;  // tie 0 / 0.5 -> even code 0
    } else if (a < 0.75f) {
      code = 1;
    } else if (a == 0.75f) {
      code = 2;  // tie 0.5 / 1 -> even code 2
    } else if (a < 1.25f) {
      code = 2;
    } else if (a == 1.25f) {
      code = 2;  // tie 1 / 1.5 -> even code 2
    } else if (a < 1.75f) {
      code = 3;
    } else if (a == 1.75f) {
      code = 4;  // tie 1.5 / 2 -> even code 4
    } else if (a < 2.5f) {
      code = 4;
    } else if (a == 2.5f) {
      code = 4;  // tie 2 / 3 -> even code 4
    } else if (a < 3.5f) {
      code = 5;
    } else if (a == 3.5f) {
      code = 6;  // tie 3 / 4 -> even code 6
    } else if (a < 5.0f) {
      code = 6;
    } else if (a == 5.0f) {
      code = 6;  // tie 4 / 6 -> even code 6
    } else {
      code = 7;  // > 5, including saturation past 6
    }
    return sign | code;
  }
};

// E8M0 block scale: a power of two, value 2^(e - 127); 0xFF is NaN.
inline float e8m0_to_float(uint8_t e) {
  uint32_t f;
  if (e == 0) {
    f = 0x00400000;  // 2^-127, an f32 denormal
  } else if (e == 0xFF) {
    f = 0x7FC00000;  // NaN
  } else {
    f = static_cast<uint32_t>(e) << 23;
  }
  return mx_detail::bits_to_float(f);
}

inline uint8_t e8m0_from_float(float x) {
  // Round to the nearest power of two (ties up), clamped to [2^-127, 2^127].
  uint32_t f;
  std::memcpy(&f, &x, 4);
  if ((f & 0x7FFFFFFF) > 0x7F800000) return 0xFF;  // NaN
  const uint32_t fexp = (f >> 23) & 0xFF;
  if (fexp == 0) return 0;  // zero or f32-denormal: clamp to 2^-127
  const int e = static_cast<int>(fexp) - 127;
  const uint32_t mant = f & 0x7FFFFF;
  int r = e + (mant > 0x3504F3 ? 1 : 0);  // 2^0.5 = 1.41421... -> mantissa 0x3504F3
  if (r < -127) r = -127;
  if (r > 127) r = 127;
  return static_cast<uint8_t>(r + 127);
}

// One MX operand: packed elements plus one E8M0 scale per 32 elements along
// k. Both arrays use the operand's matrix layout (A: m x k row-major;
// B: k x n column-major — k must be the contiguous dimension of both).
//
//   A element (i, k): data[(i * ld + k) / elems_per_byte]
//   A scale   (i, kg): scales[i * ld_scales + kg], kg = k / 32
//
// For fp4 (2 elements per byte), ld must be even so every row starts on a
// byte boundary.
template <typename T>  // fp8e4m3_t | fp8e5m2_t | fp4e2m1_t
struct MxTensorRef {
  const uint8_t* data = nullptr;
  const uint8_t* scales = nullptr;
  int ld = 0;         // leading dimension in elements
  int ld_scales = 0;  // leading dimension in scale entries
};

}  // namespace cpu_ops
