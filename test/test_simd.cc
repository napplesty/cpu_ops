#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/numeric_types.h"

namespace {

int g_failures = 0;

template <typename T, int N>
void test_vec(const char* name) {
  std::mt19937 rng(static_cast<unsigned>(N * 7 + sizeof(T)));
  std::uniform_real_distribution<T> dist(T(-1), T(1));

  std::vector<T> x(N), y(N), z(N), out(N, T(0));
  for (int i = 0; i < N; ++i) {
    x[i] = dist(rng);
    y[i] = dist(rng);
    z[i] = dist(rng);
  }

  using V = cpu_ops::simd::Vec<T, N>;

  // load / fmadd / store round trip
  const V r = cpu_ops::simd::fmadd(V::load(x.data()), V::load(y.data()), V::load(z.data()));
  r.store(out.data());
  const T eps = std::numeric_limits<T>::epsilon();
  for (int i = 0; i < N; ++i) {
    const T want = x[i] * y[i] + z[i];
    const T tol = T(4) * eps * (std::fabs(x[i] * y[i]) + std::fabs(z[i])) + T(1e-30);
    if (!(std::fabs(out[i] - want) <= tol)) {
      ++g_failures;
      std::printf("FAIL [%s] fmadd lane %d: got %g want %g\n", name, i, (double)out[i],
                  (double)want);
    }
  }

  // set1
  const V s = V::set1(T(3.25));
  s.store(out.data());
  for (int i = 0; i < N; ++i) {
    if (out[i] != T(3.25)) {
      ++g_failures;
      std::printf("FAIL [%s] set1 lane %d: got %g\n", name, i, (double)out[i]);
    }
  }
}

// 4-way u8 x s8 dot product accumulated per 32-bit lane.
template <int N>
void test_dpbusd(const char* name) {
  std::mt19937 rng(static_cast<unsigned>(N * 31));
  std::vector<uint8_t> a_bytes(4 * N);
  std::vector<int8_t> b_bytes(4 * N);
  std::vector<int32_t> acc(N), out(N, 0);
  for (auto& x : a_bytes) x = static_cast<uint8_t>(rng() % 256);
  for (auto& x : b_bytes) x = static_cast<int8_t>(rng() % 256 - 128);
  for (auto& x : acc) x = static_cast<int32_t>(rng() % 65536 - 32768);

  using V = cpu_ops::simd::Vec<int32_t, N>;
  const V r = cpu_ops::simd::dpbusd(V::load(acc.data()),
                                    V::load(reinterpret_cast<const int32_t*>(a_bytes.data())),
                                    V::load(reinterpret_cast<const int32_t*>(b_bytes.data())));
  r.store(out.data());
  for (int l = 0; l < N; ++l) {
    int32_t want = acc[l];
    for (int t = 0; t < 4; ++t) want += a_bytes[4 * l + t] * b_bytes[4 * l + t];
    if (out[l] != want) {
      ++g_failures;
      std::printf("FAIL [%s] dpbusd lane %d: got %d want %d\n", name, l, out[l], want);
    }
  }
}

// 2-way bf16 dot product accumulated per f32 lane; each lane's 32 bits pack
// two bf16 k-slices (low half = the even k).
template <int N>
void test_dpbf16ps(const char* name) {
  std::mt19937 rng(static_cast<unsigned>(N * 47));
  std::uniform_real_distribution<float> dist(-4.f, 4.f);
  const auto trunc16 = [](float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u >> 16;
  };
  const auto from_bits = [](uint32_t u) {
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };

  std::vector<uint32_t> aw(N), bw(N);
  std::vector<float> acc(N), out(N, 0.f);
  std::vector<float> a0(N), a1(N), b0(N), b1(N);
  for (int l = 0; l < N; ++l) {
    const uint32_t ha0 = trunc16(dist(rng)), ha1 = trunc16(dist(rng));
    const uint32_t hb0 = trunc16(dist(rng)), hb1 = trunc16(dist(rng));
    aw[l] = ha0 | (ha1 << 16);
    bw[l] = hb0 | (hb1 << 16);
    a0[l] = from_bits(ha0 << 16);
    a1[l] = from_bits(ha1 << 16);
    b0[l] = from_bits(hb0 << 16);
    b1[l] = from_bits(hb1 << 16);
    acc[l] = dist(rng);
  }

  using V = cpu_ops::simd::Vec<float, N>;
  const V r = cpu_ops::simd::dpbf16ps(
      V::load(acc.data()), V::load(reinterpret_cast<const float*>(aw.data())),
      V::load(reinterpret_cast<const float*>(bw.data())));
  r.store(out.data());
  for (int l = 0; l < N; ++l) {
    // Scalar reference: two fused steps, (acc + a0*b0) + a1*b1. A hardware
    // vdpbf16ps rounds the whole sum once instead, so allow a few ulp.
    const float want = std::fma(a1[l], b1[l], std::fma(a0[l], b0[l], acc[l]));
    const float tol = 8.f * std::numeric_limits<float>::epsilon() *
                          (std::fabs(a0[l] * b0[l]) + std::fabs(a1[l] * b1[l]) +
                           std::fabs(acc[l])) +
                      1e-30f;
    if (!(std::fabs(out[l] - want) <= tol)) {
      ++g_failures;
      std::printf("FAIL [%s] dpbf16ps lane %d: got %g want %g\n", name, l, out[l],
                  want);
    }
  }
}

// Widening loads: f16/bf16 bit patterns (random, ±0, denormals, inf/nan)
// widened to f32 and compared against the scalar conversions, at the native
// width (ISA specialization) and generic fallback widths.
template <int W>
void check_widen_width(const std::vector<uint16_t>& bits) {
  const auto& f16 = cpu_ops::float16_t::to_float;
  const auto& bf16 = cpu_ops::bfloat16_t::to_float;
  for (std::size_t i = 0; i + W <= bits.size(); i += W) {
    float of[W], ob[W];
    cpu_ops::simd::widen_f16<W>(bits.data() + i).store(of);
    cpu_ops::simd::widen_bf16<W>(bits.data() + i).store(ob);
    for (int l = 0; l < W; ++l) {
      const float want_f = f16(bits[i + l]);
      const float want_b = bf16(bits[i + l]);
      const bool ok_f = (of[l] == want_f) || (std::isnan(of[l]) && std::isnan(want_f));
      const bool ok_b = (ob[l] == want_b) || (std::isnan(ob[l]) && std::isnan(want_b));
      if (!ok_f) {
        ++g_failures;
        std::printf("FAIL [widen_f16<%d>] bits %04x: got %g want %g\n", W, bits[i + l],
                    of[l], want_f);
      }
      if (!ok_b) {
        ++g_failures;
        std::printf("FAIL [widen_bf16<%d>] bits %04x: got %g want %g\n", W,
                    bits[i + l], ob[l], want_b);
      }
    }
  }
}

void test_widen() {
  std::mt19937 rng(1234);
  const uint16_t specials[] = {
      0x0000, 0x8000,          // ±0
      0x0001, 0x8003,          // f16 denormals
      0x0400, 0x7BFF,          // smallest / largest finite f16
      0x3C00, 0xC000, 0x3555,  // 1, -2, ~1/3
      0x7C00, 0xFC00,          // ±inf
      0x7E00, 0xFE00,          // NaNs
  };
  std::vector<uint16_t> bits(specials, specials + sizeof(specials) / sizeof(uint16_t));
  for (int i = 0; i < 4096; ++i) bits.push_back(static_cast<uint16_t>(rng() % 65536));
  bits.resize(bits.size() + 16 - bits.size() % 16);

  check_widen_width<1>(bits);
  check_widen_width<3>(bits);
  check_widen_width<4>(bits);
  check_widen_width<8>(bits);
  check_widen_width<16>(bits);
}

}  // namespace

int main() {
  constexpr int wf = cpu_ops::simd::native_width<float>();
  constexpr int wd = cpu_ops::simd::native_width<double>();
  static_assert(wf >= 1 && wd >= 1, "native width must be positive");
  std::printf("native_width: float=%d double=%d\n", wf, wd);

  test_vec<float, wf>("native float");
  test_vec<double, wd>("native double");
  test_vec<float, 3>("generic float<3>");
  test_vec<double, 5>("generic double<5>");
  test_vec<float, 16>("generic float<16>");

  test_dpbusd<cpu_ops::simd::native_width<int32_t>()>("native int32");
  test_dpbusd<3>("generic int32<3>");

  test_dpbf16ps<cpu_ops::simd::native_width<float>()>("native float");
  test_dpbf16ps<3>("generic float<3>");
#if !defined(CPU_OPS_SIMD_SVE) || (__ARM_FEATURE_SVE_BITS != 512)
  // On SVE-512 Vec<float,16> is the (array-less) SVE specialization, which
  // has no dpbf16ps overload; everywhere else width 16 is generic or covered.
  test_dpbf16ps<16>("float<16>");
#endif

  test_widen();

  if (g_failures == 0) {
    std::printf("test_simd: all passed\n");
    return 0;
  }
  std::printf("test_simd: %d failures\n", g_failures);
  return 1;
}
