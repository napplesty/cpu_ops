#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "cpu_ops/detail/simd.h"
#include "cpu_ops/element_types.h"

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

  test_widen();

  if (g_failures == 0) {
    std::printf("test_simd: all passed\n");
    return 0;
  }
  std::printf("test_simd: %d failures\n", g_failures);
  return 1;
}
