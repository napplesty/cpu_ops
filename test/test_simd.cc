#include <cmath>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <random>
#include <vector>

#include "cpu_ops/detail/simd.h"

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

  if (g_failures == 0) {
    std::printf("test_simd: all passed\n");
    return 0;
  }
  std::printf("test_simd: %d failures\n", g_failures);
  return 1;
}
