// Tests for the ops layer: activation (gelu/silu), normalization (rmsnorm,
// fused add+rmsnorm, layernorm) and exact top-k selection. Activations and
// norms are checked against double references of the same formulas on the
// storage-rounded inputs; topk (f32 and f64) is checked BITWISE against the
// partial_sort formulation of its (value desc, index asc) contract —
// including tie-heavy inputs and the -0/+0 float-equal case. Determinism
// contracts: outputs are bit-identical across thread counts and across runs.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"

namespace {

int g_failures = 0;
int g_cases = 0;

#define CHECK(cond, ...)                                                  \
  do {                                                                    \
    if (!(cond)) {                                                        \
      ++g_failures;                                                       \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                    \
      std::printf(__VA_ARGS__);                                           \
      std::printf("\n");                                                  \
    }                                                                     \
  } while (0)

double ref_gelu(double x) {
  const double inner = 0.7978845608028654 * (x + 0.044715 * x * x * x);
  return 0.5 * x * (1.0 + std::tanh(inner));
}

double ref_silu(double x) { return x / (1.0 + std::exp(-x)); }

template <typename T>
void test_activation(int threads, double tol, const char* tag) {
  ++g_cases;
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);
  const std::size_t n = 100003;  // odd: covers vector tails
  std::vector<float> xf(n);
  for (auto& x : xf) x = dist(rng);
  std::vector<T> x(n), y(n);
  for (std::size_t i = 0; i < n; ++i) x[i] = T(xf[i]);
  // Reference sees the same rounded values.
  for (std::size_t i = 0; i < n; ++i) xf[i] = static_cast<float>(x[i]);

  CHECK(cpu_ops::ops::gelu<T>(x.data(), y.data(), n, threads) ==
            cpu_ops::Status::kSuccess,
        "gelu %s status", tag);
  double max_err = 0;
  for (std::size_t i = 0; i < n; ++i)
    max_err = std::max(max_err,
                       std::fabs(static_cast<float>(y[i]) - ref_gelu(xf[i])));
  CHECK(max_err <= tol, "gelu %s threads=%d max_err %g > %g", tag, threads, max_err,
        tol);

  CHECK(cpu_ops::ops::silu<T>(x.data(), y.data(), n, threads) ==
            cpu_ops::Status::kSuccess,
        "silu %s status", tag);
  max_err = 0;
  for (std::size_t i = 0; i < n; ++i)
    max_err = std::max(max_err,
                       std::fabs(static_cast<float>(y[i]) - ref_silu(xf[i])));
  CHECK(max_err <= tol, "silu %s threads=%d max_err %g > %g", tag, threads, max_err,
        tol);
}

template <typename T>
void test_norm(int threads, double tol, const char* tag) {
  ++g_cases;
  std::mt19937 rng(321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int rows = 33, cols = 517;  // odd sizes: tails + long rows
  const float eps = 1e-6f;
  std::vector<float> xf((std::size_t)rows * cols), rf((std::size_t)rows * cols),
      wf(cols);
  for (auto& x : xf) x = dist(rng);
  for (auto& x : rf) x = dist(rng);
  for (auto& x : wf) x = dist(rng) * 0.5f + 0.75f;
  std::vector<T> x((std::size_t)rows * cols), r((std::size_t)rows * cols), w(cols),
      y((std::size_t)rows * cols), y2((std::size_t)rows * cols),
      ro((std::size_t)rows * cols);
  for (std::size_t i = 0; i < x.size(); ++i) x[i] = T(xf[i]);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] = T(rf[i]);
  for (int i = 0; i < cols; ++i) w[i] = T(wf[i]);
  for (std::size_t i = 0; i < xf.size(); ++i) xf[i] = static_cast<float>(x[i]);
  for (std::size_t i = 0; i < rf.size(); ++i) rf[i] = static_cast<float>(r[i]);
  for (int i = 0; i < cols; ++i) wf[i] = static_cast<float>(w[i]);

  // rmsnorm
  CHECK(cpu_ops::ops::rmsnorm<T>(x.data(), w.data(), y.data(), rows, cols, eps,
                                 threads) == cpu_ops::Status::kSuccess,
        "rmsnorm %s status", tag);
  double max_err = 0;
  for (int i = 0; i < rows; ++i) {
    double ss = 0;
    for (int j = 0; j < cols; ++j) {
      const double v = xf[(std::size_t)i * cols + j];
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / cols + eps);
    for (int j = 0; j < cols; ++j) {
      const double want = xf[(std::size_t)i * cols + j] * inv * wf[j];
      max_err = std::max(max_err,
                         std::fabs(static_cast<float>(y[(std::size_t)i * cols + j]) - want));
    }
  }
  CHECK(max_err <= tol, "rmsnorm %s threads=%d max_err %g > %g", tag, threads,
        max_err, tol);

  // fused add + rmsnorm: y and residual sum, one pass over the inputs
  CHECK(cpu_ops::ops::fused_add_rmsnorm<T>(x.data(), r.data(), w.data(), y2.data(),
                                           ro.data(), rows, cols, eps,
                                           threads) == cpu_ops::Status::kSuccess,
        "fused_add_rmsnorm %s status", tag);
  max_err = 0;
  double max_err_r = 0;
  for (int i = 0; i < rows; ++i) {
    std::vector<double> sum(cols);
    double ss = 0;
    for (int j = 0; j < cols; ++j) {
      sum[j] = xf[(std::size_t)i * cols + j] + rf[(std::size_t)i * cols + j];
      ss += sum[j] * sum[j];
    }
    const double inv = 1.0 / std::sqrt(ss / cols + eps);
    for (int j = 0; j < cols; ++j) {
      const double want = sum[j] * inv * wf[j];
      max_err = std::max(max_err,
                         std::fabs(static_cast<float>(y2[(std::size_t)i * cols + j]) - want));
      max_err_r = std::max(max_err_r,
                           std::fabs(static_cast<float>(ro[(std::size_t)i * cols + j]) - sum[j]));
    }
  }
  CHECK(max_err <= tol, "fused_add_rmsnorm %s threads=%d y max_err %g > %g", tag,
        threads, max_err, tol);
  CHECK(max_err_r <= tol, "fused_add_rmsnorm %s threads=%d res max_err %g > %g", tag,
        threads, max_err_r, tol);

  // layernorm
  CHECK(cpu_ops::ops::layernorm<T>(x.data(), w.data(), y.data(), rows, cols, eps,
                                   threads) == cpu_ops::Status::kSuccess,
        "layernorm %s status", tag);
  max_err = 0;
  for (int i = 0; i < rows; ++i) {
    double s = 0, ss = 0;
    for (int j = 0; j < cols; ++j) {
      const double v = xf[(std::size_t)i * cols + j];
      s += v;
      ss += v * v;
    }
    const double mu = s / cols;
    const double var = std::max(0.0, ss / cols - mu * mu);
    const double inv = 1.0 / std::sqrt(var + eps);
    for (int j = 0; j < cols; ++j) {
      const double want = (xf[(std::size_t)i * cols + j] - mu) * inv * wf[j];
      max_err = std::max(max_err,
                         std::fabs(static_cast<float>(y[(std::size_t)i * cols + j]) - want));
    }
  }
  CHECK(max_err <= tol, "layernorm %s threads=%d max_err %g > %g", tag, threads,
        max_err, tol);
}

// Reference formulation of the topk contract; bitwise comparable.
template <typename T>
void ref_topk(const std::vector<T>& v, int k, std::vector<int32_t>* out) {
  const int n = static_cast<int>(v.size());
  if (k > n) k = n;
  out->resize(k);
  std::vector<int32_t> id(n);
  for (int i = 0; i < n; ++i) id[i] = i;
  const auto by_value = [&v](int32_t a, int32_t b) {
    if (v[a] != v[b]) return v[a] > v[b];
    return a < b;
  };
  std::partial_sort(id.begin(), id.begin() + k, id.end(), by_value);
  std::memcpy(out->data(), id.data(), sizeof(int32_t) * k);
}

template <typename T>
void test_topk_case(const char* name, const std::vector<T>& v, int k) {
  ++g_cases;
  if (k > (int)v.size()) k = (int)v.size();  // same clamp as the operator
  std::vector<int32_t> got(k), want;
  const cpu_ops::Status st = cpu_ops::ops::topk_indices(
      v.data(), static_cast<int>(v.size()), k, got.data());
  CHECK(st == cpu_ops::Status::kSuccess, "%s status %d", name, (int)st);
  ref_topk(v, k, &want);
  CHECK(got.size() == want.size() &&
            std::memcmp(got.data(), want.data(), sizeof(int32_t) * want.size()) == 0,
        "%s: selection differs from the partial_sort contract (k=%d)", name, k);
}

void test_topk() {
  std::mt19937 rng(777);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  {  // radix path, generic random
    std::vector<float> v(100000);
    for (auto& x : v) x = dist(rng);
    test_topk_case("topk_random", v, 2048);
    test_topk_case("topk_k1", v, 1);
    test_topk_case("topk_kn", v, 100000);
    test_topk_case("topk_k_gt_n", v, 200000);
  }
  {  // tie-heavy: quantized values force exact float ties
    std::vector<float> v(100000);
    for (auto& x : v) x = std::floor(dist(rng) * 8.0f) / 8.0f;
    test_topk_case("topk_ties", v, 3333);
    test_topk_case("topk_ties_k1", v, 1);
  }
  {  // -0 / +0 are float-equal: tie must resolve by index
    std::vector<float> v(5000);
    for (int i = 0; i < 10; ++i) v[i] = 3.0f + i;
    for (int i = 10; i < 5000; ++i)
      v[i] = (i % 2) ? -0.0f : +0.0f;  // alternating zero signs
    test_topk_case("topk_zeros", v, 500);
    test_topk_case("topk_zeros_boundary", v, 11);
  }
  {  // small-n path
    std::vector<float> v(100);
    for (auto& x : v) x = dist(rng);
    test_topk_case("topk_small", v, 17);
  }
  {  // error paths
    ++g_cases;
    std::vector<float> v(16, 0.5f);
    std::vector<int32_t> out(4);
    CHECK(cpu_ops::ops::topk_indices(static_cast<const float*>(nullptr), 16, 4,
                                     out.data()) ==
              cpu_ops::Status::kErrorInvalidArguments,
          "topk null accepted");
    CHECK(cpu_ops::ops::topk_indices(v.data(), 16, 0, out.data()) ==
              cpu_ops::Status::kErrorInvalidProblem,
          "topk k=0 accepted");
    v[3] = std::nanf("");
    CHECK(cpu_ops::ops::topk_indices(v.data(), 16, 4, out.data()) ==
              cpu_ops::Status::kErrorInvalidArguments,
          "topk NaN accepted");
  }
}

void test_topk_f64() {
  std::mt19937 rng(888);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);

  {  // radix path, generic random
    std::vector<double> v(100000);
    for (auto& x : v) x = dist(rng);
    test_topk_case("topk_f64_random", v, 2048);
    test_topk_case("topk_f64_k1", v, 1);
    test_topk_case("topk_f64_kn", v, 100000);
    test_topk_case("topk_f64_k_gt_n", v, 200000);
  }
  {  // tie-heavy: quantized values force exact double ties
    std::vector<double> v(100000);
    for (auto& x : v) x = std::floor(dist(rng) * 8.0) / 8.0;
    test_topk_case("topk_f64_ties", v, 3333);
    test_topk_case("topk_f64_ties_k1", v, 1);
  }
  {  // -0 / +0 are float-equal: tie must resolve by index
    std::vector<double> v(5000);
    for (int i = 0; i < 10; ++i) v[i] = 3.0 + i;
    for (int i = 10; i < 5000; ++i)
      v[i] = (i % 2) ? -0.0 : +0.0;  // alternating zero signs
    test_topk_case("topk_f64_zeros", v, 500);
    test_topk_case("topk_f64_zeros_boundary", v, 11);
  }
  {  // small-n path
    std::vector<double> v(100);
    for (auto& x : v) x = dist(rng);
    test_topk_case("topk_f64_small", v, 17);
  }
  {  // error paths
    ++g_cases;
    std::vector<double> v(16, 0.5);
    std::vector<int32_t> out(4);
    CHECK(cpu_ops::ops::topk_indices(static_cast<const double*>(nullptr), 16, 4,
                                     out.data()) ==
              cpu_ops::Status::kErrorInvalidArguments,
          "topk_f64 null accepted");
    CHECK(cpu_ops::ops::topk_indices(v.data(), 16, 0, out.data()) ==
              cpu_ops::Status::kErrorInvalidProblem,
          "topk_f64 k=0 accepted");
    v[3] = std::nan("");
    CHECK(cpu_ops::ops::topk_indices(v.data(), 16, 4, out.data()) ==
              cpu_ops::Status::kErrorInvalidArguments,
          "topk_f64 NaN accepted");
  }
}

template <typename F>
double time_best(F&& f, int reps) {
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    f();
    const auto t1 = std::chrono::steady_clock::now();
    best = std::min(best, std::chrono::duration<double>(t1 - t0).count());
  }
  return best;
}

void run_benchmark() {
  using bf16_t = cpu_ops::bfloat16_t;
  std::printf("\nops benchmark (4T unless noted, best of 5):\n");

  {  // silu vs naive scalar loop
    const std::size_t n = 64u << 20;
    std::vector<float> x(n), y(n), y2(n);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
    for (std::size_t i = 0; i < n; i += 4096) x[i] = dist(rng);
    const double tf = time_best([&] { cpu_ops::ops::silu<float>(x.data(), y.data(), n, 4); }, 5);
    const double tn = time_best(
        [&] {
          float acc = 0;
          for (std::size_t i = 0; i < n; ++i) {
            y2[i] = x[i] / (1.0f + std::expf(-x[i]));
            acc += y2[i];
          }
          if (acc == 1.2345e30f) std::printf("x");
        }, 3);
    std::printf("silu f32     64M: %7.1f ms (%5.1f GB/s) | naive %7.1f ms (%5.1f GB/s) | %4.1fx\n",
                tf * 1e3, 2.0 * n * 4 / tf / 1e9, tn * 1e3, 2.0 * n * 4 / tn / 1e9, tn / tf);
    std::vector<bf16_t> xb(n), yb(n);
    for (std::size_t i = 0; i < n; i += 4096) xb[i] = bf16_t(dist(rng));
    const double tb = time_best([&] { cpu_ops::ops::silu<bf16_t>(xb.data(), yb.data(), n, 4); }, 5);
    std::printf("silu bf16    64M: %7.1f ms (%5.1f GB/s)\n", tb * 1e3,
                2.0 * n * 2 / tb / 1e9);
  }
  {  // rmsnorm / fused vs naive
    const int rows = 8192, cols = 7168;
    std::vector<float> x((std::size_t)rows * cols), r((std::size_t)rows * cols),
        w(cols), y((std::size_t)rows * cols), ro((std::size_t)rows * cols);
    std::mt19937 rng(6);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : x) v = dist(rng);
    for (auto& v : r) v = dist(rng);
    for (auto& v : w) v = dist(rng) * 0.5f + 0.75f;
    const double bytes = 3.0 * rows * cols * 4;  // in x (+res), out y (+res_out)/2ish
    const double tf = time_best([&] { cpu_ops::ops::rmsnorm<float>(x.data(), w.data(), y.data(), rows, cols, 1e-6f, 4); }, 5);
    const double tfa = time_best([&] { cpu_ops::ops::fused_add_rmsnorm<float>(x.data(), r.data(), w.data(), y.data(), ro.data(), rows, cols, 1e-6f, 4); }, 5);
    const double tn = time_best(
        [&] {
          for (int i = 0; i < rows; ++i) {
            const float* xr = x.data() + (std::size_t)i * cols;
            float ss = 0;
            for (int j = 0; j < cols; ++j) ss += xr[j] * xr[j];
            const float inv = 1.0f / std::sqrtf(ss / cols + 1e-6f);
            for (int j = 0; j < cols; ++j)
              y[(std::size_t)i * cols + j] = xr[j] * inv * w[j];
          }
        }, 3);
    std::printf("rmsnorm 8192x7168 f32: %7.2f ms | fused_add %7.2f ms | naive %7.2f ms | %4.1fx\n",
                tf * 1e3, tfa * 1e3, tn * 1e3, tn / tf);
    (void)bytes;
  }
  {  // topk vs partial_sort (DSA decode shape)
    const int n = 65536;
    std::vector<float> v(n);
    std::mt19937 rng(9);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : v) x = dist(rng);
    std::vector<int32_t> out(2048);
    const double tt = time_best(
        [&] { cpu_ops::ops::topk_indices(v.data(), n, 2048, out.data()); }, 20);
    std::vector<int32_t> id(n);
    const double tp = time_best(
        [&] {
          for (int i = 0; i < n; ++i) id[i] = i;
          const auto by = [&v](int32_t a, int32_t b) {
            if (v[a] != v[b]) return v[a] > v[b];
            return a < b;
          };
          std::partial_sort(id.begin(), id.begin() + 2048, id.end(), by);
        }, 20);
    std::printf("topk n=64k k=2048: %7.1f us | partial_sort %7.1f us | %4.1fx\n",
                tt * 1e6, tp * 1e6, tp / tt);
  }
}

}  // namespace

int main(int argc, char** argv) {
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;
  using f16_t = float16_t;
  using bf16_t = bfloat16_t;

  for (int threads : {1, 4}) {
    test_activation<float>(threads, 2e-3, "f32");
    test_activation<f16_t>(threads, 8e-3, "f16");
    test_activation<bf16_t>(threads, 4e-2, "bf16");
    test_norm<float>(threads, 1e-3, "f32");
    test_norm<f16_t>(threads, 8e-3, "f16");
    test_norm<bf16_t>(threads, 3e-2, "bf16");
  }

  test_topk();
  test_topk_f64();

  {  // determinism: bit-identical across thread counts and runs
    ++g_cases;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const std::size_t n = 65537;
    std::vector<float> x(n);
    for (auto& v : x) v = dist(rng);
    std::vector<f16_t> xh(n), y1(n), y4(n), y4b(n);
    for (std::size_t i = 0; i < n; ++i) xh[i] = f16_t(x[i]);
    cpu_ops::ops::silu<f16_t>(xh.data(), y1.data(), n, 1);
    cpu_ops::ops::silu<f16_t>(xh.data(), y4.data(), n, 4);
    cpu_ops::ops::silu<f16_t>(xh.data(), y4b.data(), n, 4);
    CHECK(std::memcmp(y1.data(), y4.data(), n * sizeof(f16_t)) == 0,
          "silu 1T vs 4T differ");
    CHECK(std::memcmp(y4.data(), y4b.data(), n * sizeof(f16_t)) == 0,
          "silu not deterministic across runs");

    const int rows = 9, cols = 1000;
    std::vector<float> w(cols);
    for (auto& v : w) v = dist(rng) + 1.5f;
    std::vector<bf16_t> xb((std::size_t)rows * cols), rb((std::size_t)rows * cols),
        wb(cols), yb1((std::size_t)rows * cols), yb4((std::size_t)rows * cols),
        ob1((std::size_t)rows * cols), ob4((std::size_t)rows * cols);
    for (auto& v : xb) v = bf16_t(dist(rng));
    for (auto& v : rb) v = bf16_t(dist(rng));
    for (auto& v : wb) v = bf16_t(1.0f);
    cpu_ops::ops::fused_add_rmsnorm<bf16_t>(xb.data(), rb.data(), wb.data(),
                                            yb1.data(), ob1.data(), rows, cols, 1e-6f, 1);
    cpu_ops::ops::fused_add_rmsnorm<bf16_t>(xb.data(), rb.data(), wb.data(),
                                            yb4.data(), ob4.data(), rows, cols, 1e-6f, 4);
    CHECK(std::memcmp(yb1.data(), yb4.data(), yb1.size() * sizeof(bf16_t)) == 0,
          "fused_add_rmsnorm 1T vs 4T differ");
    CHECK(std::memcmp(ob1.data(), ob4.data(), ob1.size() * sizeof(bf16_t)) == 0,
          "fused_add residual 1T vs 4T differ");
  }

  const bool want_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
  if (want_bench) run_benchmark();
  std::printf("ops: %d cases, %d failures%s\n", g_cases, g_failures,
              want_bench ? "" : " (run with --bench for the benchmark)");
  return g_failures == 0 ? 0 : 1;
}
