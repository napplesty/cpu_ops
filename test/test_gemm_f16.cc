// Tests for the half-precision storage types (float16_t, bfloat16_t) and the
// widening GEMM policies (GemmF16F32 / GemmBF16F32): conversions are checked
// against exact expectations, GEMM results against a double-precision
// reference computed over the widened operand values.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"

namespace {

int g_failures = 0;
int g_cases = 0;

#define CHECK(cond, ...)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      ++g_failures;                                            \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);         \
      std::printf(__VA_ARGS__);                                \
      std::printf("\n");                                       \
    }                                                          \
  } while (0)

using cpu_ops::bfloat16_t;
using cpu_ops::float16_t;

void check_f16_conversions() {
  ++g_cases;
  // Exactly representable values round-trip.
  const float exact[] = {0.0f,  -0.0f, 1.0f,   -2.0f,  0.5f,    0.33325195f,
                         65504.f, -65504.f, 6.103515625e-05f,  5.9604644775390625e-08f};
  for (float f : exact) {
    const float rt = float16_t::to_float(float16_t::from_float(f));
    CHECK(std::memcmp(&rt, &f, 4) == 0, "f16 round-trip of %g gave %g", f, rt);
  }
  // Infinity.
  const float inf = std::numeric_limits<float>::infinity();
  CHECK(float16_t::to_float(float16_t::from_float(inf)) == inf, "f16 +inf lost");
  CHECK(float16_t::to_float(float16_t::from_float(-inf)) == -inf, "f16 -inf lost");
  // Overflow rounds to infinity.
  CHECK(float16_t::from_float(70000.f) == uint16_t(0x7C00), "f16 overflow not inf");
  CHECK(float16_t::from_float(-70000.f) == uint16_t(0xFC00), "f16 -overflow not -inf");
  // NaN stays NaN.
  CHECK(std::isnan(float16_t::to_float(float16_t::from_float(
            std::numeric_limits<float>::quiet_NaN()))),
        "f16 nan lost");
  // Underflow rounds to zero.
  CHECK(float16_t::from_float(1e-10f) == uint16_t(0), "f16 underflow not zero");
  // Round-to-nearest-even at a halfway point: 1 + 2^-11 is exactly between
  // 1.0 and 1+2^-10; ties-to-even rounds down to 1.0.
  CHECK(float16_t::from_float(1.0f + 0x1p-11f) == uint16_t(0x3C00),
        "f16 RNE tie at 1+2^-11 mishandled");
  // 1 + 3*2^-11 = (1 + 2^-10) + 2^-11 sits exactly on the tie between
  // 1+2^-10 and 1+2^-9+... i.e. between mantissas 1 and 2; ties-to-even
  // rounds up to 2.
  CHECK(float16_t::from_float(1.0f + 3.0f * 0x1p-11f) == uint16_t(0x3C02),
        "f16 RNE tie at odd mantissa mishandled");
  // Just above the first halfway point rounds up.
  CHECK(float16_t::from_float(1.0f + 0x1p-11f + 0x1p-20f) == uint16_t(0x3C01),
        "f16 RNE above tie mishandled");
}

void check_bf16_conversions() {
  ++g_cases;
  // Every bf16 value is exactly representable in f32; f32 values with zero
  // low 16 bits round-trip exactly.
  const float exact[] = {0.0f,      -0.0f,     1.0f,  -2.0f,
                         0.5f,      3.0f,      -0x1.7p+127f, 0x1p-126f};
  for (float f : exact) {
    const float rt = bfloat16_t::to_float(bfloat16_t::from_float(f));
    CHECK(std::memcmp(&rt, &f, 4) == 0, "bf16 round-trip of %g gave %g", f, rt);
  }
  const float inf = std::numeric_limits<float>::infinity();
  CHECK(bfloat16_t::to_float(bfloat16_t::from_float(inf)) == inf, "bf16 +inf lost");
  CHECK(std::isnan(bfloat16_t::to_float(
            bfloat16_t::from_float(std::numeric_limits<float>::quiet_NaN()))),
        "bf16 nan lost");
  // RNE tie: 1 + 2^-8 is halfway between 1.0 and 1+2^-7; ties-to-even -> 1.0.
  CHECK(bfloat16_t::from_float(1.0f + 0x1p-8f) == uint16_t(0x3F80),
        "bf16 RNE tie mishandled");
  // 1 + 3*2^-8 sits exactly on the tie between mantissas 1 and 2; even -> 2.
  CHECK(bfloat16_t::from_float(1.0f + 3.0f * 0x1p-8f) == uint16_t(0x3F82),
        "bf16 RNE tie at odd mantissa mishandled");
  // Just above the halfway point rounds up.
  CHECK(bfloat16_t::from_float(1.0f + 0x1p-8f + 0x1p-20f) == uint16_t(0x3F81),
        "bf16 RNE above tie mishandled");
  // No overflow-to-inf issue: bf16 has the f32 exponent range.
  CHECK(bfloat16_t::to_float(bfloat16_t::from_float(3.0e38f)) > 2.9e38f,
        "bf16 large value mangled");
}

template <typename StorageT>
struct Lab;
template <>
struct Lab<float16_t> {
  static const char* name() { return "f16"; }
};
template <>
struct Lab<bfloat16_t> {
  static const char* name() { return "bf16"; }
};

template <typename StorageT, typename LA, typename LB, typename LC,
          typename Config = cpu_ops::gemm::device::GemmConfig<float>,
          typename Policy = cpu_ops::mma::WidenPolicy<StorageT>>
void run_case(int m, int n, int k, float alpha, float beta, int num_threads) {
  ++g_cases;
  const int lda = (LA::kIsRowMajor ? k : m) + 3;
  const int ldb = (LB::kIsRowMajor ? n : k) + 3;
  const int ldc = (LC::kIsRowMajor ? n : m) + 3;
  const size_t size_a = static_cast<size_t>(LA::kIsRowMajor ? m : k) * lda;
  const size_t size_b = static_cast<size_t>(LB::kIsRowMajor ? k : n) * ldb;
  const size_t size_c = static_cast<size_t>(LC::kIsRowMajor ? m : n) * ldc;

  // Operands are generated in f32 and quantized to the storage type; the
  // reference consumes the quantized (exactly representable) values.
  std::vector<float> A32(size_a), B32(size_b);
  std::vector<StorageT> A(size_a), B(size_b);
  std::vector<float> C(size_c, 0.f), D(size_c, std::numeric_limits<float>::quiet_NaN());

  std::mt19937 rng(static_cast<unsigned>(m * 131 + n * 17 + k));
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (size_t i = 0; i < size_a; ++i) {
    A[i] = StorageT(dist(rng));
    A32[i] = float(A[i]);
  }
  for (size_t i = 0; i < size_b; ++i) {
    B[i] = StorageT(dist(rng));
    B32[i] = float(B[i]);
  }
  for (size_t i = 0; i < size_c; ++i) C[i] = dist(rng);

  std::vector<double> D_ref(static_cast<size_t>(m) * n, 0.0);
  std::vector<double> bound(static_cast<size_t>(m) * n, 0.0);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0, bnd = 0.0;
      for (int p = 0; p < k; ++p) {
        const double a = A32[LA::offset(i, p, lda)];
        const double b = B32[LB::offset(p, j, ldb)];
        acc += a * b;
        bnd += std::fabs(a * b);
      }
      double v = static_cast<double>(alpha) * acc;
      if (beta != 0.f) v += static_cast<double>(beta) * C[LC::offset(i, j, ldc)];
      D_ref[static_cast<size_t>(i) * n + j] = v;
      bound[static_cast<size_t>(i) * n + j] = bnd;
    }
  }

  using Gemm = cpu_ops::gemm::device::Gemm<
      StorageT, LA, StorageT, LB, float, LC, float,
      cpu_ops::epilogue::LinearCombination<float>, Config, Policy>;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), lda},
                                {B.data(), ldb},
                                {C.data(), ldc},
                                {D.data(), ldc},
                                {alpha, beta}};
  Gemm op;
  CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess,
        "%s (%dx%dx%d) threads=%d: bad status", Lab<StorageT>::name(), m, n, k,
        num_threads);

  // The accumulation is f32; bound the error like the f32 suite.
  const double eps = std::numeric_limits<float>::epsilon();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const size_t off = static_cast<size_t>(LC::offset(i, j, ldc));
      const double want = D_ref[static_cast<size_t>(i) * n + j];
      const double allowed = 8.0 * k * eps * bound[static_cast<size_t>(i) * n + j] +
                             16.0 * eps * std::fabs(want) + 1e-12;
      const double err = std::fabs(static_cast<double>(D[off]) - want);
      CHECK(err <= allowed, "%s (%dx%dx%d) a=%g b=%g threads=%d C(%d,%d): got %g want %g",
            Lab<StorageT>::name(), m, n, k, (double)alpha, (double)beta, num_threads, i,
            j, static_cast<double>(D[off]), want);
    }
  }
}

template <typename StorageT>
void run_all_layouts(int m, int n, int k, float alpha, float beta, int num_threads) {
  using cpu_ops::layout::ColumnMajor;
  using cpu_ops::layout::RowMajor;
  run_case<StorageT, RowMajor, RowMajor, RowMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, RowMajor, RowMajor, ColumnMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, RowMajor, ColumnMajor, RowMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, RowMajor, ColumnMajor, ColumnMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, ColumnMajor, RowMajor, RowMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, ColumnMajor, RowMajor, ColumnMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, ColumnMajor, ColumnMajor, RowMajor>(m, n, k, alpha, beta, num_threads);
  run_case<StorageT, ColumnMajor, ColumnMajor, ColumnMajor>(m, n, k, alpha, beta, num_threads);
}

// Forced split-k through the widening path (slice counts not dividing k).
template <typename StorageT,
          typename Config = cpu_ops::gemm::device::GemmConfig<float>,
          typename Policy = cpu_ops::mma::WidenPolicy<StorageT>>
void check_splitk(int num_threads) {
  using cpu_ops::layout::RowMajor;
  const int m = 24, n = 40, k = 3000;
  const int lda = k + 1, ldb = n + 1, ldc = n + 1;
  std::vector<StorageT> A(static_cast<size_t>(m) * lda), B(static_cast<size_t>(k) * ldb);
  std::vector<float> C(static_cast<size_t>(m) * ldc, 0.f),
      D(static_cast<size_t>(m) * ldc);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  for (auto& x : A) x = StorageT(dist(rng));
  for (auto& x : B) x = StorageT(dist(rng));
  for (auto& x : C) x = dist(rng);

  using Gemm = cpu_ops::gemm::device::Gemm<
      StorageT, RowMajor, StorageT, RowMajor, float, RowMajor, float,
      cpu_ops::epilogue::LinearCombination<float>, Config, Policy>;
  Gemm op;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), lda},
                                {B.data(), ldb},
                                {C.data(), ldc},
                                {D.data(), ldc},
                                {1.25f, 0.5f},
                                5};
  CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess,
        "%s splitk: bad status", Lab<StorageT>::name());
  // Spot-check a few entries against a serial reference.
  for (int i = 0; i < m; i += 7) {
    for (int j = 0; j < n; j += 9) {
      double acc = 0.0;
      for (int p = 0; p < k; ++p)
        acc += static_cast<double>(float(A[static_cast<size_t>(i) * lda + p])) *
               static_cast<double>(float(B[static_cast<size_t>(p) * ldb + j]));
      const double want = 1.25 * acc + 0.5 * C[static_cast<size_t>(i) * ldc + j];
      const double err = std::fabs(static_cast<double>(D[static_cast<size_t>(i) * ldc + j]) - want);
      CHECK(err <= 1e-3 * std::fabs(want) + 1e-3,
            "%s splitk C(%d,%d): got %g want %g", Lab<StorageT>::name(), i, j,
            static_cast<double>(D[static_cast<size_t>(i) * ldc + j]), want);
    }
  }
  ++g_cases;
}

}  // namespace

int main() {
  // Block scope keeps the names out of global lookup, where ARM's arm_bf16.h
  // typedefs a conflicting ::bfloat16_t.
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;

  check_f16_conversions();
  check_bf16_conversions();

  struct ShapeCfg {
    int m, n, k;
  };
  const ShapeCfg shapes[] = {{1, 1, 1}, {5, 7, 3},   {16, 6, 33}, {64, 64, 64},
                             {65, 63, 67}, {31, 17, 257}, {128, 256, 96}};

  for (const ShapeCfg& s : shapes) {
    for (int threads : {1, 8}) {
      run_all_layouts<float16_t>(s.m, s.n, s.k, 1.0f, 0.0f, threads);
      run_all_layouts<float16_t>(s.m, s.n, s.k, 2.0f, -0.5f, threads);
      run_all_layouts<bfloat16_t>(s.m, s.n, s.k, 1.0f, 0.0f, threads);
      run_all_layouts<bfloat16_t>(s.m, s.n, s.k, 2.0f, -0.5f, threads);
    }
  }

  // Automatic split-k (deep k) plus a forced count.
  for (int threads : {1, 8}) {
    run_all_layouts<float16_t>(32, 32, 4096, 1.0f, 0.5f, threads);
    run_all_layouts<bfloat16_t>(32, 32, 4096, 1.0f, 0.5f, threads);
  }
  check_splitk<float16_t>(8);
  check_splitk<bfloat16_t>(8);

  // The bf16 dot-product policy (GemmBF16F32's default on AVX512-BF16
  // targets), exercised explicitly so it is covered on every host through
  // its portable two-FMA dpbf16ps fallback.
  {
    using cpu_ops::layout::ColumnMajor;
    using cpu_ops::layout::RowMajor;
    using cpu_ops::mma::Bf16GemmConfig;
    using cpu_ops::mma::Bf16Policy;
    for (const ShapeCfg& s : shapes) {
      for (int threads : {1, 8}) {
        run_case<bfloat16_t, RowMajor, RowMajor, RowMajor, Bf16GemmConfig, Bf16Policy>(
            s.m, s.n, s.k, 1.0f, 0.0f, threads);
        run_case<bfloat16_t, ColumnMajor, ColumnMajor, ColumnMajor, Bf16GemmConfig,
                 Bf16Policy>(s.m, s.n, s.k, 2.0f, -0.5f, threads);
      }
    }
    check_splitk<bfloat16_t, Bf16GemmConfig, Bf16Policy>(8);
  }

  std::printf("test_gemm_f16: %d cases, %d failures\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
