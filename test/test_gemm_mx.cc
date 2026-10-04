// Tests for the MX formats (fp8 e4m3/e5m2, fp4 e2m1, E8M0 scales) and
// GemmMx: decode tables are checked against exact values, GEMM results
// against a double-precision reference over the decoded (exact) operands.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <type_traits>
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
using cpu_ops::e8m0_from_float;
using cpu_ops::e8m0_to_float;
using cpu_ops::fp4e2m1_t;
using cpu_ops::fp8e4m3_t;
using cpu_ops::fp8e5m2_t;

void check_decode_tables() {
  ++g_cases;
  // e4m3
  CHECK(fp8e4m3_t::to_float(0x00) == 0.0f, "e4m3 zero");
  CHECK(fp8e4m3_t::to_float(0x38) == 1.0f, "e4m3 one");
  CHECK(fp8e4m3_t::to_float(0x7E) == 448.0f, "e4m3 max");
  CHECK(std::isnan(fp8e4m3_t::to_float(0x7F)), "e4m3 nan");
  CHECK(fp8e4m3_t::to_float(0x01) == 0x1p-9f, "e4m3 min denormal");
  CHECK(fp8e4m3_t::to_float(0x80) == -0.0f, "e4m3 -0");
  CHECK(fp8e4m3_t::to_float(0xB8) == -1.0f, "e4m3 -1");
  // e5m2
  CHECK(fp8e5m2_t::to_float(0x3C) == 1.0f, "e5m2 one");
  CHECK(fp8e5m2_t::to_float(0x7B) == 57344.0f, "e5m2 max");
  CHECK(fp8e5m2_t::to_float(0x7C) == std::numeric_limits<float>::infinity(),
        "e5m2 inf");
  CHECK(std::isnan(fp8e5m2_t::to_float(0x7D)), "e5m2 nan");
  CHECK(fp8e5m2_t::to_float(0x01) == 0x1p-16f, "e5m2 min denormal");
  // e2m1
  CHECK(fp4e2m1_t::to_float(0x0) == 0.0f, "e2m1 zero");
  CHECK(fp4e2m1_t::to_float(0x1) == 0.5f, "e2m1 denormal");
  CHECK(fp4e2m1_t::to_float(0x2) == 1.0f, "e2m1 one");
  CHECK(fp4e2m1_t::to_float(0x7) == 6.0f, "e2m1 max");
  CHECK(fp4e2m1_t::to_float(0xF) == -6.0f, "e2m1 -max");
  // E8M0
  CHECK(e8m0_to_float(127) == 1.0f, "e8m0 one");
  CHECK(e8m0_to_float(128) == 2.0f, "e8m0 two");
  CHECK(e8m0_to_float(0) == 0x1p-127f, "e8m0 min");
  CHECK(e8m0_to_float(254) == 0x1p127f, "e8m0 max");
  CHECK(std::isnan(e8m0_to_float(0xFF)), "e8m0 nan");
}

template <typename T>
void check_encode_roundtrip(const char* name) {
  ++g_cases;
  // Every non-NaN non-negative encoding decodes to a value that re-encodes
  // to itself. Half the code space is non-negative.
  const int limit = static_cast<int>(T::kLut.size()) / 2;
  for (int v = 0; v < limit; ++v) {
    const float f = T::to_float(static_cast<uint8_t>(v));
    if (std::isnan(f) || std::isinf(f)) continue;
    const uint8_t back = T::from_float(f);
    CHECK(back == static_cast<uint8_t>(v),
          "%s round-trip: 0x%02x -> %g -> 0x%02x", name, v, f, back);
  }
}

void check_e8m0_roundtrip() {
  ++g_cases;
  for (int e = -127; e <= 127; ++e) {
    const float p = std::ldexp(1.0f, e);
    CHECK(e8m0_from_float(p) == static_cast<uint8_t>(e + 127),
          "e8m0 round-trip at 2^%d", e);
  }
}

template <typename T>
struct MxLab;
template <>
struct MxLab<fp8e4m3_t> {
  static const char* name() { return "e4m3"; }
};
template <>
struct MxLab<fp8e5m2_t> {
  static const char* name() { return "e5m2"; }
};
template <>
struct MxLab<fp4e2m1_t> {
  static const char* name() { return "e2m1"; }
};

// Element decode from raw storage, mirroring the layout contract.
template <typename T>
float decode_elem(const uint8_t* base, int row, int k, int ld) {
  if constexpr (std::is_same<T, fp4e2m1_t>::value) {
    const uint8_t byte = base[(static_cast<size_t>(row) * ld + k) / 2];
    return fp4e2m1_t::to_float((k & 1) ? (byte >> 4) : (byte & 0xF));
  } else {
    return T::to_float(base[static_cast<size_t>(row) * ld + k]);
  }
}

template <typename T>
void store_elem(uint8_t* base, int row, int k, int ld, uint8_t code) {
  if constexpr (std::is_same<T, fp4e2m1_t>::value) {
    uint8_t& byte = base[(static_cast<size_t>(row) * ld + k) / 2];
    if (k & 1) {
      byte = static_cast<uint8_t>((byte & 0x0F) | (code << 4));
    } else {
      byte = static_cast<uint8_t>((byte & 0xF0) | code);
    }
  } else {
    base[static_cast<size_t>(row) * ld + k] = code;
  }
}

template <typename T>
void fill_operand(std::mt19937& rng, uint8_t* data, uint8_t* scales, int rows, int k,
                  int ld, int ld_scales) {
  // Random representable values in a moderate range, random per-block scales.
  std::uniform_real_distribution<float> dist(-1.f, 1.f);
  std::uniform_int_distribution<int> sdist(124, 130);  // 2^-3 .. 2^3
  for (int r = 0; r < rows; ++r) {
    for (int kk = 0; kk < k; ++kk) {
      store_elem<T>(data, r, kk, ld, T::from_float(dist(rng)));
    }
    for (int kg = 0; kg < (k + 31) / 32; ++kg) {
      scales[static_cast<size_t>(r) * ld_scales + kg] =
          static_cast<uint8_t>(sdist(rng));
    }
  }
}

template <typename T, typename LC,
          typename Config =
              typename cpu_ops::gemm::device::detail::GemmMxDefaults<T, T>::Config,
          typename Policy =
              typename cpu_ops::gemm::device::detail::GemmMxDefaults<T, T>::Policy>
void run_case(int m, int n, int k, float alpha, float beta, int num_threads,
              int split_k_slices = 0) {
  ++g_cases;
  // fp4 needs an even ld; keep k even there too so padding stays simple.
  const bool is_fp4 = std::is_same<T, fp4e2m1_t>::value;
  const int lda = (k + 3) & ~(is_fp4 ? 1 : 0);
  const int ldb = (k + 3) & ~(is_fp4 ? 1 : 0);
  const int ldc = (LC::kIsRowMajor ? n : m) + 3;
  const int kg = (k + 31) / 32;
  const int ldsa = kg + 1, ldsb = kg + 1;

  std::vector<uint8_t> A(static_cast<size_t>(m) * (lda / (is_fp4 ? 2 : 1)), 0);
  std::vector<uint8_t> B(static_cast<size_t>(n) * (ldb / (is_fp4 ? 2 : 1)), 0);
  std::vector<uint8_t> SA(static_cast<size_t>(m) * ldsa, 0);
  std::vector<uint8_t> SB(static_cast<size_t>(n) * ldsb, 0);
  const size_t size_c = static_cast<size_t>(LC::kIsRowMajor ? m : n) * ldc;
  std::vector<float> C(size_c, 0.f), D(size_c, std::numeric_limits<float>::quiet_NaN());

  std::mt19937 rng(static_cast<unsigned>(m * 131 + n * 17 + k));
  fill_operand<T>(rng, A.data(), SA.data(), m, k, lda, ldsa);
  fill_operand<T>(rng, B.data(), SB.data(), n, k, ldb, ldsb);
  std::uniform_real_distribution<float> cdist(-1.f, 1.f);
  for (auto& x : C) x = cdist(rng);

  // Reference: double accumulation over decoded values with scales applied.
  std::vector<double> D_ref(static_cast<size_t>(m) * n), bound(static_cast<size_t>(m) * n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0, bnd = 0.0;
      for (int p = 0; p < k; ++p) {
        const double a = static_cast<double>(decode_elem<T>(A.data(), i, p, lda)) *
                         e8m0_to_float(SA[static_cast<size_t>(i) * ldsa + (p >> 5)]);
        const double b = static_cast<double>(decode_elem<T>(B.data(), j, p, ldb)) *
                         e8m0_to_float(SB[static_cast<size_t>(j) * ldsb + (p >> 5)]);
        acc += a * b;
        bnd += std::fabs(a * b);
      }
      double v = static_cast<double>(alpha) * acc;
      if (beta != 0.f) v += static_cast<double>(beta) * C[LC::offset(i, j, ldc)];
      D_ref[static_cast<size_t>(i) * n + j] = v;
      bound[static_cast<size_t>(i) * n + j] = bnd;
    }
  }

  using cpu_ops::gemm::device::GemmMx;
  using Op = GemmMx<T, T, LC, cpu_ops::epilogue::LinearCombination<float>, Config, Policy>;
  typename Op::Arguments args;
  args.problem_size = {m, n, k};
  args.ref_A = {A.data(), SA.data(), lda, ldsa};
  args.ref_B = {B.data(), SB.data(), ldb, ldsb};
  args.ref_C = {C.data(), ldc};
  args.ref_D = {D.data(), ldc};
  args.epilogue = {alpha, beta};
  args.split_k_slices = split_k_slices;
  Op op;
  CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess,
        "%s (%dx%dx%d) threads=%d slices=%d: bad status", MxLab<T>::name(), m, n, k,
        num_threads, split_k_slices);

  const double eps = std::numeric_limits<float>::epsilon();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const size_t off = static_cast<size_t>(LC::offset(i, j, ldc));
      const double want = D_ref[static_cast<size_t>(i) * n + j];
      const double allowed = 8.0 * k * eps * bound[static_cast<size_t>(i) * n + j] +
                             16.0 * eps * std::fabs(want) + 1e-12;
      const double err = std::fabs(static_cast<double>(D[off]) - want);
      CHECK(err <= allowed, "%s (%dx%dx%d) a=%g b=%g threads=%d slices=%d C(%d,%d): "
                            "got %g want %g",
            MxLab<T>::name(), m, n, k, (double)alpha, (double)beta, num_threads,
            split_k_slices, i, j, static_cast<double>(D[off]), want);
    }
  }
}

void check_status() {
  ++g_cases;
  using cpu_ops::layout::RowMajor;
  using Op = cpu_ops::gemm::device::GemmMx<fp4e2m1_t, fp4e2m1_t, RowMajor>;
  Op op;
  std::vector<uint8_t> data(64, 0), scales(8, 127);
  std::vector<float> c(64, 0.f);
  typename Op::Arguments args;
  args.problem_size = {4, 4, 32};
  args.ref_A = {data.data(), scales.data(), 32, 1};
  args.ref_B = {data.data(), scales.data(), 32, 1};
  args.ref_C = {c.data(), 4};
  args.ref_D = {c.data(), 4};

  CHECK(op(args, 1) == cpu_ops::Status::kSuccess, "valid fp4 problem rejected");

  args.ref_A.scales = nullptr;
  CHECK(op(args, 1) == cpu_ops::Status::kErrorInvalidArguments, "null scales accepted");
  args.ref_A.scales = scales.data();

  args.ref_A.ld = 33;  // odd ld is illegal for fp4
  CHECK(op(args, 1) == cpu_ops::Status::kErrorInvalidArguments, "odd fp4 ld accepted");
  args.ref_A.ld = 32;

  args.problem_size = {-1, 4, 32};
  CHECK(op(args, 1) == cpu_ops::Status::kErrorInvalidProblem, "negative dim accepted");
}

}  // namespace

int main() {
  check_decode_tables();
  check_encode_roundtrip<fp8e4m3_t>("e4m3");
  check_encode_roundtrip<fp8e5m2_t>("e5m2");
  check_encode_roundtrip<fp4e2m1_t>("e2m1");
  check_e8m0_roundtrip();

  struct ShapeCfg {
    int m, n, k;
  };
  const ShapeCfg shapes[] = {{1, 1, 32},   {5, 7, 96},   {16, 6, 33}, {64, 64, 64},
                             {65, 63, 67}, {31, 17, 258}, {128, 256, 96}};

  for (const ShapeCfg& s : shapes) {
    for (int threads : {1, 8}) {
      run_case<fp8e4m3_t, cpu_ops::layout::RowMajor>(s.m, s.n, s.k, 1.0f, 0.0f, threads);
      run_case<fp8e4m3_t, cpu_ops::layout::ColumnMajor>(s.m, s.n, s.k, 2.0f, -0.5f,
                                                        threads);
      run_case<fp8e5m2_t, cpu_ops::layout::RowMajor>(s.m, s.n, s.k, 1.0f, 0.25f, threads);
      run_case<fp4e2m1_t, cpu_ops::layout::RowMajor>(s.m, s.n, s.k & ~1, 1.0f, 0.0f,
                                                     threads);
      run_case<fp4e2m1_t, cpu_ops::layout::ColumnMajor>(s.m, s.n, s.k & ~1, 0.5f, -0.5f,
                                                        threads);
    }
  }

  // Automatic split-k (deep k), and forced slices with boundaries that are
  // neither 32-aligned nor even (exercises fp4 nibble alignment).
  for (int threads : {1, 8}) {
    run_case<fp8e4m3_t, cpu_ops::layout::RowMajor>(32, 32, 4096, 1.0f, 0.5f, threads);
    run_case<fp4e2m1_t, cpu_ops::layout::RowMajor>(32, 32, 4096, 1.0f, 0.5f, threads);
  }
  run_case<fp8e4m3_t, cpu_ops::layout::RowMajor>(24, 40, 3000, 1.25f, 0.5f, 8, 7);
  run_case<fp8e5m2_t, cpu_ops::layout::RowMajor>(24, 40, 3000, 1.25f, 0.5f, 8, 7);
  run_case<fp4e2m1_t, cpu_ops::layout::RowMajor>(24, 40, 2998, 1.25f, 0.5f, 8, 7);

  // 512-bit mainloop (VLEN = 16, Fma512 tile shape): the device-level default
  // on AVX-512F hosts; elsewhere the portable Vec<float, 16> fallback gives
  // the same coverage.
  using cpu_ops::mma::Fma512GemmConfig;
  using cpu_ops::mma::MxPolicy;
  for (const ShapeCfg& s : shapes) {
    for (int threads : {1, 8}) {
      run_case<fp8e4m3_t, cpu_ops::layout::RowMajor, Fma512GemmConfig,
               MxPolicy<fp8e4m3_t, fp8e4m3_t, 16>>(s.m, s.n, s.k, 1.0f, 0.25f, threads);
      run_case<fp8e5m2_t, cpu_ops::layout::RowMajor, Fma512GemmConfig,
               MxPolicy<fp8e5m2_t, fp8e5m2_t, 16>>(s.m, s.n, s.k, 1.0f, 0.25f, threads);
      run_case<fp4e2m1_t, cpu_ops::layout::RowMajor, Fma512GemmConfig,
               MxPolicy<fp4e2m1_t, fp4e2m1_t, 16>>(s.m, s.n, s.k & ~1, 1.0f, 0.25f,
                                                  threads);
    }
  }

  check_status();

  std::printf("test_gemm_mx: %d cases, %d failures\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
