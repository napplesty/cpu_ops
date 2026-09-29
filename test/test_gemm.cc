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

template <typename T>
struct Lab {
  static const char* name();
};
template <>
const char* Lab<float>::name() {
  return "f32";
}
template <>
const char* Lab<double>::name() {
  return "f64";
}

// Reference implementation with double-precision accumulation.
template <typename T, typename LA, typename LB, typename LC>
void ref_gemm(int m, int n, int k, T alpha, const std::vector<T>& A, int lda,
              const std::vector<T>& B, int ldb, T beta, const std::vector<T>& C, int ldc,
              std::vector<T>& D, int ldd, std::vector<double>& bound, bool relu) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      double acc = 0.0;
      double bnd = 0.0;
      for (int p = 0; p < k; ++p) {
        const double a = A[LA::offset(i, p, lda)];
        const double b = B[LB::offset(p, j, ldb)];
        acc += a * b;
        bnd += std::fabs(a * b);
      }
      double v = static_cast<double>(alpha) * acc;
      if (beta != T(0)) v += static_cast<double>(beta) * C[LC::offset(i, j, ldc)];
      if (relu && v < 0.0) v = 0.0;
      D[LC::offset(i, j, ldd)] = static_cast<T>(v);
      bound[static_cast<size_t>(i) * n + j] = bnd;
    }
  }
}

template <typename T, typename LA, typename LB, typename LC>
void run_case(int m, int n, int k, T alpha, T beta, bool relu, int num_threads) {
  ++g_cases;
  // Padded leading dimensions on purpose: the library must not assume
  // tightly packed operands.
  const int lda = (LA::kIsRowMajor ? k : m) + 3;
  const int ldb = (LB::kIsRowMajor ? n : k) + 3;
  const int ldc = (LC::kIsRowMajor ? n : m) + 3;
  const size_t size_a = static_cast<size_t>(LA::kIsRowMajor ? m : k) * lda;
  const size_t size_b = static_cast<size_t>(LB::kIsRowMajor ? k : n) * ldb;
  const size_t size_c = static_cast<size_t>(LC::kIsRowMajor ? m : n) * ldc;

  std::vector<T> A(size_a, T(0));
  std::vector<T> B(size_b, T(0));
  std::vector<T> C(size_c, T(0));
  std::vector<T> D(size_c, std::numeric_limits<T>::quiet_NaN());
  std::vector<T> D_ref(size_c, T(0));
  std::vector<double> bound(static_cast<size_t>(m) * n, 0.0);

  std::mt19937 rng(static_cast<unsigned>(m * 131 + n * 17 + k));
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  for (size_t i = 0; i < size_a; ++i) A[i] = dist(rng);
  for (size_t i = 0; i < size_b; ++i) B[i] = dist(rng);
  for (size_t i = 0; i < size_c; ++i) C[i] = dist(rng);

  ref_gemm<T, LA, LB, LC>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D_ref, ldc, bound,
                          relu);

  using Gemm = cpu_ops::gemm::device::Gemm<T, LA, T, LB, T, LC>;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), lda},
                                {B.data(), ldb},
                                {C.data(), ldc},
                                {D.data(), ldc},
                                {alpha, beta,
                                 relu ? cpu_ops::epilogue::Activation::kRelu
                                      : cpu_ops::epilogue::Activation::kNone}};
  Gemm op;
  CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess,
        "%s (%dx%dx%d) threads=%d: bad status", Lab<T>::name(), m, n, k, num_threads);

  // In-place run: C and D aliased.
  std::vector<T> D2 = C;
  typename Gemm::Arguments args2{{m, n, k},
                                 {A.data(), lda},
                                 {B.data(), ldb},
                                 {D2.data(), ldc},
                                 {D2.data(), ldc},
                                 args.epilogue};
  CHECK(op(args2, num_threads) == cpu_ops::Status::kSuccess,
        "%s (%dx%dx%d) threads=%d: bad status (in-place)", Lab<T>::name(), m, n, k,
        num_threads);

  const double eps = std::numeric_limits<T>::epsilon();
  double max_err = 0.0;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const size_t off = static_cast<size_t>(LC::offset(i, j, ldc));
      const double want = D_ref[off];
      const double allowed = 8.0 * k * eps * bound[static_cast<size_t>(i) * n + j] +
                             16.0 * eps * std::fabs(want) + 1e-12;
      for (const T* d : {&D[off], &D2[off]}) {
        const double err = std::fabs(static_cast<double>(*d) - want);
        if (err > max_err) max_err = err;
        CHECK(err <= allowed,
              "%s %s (%dx%dx%d) a=%g b=%g relu=%d threads=%d C(%d,%d): got %g want %g",
              Lab<T>::name(), LC::kIsRowMajor ? "row" : "col", m, n, k, (double)alpha,
              (double)beta, (int)relu, num_threads, i, j, *d, want);
      }
    }
  }
  if (max_err > 0.0) {
    std::printf("ok   %s %-3s (%4dx%4dx%4d) a=%4.2f b=%5.2f relu=%d threads=%d max_err=%.3g\n",
                Lab<T>::name(), LC::kIsRowMajor ? "row" : "col", m, n, k, (double)alpha,
                (double)beta, (int)relu, num_threads, max_err);
  }
}

template <typename T>
void run_all_layouts(int m, int n, int k, T alpha, T beta, bool relu, int num_threads) {
  using cpu_ops::layout::ColumnMajor;
  using cpu_ops::layout::RowMajor;
  run_case<T, RowMajor, RowMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, RowMajor, RowMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, RowMajor, ColumnMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, RowMajor, ColumnMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, ColumnMajor, RowMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, ColumnMajor, RowMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, ColumnMajor, ColumnMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<T, ColumnMajor, ColumnMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
}

// Forced split-k: split_k_slices in Arguments, including slice counts that do
// not divide k evenly. Summation order differs from the serial reference, so
// compare with the usual error bound.
template <typename T, typename LC>
void check_splitk_forced(int m, int n, int k, T alpha, T beta, int slices,
                         int num_threads) {
  using cpu_ops::layout::RowMajor;
  ++g_cases;
  const int lda = k + 3, ldb = n + 3, ldc = (LC::kIsRowMajor ? n : m) + 3;
  std::vector<T> A(static_cast<size_t>(m) * lda), B(static_cast<size_t>(k) * ldb);
  const size_t size_c = static_cast<size_t>(LC::kIsRowMajor ? m : n) * ldc;
  std::vector<T> C(size_c), D(size_c), D_ref(size_c);
  std::vector<double> bound(static_cast<size_t>(m) * n);

  std::mt19937 rng(static_cast<unsigned>(m * 131 + n * 17 + k + slices));
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);
  for (auto& x : C) x = dist(rng);

  ref_gemm<T, RowMajor, RowMajor, LC>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D_ref,
                                      ldc, bound, false);

  using Gemm = cpu_ops::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, LC>;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), lda},
                                {B.data(), ldb},
                                {C.data(), ldc},
                                {D.data(), ldc},
                                {alpha, beta},
                                slices};
  Gemm op;
  CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess,
        "splitk (%dx%dx%d) slices=%d: bad status", m, n, k, slices);

  const double eps = std::numeric_limits<T>::epsilon();
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const size_t off = static_cast<size_t>(LC::offset(i, j, ldc));
      const double want = D_ref[off];
      const double allowed = 8.0 * k * eps * bound[static_cast<size_t>(i) * n + j] +
                             16.0 * eps * std::fabs(want) + 1e-12;
      const double err = std::fabs(static_cast<double>(D[off]) - want);
      CHECK(err <= allowed, "splitk (%dx%dx%d) slices=%d threads=%d C(%d,%d): got %g want %g",
            m, n, k, slices, num_threads, i, j, static_cast<double>(D[off]), want);
    }
  }
}

// 1-thread and 8-thread runs must agree bit for bit: partitioning changes who
// computes a tile, never the arithmetic inside it.
template <typename T>
void check_thread_determinism(int m, int n, int k) {
  using cpu_ops::layout::RowMajor;
  const int lda = k, ldb = n, ldc = n;
  std::vector<T> A(static_cast<size_t>(m) * lda), B(static_cast<size_t>(k) * ldb),
      C(static_cast<size_t>(m) * ldc, T(0)), D1(static_cast<size_t>(m) * ldc, T(0)),
      D8(static_cast<size_t>(m) * ldc, T(0));
  std::mt19937 rng(7);
  std::uniform_real_distribution<T> dist(T(-1), T(1));
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);
  for (auto& x : C) x = dist(rng);

  using Gemm = cpu_ops::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, RowMajor>;
  Gemm op;
  typename Gemm::Arguments args1{{m, n, k}, {A.data(), lda}, {B.data(), ldb},
                                 {C.data(), ldc}, {D1.data(), ldc}, {T(1.5), T(-0.25)}};
  typename Gemm::Arguments args8{{m, n, k}, {A.data(), lda}, {B.data(), ldb},
                                 {C.data(), ldc}, {D8.data(), ldc}, {T(1.5), T(-0.25)}};
  CHECK(op(args1, 1) == cpu_ops::Status::kSuccess, "determinism: 1-thread run failed");
  CHECK(op(args8, 8) == cpu_ops::Status::kSuccess, "determinism: 8-thread run failed");
  for (size_t i = 0; i < D1.size(); ++i) {
    CHECK(std::memcmp(&D1[i], &D8[i], sizeof(T)) == 0,
          "%s determinism mismatch at flat index %zu", Lab<T>::name(), i);
  }
  ++g_cases;
}

template <typename T>
void check_status() {
  using cpu_ops::layout::RowMajor;
  using Gemm = cpu_ops::gemm::device::Gemm<T, RowMajor, T, RowMajor, T, RowMajor>;
  Gemm op;
  std::vector<T> buf(64, T(0));
  typename Gemm::Arguments args{{4, 4, 4}, {buf.data(), 4}, {buf.data(), 4},
                                {buf.data(), 4}, {buf.data(), 4}, {T(1), T(0)}};

  args.problem_size = {-1, 4, 4};
  CHECK(op(args, 1) == cpu_ops::Status::kErrorInvalidProblem, "negative dim accepted");

  args.problem_size = {4, 4, 4};
  args.ref_A = {nullptr, 4};
  CHECK(op(args, 1) == cpu_ops::Status::kErrorInvalidArguments, "null A accepted");

  args.ref_A = {buf.data(), 3};  // lda < k for row-major A
  CHECK(op(args, 1) == cpu_ops::Status::kErrorInvalidArguments, "bad lda accepted");

  args.ref_A = {buf.data(), 4};
  args.problem_size = {0, 4, 4};  // empty problem is a no-op success
  CHECK(op(args, 1) == cpu_ops::Status::kSuccess, "m=0 rejected");
  ++g_cases;
}

}  // namespace

int main() {
  struct ShapeCfg {
    int m, n, k;
  };
  const ShapeCfg shapes[] = {{1, 1, 1},   {1, 1, 9},     {6, 16, 1}, {1, 64, 64},
                             {5, 7, 3},   {16, 6, 33},   {64, 64, 64}, {65, 63, 67},
                             {31, 17, 257}, {128, 256, 96}, {300, 257, 129}};

  for (const ShapeCfg& s : shapes) {
    for (int threads : {1, 8}) {
      run_all_layouts<float>(s.m, s.n, s.k, 1.0f, 0.0f, false, threads);
      run_all_layouts<float>(s.m, s.n, s.k, 2.0f, -0.5f, false, threads);
    }
  }

  // ReLU epilogue
  for (int threads : {1, 8}) {
    run_all_layouts<float>(64, 64, 64, 1.0f, 0.25f, true, threads);
    run_all_layouts<float>(64, 64, 64, 1.0f, 0.0f, true, threads);
  }

  // Degenerate k == 0: D = beta * C
  run_all_layouts<float>(16, 24, 0, 1.0f, 0.75f, false, 4);

  // Double precision spot checks
  for (int threads : {1, 8}) {
    run_all_layouts<double>(64, 64, 64, 1.0, 0.25, false, threads);
    run_all_layouts<double>(65, 63, 67, 1.5, -0.5, true, threads);
  }

  // Split-k: small outputs with deep k trigger automatic k slicing.
  for (int threads : {1, 8}) {
    run_all_layouts<float>(16, 16, 4096, 1.0f, 0.5f, false, threads);
    run_all_layouts<float>(32, 32, 8192, 2.0f, -0.25f, true, threads);
    run_all_layouts<float>(48, 8, 2048, 1.0f, 0.0f, false, threads);
  }
  run_all_layouts<double>(24, 40, 4096, 1.5, -0.5, false, 8);

  // Forced slice counts, including ones that do not divide k evenly.
  check_splitk_forced<float, cpu_ops::layout::RowMajor>(40, 24, 3000, 1.25f, 0.5f, 3, 8);
  check_splitk_forced<float, cpu_ops::layout::RowMajor>(40, 24, 3000, 1.25f, 0.5f, 7, 8);
  check_splitk_forced<float, cpu_ops::layout::ColumnMajor>(17, 33, 1027, 1.0f, -1.0f, 5, 4);

  check_thread_determinism<float>(129, 255, 96);
  check_thread_determinism<double>(96, 130, 40);
  check_status<float>();

  std::printf("test_gemm: %d cases, %d failures\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
