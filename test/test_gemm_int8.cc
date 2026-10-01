#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"
#include "cpu_ops/gemm/threadblock/mma_policy_amx.h"

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

// Exact reference: int64 accumulation, integer alpha/beta epilogue.
template <typename LA, typename LB, typename LC>
void ref_gemm_int8(int m, int n, int k, int32_t alpha, const std::vector<uint8_t>& A,
                   int lda, const std::vector<int8_t>& B, int ldb, int32_t beta,
                   const std::vector<int32_t>& C, int ldc, std::vector<int32_t>& D, int ldd,
                   bool relu) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      int64_t acc = 0;
      for (int p = 0; p < k; ++p) {
        acc += static_cast<int64_t>(A[LA::offset(i, p, lda)]) *
               static_cast<int64_t>(B[LB::offset(p, j, ldb)]);
      }
      int64_t v = static_cast<int64_t>(alpha) * acc +
                  static_cast<int64_t>(beta) * C[LC::offset(i, j, ldc)];
      if (relu && v < 0) v = 0;
      D[LC::offset(i, j, ldd)] = static_cast<int32_t>(v);
    }
  }
}

template <typename LA, typename LB, typename LC,
          typename Config = cpu_ops::gemm::device::GemmConfig<int32_t>,
          typename Policy = cpu_ops::mma::VnniPolicy>
void run_case(int m, int n, int k, int32_t alpha, int32_t beta, bool relu,
              int num_threads) {
  ++g_cases;
  const int lda = (LA::kIsRowMajor ? k : m) + 3;  // padding on purpose
  const int ldb = (LB::kIsRowMajor ? n : k) + 3;
  const int ldc = (LC::kIsRowMajor ? n : m) + 3;
  const size_t size_a = static_cast<size_t>(LA::kIsRowMajor ? m : k) * lda;
  const size_t size_b = static_cast<size_t>(LB::kIsRowMajor ? k : n) * ldb;
  const size_t size_c = static_cast<size_t>(LC::kIsRowMajor ? m : n) * ldc;

  std::vector<uint8_t> A(size_a, 0);
  std::vector<int8_t> B(size_b, 0);
  std::vector<int32_t> C(size_c, 0);
  std::vector<int32_t> D(size_c, INT32_MIN);  // poison: every element must be written
  std::vector<int32_t> D_ref(size_c, 0);

  std::mt19937 rng(static_cast<unsigned>(m * 131 + n * 17 + k));
  for (auto& x : A) x = static_cast<uint8_t>(rng() % 16);        // [0, 15]
  for (auto& x : B) x = static_cast<int8_t>(rng() % 17 - 8);     // [-8, 8]
  for (auto& x : C) x = static_cast<int32_t>(rng() % 201 - 100); // [-100, 100]

  ref_gemm_int8<LA, LB, LC>(m, n, k, alpha, A, lda, B, ldb, beta, C, ldc, D_ref, ldc, relu);

  using Gemm = cpu_ops::gemm::device::Gemm<uint8_t, LA, int8_t, LB, int32_t, LC, int32_t,
                                           cpu_ops::epilogue::LinearCombination<int32_t>,
                                           Config, Policy>;
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
        "int8 (%dx%dx%d) threads=%d: bad status", m, n, k, num_threads);

  // In-place run: C and D aliased.
  std::vector<int32_t> D2 = C;
  typename Gemm::Arguments args2{{m, n, k},
                                 {A.data(), lda},
                                 {B.data(), ldb},
                                 {D2.data(), ldc},
                                 {D2.data(), ldc},
                                 args.epilogue};
  CHECK(op(args2, num_threads) == cpu_ops::Status::kSuccess,
        "int8 (%dx%dx%d) threads=%d: bad status (in-place)", m, n, k, num_threads);

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const size_t off = static_cast<size_t>(LC::offset(i, j, ldc));
      CHECK(D[off] == D_ref[off],
            "int8 (%dx%dx%d) a=%d b=%d relu=%d threads=%d C(%d,%d): got %d want %d", m, n, k,
            alpha, beta, (int)relu, num_threads, i, j, D[off], D_ref[off]);
      CHECK(D2[off] == D_ref[off],
            "int8 in-place (%dx%dx%d) a=%d b=%d relu=%d threads=%d C(%d,%d): got %d want %d",
            m, n, k, alpha, beta, (int)relu, num_threads, i, j, D2[off], D_ref[off]);
    }
  }
}

void run_all_layouts(int m, int n, int k, int32_t alpha, int32_t beta, bool relu,
                     int num_threads) {
  using cpu_ops::layout::ColumnMajor;
  using cpu_ops::layout::RowMajor;
  run_case<RowMajor, RowMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<RowMajor, RowMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<RowMajor, ColumnMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<RowMajor, ColumnMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<ColumnMajor, RowMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<ColumnMajor, RowMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<ColumnMajor, ColumnMajor, RowMajor>(m, n, k, alpha, beta, relu, num_threads);
  run_case<ColumnMajor, ColumnMajor, ColumnMajor>(m, n, k, alpha, beta, relu, num_threads);
}

void check_thread_determinism(int m, int n, int k) {
  using cpu_ops::layout::RowMajor;
  const int lda = k, ldb = n, ldc = n;
  std::vector<uint8_t> A(static_cast<size_t>(m) * lda);
  std::vector<int8_t> B(static_cast<size_t>(k) * ldb);
  std::vector<int32_t> C(static_cast<size_t>(m) * ldc, 0), D1(static_cast<size_t>(m) * ldc),
      D8(static_cast<size_t>(m) * ldc);
  std::mt19937 rng(11);
  for (auto& x : A) x = static_cast<uint8_t>(rng() % 16);
  for (auto& x : B) x = static_cast<int8_t>(rng() % 17 - 8);
  for (auto& x : C) x = static_cast<int32_t>(rng() % 201 - 100);

  using Gemm = cpu_ops::gemm::device::GemmU8S8S32<RowMajor, RowMajor, RowMajor>;
  Gemm op;
  typename Gemm::Arguments args1{{m, n, k}, {A.data(), lda}, {B.data(), ldb},
                                 {C.data(), ldc}, {D1.data(), ldc}, {3, -2}};
  typename Gemm::Arguments args8{{m, n, k}, {A.data(), lda}, {B.data(), ldb},
                                 {C.data(), ldc}, {D8.data(), ldc}, {3, -2}};
  CHECK(op(args1, 1) == cpu_ops::Status::kSuccess, "int8 determinism: 1-thread failed");
  CHECK(op(args8, 8) == cpu_ops::Status::kSuccess, "int8 determinism: 8-thread failed");
  for (size_t i = 0; i < D1.size(); ++i) {
    CHECK(D1[i] == D8[i], "int8 determinism mismatch at flat index %zu", i);
  }
  ++g_cases;
}

}  // namespace

int main() {
  struct ShapeCfg {
    int m, n, k;
  };
  // K values cover every residue mod 4 (1, 5, 7, 4, 9, ...) plus k-block
  // boundaries around KC = 256.
  const ShapeCfg shapes[] = {{1, 1, 1},  {1, 1, 5},    {6, 16, 4},  {5, 7, 9},
                             {3, 5, 7},  {16, 16, 16}, {64, 64, 64}, {65, 63, 67},
                             {31, 17, 255}, {17, 31, 257}, {128, 256, 96}};

  for (const ShapeCfg& s : shapes) {
    for (int threads : {1, 8}) {
      run_all_layouts(s.m, s.n, s.k, 1, 0, false, threads);
      run_all_layouts(s.m, s.n, s.k, 3, -2, false, threads);
    }
  }
  for (int threads : {1, 8}) {
    run_all_layouts(64, 64, 64, 1, 1, true, threads);
  }
  run_all_layouts(16, 24, 0, 1, 5, false, 4);

  // Split-k paths: small outputs with deep k. Exact integer arithmetic makes
  // the slice reduction order irrelevant.
  for (int threads : {1, 8}) {
    run_all_layouts(16, 16, 4096, 3, -2, false, threads);
    run_all_layouts(32, 32, 8192, 1, 1, true, threads);
  }

  check_thread_determinism(129, 255, 97);

  // The experimental AMX-INT8 policy: exact integer arithmetic through the
  // same packed layouts the tile path consumes. On hosts without AMX (or
  // without OS tile permission) the atom's scalar fallback is what runs.
  {
    using cpu_ops::layout::ColumnMajor;
    using cpu_ops::layout::RowMajor;
    using cpu_ops::mma::AmxGemmConfig;
    using cpu_ops::mma::AmxPolicy;
    for (const ShapeCfg& s : shapes) {
      for (int threads : {1, 8}) {
        run_case<RowMajor, RowMajor, RowMajor, AmxGemmConfig, AmxPolicy>(
            s.m, s.n, s.k, 1, 0, false, threads);
        run_case<ColumnMajor, ColumnMajor, ColumnMajor, AmxGemmConfig, AmxPolicy>(
            s.m, s.n, s.k, 3, -2, false, threads);
      }
    }
    run_case<RowMajor, RowMajor, RowMajor, AmxGemmConfig, AmxPolicy>(16, 24, 0, 1, 5,
                                                                     false, 4);
    // Split-k with a slice count not dividing k.
    run_case<RowMajor, RowMajor, RowMajor, AmxGemmConfig, AmxPolicy>(32, 32, 8192, 1, 1,
                                                                     true, 8);
  }

  std::printf("test_gemm_int8: %d cases, %d failures\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
