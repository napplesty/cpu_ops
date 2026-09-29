// Fused epilogue correctness: a fused single pass must match a plain Gemm
// followed by the same ops applied by hand. Comparison is by tight tolerance
// rather than bitwise: GCC's default -ffp-contract=fast may fuse mul+add into
// fma in one epilogue instantiation and not the other (1-ulp differences).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "cpu_ops/gemm/device/gemm_fused.h"
#include "cpu_ops/layout.h"

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

using cpu_ops::layout::ColumnMajor;
using cpu_ops::layout::RowMajor;
namespace epilogue = cpu_ops::epilogue;

template <typename LayoutC, typename... Ops>
void run_fused_case(int m, int n, int k, float alpha, float beta,
                    epilogue::Chain<Ops...> chain, int num_threads) {
  ++g_cases;
  const int lda = k + 2;
  const int ldb = n + 1;
  const int ldc = (LayoutC::kIsRowMajor ? n : m) + 3;
  std::vector<float> A(static_cast<size_t>(m) * lda);
  std::vector<float> B(static_cast<size_t>(k) * ldb);
  const size_t size_c = static_cast<size_t>(LayoutC::kIsRowMajor ? m : n) * ldc;
  std::vector<float> C(size_c), D_ref(size_c), D_fused(size_c);

  unsigned seed = static_cast<unsigned>(m * 131 + n * 17 + k);
  auto rnd = [&seed]() {
    seed = seed * 1664525u + 1013904223u;
    return static_cast<float>(static_cast<int>(seed >> 9) % 2001 - 1000) * 0.001f;
  };
  for (auto& x : A) x = rnd();
  for (auto& x : B) x = rnd();
  for (auto& x : C) x = rnd();

  // Reference: plain gemm (alpha, beta), then the chain applied by hand.
  {
    using Plain = cpu_ops::gemm::device::Gemm<float, RowMajor, float, RowMajor, float,
                                              LayoutC>;
    typename Plain::Arguments args{{m, n, k},
                                   {A.data(), lda},
                                   {B.data(), ldb},
                                   {C.data(), ldc},
                                   {D_ref.data(), ldc},
                                   {alpha, beta}};
    Plain op;
    CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess, "plain gemm failed");
  }
  // The reference chain works on logical (row, col); walk elements through the
  // layout instead of a flat loop.
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      float& v = D_ref[LayoutC::offset(i, j, ldc)];
      v = chain.apply(v, i, j);
    }

  // Fused single pass.
  {
    using Fused = cpu_ops::gemm::device::GemmFusedF32<RowMajor, RowMajor, LayoutC, Ops...>;
    typename Fused::Arguments args{{m, n, k},
                                   {A.data(), lda},
                                   {B.data(), ldb},
                                   {C.data(), ldc},
                                   {D_fused.data(), ldc},
                                   {alpha, beta, chain}};
    Fused op;
    CHECK(op(args, num_threads) == cpu_ops::Status::kSuccess, "fused gemm failed");
  }

  for (size_t t = 0; t < size_c; ++t) {
    const double ref = D_ref[t];
    const double got = D_fused[t];
    const double tol = 1e-5 * std::max(1.0, std::fabs(ref));
    CHECK(std::fabs(got - ref) <= tol,
          "(%dx%dx%d) a=%g b=%g threads=%d %s: fused != plain+chain at flat %zu: %g vs %g",
          m, n, k, (double)alpha, (double)beta, num_threads,
          LayoutC::kIsRowMajor ? "row" : "col", t, got, ref);
  }
}

}  // namespace

int main() {
  const int shapes[][3] = {{5, 7, 3},    {64, 64, 64},   {65, 63, 67},
                           {31, 17, 257}, {128, 256, 96}, {1, 64, 300}};

  std::vector<float> bias(512), row_scale(512), col_scale(512);
  for (int i = 0; i < 512; ++i) {
    bias[i] = static_cast<float>(i % 13 - 6) * 0.25f;
    row_scale[i] = 0.5f + 0.001f * static_cast<float>(i);
    col_scale[i] = 1.5f - 0.001f * static_cast<float>(i);
  }

  for (const auto& s : shapes) {
    const int m = s[0], n = s[1], k = s[2];
    for (int threads : {1, 8}) {
      run_fused_case<RowMajor>(m, n, k, 1.0f, 0.0f,
                               epilogue::Chain<epilogue::BiasAdd<float>, epilogue::Relu>{
                                   epilogue::BiasAdd<float>{bias.data()}, epilogue::Relu{}},
                               threads);
      run_fused_case<RowMajor>(m, n, k, 2.0f, -0.5f,
                               epilogue::Chain<epilogue::ScalePerRow<float>>{
                                   epilogue::ScalePerRow<float>{row_scale.data()}},
                               threads);
      run_fused_case<RowMajor>(
          m, n, k, 1.0f, 0.25f,
          epilogue::Chain<epilogue::BiasAdd<float>, epilogue::ScalePerCol<float>,
                          epilogue::Clamp<float>>{
              epilogue::BiasAdd<float>{bias.data()},
              epilogue::ScalePerCol<float>{col_scale.data()},
              epilogue::Clamp<float>{-0.5f, 2.0f}},
          threads);
      // Empty chain degenerates to the plain epilogue.
      run_fused_case<RowMajor>(m, n, k, 1.5f, 0.75f, epilogue::Chain<>{}, threads);
      // Column-major C takes the scalar epilogue path.
      run_fused_case<ColumnMajor>(
          m, n, k, 1.0f, 0.5f,
          epilogue::Chain<epilogue::BiasAdd<float>, epilogue::Relu>{
              epilogue::BiasAdd<float>{bias.data()}, epilogue::Relu{}},
          threads);
    }
  }

  std::printf("test_fusion: %d cases, %d failures\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
