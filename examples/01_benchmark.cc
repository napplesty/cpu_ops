#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"

namespace {

// Plain triple-loop baseline (i-k-j order, auto-vectorized by the compiler).
void naive_gemm(int m, int n, int k, const float* A, const float* B, float* C) {
  for (int i = 0; i < m; ++i) {
    for (int p = 0; p < k; ++p) {
      const float a = A[static_cast<size_t>(i) * k + p];
      for (int j = 0; j < n; ++j) {
        C[static_cast<size_t>(i) * n + j] += a * B[static_cast<size_t>(p) * n + j];
      }
    }
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

double bench_cpu_ops(int m, int n, int k, int num_threads, int reps) {
  namespace layout = cpu_ops::layout;
  std::vector<float> A(static_cast<size_t>(m) * k), B(static_cast<size_t>(k) * n),
      C(static_cast<size_t>(m) * n, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);

  using Gemm = cpu_ops::gemm::device::Gemm<float, layout::RowMajor, float, layout::RowMajor,
                                           float, layout::RowMajor>;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), k},
                                {B.data(), n},
                                {C.data(), n},
                                {C.data(), n},
                                {1.0f, 0.0f}};
  Gemm gemm;
  gemm(args, num_threads);  // warmup
  const double seconds = time_best([&] { gemm(args, num_threads); }, reps);
  const volatile float sink = C[static_cast<size_t>(m) * n / 2];
  (void)sink;
  return 2.0 * m * n * k / seconds / 1e9;
}

double bench_naive(int m, int n, int k) {
  std::vector<float> A(static_cast<size_t>(m) * k), B(static_cast<size_t>(k) * n),
      C(static_cast<size_t>(m) * n, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : A) x = dist(rng);
  for (auto& x : B) x = dist(rng);
  naive_gemm(m, n, k, A.data(), B.data(), C.data());  // warmup
  const double seconds =
      time_best([&] { naive_gemm(m, n, k, A.data(), B.data(), C.data()); }, 2);
  const volatile float sink = C[static_cast<size_t>(m) * n / 2];
  (void)sink;
  return 2.0 * m * n * k / seconds / 1e9;
}

// uint8 x int8 -> int32 path (4-way byte dot products, VNNI when available).
// Returns GOPS where one "op" is one multiply-add.
double bench_cpu_ops_int8(int m, int n, int k, int num_threads, int reps) {
  namespace layout = cpu_ops::layout;
  std::vector<uint8_t> A(static_cast<size_t>(m) * k);
  std::vector<int8_t> B(static_cast<size_t>(k) * n);
  std::vector<int32_t> C(static_cast<size_t>(m) * n, 0);
  std::mt19937 rng(42);
  for (auto& x : A) x = static_cast<uint8_t>(rng() % 16);
  for (auto& x : B) x = static_cast<int8_t>(rng() % 17 - 8);

  using Gemm = cpu_ops::gemm::device::GemmU8S8S32<layout::RowMajor, layout::RowMajor,
                                                  layout::RowMajor>;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), k},
                                {B.data(), n},
                                {C.data(), n},
                                {C.data(), n},
                                {1, 0}};
  Gemm gemm;
  gemm(args, num_threads);  // warmup
  const double seconds = time_best([&] { gemm(args, num_threads); }, reps);
  const volatile int32_t sink = C[static_cast<size_t>(m) * n / 2];
  (void)sink;
  return 2.0 * m * n * k / seconds / 1e9;
}

}  // namespace

int main() {
  const int sizes[] = {512, 1024, 2048};
  const int threads[] = {1, 2, 4, 8};

  std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
  std::printf("-------------+----------------------------------------\n");
  for (int s : sizes) {
    std::printf("%4d^3 GFLOPS |", s);
    for (int t : threads) {
      std::printf(" %8.1f", bench_cpu_ops(s, s, s, t, 3));
      std::fflush(stdout);
    }
    std::printf("\n");
  }

  std::printf("\nnaive triple loop, 512^3, 1 thread: %.2f GFLOPS\n", bench_naive(512, 512, 512));

  std::printf("\nint8 (u8 x s8 -> s32) GOPS:\n");
  std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
  std::printf("-------------+----------------------------------------\n");
  for (int s : {1024, 2048}) {
    std::printf("%4d^3 GOPS   |", s);
    for (int t : threads) {
      std::printf(" %8.1f", bench_cpu_ops_int8(s, s, s, t, 3));
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  return 0;
}
