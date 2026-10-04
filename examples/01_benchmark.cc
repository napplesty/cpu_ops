#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <type_traits>
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

// Narrow-input (f16/bf16) -> f32 GEMM path. StorageT is the on-disk element
// type; compute and the epilogue run in f32.
template <typename StorageT, typename Gemm>
double bench_cpu_ops_half(int m, int n, int k, int num_threads, int reps) {
  std::vector<StorageT> A(static_cast<size_t>(m) * k), B(static_cast<size_t>(k) * n);
  std::vector<float> C(static_cast<size_t>(m) * n, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : A) x = StorageT(dist(rng));
  for (auto& x : B) x = StorageT(dist(rng));

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

// MX block-scaled path (fp8 e4m3/e5m2, fp4 e2m1 + E8M0 scales every 32
// elements along k): operands are decoded and scaled into f32 panels at pack
// time, then the mainloop is plain f32 FMA.
template <typename T>
double bench_cpu_ops_mx(int m, int n, int k, int num_threads, int reps) {
  namespace layout = cpu_ops::layout;
  constexpr bool is_fp4 = std::is_same<T, cpu_ops::fp4e2m1_t>::value;
  const int row_bytes = is_fp4 ? k / 2 : k;  // k is even in the shapes below
  const int kg = (k + 31) / 32;
  std::vector<uint8_t> A(static_cast<size_t>(m) * row_bytes, 0);
  std::vector<uint8_t> B(static_cast<size_t>(n) * row_bytes, 0);
  std::vector<uint8_t> SA(static_cast<size_t>(m) * kg, 127);  // scales: 2^0
  std::vector<uint8_t> SB(static_cast<size_t>(n) * kg, 127);
  std::vector<float> C(static_cast<size_t>(m) * n, 0.0f);
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  auto fill = [&](uint8_t* base, int rows) {
    for (int r = 0; r < rows; ++r) {
      uint8_t* row = base + static_cast<size_t>(r) * row_bytes;
      for (int p = 0; p < k; ++p) {
        const uint8_t code = T::from_float(dist(rng));
        if constexpr (is_fp4) {  // even element -> low nibble
          row[p / 2] = static_cast<uint8_t>(row[p / 2] | (code << ((p & 1) * 4)));
        } else {
          row[p] = code;
        }
      }
    }
  };
  fill(A.data(), m);
  fill(B.data(), n);

  using Gemm = cpu_ops::gemm::device::GemmMx<T, T, layout::RowMajor>;
  typename Gemm::Arguments args{{m, n, k},
                                {A.data(), SA.data(), k, kg},
                                {B.data(), SB.data(), k, kg},
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

}  // namespace

int main() {
  // Report the SIMD ISA this translation unit was compiled for (the library
  // is header-only, so the consumer's flags decide the active paths).
#if defined(CPU_OPS_SIMD_AVX512BF16)
  std::printf("ISA: AVX-512F + VNNI + BF16\n");
#elif defined(CPU_OPS_SIMD_AVX512F)
  std::printf("ISA: AVX-512F\n");
#elif defined(CPU_OPS_SIMD_VNNI)
  std::printf("ISA: AVX2 + VNNI\n");
#elif defined(CPU_OPS_SIMD_AVX2)
  std::printf("ISA: AVX2\n");
#elif defined(CPU_OPS_SIMD_SVE)
  std::printf("ISA: ARM SVE-%d\n", __ARM_FEATURE_SVE_BITS);
#elif defined(CPU_OPS_SIMD_NEON)
  std::printf("ISA: ARM NEON\n");
#else
  std::printf("ISA: scalar\n");
#endif

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

  {
    namespace layout = cpu_ops::layout;
    using GemmF16 = cpu_ops::gemm::device::GemmF16F32<layout::RowMajor, layout::RowMajor,
                                                      layout::RowMajor>;
    using GemmBF16 = cpu_ops::gemm::device::GemmBF16F32<layout::RowMajor, layout::RowMajor,
                                                        layout::RowMajor>;
    std::printf("\nf16 -> f32 GFLOPS:\n");
    std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
    std::printf("-------------+----------------------------------------\n");
    for (int s : {1024, 2048}) {
      std::printf("%4d^3 GFLOPS |", s);
      for (int t : threads) {
        std::printf(" %8.1f",
                    bench_cpu_ops_half<cpu_ops::float16_t, GemmF16>(s, s, s, t, 3));
        std::fflush(stdout);
      }
      std::printf("\n");
    }
    std::printf("\nbf16 -> f32 GFLOPS (native vdpbf16ps on AVX512-BF16):\n");
    std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
    std::printf("-------------+----------------------------------------\n");
    for (int s : {1024, 2048}) {
      std::printf("%4d^3 GFLOPS |", s);
      for (int t : threads) {
        std::printf(" %8.1f",
                    bench_cpu_ops_half<cpu_ops::bfloat16_t, GemmBF16>(s, s, s, t, 3));
        std::fflush(stdout);
      }
      std::printf("\n");
    }
  }

  std::printf("\nmxfp8 e4m3 -> f32 GFLOPS (decode at pack, f32 FMA mainloop):\n");
  std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
  std::printf("-------------+----------------------------------------\n");
  for (int s : {1024, 2048}) {
    std::printf("%4d^3 GFLOPS |", s);
    for (int t : threads) {
      std::printf(" %8.1f", bench_cpu_ops_mx<cpu_ops::fp8e4m3_t>(s, s, s, t, 3));
      std::fflush(stdout);
    }
    std::printf("\n");
  }

  std::printf("\nmxfp8 e5m2 -> f32 GFLOPS (decode at pack, f32 FMA mainloop):\n");
  std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
  std::printf("-------------+----------------------------------------\n");
  for (int s : {1024, 2048}) {
    std::printf("%4d^3 GFLOPS |", s);
    for (int t : threads) {
      std::printf(" %8.1f", bench_cpu_ops_mx<cpu_ops::fp8e5m2_t>(s, s, s, t, 3));
      std::fflush(stdout);
    }
    std::printf("\n");
  }

  std::printf("\nmxfp4 e2m1 -> f32 GFLOPS (decode at pack, f32 FMA mainloop):\n");
  std::printf("%12s | %8s %8s %8s %8s\n", "size", "1T", "2T", "4T", "8T");
  std::printf("-------------+----------------------------------------\n");
  for (int s : {1024, 2048}) {
    std::printf("%4d^3 GFLOPS |", s);
    for (int t : threads) {
      std::printf(" %8.1f", bench_cpu_ops_mx<cpu_ops::fp4e2m1_t>(s, s, s, t, 3));
      std::fflush(stdout);
    }
    std::printf("\n");
  }
  return 0;
}
