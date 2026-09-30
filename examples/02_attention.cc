// Attention benchmark: prefill (compute-bound) and decode (bandwidth-bound)
// shapes on the fused kernel. Threads are capped at 4 by default to stay
// polite on shared machines.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"

namespace {

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

// Returns ms for one attention call; tensors packed [b][h][seq][d].
template <typename T>
double bench(int b, int hq, int hkv, int sq, int skv, int d, bool causal, int threads,
             int reps) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<T> q(static_cast<size_t>(b) * hq * sq * d), o(q.size());
  const size_t kv_elems = static_cast<size_t>(b) * hkv * skv * d;
  std::vector<T> k(kv_elems), v(kv_elems);
  for (auto& x : q) x = T(dist(rng));
  for (auto& x : k) x = T(dist(rng));
  for (auto& x : v) x = T(dist(rng));

  using Attn = cpu_ops::attention::Attention<T>;
  typename Attn::Arguments args;
  args.batch = b;
  args.heads_q = hq;
  args.heads_kv = hkv;
  args.seq_q = sq;
  args.seq_kv = skv;
  args.dim = d;
  args.q = q.data();
  args.q_stride_b = hq * sq * d;
  args.q_stride_h = sq * d;
  args.q_ld = d;
  args.k = k.data();
  args.k_stride_b = hkv * skv * d;
  args.k_stride_h = skv * d;
  args.k_ld = d;
  args.v = v.data();
  args.v_stride_b = hkv * skv * d;
  args.v_stride_h = skv * d;
  args.v_ld = d;
  args.o = o.data();
  args.o_stride_b = hq * sq * d;
  args.o_stride_h = sq * d;
  args.o_ld = d;
  args.scale = 1.0f / std::sqrt(static_cast<float>(d));
  args.causal = causal;

  Attn attn;
  attn(args, threads);  // warmup
  const double s = time_best([&] { attn(args, threads); }, reps);
  const volatile float sink = static_cast<float>(o[o.size() / 2]);
  (void)sink;
  return s * 1e3;
}

}  // namespace

int main() {
  std::printf("prefill, causal, B=1 H=16 d=128 f32 (best of 3):\n");
  std::printf("%10s | %10s %10s | %10s %10s\n", "seq", "1T ms", "4T ms", "1T GFLOPS",
              "4T GFLOPS");
  for (int n : {512, 1024, 2048}) {
    const double flops = 2.0 * 2.0 * n * n * 128 * 16 * 0.5;  // two GEMMs, causal half
    const double t1 = bench<float>(1, 16, 16, n, n, 128, true, 1, 3);
    const double t4 = bench<float>(1, 16, 16, n, n, 128, true, 4, 3);
    std::printf("%10d | %10.2f %10.2f | %10.1f %10.1f\n", n, t1, t4, flops / (t1 * 1e-3) / 1e9,
                flops / (t4 * 1e-3) / 1e9);
    std::fflush(stdout);
  }

  std::printf("\nprefill same shapes with bf16 operands:\n");
  for (int n : {1024, 2048}) {
    const double flops = 2.0 * 2.0 * n * n * 128 * 16 * 0.5;
    const double t4 = bench<cpu_ops::bfloat16_t>(1, 16, 16, n, n, 128, true, 4, 3);
    std::printf("%10d | %10s %10.2f | %10s %10.1f\n", n, "-", t4, "-",
                flops / (t4 * 1e-3) / 1e9);
    std::fflush(stdout);
  }

  std::printf("\ndecode, seq_q=1, GQA H=32/Hkv=8 d=128 bf16 (kv auto-split):\n");
  std::printf("%10s | %10s %10s | %10s\n", "kv len", "1T ms", "4T ms", "4T KV GB/s");
  for (int n : {4096, 16384}) {
    const double kv_bytes = 2.0 * n * 128 * 8 * 2;  // K+V, 2 bytes/elem
    const double t1 = bench<cpu_ops::bfloat16_t>(1, 32, 8, 1, n, 128, true, 1, 5);
    const double t4 = bench<cpu_ops::bfloat16_t>(1, 32, 8, 1, n, 128, true, 4, 5);
    std::printf("%10d | %10.3f %10.3f | %10.1f\n", n, t1, t4, kv_bytes / (t4 * 1e-3) / 1e9);
    std::fflush(stdout);
  }
  return 0;
}
