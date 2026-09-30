// MLA benchmark: DeepSeek-shaped (dn=128, dp=64, dc=512, shared 576-wide
// latent cache), prefill and decode, vs a plain GQA baseline for cache
// traffic. Threads capped at 4.

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

template <typename T>
double bench_mla(int b, int h, int sq, int skv, int dn, int dp, int dc, bool causal,
                 int threads, int reps) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int dcat = dc + dp;
  std::vector<T> qn((size_t)b * h * sq * dn), qp((size_t)b * h * sq * dp),
      kv((size_t)b * skv * dcat), uk((size_t)h * dc * dn), uv((size_t)h * dn * dc),
      o((size_t)b * h * sq * dn);
  for (auto& x : qn) x = T(dist(rng));
  for (auto& x : qp) x = T(dist(rng));
  for (auto& x : kv) x = T(dist(rng));
  for (auto& x : uk) x = T(dist(rng) * 0.2f);
  for (auto& x : uv) x = T(dist(rng) * 0.2f);

  using Mla = cpu_ops::attention::MlaAttention<T>;
  typename Mla::Arguments args;
  args.batch = b;
  args.heads = h;
  args.seq_q = sq;
  args.seq_kv = skv;
  args.dim_nope = dn;
  args.dim_pe = dp;
  args.dim_latent = dc;
  args.q_nope = qn.data();
  args.q_nope_stride_b = h * sq * dn;
  args.q_nope_stride_h = sq * dn;
  args.q_nope_ld = dn;
  args.q_pe = qp.data();
  args.q_pe_stride_b = h * sq * dp;
  args.q_pe_stride_h = sq * dp;
  args.q_pe_ld = dp;
  args.kv_cache = kv.data();
  args.kv_stride_b = skv * dcat;
  args.kv_ld = dcat;
  args.w_uk = uk.data();
  args.w_uk_stride_h = dc * dn;
  args.w_uv = uv.data();
  args.w_uv_stride_h = dn * dc;
  args.o = o.data();
  args.o_stride_b = h * sq * dn;
  args.o_stride_h = sq * dn;
  args.o_ld = dn;
  args.scale = 1.0f / std::sqrt((float)(dn + dp));
  args.causal = causal;

  Mla mla;
  mla(args, threads);  // warmup
  const double s = time_best([&] { mla(args, threads); }, reps);
  const volatile float sink = static_cast<float>(o[o.size() / 2]);
  (void)sink;
  return s * 1e3;
}

}  // namespace

int main() {
  const int H = 32, DN = 128, DP = 64, DC = 512;

  std::printf("MLA prefill, causal, B=1 H=%d dn=%d dp=%d dc=%d (best of 3):\n", H, DN, DP,
              DC);
  std::printf("%10s | %10s %10s | %10s %10s\n", "seq", "f32 1T ms", "f32 4T ms",
              "bf16 4T ms", "4T GFLOPS");
  for (int sq : {1024, 2048}) {
    const double t1 = bench_mla<float>(1, H, sq, sq, DN, DP, DC, true, 1, 3);
    const double t4 = bench_mla<float>(1, H, sq, sq, DN, DP, DC, true, 4, 3);
    const double tb = bench_mla<cpu_ops::bfloat16_t>(1, H, sq, sq, DN, DP, DC, true, 4, 3);
    // Core: H·sq²·((dc+dp)+dc)·2·2 ops, causal half; absorb: 2·H·sq·dn·dc·2.
    const double flops = (double)H * sq * sq * ((double)DC + DP + DC) * 2.0 * 0.5 +
                         2.0 * H * sq * DN * DC * 2.0;
    std::printf("%10d | %10.1f %10.1f | %10.1f %10.1f\n", sq, t1, t4, tb,
                flops / (t4 * 1e-3) / 1e9);
    std::fflush(stdout);
  }

  std::printf("\nMLA decode, seq_q=1, bf16 cache (bytes/token = (dc+dp)*2 = %d B):\n",
              (DC + DP) * 2);
  std::printf("%10s | %10s %10s | %10s\n", "kv len", "1T ms", "4T ms", "4T cache GB/s");
  for (int skv : {4096, 16384}) {
    const double t1 =
        bench_mla<cpu_ops::bfloat16_t>(1, H, 1, skv, DN, DP, DC, true, 1, 10);
    const double t4 =
        bench_mla<cpu_ops::bfloat16_t>(1, H, 1, skv, DN, DP, DC, true, 4, 10);
    const double bytes = (double)skv * (DC + DP) * 2;
    std::printf("%10d | %10.3f %10.3f | %10.1f\n", skv, t1, t4, bytes / (t4 * 1e-3) / 1e9);
    std::fflush(stdout);
  }
  return 0;
}
