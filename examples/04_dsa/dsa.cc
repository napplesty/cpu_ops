// Tests for DsaAttention (DeepSeek-V3.2 sparse attention over the MLA
// latent cache). The reference recomputes the pipeline in double — indexer,
// selection, CLS, absorbed attention, un-absorb — with one spec anchor: the
// selected set and the CLS weights are *defined* on the operator's f32
// indexer-score mirror (top-k boundaries are a property of the deployed f32
// scores, not of a double re-derivation), so the reference selects and
// soft-maxes on the mirror while everything downstream runs in double. The
// mirror itself is separately checked against a double recomputation of the
// indexer formula (wiring check). Covers chunked/decode/non-causal shapes,
// k > prefix clamping, use_cls on/off, narrow storage, the bit-identical
// across-thread-counts contract, and error paths.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "../03_mla/mla.h"
#include "dsa.h"

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

struct Problem {
  const char* name;
  int b, h, sq, skv, dn, dp, dc;  // MLA dims
  int hi, di, topk;               // indexer dims / selection width
  bool causal, use_cls;
};

template <typename T>
struct DsaTensors {
  std::vector<T> qn, qp, kv, uk, uv, qi, wm, ki, o;
  std::vector<float> mirror;  // [b][sq][skv] indexer score mirror
  typename cpu_ops::attention::DsaAttention<T>::Arguments args;

  void init(const Problem& pr, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int dcat = pr.dc + pr.dp;
    const auto gen = [&](std::vector<T>& v, std::size_t n, float s) {
      v.resize(n);
      for (auto& x : v) x = T(dist(rng) * s);
    };
    gen(qn, (std::size_t)pr.b * pr.h * pr.sq * pr.dn, 1.0f);
    gen(qp, (std::size_t)pr.b * pr.h * pr.sq * pr.dp, 1.0f);
    gen(kv, (std::size_t)pr.b * pr.skv * dcat, 1.0f);
    gen(uk, (std::size_t)pr.h * pr.dc * pr.dn, 0.2f);
    gen(uv, (std::size_t)pr.h * pr.dn * pr.dc, 0.2f);
    gen(qi, (std::size_t)pr.b * pr.sq * pr.hi * pr.di, 1.0f);
    gen(wm, (std::size_t)pr.b * pr.sq * pr.hi, 1.0f);  // signed mixing weights
    gen(ki, (std::size_t)pr.b * pr.skv * pr.di, 1.0f);
    o.assign((std::size_t)pr.b * pr.h * pr.sq * pr.dn, T(0.0f));
    mirror.assign((std::size_t)pr.b * pr.sq * pr.skv, 0.0f);

    args.batch = pr.b;
    args.heads = pr.h;
    args.seq_q = pr.sq;
    args.seq_kv = pr.skv;
    args.dim_nope = pr.dn;
    args.dim_pe = pr.dp;
    args.dim_latent = pr.dc;
    args.heads_idx = pr.hi;
    args.dim_idx = pr.di;
    args.topk = pr.topk;
    args.q_nope = qn.data();
    args.q_nope_stride_b = pr.h * pr.sq * pr.dn;
    args.q_nope_stride_h = pr.sq * pr.dn;
    args.q_nope_ld = pr.dn;
    args.q_pe = qp.data();
    args.q_pe_stride_b = pr.h * pr.sq * pr.dp;
    args.q_pe_stride_h = pr.sq * pr.dp;
    args.q_pe_ld = pr.dp;
    args.kv_cache = kv.data();
    args.kv_stride_b = pr.skv * dcat;
    args.kv_ld = dcat;
    args.w_uk = uk.data();
    args.w_uk_stride_h = pr.dc * pr.dn;
    args.w_uv = uv.data();
    args.w_uv_stride_h = pr.dn * pr.dc;
    args.q_idx = qi.data();
    args.w_mix = wm.data();
    args.k_idx = ki.data();
    args.indexer_scores_out = mirror.data();
    args.o = o.data();
    args.o_stride_b = pr.h * pr.sq * pr.dn;
    args.o_stride_h = pr.sq * pr.dn;
    args.o_ld = pr.dn;
    args.scale = 1.0f / std::sqrt((float)(pr.dn + pr.dp));
    args.indexer_scale = 1.0f / std::sqrt((float)pr.di);
    args.causal = pr.causal;
    args.use_cls = pr.use_cls;
  }
};

// Double reference on already-rounded operands; selection and the CLS
// softmax consume the operator's f32 score mirror (see file comment).
void ref_dsa(const Problem& pr, const std::vector<float>& qn,
             const std::vector<float>& qp, const std::vector<float>& kv,
             const std::vector<float>& uk, const std::vector<float>& uv,
             const std::vector<float>& qi, const std::vector<float>& wm,
             const std::vector<float>& ki, const std::vector<float>& mirror,
             float scale, float indexer_scale, std::vector<float>& o,
             double* score_wiring_err) {
  const int dcat = pr.dc + pr.dp;
  std::vector<double> sc(pr.skv), wc(pr.skv), row_vals(dcat);
  std::vector<int> id(pr.skv);
  *score_wiring_err = 0.0;

  for (int bi = 0; bi < pr.b; ++bi) {
    for (int t = 0; t < pr.sq; ++t) {
      const int n_i = pr.causal ? pr.skv - pr.sq + t + 1 : pr.skv;
      const float* mir = mirror.data() + ((std::size_t)bi * pr.sq + t) * pr.skv;
      const float* wm_t = wm.data() + ((std::size_t)bi * pr.sq + t) * pr.hi;
      const float* qi_t = qi.data() + ((std::size_t)bi * pr.sq + t) * pr.hi * pr.di;
      const float* ki_b = ki.data() + (std::size_t)bi * pr.skv * pr.di;

      for (int s = 0; s < n_i; ++s) {
        double d = 0;
        for (int j = 0; j < pr.hi; ++j) {
          double dot = 0;
          const float* qj = qi_t + (std::size_t)j * pr.di;
          const float* ks = ki_b + (std::size_t)s * pr.di;
          for (int x = 0; x < pr.di; ++x) dot += (double)qj[x] * ks[x];
          d += (double)wm_t[j] * std::max(0.0, (double)indexer_scale * dot);
        }
        sc[s] = (double)mir[s];  // spec: selection/CLS on the f32 scores
        *score_wiring_err = std::max(*score_wiring_err, std::fabs(d - (double)mir[s]));
      }

      // Exact top-k with the operator's tie rule, then ascending order.
      const int k_eff = std::min(pr.topk, n_i);
      for (int s = 0; s < n_i; ++s) id[s] = s;
      const auto by_score = [&sc](int x, int y) {
        return sc[x] != sc[y] ? sc[x] > sc[y] : x < y;
      };
      std::partial_sort(id.data(), id.data() + k_eff, id.data() + n_i, by_score);
      std::sort(id.data(), id.data() + k_eff);
      const int nsel = k_eff + (pr.use_cls ? 1 : 0);

      // CLS row: softmax(mirror) weighted sum of the whole prefix.
      std::vector<double> cls(dcat, 0.0);
      if (pr.use_cls) {
        double m = sc[0];
        for (int s = 1; s < n_i; ++s) m = std::max(m, sc[s]);
        double denom = 0;
        for (int s = 0; s < n_i; ++s) {
          wc[s] = std::exp(sc[s] - m);
          denom += wc[s];
        }
        const float* kvb = kv.data() + (std::size_t)bi * pr.skv * dcat;
        for (int s = 0; s < n_i; ++s) {
          const float* row = kvb + (std::size_t)s * dcat;
          for (int d = 0; d < dcat; ++d) cls[d] += wc[s] * (double)row[d];
        }
        for (int d = 0; d < dcat; ++d) cls[d] /= denom;
      }

      const float* kvb = kv.data() + (std::size_t)bi * pr.skv * dcat;
      for (int h = 0; h < pr.h; ++h) {
        const float* qn_t =
            qn.data() + ((std::size_t)bi * pr.h + h) * pr.sq * pr.dn + (std::size_t)t * pr.dn;
        const float* qp_t =
            qp.data() + ((std::size_t)bi * pr.h + h) * pr.sq * pr.dp + (std::size_t)t * pr.dp;
        const float* ukh = uk.data() + (std::size_t)h * pr.dc * pr.dn;
        const float* uvh = uv.data() + (std::size_t)h * pr.dn * pr.dc;

        std::vector<double> qt(dcat), O(pr.dc, 0.0);
        for (int c = 0; c < pr.dc; ++c) {
          double v = 0;
          for (int x = 0; x < pr.dn; ++x)
            v += (double)qn_t[x] * ukh[(std::size_t)c * pr.dn + x];
          qt[c] = v;
        }
        for (int x = 0; x < pr.dp; ++x) qt[pr.dc + x] = qp_t[x];

        double m = -1e300, l = 0;
        std::vector<double> srow(nsel);
        for (int r2 = 0; r2 < nsel; ++r2) {
          const float* row = r2 < k_eff ? kvb + (std::size_t)id[r2] * dcat : nullptr;
          if (row) {
            for (int d = 0; d < dcat; ++d) row_vals[d] = row[d];
          } else {
            for (int d = 0; d < dcat; ++d) row_vals[d] = cls[d];
          }
          double s = 0;
          for (int d = 0; d < dcat; ++d) s += qt[d] * row_vals[d];
          srow[r2] = s * scale;
          m = std::max(m, srow[r2]);
        }
        for (int r2 = 0; r2 < nsel; ++r2) {
          const float* row = r2 < k_eff ? kvb + (std::size_t)id[r2] * dcat : nullptr;
          if (row) {
            for (int d = 0; d < dcat; ++d) row_vals[d] = row[d];
          } else {
            for (int d = 0; d < dcat; ++d) row_vals[d] = cls[d];
          }
          const double p = std::exp(srow[r2] - m);
          l += p;
          for (int d = 0; d < pr.dc; ++d) O[d] += p * row_vals[d];
        }

        float* oi =
            o.data() + ((std::size_t)bi * pr.h + h) * pr.sq * pr.dn + (std::size_t)t * pr.dn;
        for (int e = 0; e < pr.dn; ++e) {
          double v = 0;
          for (int d = 0; d < pr.dc; ++d)
            v += (O[d] / l) * (double)uvh[(std::size_t)e * pr.dc + d];
          oi[e] = (float)v;
        }
      }
    }
  }
}

template <typename T>
void run_case(const Problem& pr, int threads, double tol, const char* tag) {
  ++g_cases;
  DsaTensors<T> ten;
  ten.init(pr, 4321);

  cpu_ops::attention::DsaAttention<T> dsa;
  const cpu_ops::Status st = dsa(ten.args, threads);
  CHECK(st == cpu_ops::Status::kSuccess, "%s %s threads=%d: status %d", pr.name, tag,
        threads, (int)st);

  // Reference sees the same storage-rounded values.
  const auto round_back = [](std::vector<float>& f, const std::vector<T>& t) {
    f.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) f[i] = static_cast<float>(t[i]);
  };
  std::vector<float> qn_f, qp_f, kv_f, uk_f, uv_f, qi_f, wm_f, ki_f;
  round_back(qn_f, ten.qn);
  round_back(qp_f, ten.qp);
  round_back(kv_f, ten.kv);
  round_back(uk_f, ten.uk);
  round_back(uv_f, ten.uv);
  round_back(qi_f, ten.qi);
  round_back(wm_f, ten.wm);
  round_back(ki_f, ten.ki);

  std::vector<float> ref(ten.o.size());
  double wiring_err = 0;
  ref_dsa(pr, qn_f, qp_f, kv_f, uk_f, uv_f, qi_f, wm_f, ki_f, ten.mirror,
          ten.args.scale, ten.args.indexer_scale, ref, &wiring_err);
  CHECK(wiring_err <= 1e-3, "%s %s threads=%d: indexer mirror err %g", pr.name, tag,
        threads, wiring_err);

  double max_err = 0;
  for (std::size_t i = 0; i < ten.o.size(); ++i) {
    const double got = static_cast<double>(static_cast<float>(ten.o[i]));
    max_err = std::max(max_err, std::fabs(got - ref[i]));
  }
  CHECK(max_err <= tol, "%s %s threads=%d: max_err %g > %g", pr.name, tag, threads,
        max_err, tol);
}

template <typename T>
void run_stored(const Problem& pr, int threads, std::vector<T>* out,
                std::vector<float>* mirror) {
  DsaTensors<T> ten;
  ten.init(pr, 4321);
  cpu_ops::attention::DsaAttention<T> dsa;
  CHECK(dsa(ten.args, threads) == cpu_ops::Status::kSuccess, "run_stored status");
  *out = std::move(ten.o);
  *mirror = std::move(ten.mirror);
}

void check_error_paths() {
  ++g_cases;
  using Dsa = cpu_ops::attention::DsaAttention<float>;
  float buf[256] = {0}, obuf[16] = {0};
  Problem pr{"err", 1, 2, 4, 4, 8, 4, 16, 2, 8, 4, true, true};
  DsaTensors<float> ten;
  ten.init(pr, 7);
  Dsa dsa;

  typename Dsa::Arguments a = ten.args;
  CHECK(dsa(a) == cpu_ops::Status::kSuccess, "valid args rejected");

  a = ten.args;
  a.topk = 0;
  CHECK(dsa(a) == cpu_ops::Status::kErrorInvalidProblem, "topk=0 accepted");
  a = ten.args;
  a.heads_idx = 0;
  CHECK(dsa(a) == cpu_ops::Status::kErrorInvalidProblem, "heads_idx=0 accepted");
  a = ten.args;
  a.k_idx = nullptr;
  CHECK(dsa(a) == cpu_ops::Status::kErrorInvalidArguments, "null k_idx accepted");
  a = ten.args;
  a.seq_kv = 2;  // causal with skv < sq
  CHECK(dsa(a) == cpu_ops::Status::kErrorInvalidArguments, "causal skv < sq accepted");
  a = ten.args;
  a.kv_ld = 19;  // < dc + dp = 20
  CHECK(dsa(a) == cpu_ops::Status::kErrorInvalidArguments, "kv_ld < dc+dp accepted");
  (void)buf;
  (void)obuf;
}

}  // namespace

// ---------------------------------------------------------------------------
// Benchmark (opt-in via --bench): DeepSeek-shaped dims (dn=128, dp=64,
// dc=512 shared latent; indexer hi=64 x di=128; topk=2048), threads <= 4.
// DSA decode is compared against dense MLA on the same cache.
// ---------------------------------------------------------------------------

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
double bench_dsa(int h, int sq, int skv, int hi, int di, int topk, int threads,
                 int reps) {
  const Problem pr{"bench", 1, h, sq, skv, 128, 64, 512, hi, di, topk, true, true};
  DsaTensors<T> ten;
  ten.init(pr, 11);
  cpu_ops::attention::DsaAttention<T> dsa;
  dsa(ten.args, threads);  // warmup
  return time_best([&] { dsa(ten.args, threads); }, reps) * 1e3;
}

template <typename T>
double bench_mla(int h, int sq, int skv, int threads, int reps) {
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int dn = 128, dp = 64, dc = 512, dcat = dc + dp;
  std::vector<T> qn((std::size_t)h * sq * dn), qp((std::size_t)h * sq * dp),
      kv((std::size_t)skv * dcat), uk((std::size_t)h * dc * dn),
      uv((std::size_t)h * dn * dc), o((std::size_t)h * sq * dn);
  for (auto& x : qn) x = T(dist(rng));
  for (auto& x : qp) x = T(dist(rng));
  for (auto& x : kv) x = T(dist(rng));
  for (auto& x : uk) x = T(dist(rng) * 0.2f);
  for (auto& x : uv) x = T(dist(rng) * 0.2f);

  typename cpu_ops::attention::MlaAttention<T>::Arguments args;
  args.batch = 1;
  args.heads = h;
  args.seq_q = sq;
  args.seq_kv = skv;
  args.dim_nope = dn;
  args.dim_pe = dp;
  args.dim_latent = dc;
  args.q_nope = qn.data();
  args.q_nope_stride_h = sq * dn;
  args.q_nope_ld = dn;
  args.q_pe = qp.data();
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
  args.o_stride_h = sq * dn;
  args.o_ld = dn;
  args.scale = 1.0f / std::sqrt((float)(dn + dp));
  args.causal = true;

  cpu_ops::attention::MlaAttention<T> mla;
  mla(args, threads);  // warmup
  return time_best([&] { mla(args, threads); }, reps) * 1e3;
}

void run_benchmark() {
  using bf16_t = cpu_ops::bfloat16_t;
  const int H = 32, DN = 128, DP = 64, DC = 512, HI = 64, DI = 128, TOPK = 2048;
  const int dcat = DC + DP;

  std::printf("\nDSA decode vs dense MLA, seq_q=1, causal, B=1 H=%d\n", H);
  std::printf("dims: dn=%d dp=%d dc=%d, indexer %dx%d, topk=%d, bf16 cache\n", DN, DP,
              DC, HI, DI, TOPK);
  std::printf("%10s | %12s | %10s %10s | %8s %10s\n", "kv len", "MLA 4T ms",
              "DSA 1T ms", "DSA 4T ms", "speedup", "GB/s eff");
  for (int skv : {8192, 32768, 65536}) {
    const double tmla = bench_mla<bf16_t>(H, 1, skv, 4, 5);
    const double t1 = bench_dsa<bf16_t>(H, 1, skv, HI, DI, TOPK, 1, 5);
    const double t4 = bench_dsa<bf16_t>(H, 1, skv, HI, DI, TOPK, 4, 5);
    // bytes the operator must touch per token: indexer key stream (hi
    // passes), one CLS sweep of the latent cache, the gathered panel, and
    // the per-head absorb weights.
    const double bytes = (double)HI * skv * DI * 2 + (double)skv * dcat * 2 +
                         (double)(TOPK + 1) * dcat * 2 +
                         (double)H * (DC * DN + DN * DC) * 2;
    std::printf("%10d | %12.1f | %10.1f %10.1f | %8.2f %10.1f\n", skv, tmla, t1, t4,
                tmla / t4, bytes / (t4 * 1e-3) / 1e9);
    std::fflush(stdout);
  }

  std::printf("\nDSA prefill (chunked continuation), causal, B=1 H=%d, topk=%d:\n", H,
              TOPK);
  std::printf("%10s %10s | %12s | %10s %10s | %8s\n", "seq_q", "kv len", "MLA 4T ms",
              "DSA 1T ms", "DSA 4T ms", "speedup");
  const int sq = 512, skv = 8192;
  const double tmla = bench_mla<bf16_t>(H, sq, skv, 4, 2);
  const double t1 = bench_dsa<bf16_t>(H, sq, skv, HI, DI, TOPK, 1, 2);
  const double t4 = bench_dsa<bf16_t>(H, sq, skv, HI, DI, TOPK, 4, 2);
  std::printf("%10d %10d | %12.1f | %10.1f %10.1f | %8.2f\n", sq, skv, tmla, t1, t4,
              tmla / t4);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;
  using f16_t = float16_t;
  using bf16_t = bfloat16_t;

  const Problem probs[] = {
      {"dsa_causal", 1, 4, 17, 131, 32, 16, 64, 3, 48, 37, true, true},
      {"dsa_chunked", 1, 4, 13, 100, 24, 8, 48, 2, 32, 64, true, true},
      {"dsa_full_b2", 2, 2, 40, 40, 32, 16, 64, 2, 32, 128, false, true},
      {"dsa_decode", 1, 4, 1, 300, 16, 8, 32, 3, 24, 128, true, true},
      {"dsa_nocls", 1, 3, 20, 77, 24, 8, 48, 2, 32, 25, true, false},
  };

  for (const Problem& pr : probs) {
    for (int threads : {1, 4}) {
      // Tolerances mirror the MLA example: f32 covers the absorbed dots;
      // narrow types add the CLS/gather storage rounding.
      run_case<float>(pr, threads, 1.5e-3, "f32");
      run_case<f16_t>(pr, threads, 8e-3, "f16");
      run_case<bf16_t>(pr, threads, 3e-2, "bf16");
    }
  }

  {  // bit-identical across thread counts, deterministic across runs
    const Problem pr{"dsa_bitwise", 1, 8, 100, 100, 32, 16, 64, 3, 32, 50, true, true};
    std::vector<float> o1, m1, o4, m4, o4b, m4b;
    run_stored<float>(pr, 1, &o1, &m1);
    run_stored<float>(pr, 4, &o4, &m4);
    run_stored<float>(pr, 4, &o4b, &m4b);
    CHECK(o1.size() == o4.size() &&
              std::memcmp(o1.data(), o4.data(), o1.size() * sizeof(float)) == 0,
          "DSA 1T vs 4T outputs differ");
    CHECK(std::memcmp(m1.data(), m4.data(), m1.size() * sizeof(float)) == 0,
          "DSA 1T vs 4T indexer mirror differs");
    CHECK(std::memcmp(o4.data(), o4b.data(), o4.size() * sizeof(float)) == 0,
          "DSA output not deterministic across runs");
  }

  check_error_paths();

  const bool want_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
  if (want_bench) run_benchmark();
  std::printf("dsa: %d cases, %d failures%s\n", g_cases, g_failures,
              want_bench ? "" : " (run with --bench for the benchmark)");
  return g_failures == 0 ? 0 : 1;
}
