// Tests for CsaAttention (NSA-style three-branch compressed sparse
// attention over the MLA latent cache). The reference recomputes the whole
// pipeline in double — centroids, per-head compressed softmax, head-summed
// block scores, selection with the fixed-activation rule, gathered /
// window attentions, sigmoid gates, un-absorb — with one spec anchor: the
// selected block set is *defined* on the operator's f32 block-score mirror
// (block_scores_out), which is itself checked against the double head-sum
// (wiring). Covers causal chunked/decode/non-causal shapes, empty-prefix
// rows (no completed block yet), n < fixed-count budgets, the precomputed
// compressed-cache input, narrow storage, bit-identical outputs across
// thread counts, and error paths.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "csa.h"

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
  int block, nsel, window;        // NSA knobs
  bool causal;
  bool compressed_input;  // feed kv_compressed instead of on-the-fly means
};

template <typename T>
struct CsaTensors {
  std::vector<T> qn, qp, kv, uk, uv, kc, gl, o;
  std::vector<float> mirror;  // [b][sq][nblk] head-summed block scores
  typename cpu_ops::attention::CsaAttention<T>::Arguments args;

  void init(const Problem& pr, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int dcat = pr.dc + pr.dp;
    const int nblk = (pr.skv + pr.block - 1) / pr.block;
    const auto gen = [&](std::vector<T>& v, std::size_t n, float s) {
      v.resize(n);
      for (auto& x : v) x = T(dist(rng) * s);
    };
    gen(qn, (std::size_t)pr.b * pr.h * pr.sq * pr.dn, 1.0f);
    gen(qp, (std::size_t)pr.b * pr.h * pr.sq * pr.dp, 1.0f);
    gen(kv, (std::size_t)pr.b * pr.skv * dcat, 1.0f);
    gen(uk, (std::size_t)pr.h * pr.dc * pr.dn, 0.2f);
    gen(uv, (std::size_t)pr.h * pr.dn * pr.dc, 0.2f);
    gen(gl, (std::size_t)pr.b * pr.h * pr.sq * 3, 2.0f);  // signed logits
    if (pr.compressed_input) {
      gen(kc, (std::size_t)pr.b * nblk * dcat, 1.0f);
    }
    o.assign((std::size_t)pr.b * pr.h * pr.sq * pr.dn, T(0.0f));
    mirror.assign((std::size_t)pr.b * pr.sq * nblk, 0.0f);

    args.batch = pr.b;
    args.heads = pr.h;
    args.seq_q = pr.sq;
    args.seq_kv = pr.skv;
    args.dim_nope = pr.dn;
    args.dim_pe = pr.dp;
    args.dim_latent = pr.dc;
    args.block = pr.block;
    args.select_blocks = pr.nsel;
    args.window = pr.window;
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
    args.kv_compressed = pr.compressed_input ? kc.data() : nullptr;
    args.compressed_stride_b = nblk * dcat;
    args.compressed_ld = dcat;
    args.gate_logits = gl.data();
    args.block_scores_out = mirror.data();
    args.o = o.data();
    args.o_stride_b = pr.h * pr.sq * pr.dn;
    args.o_stride_h = pr.sq * pr.dn;
    args.o_ld = pr.dn;
    args.scale = 1.0f / std::sqrt((float)(pr.dn + pr.dp));
    args.causal = pr.causal;
  }
};

// Double reference; selection consumes the operator's f32 block-score
// mirror (see file comment).
void ref_csa(const Problem& pr, const std::vector<float>& qn,
             const std::vector<float>& qp, const std::vector<float>& kv,
             const std::vector<float>& uk, const std::vector<float>& uv,
             const std::vector<float>& kc, const std::vector<float>& gl,
             const std::vector<float>& mirror, float scale, std::vector<float>& o,
             double* wiring_err) {
  const int dcat = pr.dc + pr.dp;
  const int nblk = (pr.skv + pr.block - 1) / pr.block;
  const int g = pr.block, kv_off = pr.skv - pr.sq;
  std::vector<double> qt(dcat), bs(nblk), row(dcat);
  *wiring_err = 0.0;

  for (int bi = 0; bi < pr.b; ++bi) {
    // Centroids: input cache if provided, else block means.
    std::vector<double> cent((std::size_t)nblk * dcat);
    const float* kvb = kv.data() + (std::size_t)bi * pr.skv * dcat;
    for (int l = 0; l < nblk; ++l) {
      double* out = cent.data() + (std::size_t)l * dcat;
      if (pr.compressed_input) {
        const float* src = kc.data() + ((std::size_t)bi * nblk + l) * dcat;
        for (int d = 0; d < dcat; ++d) out[d] = src[d];
      } else {
        const int s0 = l * g;
        const int cnt = std::min(g, pr.skv - s0);
        for (int d = 0; d < dcat; ++d) out[d] = 0;
        for (int s = 0; s < cnt; ++s)
          for (int d = 0; d < dcat; ++d) out[d] += kvb[(std::size_t)(s0 + s) * dcat + d];
        for (int d = 0; d < dcat; ++d) out[d] /= cnt;
      }
    }

    for (int t = 0; t < pr.sq; ++t) {
      const int p_t = pr.causal ? kv_off + t : pr.skv - 1;
      const int n_elig = pr.causal ? (p_t + 1) / g : nblk;
      const float* mir = mirror.data() + ((std::size_t)bi * pr.sq + t) * nblk;

      // Per head: q̃, compressed scores, softmax; head-sum into bs (double)
      // for the wiring check only — selection reads the mirror.
      std::vector<std::vector<double>> oc(pr.h, std::vector<double>(pr.dc, 0.0));
      for (int d = 0; d < nblk; ++d) bs[d] = 0.0;
      std::vector<double> s_h(nblk);
      for (int h = 0; h < pr.h; ++h) {
        const float* qn_t = qn.data() +
                            ((std::size_t)bi * pr.h + h) * pr.sq * pr.dn + (std::size_t)t * pr.dn;
        const float* qp_t = qp.data() +
                            ((std::size_t)bi * pr.h + h) * pr.sq * pr.dp + (std::size_t)t * pr.dp;
        const float* ukh = uk.data() + (std::size_t)h * pr.dc * pr.dn;
        for (int c = 0; c < pr.dc; ++c) {
          double v = 0;
          for (int x = 0; x < pr.dn; ++x)
            v += (double)qn_t[x] * ukh[(std::size_t)c * pr.dn + x];
          qt[c] = v;
        }
        for (int x = 0; x < pr.dp; ++x) qt[pr.dc + x] = qp_t[x];
        if (n_elig == 0) continue;
        double m = -1e300;
        for (int l = 0; l < n_elig; ++l) {
          const double* cl = cent.data() + (std::size_t)l * dcat;
          double s = 0;
          for (int d = 0; d < dcat; ++d) s += qt[d] * cl[d];
          s_h[l] = s * scale;
          m = std::max(m, s_h[l]);
        }
        double l_den = 0;
        for (int l = 0; l < n_elig; ++l) {
          s_h[l] = std::exp(s_h[l] - m);
          l_den += s_h[l];
        }
        for (int l = 0; l < n_elig; ++l) {
          const double p = s_h[l] / l_den;
          bs[l] += p;
          const double* cl = cent.data() + (std::size_t)l * dcat;
          for (int d = 0; d < pr.dc; ++d) oc[h][d] += p * cl[d];
        }
      }
      for (int l = 0; l < n_elig; ++l) {
        *wiring_err = std::max(*wiring_err, std::fabs(bs[l] - (double)mir[l]));
      }

      // Selection on the mirror, fixed activations inside the budget —
      // the same rule as the operator.
      std::vector<int> sel;
      {
        const int n_take = std::min(pr.nsel, n_elig);
        if (n_elig >= 1 && (int)sel.size() < n_take) sel.push_back(0);
        if (n_elig >= 2 && (int)sel.size() < n_take) sel.push_back(n_elig - 1);
        if (n_elig >= 3 && (int)sel.size() < n_take) sel.push_back(n_elig - 2);
        std::vector<int> id(n_elig);
        for (int l = 0; l < n_elig; ++l) id[l] = l;
        const double* md = nullptr;  // comparator on the mirror as double
        std::vector<double> mird(n_elig);
        for (int l = 0; l < n_elig; ++l) mird[l] = (double)mir[l];
        md = mird.data();
        std::sort(id.begin(), id.end(), [&md](int x, int y) {
          return md[x] != md[y] ? md[x] > md[y] : x < y;
        });
        for (int i = 0; i < n_elig && (int)sel.size() < n_take; ++i) {
          const int l = id[i];
          bool dup = false;
          for (int j : sel) dup |= (j == l);
          if (!dup) sel.push_back(l);
        }
        std::sort(sel.begin(), sel.end());
      }
      int k_rows = (int)sel.size() * g;
      if (!pr.causal && !sel.empty()) {
        const int last = sel.back();
        k_rows -= g - std::min(g, pr.skv - last * g);
      }
      const int w0 = pr.causal ? std::max(0, p_t - pr.window + 1)
                               : std::max(0, pr.skv - pr.window);
      const int n_win = (pr.causal ? p_t : pr.skv - 1) - w0 + 1;

      for (int h = 0; h < pr.h; ++h) {
        // q̃ again for this head.
        const float* qn_t = qn.data() +
                            ((std::size_t)bi * pr.h + h) * pr.sq * pr.dn + (std::size_t)t * pr.dn;
        const float* qp_t = qp.data() +
                            ((std::size_t)bi * pr.h + h) * pr.sq * pr.dp + (std::size_t)t * pr.dp;
        const float* ukh = uk.data() + (std::size_t)h * pr.dc * pr.dn;
        const float* uvh = uv.data() + (std::size_t)h * pr.dn * pr.dc;
        for (int c = 0; c < pr.dc; ++c) {
          double v = 0;
          for (int x = 0; x < pr.dn; ++x)
            v += (double)qn_t[x] * ukh[(std::size_t)c * pr.dn + x];
          qt[c] = v;
        }
        for (int x = 0; x < pr.dp; ++x) qt[pr.dc + x] = qp_t[x];

        // Selected + window branch softmaxes (double), two passes each.
        const auto gather_row = [&](int r2) {
          const int s0 = sel[r2 / g] * g + r2 % g;
          for (int d = 0; d < dcat; ++d) row[d] = kvb[(std::size_t)s0 * dcat + d];
        };
        double m = -1e300;
        std::vector<double> sr(k_rows), sw(n_win);
        for (int r2 = 0; r2 < k_rows; ++r2) {
          gather_row(r2);
          double s = 0;
          for (int d = 0; d < dcat; ++d) s += qt[d] * row[d];
          sr[r2] = s * scale;
          m = std::max(m, sr[r2]);
        }
        for (int r2 = 0; r2 < n_win; ++r2) {
          const float* wrow = kvb + (std::size_t)(w0 + r2) * dcat;
          double s = 0;
          for (int d = 0; d < dcat; ++d) s += qt[d] * (double)wrow[d];
          sw[r2] = s * scale;
          m = std::max(m, sw[r2]);
        }
        std::vector<double> Os(pr.dc, 0.0), Ow(pr.dc, 0.0), ot(pr.dc, 0.0);
        double l_s = 0, l_w = 0;
        for (int r2 = 0; r2 < k_rows; ++r2) {
          gather_row(r2);
          const double p = std::exp(sr[r2] - m);
          l_s += p;
          for (int d = 0; d < pr.dc; ++d) Os[d] += p * row[d];
        }
        for (int r2 = 0; r2 < n_win; ++r2) {
          const float* wrow = kvb + (std::size_t)(w0 + r2) * dcat;
          const double p = std::exp(sw[r2] - m);
          l_w += p;
          for (int d = 0; d < pr.dc; ++d) Ow[d] += p * (double)wrow[d];
        }

        const float* glh =
            gl.data() + (((std::size_t)bi * pr.h + h) * pr.sq + t) * 3;
        const double gc = 1.0 / (1.0 + std::exp(-(double)glh[0]));
        const double gs = 1.0 / (1.0 + std::exp(-(double)glh[1]));
        const double gw = 1.0 / (1.0 + std::exp(-(double)glh[2]));
        for (int d = 0; d < pr.dc; ++d) {
          ot[d] = gc * oc[h][d] + gs * (l_s > 0 ? Os[d] / l_s : 0.0) +
                  gw * (l_w > 0 ? Ow[d] / l_w : 0.0);
        }
        float* oi =
            o.data() + ((std::size_t)bi * pr.h + h) * pr.sq * pr.dn + (std::size_t)t * pr.dn;
        for (int e = 0; e < pr.dn; ++e) {
          double v = 0;
          for (int d = 0; d < pr.dc; ++d)
            v += ot[d] * (double)uvh[(std::size_t)e * pr.dc + d];
          oi[e] = (float)v;
        }
      }
    }
  }
}

template <typename T>
void run_case(const Problem& pr, int threads, double tol, const char* tag) {
  ++g_cases;
  CsaTensors<T> ten;
  ten.init(pr, 4321);

  cpu_ops::attention::CsaAttention<T> csa;
  const cpu_ops::Status st = csa(ten.args, threads);
  CHECK(st == cpu_ops::Status::kSuccess, "%s %s threads=%d: status %d", pr.name, tag,
        threads, (int)st);

  const auto round_back = [](std::vector<float>& f, const std::vector<T>& t) {
    f.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) f[i] = static_cast<float>(t[i]);
  };
  std::vector<float> qn_f, qp_f, kv_f, uk_f, uv_f, kc_f, gl_f;
  round_back(qn_f, ten.qn);
  round_back(qp_f, ten.qp);
  round_back(kv_f, ten.kv);
  round_back(uk_f, ten.uk);
  round_back(uv_f, ten.uv);
  if (pr.compressed_input) round_back(kc_f, ten.kc);
  round_back(gl_f, ten.gl);

  std::vector<float> ref(ten.o.size());
  double wiring_err = 0;
  ref_csa(pr, qn_f, qp_f, kv_f, uk_f, uv_f, kc_f, gl_f, ten.mirror, ten.args.scale,
          ref, &wiring_err);
  CHECK(wiring_err <= 5e-3, "%s %s threads=%d: block-score mirror err %g", pr.name, tag,
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
  CsaTensors<T> ten;
  ten.init(pr, 4321);
  cpu_ops::attention::CsaAttention<T> csa;
  CHECK(csa(ten.args, threads) == cpu_ops::Status::kSuccess, "run_stored status");
  *out = std::move(ten.o);
  *mirror = std::move(ten.mirror);
}

void check_error_paths() {
  ++g_cases;
  using Csa = cpu_ops::attention::CsaAttention<float>;
  Problem pr{"err", 1, 2, 4, 40, 8, 4, 16, 8, 3, 16, true, false};
  CsaTensors<float> ten;
  ten.init(pr, 7);
  Csa csa;

  CHECK(csa(ten.args) == cpu_ops::Status::kSuccess, "valid args rejected");
  typename Csa::Arguments a = ten.args;
  a.gate_logits = nullptr;
  CHECK(csa(a) == cpu_ops::Status::kErrorInvalidArguments, "null gate_logits accepted");
  a = ten.args;
  a.block = 0;
  CHECK(csa(a) == cpu_ops::Status::kErrorInvalidProblem, "block=0 accepted");
  a = ten.args;
  a.select_blocks = 0;
  CHECK(csa(a) == cpu_ops::Status::kErrorInvalidProblem, "select_blocks=0 accepted");
  a = ten.args;
  a.window = 0;
  CHECK(csa(a) == cpu_ops::Status::kErrorInvalidProblem, "window=0 accepted");
  a = ten.args;
  a.seq_kv = 2;  // causal with skv < sq
  CHECK(csa(a) == cpu_ops::Status::kErrorInvalidArguments, "causal skv < sq accepted");
  a = ten.args;
  a.kv_compressed = ten.kv.data();  // non-null, exercise the ld check
  a.compressed_ld = 19;             // < dc + dp = 20
  CHECK(csa(a) == cpu_ops::Status::kErrorInvalidArguments,
        "compressed_ld < dc+dp accepted");
}

}  // namespace

// ---------------------------------------------------------------------------
// Benchmark (opt-in via --bench). Functionality-first example: the numbers
// are reference points, not tuned results (see the DSA example for the
// perf-focused decode discussion).
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
double bench_csa(int h, int sq, int skv, int block, int nsel, int window, int threads,
                 int reps) {
  Problem pr{"bench", 1, h, sq, skv, 128, 64, 512, block, nsel, window, true, false};
  CsaTensors<T> ten;
  ten.init(pr, 11);
  cpu_ops::attention::CsaAttention<T> csa;
  csa(ten.args, threads);  // warmup
  return time_best([&] { csa(ten.args, threads); }, reps) * 1e3;
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
  mla(args, threads);
  return time_best([&] { mla(args, threads); }, reps) * 1e3;
}

void run_benchmark() {
  using bf16_t = cpu_ops::bfloat16_t;
  const int H = 32, BLOCK = 64, NSEL = 16, W = 512;  // NSA defaults

  std::printf("\nCSA (NSA: block=%d, top-%d blocks, window=%d) vs dense MLA,\n"
              "seq_q=1 causal, B=1 H=%d, dn/dp/dc=128/64/512, bf16:\n",
              BLOCK, NSEL, W, H);
  std::printf("%10s | %12s | %10s %10s | %8s\n", "kv len", "MLA 4T ms", "CSA 1T ms",
              "CSA 4T ms", "speedup");
  for (int skv : {8192, 32768, 65536}) {
    const double tmla = bench_mla<bf16_t>(H, 1, skv, 4, 5);
    const double t1 = bench_csa<bf16_t>(H, 1, skv, BLOCK, NSEL, W, 1, 5);
    const double t4 = bench_csa<bf16_t>(H, 1, skv, BLOCK, NSEL, W, 4, 5);
    std::printf("%10d | %12.1f | %10.1f %10.1f | %8.2f\n", skv, tmla, t1, t4, tmla / t4);
    std::fflush(stdout);
  }

  std::printf("\nCSA prefill (chunked continuation 512x8192):\n");
  const double tmla = bench_mla<bf16_t>(H, 512, 8192, 4, 2);
  const double t1 = bench_csa<bf16_t>(H, 512, 8192, BLOCK, NSEL, W, 1, 2);
  const double t4 = bench_csa<bf16_t>(H, 512, 8192, BLOCK, NSEL, W, 4, 2);
  std::printf("MLA 4T %8.1f ms | CSA 1T %8.1f ms | CSA 4T %8.1f ms\n", tmla, t1, t4);
  std::fflush(stdout);
}

int main(int argc, char** argv) {
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;
  using f16_t = float16_t;
  using bf16_t = bfloat16_t;

  const Problem probs[] = {
      {"csa_causal", 1, 4, 17, 131, 32, 16, 64, 8, 5, 32, true, false},
      {"csa_small_budget", 1, 4, 13, 100, 24, 8, 48, 16, 3, 40, true, false},
      {"csa_full_b2", 2, 2, 40, 40, 32, 16, 64, 16, 4, 24, false, false},
      {"csa_short_prefix", 1, 3, 20, 20, 16, 8, 32, 8, 3, 12, true, false},
      {"csa_decode", 1, 4, 1, 300, 16, 8, 32, 16, 4, 48, true, false},
      {"csa_comp_input", 1, 2, 9, 77, 24, 8, 48, 8, 4, 20, true, true},
  };

  for (const Problem& pr : probs) {
    for (int threads : {1, 4}) {
      run_case<float>(pr, threads, 1.5e-3, "f32");
      run_case<f16_t>(pr, threads, 8e-3, "f16");
      run_case<bf16_t>(pr, threads, 3e-2, "bf16");
    }
  }

  {  // bit-identical across thread counts, deterministic across runs
    const Problem pr{"csa_bitwise", 1, 8, 100, 100, 32, 16, 64, 8, 6, 32, true, false};
    std::vector<float> o1, m1, o4, m4, o4b, m4b;
    run_stored<float>(pr, 1, &o1, &m1);
    run_stored<float>(pr, 4, &o4, &m4);
    run_stored<float>(pr, 4, &o4b, &m4b);
    CHECK(o1.size() == o4.size() &&
              std::memcmp(o1.data(), o4.data(), o1.size() * sizeof(float)) == 0,
          "CSA 1T vs 4T outputs differ");
    CHECK(std::memcmp(m1.data(), m4.data(), m1.size() * sizeof(float)) == 0,
          "CSA 1T vs 4T block-score mirror differs");
    CHECK(std::memcmp(o4.data(), o4b.data(), o4.size() * sizeof(float)) == 0,
          "CSA output not deterministic across runs");
  }

  check_error_paths();

  const bool want_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
  if (want_bench) run_benchmark();
  std::printf("csa: %d cases, %d failures%s\n", g_cases, g_failures,
              want_bench ? "" : " (run with --bench for the benchmark)");
  return g_failures == 0 ? 0 : 1;
}
