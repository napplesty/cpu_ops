// Tests for KdaAttention (Kimi Delta Attention: gated delta-rule linear
// attention). The naive recurrence *is* the spec — the double reference
// implements it directly, and both execution modes are checked against it.
// Additional contracts: chunked ≡ naive in f32 to rounding, the decode
// state round-trip (two sequential calls through s0/s_out reproduces one
// longer call — bitwise for f32 with an aligned chunk grid, since the
// intermediate state is exact), bit-identical outputs across thread
// counts, and error paths. Keys are L2-normalized in generation (as
// upstream models do), which keeps the substitution well-conditioned.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "kda.h"

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
  int b, h, sq, dk, dv, chunk;
  bool use_s0;
  float gate_scale = 4.0f;  // ±scale on the λ logits; 8+ pushes chunks
                            // past the factored-kernel range (fallback path)
};

template <typename T>
struct KdaTensors {
  std::vector<T> q, k, v, bl, gl, s0, o, s_out;
  typename cpu_ops::attention::KdaAttention<T>::Arguments args;

  void init(const Problem& pr, unsigned seed, bool want_state_out = true) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const float qk_scale = 1.0f / std::sqrt((float)pr.dk);
    const float v_scale = 1.0f / std::sqrt((float)pr.dv);
    const auto gen = [&](std::vector<T>& x, std::size_t n, float s) {
      x.resize(n);
      for (auto& e : x) e = T(dist(rng) * s);
    };
    gen(q, (std::size_t)pr.b * pr.h * pr.sq * pr.dk, qk_scale);
    gen(v, (std::size_t)pr.b * pr.h * pr.sq * pr.dv, v_scale);
    // k: unit rows (L2-normalized upstream, as Kimi Linear does).
    k.resize((std::size_t)pr.b * pr.h * pr.sq * pr.dk);
    {
      std::vector<float> kf((std::size_t)pr.b * pr.h * pr.sq * pr.dk);
      for (auto& e : kf) e = dist(rng);
      for (std::size_t i = 0; i < kf.size() / pr.dk; ++i) {
        float nrm = 0;
        for (int c = 0; c < pr.dk; ++c) nrm += kf[i * pr.dk + c] * kf[i * pr.dk + c];
        nrm = std::sqrt(nrm);
        for (int c = 0; c < pr.dk; ++c)
          k[i * pr.dk + c] = T(kf[i * pr.dk + c] / nrm);
      }
    }
    gen(bl, (std::size_t)pr.b * pr.h * pr.sq, 3.0f);   // signed β logits
    gen(gl, (std::size_t)pr.b * pr.h * pr.sq * pr.dk, pr.gate_scale);  // λ logits
    if (pr.use_s0) {
      gen(s0, (std::size_t)pr.b * pr.h * pr.dk * pr.dv, v_scale);
    }
    o.assign((std::size_t)pr.b * pr.h * pr.sq * pr.dv, T(0.0f));
    s_out.assign(want_state_out ? (std::size_t)pr.b * pr.h * pr.dk * pr.dv : 0,
                 T(0.0f));

    args.batch = pr.b;
    args.heads = pr.h;
    args.seq_q = pr.sq;
    args.dim_qk = pr.dk;
    args.dim_v = pr.dv;
    args.chunk = pr.chunk;
    args.q = q.data();
    args.q_stride_b = pr.h * pr.sq * pr.dk;
    args.q_stride_h = pr.sq * pr.dk;
    args.q_ld = pr.dk;
    args.k = k.data();
    args.k_stride_b = pr.h * pr.sq * pr.dk;
    args.k_stride_h = pr.sq * pr.dk;
    args.k_ld = pr.dk;
    args.v = v.data();
    args.v_stride_b = pr.h * pr.sq * pr.dv;
    args.v_stride_h = pr.sq * pr.dv;
    args.v_ld = pr.dv;
    args.beta_logit = bl.data();
    args.beta_stride_b = pr.h * pr.sq;
    args.beta_stride_h = pr.sq;
    args.gate_logit = gl.data();
    args.gate_stride_b = pr.h * pr.sq * pr.dk;
    args.gate_stride_h = pr.sq * pr.dk;
    args.gate_ld = pr.dk;
    args.s0 = pr.use_s0 ? s0.data() : nullptr;
    args.s_stride_b = pr.h * pr.dk * pr.dv;
    args.s_stride_h = pr.dk * pr.dv;
    args.s_out = want_state_out ? s_out.data() : nullptr;
    args.s_out_stride_b = pr.h * pr.dk * pr.dv;
    args.s_out_stride_h = pr.dk * pr.dv;
    args.o = o.data();
    args.o_stride_b = pr.h * pr.sq * pr.dv;
    args.o_stride_h = pr.sq * pr.dv;
    args.o_ld = pr.dv;
    args.scale = 1.0f / std::sqrt((float)pr.dk);
  }
};

// Double reference: the naive recurrence, straight from the spec.
void ref_kda(const Problem& pr, const std::vector<float>& q,
             const std::vector<float>& k, const std::vector<float>& v,
             const std::vector<float>& bl, const std::vector<float>& gl,
             const std::vector<float>& s0, float scale, std::vector<float>& o,
             std::vector<float>* s_out) {
  const int dk = pr.dk, dv = pr.dv;
  std::vector<double> u(dv);
  for (int bi = 0; bi < pr.b; ++bi) {
    for (int h = 0; h < pr.h; ++h) {
      const std::size_t bh = (std::size_t)bi * pr.h + h;
      std::vector<double> S((std::size_t)dk * dv, 0.0);
      if (pr.use_s0) {
        for (std::size_t i = 0; i < S.size(); ++i) S[i] = s0[bh * S.size() + i];
      }
      for (int t = 0; t < pr.sq; ++t) {
        const float* kt = k.data() + bh * pr.sq * dk + (std::size_t)t * dk;
        const float* vt = v.data() + bh * pr.sq * dv + (std::size_t)t * dv;
        const float* qt = q.data() + bh * pr.sq * dk + (std::size_t)t * dk;
        const float* gt = gl.data() + bh * pr.sq * dk + (std::size_t)t * dk;
        const double beta = 1.0 / (1.0 + std::exp(-(double)bl[bh * pr.sq + t]));
        for (int c = 0; c < dk; ++c) {
          const double lc = 1.0 / (1.0 + std::exp(-(double)gt[c]));
          for (int j = 0; j < dv; ++j) S[(std::size_t)c * dv + j] *= lc;
        }
        for (int j = 0; j < dv; ++j) u[j] = 0;
        for (int c = 0; c < dk; ++c)
          for (int j = 0; j < dv; ++j)
            u[j] += (double)kt[c] * S[(std::size_t)c * dv + j];
        for (int c = 0; c < dk; ++c) {
          const double kc = beta * kt[c];
          for (int j = 0; j < dv; ++j)
            S[(std::size_t)c * dv + j] += kc * ((double)vt[j] - u[j]);
        }
        float* ot = o.data() + bh * pr.sq * dv + (std::size_t)t * dv;
        for (int j = 0; j < dv; ++j) ot[j] = 0;
        for (int c = 0; c < dk; ++c) {
          const double qc = scale * qt[c];
          for (int j = 0; j < dv; ++j) ot[j] += (float)(qc * S[(std::size_t)c * dv + j]);
        }
      }
      if (s_out) {
        for (std::size_t i = 0; i < S.size(); ++i) (*s_out)[bh * S.size() + i] = (float)S[i];
      }
    }
  }
}

template <typename T>
void run_case(const Problem& pr, cpu_ops::attention::KdaMode mode,
              int threads, double tol, const char* tag) {
  ++g_cases;
  KdaTensors<T> ten;
  ten.init(pr, 4321);
  ten.args.mode = mode;

  cpu_ops::attention::KdaAttention<T> kda;
  const cpu_ops::Status st = kda(ten.args, threads);
  CHECK(st == cpu_ops::Status::kSuccess, "%s %s threads=%d: status %d", pr.name, tag,
        threads, (int)st);

  const auto round_back = [](std::vector<float>& f, const std::vector<T>& t) {
    f.resize(t.size());
    for (std::size_t i = 0; i < t.size(); ++i) f[i] = static_cast<float>(t[i]);
  };
  std::vector<float> q_f, k_f, v_f, bl_f, gl_f, s0_f;
  round_back(q_f, ten.q);
  round_back(k_f, ten.k);
  round_back(v_f, ten.v);
  round_back(bl_f, ten.bl);
  round_back(gl_f, ten.gl);
  if (pr.use_s0) round_back(s0_f, ten.s0);

  std::vector<float> ref(ten.o.size());
  std::vector<float> ref_s(ten.s_out.size());
  ref_kda(pr, q_f, k_f, v_f, bl_f, gl_f, s0_f, ten.args.scale, ref, &ref_s);

  double max_err = 0;
  for (std::size_t i = 0; i < ten.o.size(); ++i) {
    max_err = std::max(max_err,
                       std::fabs(static_cast<double>(static_cast<float>(ten.o[i])) -
                                 ref[i]));
  }
  CHECK(max_err <= tol, "%s %s threads=%d: output max_err %g > %g", pr.name, tag,
        threads, max_err, tol);
  double max_err_s = 0;
  for (std::size_t i = 0; i < ten.s_out.size(); ++i) {
    max_err_s = std::max(max_err_s,
                         std::fabs(static_cast<double>(static_cast<float>(ten.s_out[i])) -
                                   ref_s[i]));
  }
  CHECK(max_err_s <= tol, "%s %s threads=%d: state max_err %g > %g", pr.name, tag,
        threads, max_err_s, tol);
}

template <typename T>
void run_stored(const Problem& pr, cpu_ops::attention::KdaMode mode,
                int threads, std::vector<T>* o, std::vector<T>* s) {
  KdaTensors<T> ten;
  ten.init(pr, 4321);
  ten.args.mode = mode;
  cpu_ops::attention::KdaAttention<T> kda;
  CHECK(kda(ten.args, threads) == cpu_ops::Status::kSuccess, "run_stored status");
  *o = std::move(ten.o);
  *s = std::move(ten.s_out);
}

void check_two_stage() {
  ++g_cases;
  using Kda = cpu_ops::attention::KdaAttention<float>;
  using Args = Kda::Arguments;
  // Chunk grid aligned (C=8, split at 8): f32 two-stage must be bitwise
  // identical to one shot (the intermediate state round-trips exactly).
  // Single head: a token-window view of a [b][h][sq][*] tensor is only a
  // plain pointer shift when h == 1 (heads interleave otherwise).
  const Problem pr{"kda_2stage", 1, 1, 16, 32, 16, 8, true};
  KdaTensors<float> whole;
  whole.init(pr, 4321);
  whole.args.mode = cpu_ops::attention::KdaMode::kChunked;
  Kda kda;
  CHECK(kda(whole.args, 4) == cpu_ops::Status::kSuccess, "two-stage whole status");

  KdaTensors<float> part;
  part.init(pr, 4321);
  Args a1 = part.args;
  a1.seq_q = 8;
  a1.mode = cpu_ops::attention::KdaMode::kChunked;
  CHECK(kda(a1, 4) == cpu_ops::Status::kSuccess, "two-stage first status");

  Args a2 = part.args;
  a2.q = part.q.data() + (std::size_t)part.args.q_ld * 8;
  a2.k = part.k.data() + (std::size_t)part.args.k_ld * 8;
  a2.v = part.v.data() + (std::size_t)part.args.v_ld * 8;
  a2.beta_logit = part.bl.data() + 8;
  a2.gate_logit = part.gl.data() + (std::size_t)part.args.gate_ld * 8;
  a2.s0 = part.s_out.data();  // carried state
  a2.o = part.o.data() + (std::size_t)part.args.o_ld * 8;
  a2.seq_q = 8;
  a2.mode = cpu_ops::attention::KdaMode::kChunked;
  CHECK(kda(a2, 4) == cpu_ops::Status::kSuccess, "two-stage second status");

  CHECK(std::memcmp(part.o.data(), whole.o.data(), part.o.size() * sizeof(float)) == 0,
        "two-stage f32 outputs differ from one-shot (aligned chunk grid)");
}

void check_error_paths() {
  ++g_cases;
  using Kda = cpu_ops::attention::KdaAttention<float>;
  using Args = Kda::Arguments;
  Problem pr{"err", 1, 2, 16, 8, 4, 8, false};
  KdaTensors<float> ten;
  ten.init(pr, 7);
  Kda kda;
  CHECK(kda(ten.args) == cpu_ops::Status::kSuccess, "valid args rejected");

  Args a = ten.args;
  a.chunk = 0;
  CHECK(kda(a) == cpu_ops::Status::kErrorInvalidProblem, "chunk=0 accepted");
  a = ten.args;
  a.gate_logit = nullptr;
  CHECK(kda(a) == cpu_ops::Status::kErrorInvalidArguments, "null gate_logit accepted");
  a = ten.args;
  a.q_ld = 7;  // < dk = 8
  CHECK(kda(a) == cpu_ops::Status::kErrorInvalidArguments, "q_ld < dk accepted");
  a = ten.args;
  a.seq_q = 0;
  CHECK(kda(a) == cpu_ops::Status::kSuccess, "seq_q=0 rejected");
}

}  // namespace

// ---------------------------------------------------------------------------
// Benchmark (opt-in via --bench): naive (decode path) vs chunked, Kimi
// Linear-shaped dims (dk = dv = 128, H = 32), threads <= 4.
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
double bench_kda(int h, int sq, int dk, int dv, int chunk,
                 cpu_ops::attention::KdaMode mode, int threads,
                 int reps) {
  Problem pr{"bench", 1, h, sq, dk, dv, chunk, false};
  KdaTensors<T> ten;
  ten.init(pr, 11, /*want_state_out=*/false);
  ten.args.mode = mode;
  cpu_ops::attention::KdaAttention<T> kda;
  kda(ten.args, threads);  // warmup
  return time_best([&] { kda(ten.args, threads); }, reps) * 1e3;
}

void run_benchmark() {
  using bf16_t = cpu_ops::bfloat16_t;
  const int H = 32, DK = 128, DV = 128, C = 64;
  std::printf("\nKDA naive vs chunked (C=%d), B=1 H=%d, dk=dv=%d, bf16:\n",
              C, H, DK);
  std::printf("%10s | %12s | %12s %12s | %8s\n", "seq", "naive 4T ms", "chunk 1T ms",
              "chunk 4T ms", "speedup");
  for (int sq : {2048, 8192}) {
    const double tn = bench_kda<bf16_t>(H, sq, DK, DV, C, cpu_ops::attention::KdaMode::kNaive, 4, 2);
    const double t1 = bench_kda<bf16_t>(H, sq, DK, DV, C, cpu_ops::attention::KdaMode::kChunked, 1, 3);
    const double t4 = bench_kda<bf16_t>(H, sq, DK, DV, C, cpu_ops::attention::KdaMode::kChunked, 4, 3);
    std::printf("%10d | %12.1f | %12.1f %12.1f | %8.2f\n", sq, tn, t1, t4, tn / t4);
    std::fflush(stdout);
  }
}

int main(int argc, char** argv) {
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;
  using f16_t = float16_t;
  using bf16_t = bfloat16_t;
    const Problem probs[] = {
      {"kda_basic", 1, 4, 130, 32, 16, 16, false},
      {"kda_s0", 2, 3, 97, 24, 24, 8, true},
      {"kda_chunk1", 1, 2, 40, 16, 8, 1, false},   // degenerate chunks
      {"kda_wide", 1, 2, 33, 64, 48, 64, true},    // single partial chunk
      {"kda_small", 1, 4, 1, 8, 8, 8, false},      // decode shape
      {"kda_strong_decay", 1, 2, 70, 16, 16, 64, false, 8.0f},  // fallback kernels
  };

  for (const Problem& pr : probs) {
    for (int threads : {1, 4}) {
      for (cpu_ops::attention::KdaMode mode : {cpu_ops::attention::KdaMode::kNaive, cpu_ops::attention::KdaMode::kChunked}) {
        const char* tag = mode == cpu_ops::attention::KdaMode::kNaive ? "naive" : "chunk";
        run_case<float>(pr, mode, threads, 2e-3, tag);
        run_case<f16_t>(pr, mode, threads, 8e-3, tag);
        run_case<bf16_t>(pr, mode, threads, 3e-2, tag);
      }
    }
  }

  {  // chunked ≡ naive in f32 to rounding
    ++g_cases;
    const Problem pr{"kda_equiv", 1, 4, 200, 32, 16, 16, true};
    std::vector<float> on, sn, oc, sc;
    run_stored<float>(pr, cpu_ops::attention::KdaMode::kNaive, 4, &on, &sn);
    run_stored<float>(pr, cpu_ops::attention::KdaMode::kChunked, 4, &oc, &sc);
    double max_err = 0;
    for (std::size_t i = 0; i < on.size(); ++i)
      max_err = std::max(max_err, std::fabs((double)(on[i] - oc[i])));
    double max_err_s = 0;
    for (std::size_t i = 0; i < sn.size(); ++i)
      max_err_s = std::max(max_err_s, std::fabs((double)(sn[i] - sc[i])));
    CHECK(max_err <= 2e-3, "chunk vs naive f32 output err %g", max_err);
    CHECK(max_err_s <= 2e-3, "chunk vs naive f32 state err %g", max_err_s);
  }

  {  // bit-identical across thread counts / runs (both modes)
    const Problem pr{"kda_bitwise", 2, 4, 90, 24, 12, 16, true};
    for (cpu_ops::attention::KdaMode mode : {cpu_ops::attention::KdaMode::kNaive, cpu_ops::attention::KdaMode::kChunked}) {
      std::vector<float> o1, s1, o4, s4, o4b, s4b;
      run_stored<float>(pr, mode, 1, &o1, &s1);
      run_stored<float>(pr, mode, 4, &o4, &s4);
      run_stored<float>(pr, mode, 4, &o4b, &s4b);
      CHECK(std::memcmp(o1.data(), o4.data(), o1.size() * sizeof(float)) == 0,
            "KDA 1T vs 4T outputs differ");
      CHECK(std::memcmp(s1.data(), s4.data(), s1.size() * sizeof(float)) == 0,
            "KDA 1T vs 4T state differs");
      CHECK(std::memcmp(o4.data(), o4b.data(), o4.size() * sizeof(float)) == 0,
            "KDA output not deterministic across runs");
    }
  }

  check_two_stage();
  check_error_paths();

  const bool want_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
  if (want_bench) run_benchmark();
  std::printf("kda: %d cases, %d failures%s\n", g_cases, g_failures,
              want_bench ? "" : " (run with --bench for the benchmark)");
  return g_failures == 0 ? 0 : 1;
}
