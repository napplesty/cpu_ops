// Tests for MlaAttention (absorbed-form MLA over the fused kernel). The
// reference computes the *non-absorbed* math in double — k_nope = W_UK·c,
// v = W_UV·c per head — so the comparison also validates the absorption
// identity and the GEMM orientations. Covers MHA-shaped and chunked-prefill
// sizes, narrow storage (operands rounded through the storage type, reference
// on the same rounded values), kv-split, the determinism contracts, and
// error paths.

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"

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
  int b, h, sq, skv, dn, dp, dc;
  bool causal;
};

// Non-absorbed double reference on already-rounded operands.
void ref_mla(const Problem& pr, const std::vector<float>& q_nope,
             const std::vector<float>& q_pe, const std::vector<float>& cache,
             const std::vector<float>& w_uk, const std::vector<float>& w_uv, float scale,
             std::vector<float>& o) {
  const int dcat = pr.dc + pr.dp;
  const int kv_off = pr.skv - pr.sq;
  std::vector<double> p(pr.skv);
  for (int bi = 0; bi < pr.b; ++bi) {
    for (int h = 0; h < pr.h; ++h) {
      const float* uk = w_uk.data() + static_cast<size_t>(h) * pr.dc * pr.dn;
      const float* uv = w_uv.data() + static_cast<size_t>(h) * pr.dn * pr.dc;
      for (int i = 0; i < pr.sq; ++i) {
        const float* qn = q_nope.data() +
                          ((static_cast<size_t>(bi) * pr.h + h) * pr.sq + i) * pr.dn;
        const float* qp =
            q_pe.data() + ((static_cast<size_t>(bi) * pr.h + h) * pr.sq + i) * pr.dp;
        double m = -1e300;
        for (int t = 0; t < pr.skv; ++t) {
          if (pr.causal && t > i + kv_off) break;
          const float* ct = cache.data() + static_cast<size_t>(bi) * pr.skv * dcat +
                            static_cast<size_t>(t) * dcat;
          const float* kt = ct + pr.dc;
          double s = 0;
          for (int x = 0; x < pr.dp; ++x) s += (double)qp[x] * kt[x];  // rope part
          for (int x = 0; x < pr.dn; ++x) {  // q_nope . (W_UK c)
            double kn = 0;
            for (int y = 0; y < pr.dc; ++y) kn += (double)uk[static_cast<size_t>(y) * pr.dn + x] * ct[y];
            s += (double)qn[x] * kn;
          }
          s *= scale;
          p[t] = s;
          if (s > m) m = s;
        }
        double l = 0;
        std::vector<double> acc(pr.dn, 0.0);
        for (int t = 0; t < pr.skv; ++t) {
          if (pr.causal && t > i + kv_off) break;
          const double w = std::exp(p[t] - m);
          l += w;
          const float* ct = cache.data() + static_cast<size_t>(bi) * pr.skv * dcat +
                            static_cast<size_t>(t) * dcat;
          for (int x = 0; x < pr.dn; ++x) {  // w · (W_UV c)[x]
            double v = 0;
            for (int y = 0; y < pr.dc; ++y)
              v += (double)uv[static_cast<size_t>(x) * pr.dc + y] * ct[y];
            acc[x] += w * v;
          }
        }
        float* oi = o.data() + ((static_cast<size_t>(bi) * pr.h + h) * pr.sq + i) * pr.dn;
        for (int x = 0; x < pr.dn; ++x) oi[x] = (float)(acc[x] / l);
      }
    }
  }
}

template <typename T>
void run_case(const Problem& pr, int threads, int split_slices, double tol,
              const char* tag) {
  ++g_cases;
  std::mt19937 rng(4321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int dcat = pr.dc + pr.dp;

  std::vector<float> qn_f((size_t)pr.b * pr.h * pr.sq * pr.dn);
  std::vector<float> qp_f((size_t)pr.b * pr.h * pr.sq * pr.dp);
  std::vector<float> kv_f((size_t)pr.b * pr.skv * dcat);
  std::vector<float> uk_f((size_t)pr.h * pr.dc * pr.dn);
  std::vector<float> uv_f((size_t)pr.h * pr.dn * pr.dc);
  for (auto& x : qn_f) x = dist(rng);
  for (auto& x : qp_f) x = dist(rng);
  for (auto& x : kv_f) x = dist(rng);
  for (auto& x : uk_f) x = dist(rng) * 0.2f;
  for (auto& x : uv_f) x = dist(rng) * 0.2f;

  std::vector<T> qn(qn_f.size()), qp(qp_f.size()), kv(kv_f.size()), uk(uk_f.size()),
      uv(uv_f.size()), o((size_t)pr.b * pr.h * pr.sq * pr.dn);
  for (size_t i = 0; i < qn.size(); ++i) qn[i] = T(qn_f[i]);
  for (size_t i = 0; i < qp.size(); ++i) qp[i] = T(qp_f[i]);
  for (size_t i = 0; i < kv.size(); ++i) kv[i] = T(kv_f[i]);
  for (size_t i = 0; i < uk.size(); ++i) uk[i] = T(uk_f[i]);
  for (size_t i = 0; i < uv.size(); ++i) uv[i] = T(uv_f[i]);
  // Reference sees the same rounded values.
  auto round_back = [](std::vector<float>& f, const std::vector<T>& t) {
    for (size_t i = 0; i < f.size(); ++i) f[i] = static_cast<float>(t[i]);
  };
  round_back(qn_f, qn);
  round_back(qp_f, qp);
  round_back(kv_f, kv);
  round_back(uk_f, uk);
  round_back(uv_f, uv);

  using Mla = cpu_ops::attention::MlaAttention<T>;
  typename Mla::Arguments args;
  args.batch = pr.b;
  args.heads = pr.h;
  args.seq_q = pr.sq;
  args.seq_kv = pr.skv;
  args.dim_nope = pr.dn;
  args.dim_pe = pr.dp;
  args.dim_latent = pr.dc;
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
  args.o = o.data();
  args.o_stride_b = pr.h * pr.sq * pr.dn;
  args.o_stride_h = pr.sq * pr.dn;
  args.o_ld = pr.dn;
  args.scale = 1.0f / std::sqrt((float)(pr.dn + pr.dp));
  args.causal = pr.causal;
  args.split_kv_slices = split_slices;

  Mla mla;
  const cpu_ops::Status st = mla(args, threads);
  CHECK(st == cpu_ops::Status::kSuccess, "%s %s threads=%d slices=%d: status %d", pr.name,
        tag, threads, split_slices, (int)st);

  std::vector<float> ref(o.size());
  ref_mla(pr, qn_f, qp_f, kv_f, uk_f, uv_f, args.scale, ref);

  double max_err = 0;
  for (size_t i = 0; i < o.size(); ++i) {
    const double got = static_cast<double>(static_cast<float>(o[i]));
    max_err = std::max(max_err, std::fabs(got - ref[i]));
  }
  CHECK(max_err <= tol, "%s %s threads=%d slices=%d: max_err %g > %g", pr.name, tag,
        threads, split_slices, max_err, tol);
}

template <typename T>
void run_stored(const Problem& pr, int threads, int split_slices, std::vector<T>* out) {
  std::mt19937 rng(4321);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int dcat = pr.dc + pr.dp;
  std::vector<T> qn((size_t)pr.b * pr.h * pr.sq * pr.dn), qp((size_t)pr.b * pr.h * pr.sq * pr.dp),
      kv((size_t)pr.b * pr.skv * dcat), uk((size_t)pr.h * pr.dc * pr.dn),
      uv((size_t)pr.h * pr.dn * pr.dc), o((size_t)pr.b * pr.h * pr.sq * pr.dn);
  for (auto& x : qn) x = T(dist(rng));
  for (auto& x : qp) x = T(dist(rng));
  for (auto& x : kv) x = T(dist(rng));
  for (auto& x : uk) x = T(dist(rng) * 0.2f);
  for (auto& x : uv) x = T(dist(rng) * 0.2f);

  using Mla = cpu_ops::attention::MlaAttention<T>;
  typename Mla::Arguments args;
  args.batch = pr.b;
  args.heads = pr.h;
  args.seq_q = pr.sq;
  args.seq_kv = pr.skv;
  args.dim_nope = pr.dn;
  args.dim_pe = pr.dp;
  args.dim_latent = pr.dc;
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
  args.o = o.data();
  args.o_stride_b = pr.h * pr.sq * pr.dn;
  args.o_stride_h = pr.sq * pr.dn;
  args.o_ld = pr.dn;
  args.scale = 1.0f / std::sqrt((float)(pr.dn + pr.dp));
  args.causal = pr.causal;
  args.split_kv_slices = split_slices;
  Mla mla;
  CHECK(mla(args, threads) == cpu_ops::Status::kSuccess, "run_stored status");
  *out = std::move(o);
}

void check_error_paths() {
  ++g_cases;
  using Mla = cpu_ops::attention::MlaAttention<float>;
  float buf[64] = {0}, obuf[16] = {0};
  typename Mla::Arguments a;
  a.batch = 1;
  a.heads = 2;
  a.seq_q = a.seq_kv = 4;
  a.dim_nope = 4;
  a.dim_pe = 2;
  a.dim_latent = 8;
  a.q_nope = a.q_pe = a.kv_cache = a.w_uk = a.w_uv = buf;
  a.o = obuf;
  a.q_nope_ld = 4;
  a.q_pe_ld = 2;
  a.kv_ld = 9;  // < dc + dp = 10
  a.w_uk_stride_h = 32;
  a.w_uv_stride_h = 32;
  a.o_ld = 4;
  Mla mla;
  CHECK(mla(a) == cpu_ops::Status::kErrorInvalidArguments, "kv_ld < dc+dp accepted");
  a.kv_ld = 12;
  a.w_uk = nullptr;
  CHECK(mla(a) == cpu_ops::Status::kErrorInvalidArguments, "null w_uk accepted");
  a.w_uk = buf;
  a.seq_kv = 2;  // causal with skv < sq
  CHECK(mla(a) == cpu_ops::Status::kErrorInvalidArguments, "causal skv < sq accepted");
  a.seq_kv = 4;
  CHECK(mla(a) == cpu_ops::Status::kSuccess, "valid args rejected");
}

}  // namespace

int main() {
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;
  using f16_t = float16_t;
  using bf16_t = bfloat16_t;

  const Problem probs[] = {
      {"mla_causal", 1, 8, 50, 50, 32, 16, 64, true},
      {"mla_chunked", 1, 4, 13, 100, 24, 8, 48, true},
      {"mla_full_b2", 2, 2, 40, 40, 32, 16, 64, false},
      {"mla_decode", 1, 4, 1, 300, 16, 8, 32, true},
  };

  for (const Problem& pr : probs) {
    for (int threads : {1, 4}) {
      // Tolerances: f32 covers the two extra absorb GEMMs; narrow types add
      // one rounding of the q̃ / õ panels back to storage.
      run_case<float>(pr, threads, 1, 1.5e-3, "f32");
      run_case<f16_t>(pr, threads, 1, 8e-3, "f16");
      run_case<bf16_t>(pr, threads, 1, 3e-2, "bf16");
    }
    run_case<float>(pr, 4, 3, 1.5e-3, "f32");  // forced kv-split through MLA
  }

  {
    const Problem pr{"mla_bitwise", 1, 8, 100, 100, 32, 16, 64, true};
    std::vector<float> a, b, c;
    run_stored<float>(pr, 1, 1, &a);
    run_stored<float>(pr, 4, 1, &b);
    CHECK(a.size() == b.size() &&
              std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
          "MLA 1T vs 4T outputs differ (kv-split off)");
    run_stored<float>(pr, 4, 4, &b);
    run_stored<float>(pr, 4, 4, &c);
    CHECK(b.size() == c.size() &&
              std::memcmp(b.data(), c.data(), b.size() * sizeof(float)) == 0,
          "MLA kv-split output not deterministic across runs");
  }

  check_error_paths();

  std::printf("test_attention_mla: %d cases, %d failures\n", g_cases, g_failures);
  return g_failures == 0 ? 0 : 1;
}
