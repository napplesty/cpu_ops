// Tests for the fused attention kernel: MHA / GQA / MQA mappings, causal and
// full attention, chunked prefill (seq_kv > seq_q), odd shapes, padded
// strides, narrow storage (f16/bf16, reference computed on the same rounded
// operands), and the determinism contracts — bit-identical across thread
// counts when kv-split is off, bit-identical across runs when it is on.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu_ops/cpu_ops.h"

#include "attention.h"

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

// Double-precision two-pass reference. Operands are floats that already went
// through the storage rounding, so the comparison isolates the kernel's f32
// arithmetic.
template <typename T>
void ref_attention(int b, int hq, int hkv, int sq, int skv, int d, const float* q, int qsb,
                   int qsh, int qld, const float* k, int ksb, int ksh, int kld,
                   const float* v, int vsb, int vsh, int vld, float scale, bool causal,
                   float* o, int osb, int osh, int old_) {
  const int kv_off = skv - sq;
  std::vector<double> p(skv);
  for (int bi = 0; bi < b; ++bi) {
    for (int h = 0; h < hq; ++h) {
      const int g = static_cast<int>(static_cast<long long>(h) * hkv / hq);
      for (int i = 0; i < sq; ++i) {
        const float* qi = q + static_cast<size_t>(bi) * qsb +
                          static_cast<size_t>(h) * qsh + static_cast<size_t>(i) * qld;
        double m = -1e300;
        for (int j = 0; j < skv; ++j) {
          if (causal && j > i + kv_off) break;
          const float* kj = k + static_cast<size_t>(bi) * ksb +
                            static_cast<size_t>(g) * ksh + static_cast<size_t>(j) * kld;
          double s = 0;
          for (int x = 0; x < d; ++x) s += static_cast<double>(qi[x]) * kj[x];
          s *= scale;
          p[j] = s;
          if (s > m) m = s;
        }
        double l = 0;
        std::vector<double> acc(d, 0.0);
        for (int j = 0; j < skv; ++j) {
          if (causal && j > i + kv_off) break;
          const double w = std::exp(p[j] - m);
          l += w;
          const float* vj = v + static_cast<size_t>(bi) * vsb +
                            static_cast<size_t>(g) * vsh + static_cast<size_t>(j) * vld;
          for (int x = 0; x < d; ++x) acc[x] += w * static_cast<double>(vj[x]);
        }
        float* oi = o + static_cast<size_t>(bi) * osb + static_cast<size_t>(h) * osh +
                    static_cast<size_t>(i) * old_;
        for (int x = 0; x < d; ++x) oi[x] = static_cast<float>(acc[x] / l);
      }
    }
  }
}

struct Problem {
  const char* name;
  int b, hq, hkv, sq, skv, d;
  bool causal;
};

// Runs one problem: random operands (padded ld + extra batch/head padding),
// kernel vs reference at the requested thread count and kv-split setting.
template <typename T>
void run_case(const Problem& pr, int threads, int split_slices, double tol,
              const char* tag) {
  ++g_cases;
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  const int ld = pr.d + 3;  // padded, non-tight leading dimension
  const size_t head_elems = static_cast<size_t>(pr.sq) * ld;
  const size_t head_elems_kv = static_cast<size_t>(pr.skv) * ld;

  auto alloc_padded = [&](size_t heads, size_t head_stride_elems) {
    // Deliberately loose batch/head strides to prove the indexing.
    const size_t batch_stride = heads * head_stride_elems + 7;
    std::vector<float> buf(batch_stride * pr.b);
    for (auto& x : buf) x = dist(rng);
    return std::make_pair(buf, batch_stride);
  };

  auto [qbuf, qsb] = alloc_padded(pr.hq, head_elems + 5);
  auto [kbuf, ksb] = alloc_padded(pr.hkv, head_elems_kv + 9);
  auto [vbuf, vsb] = alloc_padded(pr.hkv, head_elems_kv + 11);
  auto [obuf, osb] = alloc_padded(pr.hq, head_elems + 13);
  for (auto& x : obuf) x = 0.0f;

  // Round operands through the storage type; the reference sees the same
  // rounded values (as floats) that the kernel consumes.
  const size_t qn = qbuf.size(), kn = kbuf.size(), vn = vbuf.size();
  std::vector<T> qT(qn), kT(kn), vT(vn), oT(obuf.size());
  for (size_t i = 0; i < qn; ++i) qT[i] = T(qbuf[i]);
  for (size_t i = 0; i < kn; ++i) kT[i] = T(kbuf[i]);
  for (size_t i = 0; i < vn; ++i) vT[i] = T(vbuf[i]);
  std::vector<float> qr(qn), kr(kn), vr(vn);
  for (size_t i = 0; i < qn; ++i) qr[i] = static_cast<float>(qT[i]);
  for (size_t i = 0; i < kn; ++i) kr[i] = static_cast<float>(kT[i]);
  for (size_t i = 0; i < vn; ++i) vr[i] = static_cast<float>(vT[i]);

  const float scale = 1.0f / std::sqrt(static_cast<float>(pr.d));
  using Attn = cpu_ops::attention::Attention<T>;
  typename Attn::Arguments args;
  args.batch = pr.b;
  args.heads_q = pr.hq;
  args.heads_kv = pr.hkv;
  args.seq_q = pr.sq;
  args.seq_kv = pr.skv;
  args.dim = pr.d;
  args.q = qT.data();
  args.q_stride_b = static_cast<int>(qsb);
  args.q_stride_h = static_cast<int>(head_elems + 5);
  args.q_ld = ld;
  args.k = kT.data();
  args.k_stride_b = static_cast<int>(ksb);
  args.k_stride_h = static_cast<int>(head_elems_kv + 9);
  args.k_ld = ld;
  args.v = vT.data();
  args.v_stride_b = static_cast<int>(vsb);
  args.v_stride_h = static_cast<int>(head_elems_kv + 11);
  args.v_ld = ld;
  args.o = oT.data();
  args.o_stride_b = static_cast<int>(osb);
  args.o_stride_h = static_cast<int>(head_elems + 13);
  args.o_ld = ld;
  args.scale = scale;
  args.causal = pr.causal;
  args.split_kv_slices = split_slices;

  Attn attn;
  const cpu_ops::Status st = attn(args, threads);
  CHECK(st == cpu_ops::Status::kSuccess, "%s %s threads=%d slices=%d: bad status %d",
        pr.name, tag, threads, split_slices, static_cast<int>(st));

  std::vector<float> ref(obuf.size());
  ref_attention<T>(pr.b, pr.hq, pr.hkv, pr.sq, pr.skv, pr.d, qr.data(), args.q_stride_b,
                   args.q_stride_h, args.q_ld, kr.data(), args.k_stride_b,
                   args.k_stride_h, args.k_ld, vr.data(), args.v_stride_b,
                   args.v_stride_h, args.v_ld, scale, pr.causal, ref.data(),
                   args.o_stride_b, args.o_stride_h, args.o_ld);

  double max_err = 0;
  for (int bi = 0; bi < pr.b; ++bi)
    for (int h = 0; h < pr.hq; ++h)
      for (int i = 0; i < pr.sq; ++i)
        for (int x = 0; x < pr.d; ++x) {
          const size_t off = static_cast<size_t>(bi) * args.o_stride_b +
                             static_cast<size_t>(h) * args.o_stride_h +
                             static_cast<size_t>(i) * args.o_ld + x;
          const double got = static_cast<double>(static_cast<float>(oT[off]));
          const double err = std::fabs(got - ref[off]);
          if (err > max_err) max_err = err;
        }
  CHECK(max_err <= tol, "%s %s threads=%d slices=%d: max_err %g > %g", pr.name, tag,
        threads, split_slices, max_err, tol);
}

// Bit-for-bit comparison of two runs (any thread counts / split settings).
template <typename T>
void run_stored(const Problem& pr, int threads, int split_slices, std::vector<T>* out) {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  const int ld = pr.d + 3;
  const size_t qh = static_cast<size_t>(pr.sq) * ld + 5;
  const size_t kh = static_cast<size_t>(pr.skv) * ld + 9;
  const size_t qsb = static_cast<size_t>(pr.hq) * qh + 7;
  const size_t ksb = static_cast<size_t>(pr.hkv) * kh + 7;
  std::vector<T> q(qsb * pr.b), k(ksb * pr.b), v(ksb * pr.b), o(qsb * pr.b);
  for (auto& x : q) x = T(dist(rng));
  for (auto& x : k) x = T(dist(rng));
  for (auto& x : v) x = T(dist(rng));
  for (auto& x : o) x = T(0);

  using Attn = cpu_ops::attention::Attention<T>;
  typename Attn::Arguments args;
  args.batch = pr.b;
  args.heads_q = pr.hq;
  args.heads_kv = pr.hkv;
  args.seq_q = pr.sq;
  args.seq_kv = pr.skv;
  args.dim = pr.d;
  args.q = q.data();
  args.q_stride_b = static_cast<int>(qsb);
  args.q_stride_h = static_cast<int>(qh);
  args.q_ld = ld;
  args.k = k.data();
  args.k_stride_b = static_cast<int>(ksb);
  args.k_stride_h = static_cast<int>(kh);
  args.k_ld = ld;
  args.v = v.data();
  args.v_stride_b = static_cast<int>(ksb);
  args.v_stride_h = static_cast<int>(kh);
  args.v_ld = ld;
  args.o = o.data();
  args.o_stride_b = static_cast<int>(qsb);
  args.o_stride_h = static_cast<int>(qh);
  args.o_ld = ld;
  args.scale = 1.0f / std::sqrt(static_cast<float>(pr.d));
  args.causal = pr.causal;
  args.split_kv_slices = split_slices;

  Attn attn;
  CHECK(attn(args, threads) == cpu_ops::Status::kSuccess, "run_stored status");
  *out = std::move(o);
}

void check_error_paths() {
  ++g_cases;
  using Attn = cpu_ops::attention::Attention<float>;
  float q[16] = {0}, k[16] = {0}, v[16] = {0}, o[16] = {0};
  typename Attn::Arguments args;
  args.batch = 1;
  args.heads_q = 3;  // not divisible by heads_kv
  args.heads_kv = 2;
  args.seq_q = args.seq_kv = 2;
  args.dim = 4;
  args.q = q;
  args.k = k;
  args.v = v;
  args.o = o;
  args.q_ld = args.k_ld = args.v_ld = args.o_ld = 4;
  Attn attn;
  CHECK(attn(args) == cpu_ops::Status::kErrorInvalidArguments, "hq%%hkv != 0 accepted");

  args.heads_kv = 1;
  args.seq_kv = 1;  // causal with shorter kv than q
  CHECK(attn(args) == cpu_ops::Status::kErrorInvalidArguments, "causal skv < sq accepted");

  args.seq_kv = 2;
  args.q = nullptr;
  CHECK(attn(args) == cpu_ops::Status::kErrorInvalidArguments, "null q accepted");
  args.q = q;
  args.dim = 8;  // ld too small
  CHECK(attn(args) == cpu_ops::Status::kErrorInvalidArguments, "ld < dim accepted");
  args.dim = 4;
  CHECK(attn(args) == cpu_ops::Status::kSuccess, "valid args rejected");
}

}  // namespace


// ---------------------------------------------------------------------------
// Benchmark (opt-in via --bench): prefill and decode shapes, threads capped
// at 4 to stay polite on shared machines.
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
double bench(int hq, int hkv, int sq, int skv, int d, bool causal, int threads,
             int reps) {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<T> q((size_t)hq * sq * d), o(q.size());
  const size_t kv_elems = (size_t)hkv * skv * d;
  std::vector<T> k(kv_elems), v(kv_elems);
  for (auto& x : q) x = T(dist(rng));
  for (auto& x : k) x = T(dist(rng));
  for (auto& x : v) x = T(dist(rng));

  typename cpu_ops::attention::Attention<T>::Arguments args;
  args.batch = 1;
  args.heads_q = hq;
  args.heads_kv = hkv;
  args.seq_q = sq;
  args.seq_kv = skv;
  args.dim = d;
  args.q = q.data();
  args.q_stride_h = sq * d;
  args.q_ld = d;
  args.k = k.data();
  args.k_stride_h = skv * d;
  args.k_ld = d;
  args.v = v.data();
  args.v_stride_h = skv * d;
  args.v_ld = d;
  args.o = o.data();
  args.o_stride_h = sq * d;
  args.o_ld = d;
  args.scale = 1.0f / std::sqrt((float)d);
  args.causal = causal;

  cpu_ops::attention::Attention<T> attn;
  attn(args, threads);  // warmup
  const double s = time_best([&] { attn(args, threads); }, reps);
  const volatile float sink = static_cast<float>(o[o.size() / 2]);
  (void)sink;
  return s * 1e3;
}

void run_benchmark() {
  std::printf("\nprefill, causal, B=1 H=16 d=128 f32 (best of 3):\n");
  std::printf("%10s | %10s %10s | %10s %10s\n", "seq", "1T ms", "4T ms", "1T GFLOPS",
              "4T GFLOPS");
  for (int n : {512, 1024, 2048}) {
    const double flops = 2.0 * 2.0 * n * n * 128 * 16 * 0.5;  // two GEMMs, causal half
    const double t1 = bench<float>(16, 16, n, n, 128, true, 1, 3);
    const double t4 = bench<float>(16, 16, n, n, 128, true, 4, 3);
    std::printf("%10d | %10.2f %10.2f | %10.1f %10.1f\n", n, t1, t4,
                flops / (t1 * 1e-3) / 1e9, flops / (t4 * 1e-3) / 1e9);
  }
  std::printf("\ndecode, seq_q=1, GQA H=32/Hkv=8 d=128 bf16:\n");
  std::printf("%10s | %10s %10s | %10s\n", "kv len", "1T ms", "4T ms", "4T KV GB/s");
  for (int n : {4096, 16384}) {
    const double kv_bytes = 2.0 * n * 128 * 8 * 2;
    const double t1 = bench<cpu_ops::bfloat16_t>(32, 8, 1, n, 128, true, 1, 5);
    const double t4 = bench<cpu_ops::bfloat16_t>(32, 8, 1, n, 128, true, 4, 5);
    std::printf("%10d | %10.3f %10.3f | %10.1f\n", n, t1, t4,
                kv_bytes / (t4 * 1e-3) / 1e9);
  }
}

int main(int argc, char** argv) {
  using cpu_ops::bfloat16_t;
  using cpu_ops::float16_t;
  // Block-scope using declarations: ARM's arm_bf16.h defines a global
  // ::bfloat16_t that a using-declaration at namespace scope would collide
  // with at unqualified lookup.
  using bf16_t = bfloat16_t;
  using f16_t = float16_t;

  const Problem probs[] = {
      {"mha_causal", 1, 4, 4, 64, 64, 32, true},
      {"gqa_chunked", 1, 8, 2, 37, 100, 40, true},   // odd sizes, kv > q
      {"mqa_full", 2, 6, 1, 50, 50, 48, false},      // batch, non-causal
      {"gqa_deep", 1, 8, 4, 200, 200, 64, true},     // several q panels
      {"mha_decode", 1, 4, 4, 3, 500, 32, true},     // split-kv territory
  };

  for (const Problem& pr : probs) {
    for (int threads : {1, 4}) {
      // Tolerances are dominated by the output quantization of the storage
      // type (half ulp at |O| <= 1: f32 ~6e-8, f16 ~2.4e-4, bf16 ~2e-3).
      run_case<float>(pr, threads, 1, 5e-4, "f32");
      run_case<f16_t>(pr, threads, 1, 1e-3, "f16");
      run_case<bf16_t>(pr, threads, 1, 4e-3, "bf16");
    }
    // Forced kv-split: correctness against the reference and (checked below)
    // run-to-run determinism. Threads > 1 is required for the split path.
    run_case<float>(pr, 4, 4, 5e-4, "f32");
  }

  // Determinism contracts.
  {
    const Problem pr{"bitwise", 1, 8, 4, 200, 200, 64, true};
    std::vector<float> a, b, c;
    run_stored<float>(pr, 1, 1, &a);  // single thread, no split
    run_stored<float>(pr, 4, 1, &b);  // 4 threads, split off: must match bit-for-bit
    CHECK(a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0,
          "1T vs 4T outputs differ (kv-split off)");
    run_stored<float>(pr, 4, 4, &b);
    run_stored<float>(pr, 4, 4, &c);  // same split config twice: bit-stable
    CHECK(b.size() == c.size() && std::memcmp(b.data(), c.data(), b.size() * sizeof(float)) == 0,
          "kv-split output not deterministic across runs");
  }

  check_error_paths();

  const bool want_bench = argc > 1 && std::strcmp(argv[1], "--bench") == 0;
  if (want_bench) run_benchmark();
  std::printf("fused_attention: %d cases, %d failures%s\n", g_cases, g_failures,
              want_bench ? "" : " (run with --bench for the benchmark)");
  return g_failures == 0 ? 0 : 1;
}
