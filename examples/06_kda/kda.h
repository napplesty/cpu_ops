#pragma once

// KDA — Kimi Delta Attention (Moonshot AI, "Kimi Linear"): a gated
// delta-rule linear attention, with both execution forms the architecture
// is known for:
//
//   * naive (recurrent) — the per-token state machine, one O(dk·dv) pass
//     per token. This is the decode path (constant work per generated
//     token, state carried across calls via s0 / s_out).
//   * chunked — the chunkwise parallel form ("Parallelizing DeltaNet"
//     training algorithm): within a chunk of C tokens the coupled delta
//     corrections decouple into a unit-lower-triangular forward
//     substitution plus attention-shaped kernels; the state update
//     amortizes to O(dk·dv) per *chunk* instead of per token.
//
// Recurrence (the spec; per head, state S ∈ R^{dk×dv}):
//
//   S ← λ_t ⊙ S                     λ_t = σ(gate_logit) ∈ (0,1)^dk,
//   δ_t = v_t − k_tᵀ S              vector-wise gating (KDA's expressivity
//   S ← S + β_t · k_t δ_tᵀ          win over scalar decay); β_t = σ(beta_
//   o_t = (scale·q_t)ᵀ S            logit) ∈ (0,1) is the update strength;
//                                    o reads the *post-update* state.
//
// Chunk algebra (chunk-local indices 1..C, Λ_t = λ_1⊙…⊙λ_t, G = log Λ
// cumulative): unrolling the recurrence gives
//
//   Δ = (I + N)⁻¹ (V − K̃ S₀),   N[t][s] = β_s · Σ_c k_t[c]k_s[c]e^{G_t−G_s}
//                               (strictly lower triangular — Δ is obtained
//                               by forward substitution, never an explicit
//                               inverse)
//   o_t = (q̃_t)ᵀS₀ + Σ_{s≤t} A[t][s]·β_sδ_s,
//         q̃_t = scale·q_t ⊙ e^{G_t},  A[t][s] = scale·Σ_c q_t[c]k_s[c]e^{G_t−G_s}
//   S_new = diag(e^{G_C}) S₀ + Σ_t (k_t ⊙ e^{G_C−G_t}) ⊗ β_tδ_t
//
// Numerics: every decay exponent that appears is a *relative* cumulative
// log-decay G_t − G_s ≤ 0 (log λ ≤ 0), so exp() never overflows and
// underflow is the mathematically correct limit. The kernels are therefore
// computed exactly with pairwise relative exponents — no FLA-style
// factorized-decay clamping approximation. Keys are consumed as given
// (Kimi Linear L2-normalizes k upstream, which keeps N contractive).
//
// Determinism: per (batch, head) tasks run the same fixed-order loops
// (substitution ascends in t, kernels accumulate over lanes in a fixed
// tree), so results are bit-identical across runs and thread counts.
// The two modes agree to float rounding, not bitwise (different order).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>

#include "../04_dsa/dsa.h"  // streaming kernels + run_batched

namespace cpu_ops {
namespace attention {
namespace kda_detail {

using dsa_detail::axpy_f32;
using dsa_detail::del_buf;
using dsa_detail::dot_f32;
using dsa_detail::exp_scale_row;
using dsa_detail::new_fbuf;
using dsa_detail::run_batched;
using dsa_detail::widen_row;

// row += kc * (v − u): the naive mode's per-token state correction.
template <typename T>
inline void delta_row(float kc, const T* v, const float* u, float* row, int n) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  int i = 0;
  for (; i + W <= n; i += W) {
    const FV d = simd::sub(dsa_detail::widen_vec<W>(v + i), FV::load(u + i));
    FV r = simd::fmadd(FV::set1(kc), d, FV::load(row + i));
    r.store(row + i);
  }
  for (; i < n; ++i) row[i] += kc * (static_cast<float>(v[i]) - u[i]);
}

inline float sigmoidf_stable(float x) {
  return 1.0f / (1.0f + std::expf(-x));
}

// log σ(x), stable for both tails: logσ(x) = −softplus(−x).
inline float log_sigmoidf(float x) {
  const float y = -x;  // softplus(y) = log(1 + e^y)
  const float sp = y > 20.0f ? y : std::log1pf(std::expf(y));
  return -sp;
}

// Decay-weighted dot: Σ_c a[c]·b[c]·exp(Ga[c] − Gs[c]). The exponent is
// ≤ 0 by construction (decays), so exp never overflows; lanes accumulate
// in a fixed order.
inline float decay_dot(const float* a, const float* b, const float* ga,
                       const float* gs, int d) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  FV acc = FV::set1(0.0f);
  const FV log2e = FV::set1(1.4426950408889634f);
  int i = 0;
  for (; i + W <= d; i += W) {
    const FV va = FV::load(a + i), vb = FV::load(b + i);
    const FV gd = simd::sub(FV::load(ga + i), FV::load(gs + i));
    const FV e = simd::exp2(simd::mul(gd, log2e));
    acc = simd::fmadd(simd::mul(va, vb), e, acc);
  }
  float s = simd::hsum(acc);
  for (; i < d; ++i) s += a[i] * b[i] * std::expf(ga[i] - gs[i]);
  return s;
}

}  // namespace kda_detail

// Execution form: the naive recurrence (decode path) or the chunkwise
// parallel algebra (prefill / training). Namespace-level so callers can
// pass one mode to any storage instantiation.
enum class KdaMode {
  kNaive,    // per-token recurrence (decode)
  kChunked,  // chunkwise parallel form (prefill / training)
};

template <typename T>
struct KdaAttention {
  static_assert(std::is_same<T, float>::value || std::is_same<T, float16_t>::value ||
                    std::is_same<T, bfloat16_t>::value,
                "KDA storage must be float, float16_t or bfloat16_t");

  struct Arguments {
    int batch = 0;
    int heads = 0;
    int seq_q = 0;

    int dim_qk = 128;  // dk: q/k width, also the state row count
    int dim_v = 128;   // dv: v/output width, state column count
    int chunk = 64;    // chunkwise length C (chunked mode only)

    const T* q = nullptr;  // [b][h][sq][dk]
    int q_stride_b = 0, q_stride_h = 0, q_ld = 0;
    const T* k = nullptr;  // [b][h][sq][dk], L2-normalized upstream
    int k_stride_b = 0, k_stride_h = 0, k_ld = 0;
    const T* v = nullptr;  // [b][h][sq][dv]
    int v_stride_b = 0, v_stride_h = 0, v_ld = 0;
    const T* beta_logit = nullptr;  // [b][h][sq]        -> β = σ(·)
    int beta_stride_b = 0, beta_stride_h = 0;
    const T* gate_logit = nullptr;  // [b][h][sq][dk]    -> λ = σ(·)
    int gate_stride_b = 0, gate_stride_h = 0, gate_ld = 0;

    const T* s0 = nullptr;  // [b][h][dk][dv] initial state (null = zero)
    int s_stride_b = 0, s_stride_h = 0;
    T* s_out = nullptr;  // [b][h][dk][dv] final state (optional)
    int s_out_stride_b = 0, s_out_stride_h = 0;

    T* o = nullptr;  // [b][h][sq][dv]
    int o_stride_b = 0, o_stride_h = 0, o_ld = 0;

    float scale = 1.0f;  // applied to q, typically 1/sqrt(dk)
    KdaMode mode = KdaMode::kChunked;
  };

  // num_threads <= 0 selects the size of the global thread pool.
  Status operator()(const Arguments& args, int num_threads = 0) const;
};

template <typename T>
Status KdaAttention<T>::operator()(const Arguments& args, int num_threads) const {
  const int b = args.batch, hq = args.heads, sq = args.seq_q;
  const int dk = args.dim_qk, dv = args.dim_v, C = args.chunk;
  if (b < 0 || hq < 1 || sq < 0 || dk < 1 || dv < 1 || C < 1) {
    return Status::kErrorInvalidProblem;
  }
  if (sq == 0) return Status::kSuccess;
  if (!args.q || !args.k || !args.v || !args.beta_logit || !args.gate_logit ||
      !args.o) {
    return Status::kErrorInvalidArguments;
  }
  if (args.q_ld < dk || args.k_ld < dk || args.v_ld < dv || args.o_ld < dv ||
      args.gate_ld < dk) {
    return Status::kErrorInvalidArguments;
  }

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  kda_detail::run_batched(pool, b * hq, p, [&](int task, int /*tid*/) {
    const int h = task % hq;
    const int bi = task / hq;
    const auto base = [&](int stride_b, int stride_h) {
      return static_cast<std::size_t>(bi) * stride_b +
             static_cast<std::size_t>(h) * stride_h;
    };
    const T* q = args.q + base(args.q_stride_b, args.q_stride_h);
    const T* k = args.k + base(args.k_stride_b, args.k_stride_h);
    const T* v = args.v + base(args.v_stride_b, args.v_stride_h);
    const T* bl = args.beta_logit + base(args.beta_stride_b, args.beta_stride_h);
    const T* gl = args.gate_logit + base(args.gate_stride_b, args.gate_stride_h);
    T* o = args.o + base(args.o_stride_b, args.o_stride_h);

    // State in f32 across the whole call; rounded to storage only at the
    // s_out boundary (both modes).
    float* S = kda_detail::new_fbuf(static_cast<std::size_t>(dk) * dv);
    const T* s0 = args.s0 ? args.s0 + base(args.s_stride_b, args.s_stride_h) : nullptr;
    for (int i = 0; i < dk * dv; ++i) {
      S[i] = s0 ? static_cast<float>(s0[i]) : 0.0f;
    }

    if (args.mode == KdaMode::kNaive) {
      float* u = kda_detail::new_fbuf(dv);
      float* ot = kda_detail::new_fbuf(dv);
      for (int t = 0; t < sq; ++t) {
        const T* kt = k + static_cast<std::size_t>(t) * args.k_ld;
        const T* vt = v + static_cast<std::size_t>(t) * args.v_ld;
        const T* qt = q + static_cast<std::size_t>(t) * args.q_ld;
        const T* gt = gl + static_cast<std::size_t>(t) * args.gate_ld;
        const float beta = kda_detail::sigmoidf_stable(static_cast<float>(bl[t]));
        for (int c = 0; c < dk; ++c) {
          const float lc = kda_detail::sigmoidf_stable(static_cast<float>(gt[c]));
          for (int j = 0; j < dv; ++j) S[c * dv + j] *= lc;
        }
        // u = kᵀ S (post-decay read)
        for (int j = 0; j < dv; ++j) u[j] = 0.0f;
        for (int c = 0; c < dk; ++c) {
          kda_detail::axpy_f32(static_cast<float>(kt[c]), S + c * dv, u, dv);
        }
        // δ = v − u;  S += β · k ⊗ δ
        for (int c = 0; c < dk; ++c) {
          kda_detail::delta_row(beta * static_cast<float>(kt[c]), vt, u, S + c * dv, dv);
        }
        // o_t = (scale·q)ᵀ S
        for (int j = 0; j < dv; ++j) ot[j] = 0.0f;
        for (int c = 0; c < dk; ++c) {
          kda_detail::axpy_f32(args.scale * static_cast<float>(qt[c]), S + c * dv, ot,
                               dv);
        }
        T* orow = o + static_cast<std::size_t>(t) * args.o_ld;
        for (int j = 0; j < dv; ++j) orow[j] = T(ot[j]);
      }
      kda_detail::del_buf(u);
      kda_detail::del_buf(ot);
    } else {
      // Chunk scratch (f32): widened q̃/k̃/ṽ (q pre-scaled), G cumulative
      // log-decay, decay-combination rows (ktil = k·e^G, qtil = q·e^G,
      // kcar = k·e^{G_C−G} — factors ≤ 1, always exact-safe), midpoint-
      // factored rows (qq/kk/khat, kernel fast path), N/A kernels,
      // rhs/Δ/bd, and [C×dv] panels for the GEMM path.
      float* qw = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* kw = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* vw = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dv);
      float* qq = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* kk = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* khat = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* ktil = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* qtil = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* kcar = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* G = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dk);
      float* gtmp = kda_detail::new_fbuf(dk);
      float* N = kda_detail::new_fbuf(static_cast<std::size_t>(C) * C);
      float* A = kda_detail::new_fbuf(static_cast<std::size_t>(C) * C);
      float* rhs = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dv);
      float* delta = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dv);
      float* bd = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dv);
      float* ot2 = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dv);
      float* abd = kda_detail::new_fbuf(static_cast<std::size_t>(C) * dv);
      float* u = kda_detail::new_fbuf(dv);
      float* ot = kda_detail::new_fbuf(dv);

      // The chunk's six matrix products run on the library GEMM when the
      // kernel factorization is safe (factored chunks); single-threaded per
      // (batch, head) task, like the MLA absorb GEMMs.
      using GemmRM_CM = typename mla_detail::AbsorbGemmFor<float>::Type;
      using GemmRM_RM = gemm::device::Gemm<float, layout::RowMajor, float,
                                           layout::RowMajor, float, layout::RowMajor>;
      using GemmCM_RM = gemm::device::Gemm<float, layout::ColumnMajor, float,
                                           layout::RowMajor, float, layout::RowMajor>;

      for (int c0 = 0; c0 < sq; c0 += C) {
        const int Ce = std::min(C, sq - c0);
        // Widen inputs; q carries the scale; G = cumulative log λ.
        for (int t = 0; t < Ce; ++t) {
          const T* qt = q + static_cast<std::size_t>(c0 + t) * args.q_ld;
          const T* kt = k + static_cast<std::size_t>(c0 + t) * args.k_ld;
          const T* vt = v + static_cast<std::size_t>(c0 + t) * args.v_ld;
          const T* gt = gl + static_cast<std::size_t>(c0 + t) * args.gate_ld;
          float* qrow = qw + static_cast<std::size_t>(t) * dk;
          float* krow = kw + static_cast<std::size_t>(t) * dk;
          float* vrow = vw + static_cast<std::size_t>(t) * dv;
          float* grow = G + static_cast<std::size_t>(t) * dk;
          const float* gprev = t ? G + static_cast<std::size_t>(t - 1) * dk : nullptr;
          kda_detail::widen_row(qt, qrow, dk, args.scale);
          kda_detail::widen_row(kt, krow, dk);
          kda_detail::widen_row(vt, vrow, dv);
          for (int c = 0; c < dk; ++c) {
            const float lg = kda_detail::log_sigmoidf(static_cast<float>(gt[c]));
            grow[c] = lg + (gprev ? gprev[c] : 0.0f);
          }
        }
        const float* glast = G + static_cast<std::size_t>(Ce - 1) * dk;

        // Decay kernels come in two exact flavors. The pairwise kernel
        // ⟨a_t, b_s⟩_decay = Σ_c a[c]·b[c]·e^{G_t[c]−G_s[c]} costs an exp
        // per (pair, channel) — O(C²·dk) exps, which would dominate. When
        // the chunk's decay range stays moderate it factors exactly around
        // the per-channel midpoint m = G_last/2 as
        //   (a_t ⊙ e^{G_t − m}) · (b_s ⊙ e^{m − G_s}),
        // both factors stay within e^±70 while max_c |G_last[c]| ≤ 140,
        // and the t ≥ s products still multiply to e^{G_t−G_s} ≤ 1. More
        // extreme chunks (|G| > 140) fall back to the pairwise kernel,
        // exact for any decay. The branch depends only on the data, so
        // bit-stability is unaffected.
        float R = 0.0f;
        for (int c = 0; c < dk; ++c) R = std::max(R, std::fabs(glast[c]));
        const bool factored = R <= 140.0f;

        // Decay-combination rows on the fast vector exp (all ≤-1 factors
        // except the factored khat side, which the range check covers).
        for (int t = 0; t < Ce; ++t) {
          const float* qrow = qw + static_cast<std::size_t>(t) * dk;
          const float* krow = kw + static_cast<std::size_t>(t) * dk;
          const float* grow = G + static_cast<std::size_t>(t) * dk;
          kda_detail::exp_scale_row(krow, grow, ktil + static_cast<std::size_t>(t) * dk, dk);
          kda_detail::exp_scale_row(qrow, grow, qtil + static_cast<std::size_t>(t) * dk, dk);
          for (int c = 0; c < dk; ++c) gtmp[c] = glast[c] - grow[c];
          kda_detail::exp_scale_row(krow, gtmp, kcar + static_cast<std::size_t>(t) * dk, dk);
        }
        if (factored) {
          for (int t = 0; t < Ce; ++t) {
            const float* krow = kw + static_cast<std::size_t>(t) * dk;
            const float* grow = G + static_cast<std::size_t>(t) * dk;
            for (int c = 0; c < dk; ++c) gtmp[c] = grow[c] - 0.5f * glast[c];
            kda_detail::exp_scale_row(qw + static_cast<std::size_t>(t) * dk, gtmp,
                                      qq + static_cast<std::size_t>(t) * dk, dk);
            kda_detail::exp_scale_row(krow, gtmp, kk + static_cast<std::size_t>(t) * dk, dk);
            for (int c = 0; c < dk; ++c) gtmp[c] = -gtmp[c];
            kda_detail::exp_scale_row(krow, gtmp, khat + static_cast<std::size_t>(t) * dk, dk);
          }
        }

        // Kernels: N[t][s] = β_s·⟨k_t, k_s⟩_decay (strict lower),
        //          A[t][s] = ⟨q_t, k_s⟩_decay (lower, diagonal included).
        if (factored) {
          // Full-square GEMMs; the discarded halves are masked right after.
          GemmRM_CM gemm;
          typename GemmRM_CM::Arguments ga;
          ga.problem_size = {Ce, Ce, dk};
          ga.ref_A = TensorRef<const float, layout::RowMajor>(qq, dk);
          ga.ref_B = TensorRef<const float, layout::ColumnMajor>(khat, dk);
          ga.ref_C = TensorRef<const float, layout::RowMajor>(A, C);
          ga.ref_D = TensorRef<float, layout::RowMajor>(A, C);
          ga.epilogue = {1.0f, 0.0f};
          (void)gemm(ga, 1);
          ga.ref_A = TensorRef<const float, layout::RowMajor>(kk, dk);
          ga.ref_C = TensorRef<const float, layout::RowMajor>(N, C);
          ga.ref_D = TensorRef<float, layout::RowMajor>(N, C);
          (void)gemm(ga, 1);
          for (int t = 0; t < Ce; ++t) {
            for (int s2 = 0; s2 < Ce; ++s2) {
              if (s2 > t) {
                A[t * C + s2] = 0.0f;
              } else if (s2 < t) {
                N[t * C + s2] *=
                    kda_detail::sigmoidf_stable(static_cast<float>(bl[c0 + s2]));
              } else {
                N[t * C + s2] = 0.0f;  // unit diagonal of (I + N)
              }
            }
          }
        } else {
          for (int t = 0; t < Ce; ++t) {
            const float* kt = kw + static_cast<std::size_t>(t) * dk;
            const float* qt = qw + static_cast<std::size_t>(t) * dk;
            const float* gt = G + static_cast<std::size_t>(t) * dk;
            for (int s2 = 0; s2 <= t; ++s2) {
              const float* ks = kw + static_cast<std::size_t>(s2) * dk;
              const float* gs = G + static_cast<std::size_t>(s2) * dk;
              A[t * C + s2] = kda_detail::decay_dot(qt, ks, gt, gs, dk);
              N[t * C + s2] =
                  (t == s2)
                      ? 0.0f
                      : kda_detail::decay_dot(kt, ks, gt, gs, dk) *
                            kda_detail::sigmoidf_stable(static_cast<float>(bl[c0 + s2]));
            }
          }
        }

        // rhs = V − K̃ S₀ (K̃ rows = ktil): GEMM on the factored path.
        if (factored) {
          GemmRM_RM gemm;
          typename GemmRM_RM::Arguments ga;
          ga.problem_size = {Ce, dv, dk};
          ga.ref_A = TensorRef<const float, layout::RowMajor>(ktil, dk);
          ga.ref_B = TensorRef<const float, layout::RowMajor>(S, dv);
          ga.ref_C = TensorRef<const float, layout::RowMajor>(vw, dv);
          ga.ref_D = TensorRef<float, layout::RowMajor>(rhs, dv);
          ga.epilogue = {-1.0f, 1.0f};  // D = −AB + C = V − K̃S
          (void)gemm(ga, 1);
        } else {
          for (int t = 0; t < Ce; ++t) {
            for (int j = 0; j < dv; ++j) u[j] = 0.0f;
            const float* ktr = ktil + static_cast<std::size_t>(t) * dk;
            for (int c = 0; c < dk; ++c) {
              kda_detail::axpy_f32(ktr[c], S + c * dv, u, dv);
            }
            float* rrow = rhs + static_cast<std::size_t>(t) * dv;
            const float* vrow = vw + static_cast<std::size_t>(t) * dv;
            for (int j = 0; j < dv; ++j) rrow[j] = vrow[j] - u[j];
          }
        }

        // Forward substitution: Δ_t = rhs_t − Σ_{s<t} N[t][s]·Δ_s.
        for (int t = 0; t < Ce; ++t) {
          float* drow = delta + static_cast<std::size_t>(t) * dv;
          const float* rrow = rhs + static_cast<std::size_t>(t) * dv;
          std::memcpy(drow, rrow, static_cast<std::size_t>(dv) * sizeof(float));
          for (int s2 = 0; s2 < t; ++s2) {
            kda_detail::axpy_f32(-N[t * C + s2], delta + static_cast<std::size_t>(s2) * dv,
                                 drow, dv);
          }
          const float beta_t =
              kda_detail::sigmoidf_stable(static_cast<float>(bl[c0 + t]));
          float* brow = bd + static_cast<std::size_t>(t) * dv;
          for (int j = 0; j < dv; ++j) brow[j] = beta_t * drow[j];
        }

        // Outputs: o_t = (q_t ⊙ e^{G_t})ᵀS₀ + Σ_{s≤t} A[t][s]·β_sδ_s.
        if (factored) {
          GemmRM_RM gemm;
          typename GemmRM_RM::Arguments ga;
          ga.problem_size = {Ce, dv, dk};
          ga.ref_A = TensorRef<const float, layout::RowMajor>(qtil, dk);
          ga.ref_B = TensorRef<const float, layout::RowMajor>(S, dv);
          ga.ref_C = TensorRef<const float, layout::RowMajor>(ot2, dv);
          ga.ref_D = TensorRef<float, layout::RowMajor>(ot2, dv);
          ga.epilogue = {1.0f, 0.0f};
          (void)gemm(ga, 1);
          ga.problem_size = {Ce, dv, Ce};
          ga.ref_A = TensorRef<const float, layout::RowMajor>(A, C);
          ga.ref_B = TensorRef<const float, layout::RowMajor>(bd, dv);
          ga.ref_C = TensorRef<const float, layout::RowMajor>(abd, dv);
          ga.ref_D = TensorRef<float, layout::RowMajor>(abd, dv);
          (void)gemm(ga, 1);
          for (int t = 0; t < Ce; ++t) {
            T* orow = o + static_cast<std::size_t>(c0 + t) * args.o_ld;
            const float* o1 = ot2 + static_cast<std::size_t>(t) * dv;
            const float* o2 = abd + static_cast<std::size_t>(t) * dv;
            for (int j = 0; j < dv; ++j) orow[j] = T(o1[j] + o2[j]);
          }
        } else {
          for (int t = 0; t < Ce; ++t) {
            const float* qtr = qtil + static_cast<std::size_t>(t) * dk;
            for (int j = 0; j < dv; ++j) ot[j] = 0.0f;
            for (int c = 0; c < dk; ++c) {
              kda_detail::axpy_f32(qtr[c], S + c * dv, ot, dv);
            }
            for (int s2 = 0; s2 <= t; ++s2) {
              kda_detail::axpy_f32(A[t * C + s2], bd + static_cast<std::size_t>(s2) * dv,
                                   ot, dv);
            }
            T* orow = o + static_cast<std::size_t>(c0 + t) * args.o_ld;
            for (int j = 0; j < dv; ++j) orow[j] = T(ot[j]);
          }
        }

        // State: S = diag(e^{G_Ce})S₀ + Σ_t (k_t ⊙ e^{G_Ce−G_t}) ⊗ bd_t —
        // the K̂ᵀ·BD rank-Ce update is a GEMM accumulating in place.
        {
          for (int c = 0; c < dk; ++c) {
            const float lc = std::expf(glast[c]);
            float* row = S + c * dv;
            for (int j = 0; j < dv; ++j) row[j] *= lc;
          }
          if (factored) {
            GemmCM_RM gemm;
            typename GemmCM_RM::Arguments ga;
            ga.problem_size = {dk, dv, Ce};
            ga.ref_A = TensorRef<const float, layout::ColumnMajor>(kcar, dk);
            ga.ref_B = TensorRef<const float, layout::RowMajor>(bd, dv);
            ga.ref_C = TensorRef<const float, layout::RowMajor>(S, dv);
            ga.ref_D = TensorRef<float, layout::RowMajor>(S, dv);
            ga.epilogue = {1.0f, 1.0f};
            (void)gemm(ga, 1);
          } else {
            for (int t = 0; t < Ce; ++t) {
              const float* kcr = kcar + static_cast<std::size_t>(t) * dk;
              const float* brow = bd + static_cast<std::size_t>(t) * dv;
              for (int c = 0; c < dk; ++c) {
                kda_detail::axpy_f32(kcr[c], brow, S + c * dv, dv);
              }
            }
          }
        }
      }
      kda_detail::del_buf(qw);
      kda_detail::del_buf(kw);
      kda_detail::del_buf(vw);
      kda_detail::del_buf(qq);
      kda_detail::del_buf(kk);
      kda_detail::del_buf(khat);
      kda_detail::del_buf(ktil);
      kda_detail::del_buf(qtil);
      kda_detail::del_buf(kcar);
      kda_detail::del_buf(G);
      kda_detail::del_buf(gtmp);
      kda_detail::del_buf(N);
      kda_detail::del_buf(A);
      kda_detail::del_buf(rhs);
      kda_detail::del_buf(delta);
      kda_detail::del_buf(bd);
      kda_detail::del_buf(ot2);
      kda_detail::del_buf(abd);
      kda_detail::del_buf(u);
      kda_detail::del_buf(ot);
    }

    if (args.s_out) {
      T* so = args.s_out + base(args.s_out_stride_b, args.s_out_stride_h);
      for (int i = 0; i < dk * dv; ++i) so[i] = T(S[i]);
    }
    kda_detail::del_buf(S);
  });
  return Status::kSuccess;
}

}  // namespace attention
}  // namespace cpu_ops
