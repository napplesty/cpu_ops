#pragma once

// DeepSeek Sparse Attention (DSA, DeepSeek-V3.2) on top of the MLA latent
// cache, for CPU.
//
// DSA replaces the dense KV sweep of MLA with three cheaper stages:
//
//   1. Lightning indexer — one shared FP8 key per token (indexer_dim wide,
//      cached alongside the latent), per-token per-head queries and mixing
//      weights (DeepSeek-V3.2-Exp: 64 heads x 128 dims, MQA — one shared
//      key head):
//
//        score(t, s) = sum_j w[t, j] * ReLU(q_idx[t, j] . k_idx[s])
//
//      The score is a single scalar per (query, token) pair, so the selected
//      set is shared by every attention head.
//   2. Token selection — the exact top-k tokens of the causal prefix
//      (k = 2048 in V3.2), ties broken toward the smaller index, selected
//      indices sorted ascending. vLLM semantics: fewer than k entries when
//      the prefix is shorter than k.
//   3. Sparse attention over the selected tokens plus one CLS token —
//      a query-specific summary of the *whole* prefix:
//
//        w_cls = softmax(score(t, :))          (causal prefix)
//        kv_cls = sum_s w_cls[s] * cache[s]    (latent ‖ rope, shared by heads)
//
//      appended after the k selected rows as the (k+1)-st KV entry. The CLS
//      weighted sum costs one streaming pass over the shared cache per query
//      row — head-count-independent, i.e. 1/H of the dense score GEMM and
//      exactly the one cache sweep dense decode already pays.
//
// The math is the MLA absorbed form: per head, score = (W_UK·q_nope)·c +
// q_pe·k_pe over the gathered 576-wide rows, V = latent prefix (512), output
// = W_UV·(softmax-weighted latent sum).
//
// CPU choices (deliberately not a port of the GPU kernels):
//   * Exact per-query top-k. A 64k-entry f32 score row is 256 KiB and lives
//     in L2; std::partial_sort is deterministic and exact, so the GPU
//     block-max coarse pass / radix-select machinery buys nothing here.
//   * Gather-then-dense. The selected rows are copied into one contiguous
//     buffer once per query row and shared by every head, instead of being
//     re-gathered inside each head's kernel.
//   * A no-pack streaming row kernel for the attention itself. Each query
//     row owns its selected set, so M = 1 per (row, head); panel packing
//     (the MC/NC blocking in the GEMM core) would re-pack the same gathered
//     panel once per head. The row kernel streams the gathered rows once,
//     keeps q̃ in f32 end to end, and is the first instance of the decode
//     small-M/no-pack kernel on the roadmap.
//   * The indexer runs in f32 (narrow storage widens at load). V3.2 scores
//     in FP8 to spare HBM bandwidth; the CPU analogue is the int8
//     dot-product GEMM primitive, left as future work.
//
// Layout: q_nope/q_pe/kv_cache/w_uk/w_uv/o exactly as MlaAttention
// (examples/03_mla); the indexer tensors are contiguous —
//   q_idx [b][sq][heads_idx][dim_idx], w_mix [b][sq][heads_idx],
//   k_idx [b][skv][dim_idx].
//
// Determinism: every reduction has a fixed evaluation order independent of
// the thread count (indexer dots ascend in s, CLS partials merge in fixed
// slice order, online softmax ascends in gathered-row order), so results
// are bit-identical across runs and thread counts.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "../03_mla/mla.h"  // AbsorbGemmFor: storage-typed GEMM with f32 output
#include "cpu_ops/detail/simd.h"
#include "cpu_ops/detail/thread_pool.h"
#include "cpu_ops/element_types.h"
#include "cpu_ops/status.h"

namespace cpu_ops {
namespace attention {
namespace dsa_detail {

// CLS sum is split into this many fixed s-ranges (independent of the thread
// count, so the merge order — and therefore the result — never varies).
constexpr int kClsSlices = 8;
// Per-row workspace budget; rows are processed in chunks so this bounds the
// peak scratch of a long prefill.
constexpr std::size_t kRowBytesCap = 128u << 20;

inline float* new_fbuf(std::size_t elems) {
  return static_cast<float*>(::operator new(elems * sizeof(float), std::align_val_t(64)));
}
inline int32_t* new_ibuf(std::size_t elems) {
  return static_cast<int32_t*>(::operator new(elems * sizeof(int32_t), std::align_val_t(64)));
}
inline void del_buf(void* p) { ::operator delete(p, std::align_val_t(64)); }

// fn(task, tid) over [0, n) on at most p threads (the pool always engages
// all workers, so tasks go out in p-sized batches).
template <typename F>
void run_batched(thread::ThreadPool& pool, int n, int p, F&& fn) {
  if (n <= 0) return;
  if (p <= 1 || n == 1) {
    for (int t = 0; t < n; ++t) fn(t, 0);
    return;
  }
  for (int base = 0; base < n; base += p) {
    const int b0 = base;
    pool.parallel_for(std::min(p, n - base),
                      [&fn, b0](int i, int tid) { fn(b0 + i, tid); });
  }
}

// Vector widen dispatch: storage-typed rows enter the streaming kernels as
// f32 lanes (single-instruction converts on NEON/AVX2, scalar elsewhere).
template <int W, typename T>
inline simd::Vec<float, W> widen_vec(const T* p) {
  if constexpr (std::is_same<T, float>::value) {
    return simd::Vec<float, W>::load(p);
  } else if constexpr (std::is_same<T, float16_t>::value) {
    return simd::widen_f16<W>(reinterpret_cast<const uint16_t*>(p));
  } else {
    return simd::widen_bf16<W>(reinterpret_cast<const uint16_t*>(p));
  }
}

// Streaming kernels: fixed ascending order, f32 accumulation, narrow
// storage widened in W-wide chunks. These are the whole point of the row
// path — no panel packing anywhere. The two operand types differ where an
// f32 accumulator row meets storage-typed data.

template <typename TA, typename TB>
inline float dot_f32(const TA* a, const TB* b, int n) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  FV acc = FV::set1(0.0f);
  int i = 0;
  for (; i + W <= n; i += W) {
    acc = simd::fmadd(widen_vec<W>(a + i), widen_vec<W>(b + i), acc);
  }
  float s = simd::hsum(acc);
  for (; i < n; ++i) s += static_cast<float>(a[i]) * static_cast<float>(b[i]);
  return s;
}

template <typename T>
inline void axpy_f32(float a, const T* x, float* y, int n) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  int i = 0;
  for (; i + W <= n; i += W) {
    FV v = simd::fmadd(FV::set1(a), widen_vec<W>(x + i), FV::load(y + i));
    v.store(y + i);
  }
  for (; i < n; ++i) y[i] += a * static_cast<float>(x[i]);
}

// Row-wise widening copy: y[i] = s * float(x[i]) (s = 1 for plain copies).
template <typename T>
inline void widen_row(const T* x, float* y, int n, float s = 1.0f) {
  constexpr int W = simd::native_width<float>();
  using FV = simd::Vec<float, W>;
  int i = 0;
  for (; i + W <= n; i += W) {
    FV v = widen_vec<W>(x + i);
    if (s != 1.0f) v = simd::mul(v, FV::set1(s));
    v.store(y + i);
  }
  for (; i < n; ++i) y[i] = s * static_cast<float>(x[i]);
}

}  // namespace dsa_detail

template <typename T>
struct DsaAttention {
  static_assert(std::is_same<T, float>::value || std::is_same<T, float16_t>::value ||
                    std::is_same<T, bfloat16_t>::value,
                "DSA storage must be float, float16_t or bfloat16_t");

  struct Arguments {
    int batch = 0;
    int heads = 0;  // H query heads; KV is shared (one latent per token)
    int seq_q = 0;
    int seq_kv = 0;

    int dim_nope = 128;    // dn: q_nope width; also the output width
    int dim_pe = 64;       // dp: RoPE width, appended to the latent cache
    int dim_latent = 512;  // dc: latent width (c_kv and the value extent)
    int heads_idx = 1;     // indexer heads (V3.2: 64)
    int dim_idx = 128;     // indexer head dim (V3.2: 128, one shared key)
    int topk = 2048;       // selected tokens per query (clamped to prefix)

    const T* q_nope = nullptr;  // [b][h][sq][dn]
    int q_nope_stride_b = 0, q_nope_stride_h = 0, q_nope_ld = 0;
    const T* q_pe = nullptr;  // [b][h][sq][dp], pre-rotated
    int q_pe_stride_b = 0, q_pe_stride_h = 0, q_pe_ld = 0;
    const T* kv_cache = nullptr;  // [b][skv][dc + dp]: c(t) ‖ k_pe(t), shared
    int kv_stride_b = 0, kv_ld = 0;
    const T* w_uk = nullptr;  // [h][dc][dn] row-major; q̃ = q_nope · W_UK_h^T
    int w_uk_stride_h = 0;
    const T* w_uv = nullptr;  // [h][dn][dc] row-major; o = õ · W_UV_h^T
    int w_uv_stride_h = 0;

    // Lightning indexer (contiguous). The score is head-summed, so one
    // selected set — and one CLS token — per query row, shared by all H
    // attention heads.
    const T* q_idx = nullptr;  // [b][sq][heads_idx][dim_idx]
    const T* w_mix = nullptr;  // [b][sq][heads_idx]
    const T* k_idx = nullptr;  // [b][skv][dim_idx], shared key cache

    // Optional f32 mirror of the indexer scores, [b][sq][skv]; entries at
    // or beyond a row's causal prefix are written as 0. The selected set
    // and CLS weights are *defined* on these f32 scores, so a reference
    // implementation reuses them instead of re-deriving (and re-rounding)
    // the dots.
    float* indexer_scores_out = nullptr;

    T* o = nullptr;  // [b][h][sq][dn]
    int o_stride_b = 0, o_stride_h = 0, o_ld = 0;

    float scale = 1.0f;          // attention scale, 1/sqrt(dn + dp)
    float indexer_scale = 1.0f;  // indexer dot scale, 1/sqrt(dim_idx)
    bool causal = true;          // query i attends kv j <= i + (seq_kv - seq_q)
    bool use_cls = true;         // append the softmax-weighted CLS token
  };

  // num_threads <= 0 selects the size of the global thread pool.
  Status operator()(const Arguments& args, int num_threads = 0) const;
};

template <typename T>
Status DsaAttention<T>::operator()(const Arguments& args, int num_threads) const {
  const int b = args.batch, hq = args.heads, sq = args.seq_q, skv = args.seq_kv;
  const int dn = args.dim_nope, dp = args.dim_pe, dc = args.dim_latent;
  const int hi = args.heads_idx, di = args.dim_idx, topk = args.topk;
  if (b < 0 || hq < 1 || sq < 0 || skv < 0 || dn < 1 || dp < 1 || dc < 1 || hi < 1 ||
      di < 1 || topk < 1) {
    return Status::kErrorInvalidProblem;
  }
  if (sq == 0) return Status::kSuccess;
  if (skv == 0) return Status::kErrorInvalidProblem;
  if (args.causal && skv < sq) return Status::kErrorInvalidArguments;
  if (!args.q_nope || !args.q_pe || !args.kv_cache || !args.w_uk || !args.w_uv ||
      !args.q_idx || !args.w_mix || !args.k_idx || !args.o) {
    return Status::kErrorInvalidArguments;
  }
  if (args.q_nope_ld < dn || args.q_pe_ld < dp || args.kv_ld < dc + dp ||
      args.o_ld < dn || args.w_uk_stride_h < dc * dn || args.w_uv_stride_h < dn * dc) {
    return Status::kErrorInvalidArguments;
  }

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  const int dcat = dc + dp;

  // Per-row workspace, in elements:
  //   dots[hi][skv] f32 | score[skv] f32 | cls[kClsSlices][dcat + 1] f32
  //   idx[skv + topk + 1] i32 | gbuf[(topk + 1) * dcat] T
  const std::size_t dots_elems = static_cast<std::size_t>(hi) * skv;
  const std::size_t score_elems = static_cast<std::size_t>(skv);
  const std::size_t cls_elems =
      static_cast<std::size_t>(dsa_detail::kClsSlices) * (dcat + 1);
  const std::size_t idx_elems = static_cast<std::size_t>(skv) + topk + 1;
  const std::size_t gbuf_elems = static_cast<std::size_t>(topk + 1) * dcat;
  const std::size_t row_bytes =
      (dots_elems + score_elems + cls_elems) * sizeof(float) +
      idx_elems * sizeof(int32_t) + gbuf_elems * sizeof(T);
  int rows_per_chunk = static_cast<int>(dsa_detail::kRowBytesCap / row_bytes);
  if (rows_per_chunk < 1) rows_per_chunk = 1;

  using AbsorbGemm = typename mla_detail::AbsorbGemmFor<T>::Type;

  // Chunks stay inside one batch (the indexer GEMM below needs a single
  // q_idx base pointer with a constant row stride).
  for (int bi = 0; bi < b; ++bi) {
    for (int r0 = 0; r0 < sq; r0 += rows_per_chunk) {
    const int rows = std::min(rows_per_chunk, sq - r0);

    float* dots = dsa_detail::new_fbuf(dots_elems * rows);
    float* score = dsa_detail::new_fbuf(score_elems * rows);
    float* cls = dsa_detail::new_fbuf(cls_elems * rows);
    int32_t* idx = dsa_detail::new_ibuf(idx_elems * rows);
    T* gbuf = static_cast<T*>(
        ::operator new(gbuf_elems * sizeof(T) * rows, std::align_val_t(64)));

    // ---- 1a: indexer dots ----------------------------------------------
    // Prefill-sized chunks run one GEMM per indexer head — [rows×di] ·
    // (k_idx as column-major [di×skv]) with the dot scale in the epilogue
    // — while decode-sized chunks (rows < 8) keep the no-pack streaming
    // GEMV, which wins when M = 1 (a GEMM call would re-pack the whole
    // key cache once per indexer head). The chunk size depends only on the
    // problem shape, so both paths stay bit-stable across thread counts.
    const T* ki_b = args.k_idx + static_cast<std::size_t>(bi) * skv * di;
    if (rows >= 8) {
      AbsorbGemm gemm;
      dsa_detail::run_batched(pool, hi, p, [&](int j, int /*tid*/) {
        // q_idx rows for this chunk, strided by the [hi][di] grouping.
        const T* qa = args.q_idx +
                      (static_cast<std::size_t>(bi) * sq + r0) * hi * di +
                      static_cast<std::size_t>(j) * di;
        float* out = dots + static_cast<std::size_t>(j) * skv;  // ld = hi*skv
        typename AbsorbGemm::Arguments ga;
        ga.problem_size = {rows, skv, di};
        ga.ref_A = TensorRef<const T, layout::RowMajor>(qa, hi * di);
        ga.ref_B = TensorRef<const T, layout::ColumnMajor>(ki_b, di);
        ga.ref_C = TensorRef<const float, layout::RowMajor>(out, hi * skv);
        ga.ref_D = TensorRef<float, layout::RowMajor>(out, hi * skv);
        ga.epilogue = {args.indexer_scale, 0.0f};
        (void)gemm(ga, 1);  // dimensions pre-validated
      });
    } else {
      // Decode-sized chunks keep a no-pack GEMV, but blocked over indexer
      // heads: one k_idx stream pass serves HB heads (the naive per-head
      // GEMV would stream the whole key cache hi times — at 64 heads and
      // 64k tokens that is ~1 GB per query row, pure bandwidth wall). Both
      // operands are widened once per pass, so the inner loop is a plain
      // f32 dot and the per-(row, head) results are bit-identical to the
      // unblocked form.
      constexpr int HB = 8;
      // Per-thread widen scratch (tid slots): tasks of one wave run
      // concurrently and would race on shared buffers. parallel_for hands
      // out the worker's pool id (0 .. pool.num_threads()-1), not a slot in
      // [0, p), so size by the pool.
      const int nslots = pool.num_threads();
      float* qf = dsa_detail::new_fbuf(static_cast<std::size_t>(nslots) * HB * di);
      float* kf = dsa_detail::new_fbuf(static_cast<std::size_t>(nslots) * di);
      const int jblocks = (hi + HB - 1) / HB;
      dsa_detail::run_batched(pool, rows * jblocks, p, [&](int task, int tid) {
        const int r = task % rows;
        const int j0 = (task / rows) * HB;
        const int jb = std::min(HB, hi - j0);
        float* qft = qf + static_cast<std::size_t>(tid) * HB * di;
        float* kft = kf + static_cast<std::size_t>(tid) * di;
        const T* qb = args.q_idx +
                      (static_cast<std::size_t>(bi) * sq + r0 + r) * hi * di +
                      static_cast<std::size_t>(j0) * di;
        for (int j = 0; j < jb; ++j) {
          dsa_detail::widen_row(qb + static_cast<std::size_t>(j) * di,
                                qft + static_cast<std::size_t>(j) * di, di);
        }
        float* out = dots + static_cast<std::size_t>(r) * dots_elems +
                     static_cast<std::size_t>(j0) * skv;
        for (int s = 0; s < skv; ++s) {
          dsa_detail::widen_row(ki_b + static_cast<std::size_t>(s) * di, kft, di);
          for (int j = 0; j < jb; ++j) {
            out[static_cast<std::size_t>(j) * skv + s] =
                args.indexer_scale *
                dsa_detail::dot_f32(qft + static_cast<std::size_t>(j) * di, kft, di);
          }
        }
      });
      dsa_detail::del_buf(qf);
      dsa_detail::del_buf(kf);
    }

    // ---- 1b: head-mix, score mirror, exact top-k, gather, CLS weights --
    // (top-k runs on the raw scores; the softmax weights overwrite the
    // score row only after selection, since exp is not injective in f32.)
    dsa_detail::run_batched(pool, rows, p, [&](int r, int /*tid*/) {
      const int t = r0 + r;
      const int n_i = args.causal ? skv - sq + t + 1 : skv;
      const T* w = args.w_mix + (static_cast<std::size_t>(bi) * sq + t) * hi;
      float* sc = score + static_cast<std::size_t>(r) * skv;

      for (int s = 0; s < n_i; ++s) sc[s] = 0.0f;
      for (int j = 0; j < hi; ++j) {
        const float wj = static_cast<float>(w[j]);
        if (wj == 0.0f) continue;  // exact: wj * ReLU(d) adds nothing
        const float* dj = dots + static_cast<std::size_t>(r) * dots_elems +
                          static_cast<std::size_t>(j) * skv;
        for (int s = 0; s < n_i; ++s) {
          sc[s] += wj * (dj[s] > 0.0f ? dj[s] : 0.0f);  // w * ReLU(dot)
        }
      }
      if (args.indexer_scores_out) {
        float* out = args.indexer_scores_out +
                     (static_cast<std::size_t>(bi) * sq + t) * skv;
        std::memcpy(out, sc, static_cast<std::size_t>(n_i) * sizeof(float));
        for (int s = n_i; s < skv; ++s) out[s] = 0.0f;
      }

      // Exact top-k over [0, n_i): (score desc, index asc) totally orders
      // the candidates, so the set is deterministic; gathered rows go back
      // to ascending order. sel[topk] publishes k_eff to phases 1c and 2.
      const int k_eff = std::min(topk, n_i);
      int32_t* id = idx + static_cast<std::size_t>(r) * idx_elems;
      for (int s = 0; s < n_i; ++s) id[s] = s;
      const auto by_score = [&sc](int32_t x, int32_t y) {
        return sc[x] != sc[y] ? sc[x] > sc[y] : x < y;
      };
      std::partial_sort(id, id + k_eff, id + n_i, by_score);
      int32_t* sel = id + skv;
      std::copy(id, id + k_eff, sel);
      std::sort(sel, sel + k_eff);
      sel[topk] = k_eff;

      T* gb = gbuf + static_cast<std::size_t>(r) * gbuf_elems;
      const T* cache = args.kv_cache + static_cast<std::size_t>(bi) * args.kv_stride_b;
      for (int r2 = 0; r2 < k_eff; ++r2) {
        std::memcpy(gb + static_cast<std::size_t>(r2) * dcat,
                    cache + static_cast<std::size_t>(sel[r2]) * args.kv_ld,
                    static_cast<std::size_t>(dcat) * sizeof(T));
      }

      // CLS weights: softmax over the causal prefix, kept unnormalized in
      // sc (the partial-sum phase carries the denominator).
      if (args.use_cls) {
        float m = sc[0];
        for (int s = 1; s < n_i; ++s) m = std::max(m, sc[s]);
        for (int s = 0; s < n_i; ++s) sc[s] = std::expf(sc[s] - m);
      }
    });

    // ---- 1c: CLS partial sums over fixed s-ranges, merged in order -----
    if (args.use_cls) {
      dsa_detail::run_batched(pool, rows * dsa_detail::kClsSlices, p,
                              [&](int task, int /*tid*/) {
        const int r = task % rows;
        const int sl = task / rows;
        const int t = r0 + r;
        const int n_i = args.causal ? skv - sq + t + 1 : skv;
        const int s0 = static_cast<int>(
            static_cast<long long>(n_i) * sl / dsa_detail::kClsSlices);
        const int s1 = static_cast<int>(
            static_cast<long long>(n_i) * (sl + 1) / dsa_detail::kClsSlices);
        const float* sc = score + static_cast<std::size_t>(r) * skv;
        float* part = cls + static_cast<std::size_t>(r) * cls_elems +
                      static_cast<std::size_t>(sl) * (dcat + 1);
        for (int d = 0; d < dcat + 1; ++d) part[d] = 0.0f;
        const T* cache =
            args.kv_cache + static_cast<std::size_t>(bi) * args.kv_stride_b;
        for (int s = s0; s < s1; ++s) {
          dsa_detail::axpy_f32(sc[s], cache + static_cast<std::size_t>(s) * args.kv_ld,
                               part, dcat);
          part[dcat] += sc[s];
        }
      });
      dsa_detail::run_batched(pool, rows, p, [&](int r, int /*tid*/) {
        const int k_eff = idx[static_cast<std::size_t>(r) * idx_elems + skv + topk];
        float* acc = dsa_detail::new_fbuf(dcat);
        for (int d = 0; d < dcat; ++d) acc[d] = 0.0f;
        float denom = 0.0f;
        for (int sl = 0; sl < dsa_detail::kClsSlices; ++sl) {  // fixed order
          const float* part = cls + static_cast<std::size_t>(r) * cls_elems +
                              static_cast<std::size_t>(sl) * (dcat + 1);
          for (int d = 0; d < dcat; ++d) acc[d] += part[d];
          denom += part[dcat];
        }
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        T* gb = gbuf + static_cast<std::size_t>(r) * gbuf_elems;
        T* cls_row = gb + static_cast<std::size_t>(k_eff) * dcat;
        for (int d = 0; d < dcat; ++d) cls_row[d] = T(acc[d] * inv);
        dsa_detail::del_buf(acc);
      });
    }

    // ---- 2: absorbed sparse attention, one (row, head) task ------------
    dsa_detail::run_batched(pool, rows * hq, p, [&](int task, int /*tid*/) {
      const int r = task % rows;
      const int h = task / rows;
      const int t = r0 + r;
      const T* qn = args.q_nope + static_cast<std::size_t>(bi) * args.q_nope_stride_b +
                    static_cast<std::size_t>(h) * args.q_nope_stride_h +
                    static_cast<std::size_t>(t) * args.q_nope_ld;
      const T* qp = args.q_pe + static_cast<std::size_t>(bi) * args.q_pe_stride_b +
                    static_cast<std::size_t>(h) * args.q_pe_stride_h +
                    static_cast<std::size_t>(t) * args.q_pe_ld;
      const T* wuk = args.w_uk + static_cast<std::size_t>(h) * args.w_uk_stride_h;
      const T* wuv = args.w_uv + static_cast<std::size_t>(h) * args.w_uv_stride_h;

      // q̃ = [W_UK_h · q_nope | q_pe], f32 throughout (no storage round-trip).
      float* qt = dsa_detail::new_fbuf(dcat);
      for (int c = 0; c < dc; ++c) {
        qt[c] = dsa_detail::dot_f32(qn, wuk + static_cast<std::size_t>(c) * dn, dn);
      }
      for (int j = 0; j < dp; ++j) qt[dc + j] = static_cast<float>(qp[j]);

      const T* gb = gbuf + static_cast<std::size_t>(r) * gbuf_elems;
      const int32_t* sel = idx + static_cast<std::size_t>(r) * idx_elems + skv;
      const int k_eff = sel[topk];
      const int nsel = k_eff + (args.use_cls ? 1 : 0);

      // Online softmax over the gathered rows, ascending row order; the
      // V extent is the latent prefix of each row.
      float* O = dsa_detail::new_fbuf(dc);
      for (int d = 0; d < dc; ++d) O[d] = 0.0f;
      float m = -1.0e30f, l = 0.0f;
      for (int r2 = 0; r2 < nsel; ++r2) {
        const T* row = gb + static_cast<std::size_t>(r2) * dcat;
        const float s = args.scale * dsa_detail::dot_f32(qt, row, dcat);
        const float m_new = std::max(m, s);
        const float rescale = m > -1.0e29f ? std::expf(m - m_new) : 0.0f;
        const float p = std::expf(s - m_new);
        if (rescale != 1.0f) {
          for (int d = 0; d < dc; ++d) O[d] *= rescale;
        }
        dsa_detail::axpy_f32(p, row, O, dc);
        l = l * rescale + p;
        m = m_new;
      }

      // o = W_UV_h · (O / l), one streaming dot per output element.
      T* out = args.o + static_cast<std::size_t>(bi) * args.o_stride_b +
               static_cast<std::size_t>(h) * args.o_stride_h +
               static_cast<std::size_t>(t) * args.o_ld;
      const float inv = l > 0.0f ? 1.0f / l : 0.0f;
      for (int d = 0; d < dc; ++d) O[d] *= inv;
      for (int e = 0; e < dn; ++e) {
        out[e] = T(dsa_detail::dot_f32(O, wuv + static_cast<std::size_t>(e) * dc, dc));
      }
      dsa_detail::del_buf(qt);
      dsa_detail::del_buf(O);
    });

    dsa_detail::del_buf(dots);
    dsa_detail::del_buf(score);
    dsa_detail::del_buf(cls);
    dsa_detail::del_buf(idx);
    ::operator delete(gbuf, std::align_val_t(64));
    }
  }
  return Status::kSuccess;
}

}  // namespace attention
}  // namespace cpu_ops
