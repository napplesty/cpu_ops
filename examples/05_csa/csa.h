#pragma once

// Compressed Sparse Attention (CSA) — the Native Sparse Attention design
// (DeepSeek, arXiv 2502.11089) assembled on the MLA latent cache, for CPU.
//
// Three branches run per query and are summed with learned sigmoid gates:
//
//   o_t = g_cmp·Attn(q_t, K̃, Ṽ) + g_slc·Attn(q_t, KV[selected blocks])
//       + g_win·Attn(q_t, KV[last w tokens])
//
//   * Compressed: one coarse row per block (the paper's φ is a learnable
//     MLP over the block with intra-block positions, l=32/d=16; here the
//     aligned degenerate case l = d = l' = one `block` parameter, and the
//     mapping is either an input cache — kv_compressed [b][nblk][dc+dp],
//     deployment-realistic, φ applied upstream — or, when null, an
//     on-the-fly block mean, documented as a simplification).
//   * Selection: block importance reuses the compressed branch *post-
//     softmax* scores, summed over heads (the paper's GQA/MQA rule) —
//     so one selected block set per query row, shared by every head. The
//     top-`select_blocks` 64-token blocks are gathered whole and attended
//     at token granularity, with the paper's fixed activations folded in:
//     the first eligible block and the two most recent eligible blocks
//     count toward the budget.
//   * Window: the last `window` tokens through p_t, contiguous, no gather.
//
// Causal-frontier rule (the paper does not spell this out): the compressed
// and selected branches see only *fully completed* blocks (block l is
// eligible for query at global position p iff (l+1)·block ≤ p+1); the
// partial block containing p is covered by the window branch. Non-causal
// problems make every block (including the tail partial block) eligible
// and take the window from the end of the sequence.
//
// Everything runs on the no-pack streaming row kernel introduced by the
// DSA example (dot_f32 / axpy_f32): per (row, head) the attention stages
// are online-softmax passes over centroid rows, gathered rows, or a
// contiguous cache range — q̃ stays f32 throughout. Per batch, phase A
// (rows × heads) computes q̃, the compressed pass and its per-head
// softmaxed block scores; phase A2 (rows) head-sums the scores, selects,
// and gathers whole blocks once for all heads; phase B (rows × heads)
// runs the selected and window passes, applies the gates and un-absorbs.
//
// Determinism: block centroids in fixed order, per-head softmax and the
// head sum over ascending blocks, top-n by (score desc, block asc) with a
// fixed reserved rule, branch passes over ascending rows — results are
// bit-identical across runs and thread counts.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <type_traits>
#include <utility>

#include "../04_dsa/dsa.h"  // dsa_detail streaming kernels + run_batched

namespace cpu_ops {
namespace attention {
namespace csa_detail {

using dsa_detail::axpy_f32;
using dsa_detail::del_buf;
using dsa_detail::dot_f32;
using dsa_detail::new_fbuf;
using dsa_detail::run_batched;

// One online-softmax pass over n_rows rows of width dcat; accumulates the
// first dc columns into O (unnormalized: O = Σ exp(s−m)·row[:dc]) and the
// raw scores into s_scratch; returns the denominator l. row_fn(r) yields
// row r's base pointer (type RT = float for centroids, T for cache rows).
template <typename RT, typename RowFn>
inline float row_attn(const float* qt, int dcat, int dc, float scale, int n_rows,
                      RowFn&& row_fn, float* O, float* s_scratch) {
  for (int d = 0; d < dc; ++d) O[d] = 0.0f;
  float m = -1.0e30f, l = 0.0f;
  for (int r = 0; r < n_rows; ++r) {
    const RT* row = row_fn(r);
    const float s = scale * dsa_detail::dot_f32(qt, row, dcat);
    s_scratch[r] = s;
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
  return l;
}

}  // namespace csa_detail

template <typename T>
struct CsaAttention {
  static_assert(std::is_same<T, float>::value || std::is_same<T, float16_t>::value ||
                    std::is_same<T, bfloat16_t>::value,
                "CSA storage must be float, float16_t or bfloat16_t");

  struct Arguments {
    int batch = 0;
    int heads = 0;  // H query heads; KV is shared (one latent per token)
    int seq_q = 0;
    int seq_kv = 0;

    int dim_nope = 128;    // dn: q_nope width; also the output width
    int dim_pe = 64;       // dp: RoPE width, appended to the latent cache
    int dim_latent = 512;  // dc: latent width (c_kv and the value extent)
    int block = 64;        // compression/selection block size (paper l' = 64)
    int select_blocks = 16;  // n: top blocks per query, fixed ones included
    int window = 512;        // w: sliding window tokens

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

    // Compressed KV cache, [b][nblk][dc + dp] with nblk = ceil(skv / block)
    // — the paper's learnable φ applied upstream. null = on-the-fly block
    // means (simplification; the branch structure is identical).
    const T* kv_compressed = nullptr;
    int compressed_stride_b = 0, compressed_ld = 0;

    // Branch gate logits (cmp, slc, win order), sigmoid applied here.
    const T* gate_logits = nullptr;  // [b][h][sq][3]

    // Optional f32 mirror of the head-summed, softmaxed block scores,
    // [b][sq][nblk]; blocks beyond a row's causal eligibility are 0. The
    // selected set is *defined* on these f32 scores (same spec-anchor
    // pattern as the DSA example's indexer mirror).
    float* block_scores_out = nullptr;

    T* o = nullptr;  // [b][h][sq][dn]
    int o_stride_b = 0, o_stride_h = 0, o_ld = 0;

    float scale = 1.0f;  // attention scale, 1/sqrt(dn + dp)
    bool causal = true;  // query i attends kv j <= i + (seq_kv - seq_q)
  };

  // num_threads <= 0 selects the size of the global thread pool.
  Status operator()(const Arguments& args, int num_threads = 0) const;
};

template <typename T>
Status CsaAttention<T>::operator()(const Arguments& args, int num_threads) const {
  const int b = args.batch, hq = args.heads, sq = args.seq_q, skv = args.seq_kv;
  const int dn = args.dim_nope, dp = args.dim_pe, dc = args.dim_latent;
  const int g = args.block, nsel = args.select_blocks, w = args.window;
  if (b < 0 || hq < 1 || sq < 0 || skv < 0 || dn < 1 || dp < 1 || dc < 1 || g < 1 ||
      nsel < 1 || w < 1) {
    return Status::kErrorInvalidProblem;
  }
  if (sq == 0) return Status::kSuccess;
  if (skv == 0) return Status::kErrorInvalidProblem;
  if (args.causal && skv < sq) return Status::kErrorInvalidArguments;
  if (!args.q_nope || !args.q_pe || !args.kv_cache || !args.w_uk || !args.w_uv ||
      !args.gate_logits || !args.o) {
    return Status::kErrorInvalidArguments;
  }
  const int dcat = dc + dp;
  const int nblk = (skv + g - 1) / g;
  if (args.kv_compressed &&
      (args.compressed_ld < dcat || args.compressed_stride_b < nblk * dcat)) {
    return Status::kErrorInvalidArguments;
  }
  if (args.q_nope_ld < dn || args.q_pe_ld < dp || args.kv_ld < dcat || args.o_ld < dn ||
      args.w_uk_stride_h < dc * dn || args.w_uv_stride_h < dn * dc) {
    return Status::kErrorInvalidArguments;
  }

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  // Per-row workspace, in elements (chunked; budget from the DSA example):
  //   qt[h][dcat] f32 | oc[h][dc] f32 | lh[h] f32 | csc[h][nblk] f32
  //   bs[nblk] f32 | sel[nsel + 1] i32 | gbuf[nsel * g * dcat] T
  const std::size_t qt_elems = static_cast<std::size_t>(hq) * dcat;
  const std::size_t oc_elems = static_cast<std::size_t>(hq) * dc;
  const std::size_t csc_elems = static_cast<std::size_t>(hq) * nblk;
  const std::size_t bs_elems = nblk;
  const std::size_t sel_elems = nsel + 1;
  const std::size_t gbuf_elems = static_cast<std::size_t>(nsel) * g * dcat;
  const std::size_t row_bytes =
      (qt_elems + oc_elems + csc_elems + bs_elems + hq) * sizeof(float) +
      sel_elems * sizeof(int32_t) + gbuf_elems * sizeof(T);
  int rows_per_chunk = static_cast<int>(dsa_detail::kRowBytesCap / row_bytes);
  if (rows_per_chunk < 1) rows_per_chunk = 1;

  const int kv_off = skv - sq;  // causal: global query position p = kv_off + t

  for (int bi = 0; bi < b; ++bi) {
    // ---- 0: block centroids, f32 [nblk][dcat], shared by rows and heads.
    float* centroids = csa_detail::new_fbuf(static_cast<std::size_t>(nblk) * dcat);
    const T* cache = args.kv_cache + static_cast<std::size_t>(bi) * args.kv_stride_b;
    if (args.kv_compressed) {
      const T* kc = args.kv_compressed +
                    static_cast<std::size_t>(bi) * args.compressed_stride_b;
      csa_detail::run_batched(pool, nblk, p, [&](int l, int /*tid*/) {
        const T* row = kc + static_cast<std::size_t>(l) * args.compressed_ld;
        float* out = centroids + static_cast<std::size_t>(l) * dcat;
        for (int d = 0; d < dcat; ++d) out[d] = static_cast<float>(row[d]);
      });
    } else {
      csa_detail::run_batched(pool, nblk, p, [&](int l, int /*tid*/) {
        const int s0 = l * g;
        const int cnt = std::min(g, skv - s0);  // tail partial block
        float* out = centroids + static_cast<std::size_t>(l) * dcat;
        for (int d = 0; d < dcat; ++d) out[d] = 0.0f;
        for (int s = 0; s < cnt; ++s) {
          csa_detail::axpy_f32(1.0f, cache + static_cast<std::size_t>(s0 + s) * args.kv_ld,
                               out, dcat);
        }
        const float inv = 1.0f / static_cast<float>(cnt);
        for (int d = 0; d < dcat; ++d) out[d] *= inv;
      });
    }

    for (int r0 = 0; r0 < sq; r0 += rows_per_chunk) {
      const int rows = std::min(rows_per_chunk, sq - r0);

      float* qt = csa_detail::new_fbuf(qt_elems * rows);
      float* oc = csa_detail::new_fbuf(oc_elems * rows);
      float* lh = csa_detail::new_fbuf(static_cast<std::size_t>(hq) * rows);
      float* csc = csa_detail::new_fbuf(csc_elems * rows);
      float* bs = csa_detail::new_fbuf(bs_elems * rows);
      int32_t* sel = static_cast<int32_t*>(
          ::operator new(sel_elems * sizeof(int32_t) * rows, std::align_val_t(64)));
      T* gbuf = static_cast<T*>(
          ::operator new(gbuf_elems * sizeof(T) * rows, std::align_val_t(64)));

      // ---- A: per (row, head) q̃ and the compressed pass ---------------
      csa_detail::run_batched(pool, rows * hq, p, [&](int task, int /*tid*/) {
        const int r = task % rows;
        const int h = task / rows;
        const int t = r0 + r;
        const int p_t = args.causal ? kv_off + t : skv - 1;
        const int n_elig = args.causal ? (p_t + 1) / g : nblk;

        const T* qn = args.q_nope + static_cast<std::size_t>(bi) * args.q_nope_stride_b +
                      static_cast<std::size_t>(h) * args.q_nope_stride_h +
                      static_cast<std::size_t>(t) * args.q_nope_ld;
        const T* qp = args.q_pe + static_cast<std::size_t>(bi) * args.q_pe_stride_b +
                      static_cast<std::size_t>(h) * args.q_pe_stride_h +
                      static_cast<std::size_t>(t) * args.q_pe_ld;
        const T* wuk = args.w_uk + static_cast<std::size_t>(h) * args.w_uk_stride_h;
        float* qt_rh = qt + static_cast<std::size_t>(r) * qt_elems +
                       static_cast<std::size_t>(h) * dcat;
        for (int c = 0; c < dc; ++c) {
          qt_rh[c] = dsa_detail::dot_f32(qn, wuk + static_cast<std::size_t>(c) * dn, dn);
        }
        for (int j = 0; j < dp; ++j) qt_rh[dc + j] = static_cast<float>(qp[j]);

        float* oc_rh = oc + static_cast<std::size_t>(r) * oc_elems +
                       static_cast<std::size_t>(h) * dc;
        float* p_row = csc + static_cast<std::size_t>(r) * csc_elems +
                       static_cast<std::size_t>(h) * nblk;
        if (n_elig == 0) {
          lh[static_cast<std::size_t>(r) * hq + h] = 0.0f;
          return;
        }
        float* s_tmp = csa_detail::new_fbuf(n_elig);
        const float l = csa_detail::row_attn<float>(
            qt_rh, dcat, dc, args.scale, n_elig,
            [&](int l2) {
              return centroids + static_cast<std::size_t>(l2) * dcat;
            },
            oc_rh, s_tmp);
        float m = s_tmp[0];
        for (int l2 = 1; l2 < n_elig; ++l2) m = std::max(m, s_tmp[l2]);
        for (int l2 = 0; l2 < n_elig; ++l2) {
          p_row[l2] = std::expf(s_tmp[l2] - m) / l;  // per-head softmaxed score
        }
        lh[static_cast<std::size_t>(r) * hq + h] = l;
        csa_detail::del_buf(s_tmp);
      });

      // ---- A2: head-sum, mirror, selection with fixed blocks, gather ---
      csa_detail::run_batched(pool, rows, p, [&](int r, int /*tid*/) {
        const int t = r0 + r;
        const int p_t = args.causal ? kv_off + t : skv - 1;
        const int n_elig = args.causal ? (p_t + 1) / g : nblk;
        float* bs_r = bs + static_cast<std::size_t>(r) * nblk;

        for (int l = 0; l < n_elig; ++l) bs_r[l] = 0.0f;
        for (int h = 0; h < hq; ++h) {
          const float* p_row = csc + static_cast<std::size_t>(r) * csc_elems +
                               static_cast<std::size_t>(h) * nblk;
          for (int l = 0; l < n_elig; ++l) bs_r[l] += p_row[l];
        }
        if (args.block_scores_out) {
          float* out = args.block_scores_out +
                       (static_cast<std::size_t>(bi) * sq + t) * nblk;
          std::memcpy(out, bs_r, static_cast<std::size_t>(n_elig) * sizeof(float));
          for (int l = n_elig; l < nblk; ++l) out[l] = 0.0f;
        }

        // Fixed activations (paper: 1 initial + 2 local) inside the budget.
        int32_t* sel_r = sel + static_cast<std::size_t>(r) * sel_elems;
        int n_take = std::min(nsel, n_elig);
        int n_sel = 0;
        const auto add_fixed = [&](int l) {
          if (n_sel < n_take) sel_r[n_sel++] = l;
        };
        if (n_elig >= 1) add_fixed(0);
        if (n_elig >= 2) add_fixed(n_elig - 1);
        if (n_elig >= 3 && n_sel < n_take) add_fixed(n_elig - 2);
        if (n_sel < n_take) {
          // Top-up from (score desc, block asc); already-fixed blocks skip.
          int32_t* id = static_cast<int32_t*>(
              ::operator new(static_cast<std::size_t>(n_elig) * sizeof(int32_t),
                             std::align_val_t(64)));
          for (int l = 0; l < n_elig; ++l) id[l] = l;
          const auto by_score = [&bs_r](int32_t x, int32_t y) {
            return bs_r[x] != bs_r[y] ? bs_r[x] > bs_r[y] : x < y;
          };
          std::sort(id, id + n_elig, by_score);  // total order → deterministic
          for (int i = 0; i < n_elig && n_sel < n_take; ++i) {
            const int l = id[i];
            bool dup = false;
            for (int j = 0; j < n_sel; ++j) dup |= (sel_r[j] == l);
            if (!dup) sel_r[n_sel++] = l;
          }
          ::operator delete(id, std::align_val_t(64));
        }
        std::sort(sel_r, sel_r + n_sel);
        sel_r[nsel] = n_sel;  // publish count

        // Gather whole blocks once for every head.
        T* gb = gbuf + static_cast<std::size_t>(r) * gbuf_elems;
        for (int i = 0; i < n_sel; ++i) {
          const int s0 = sel_r[i] * g;
          const int cnt =
              args.causal ? g : std::min(g, skv - s0);  // non-causal tail
          std::memcpy(gb + static_cast<std::size_t>(i) * g * dcat,
                      cache + static_cast<std::size_t>(s0) * args.kv_ld,
                      static_cast<std::size_t>(cnt) * dcat * sizeof(T));
        }
      });

      // ---- B: selected + window passes, gates, un-absorb ----------------
      csa_detail::run_batched(pool, rows * hq, p, [&](int task, int /*tid*/) {
        const int r = task % rows;
        const int h = task / rows;
        const int t = r0 + r;
        const int p_t = args.causal ? kv_off + t : skv - 1;
        const int32_t* sel_r = sel + static_cast<std::size_t>(r) * sel_elems;
        const int n_sel = sel_r[nsel];
        const int k_rows = n_sel * g;  // causal: every selected block is full

        const float* qt_rh = qt + static_cast<std::size_t>(r) * qt_elems +
                             static_cast<std::size_t>(h) * dcat;
        const float* oc_rh = oc + static_cast<std::size_t>(r) * oc_elems +
                             static_cast<std::size_t>(h) * dc;
        const float l_cmp = lh[static_cast<std::size_t>(r) * hq + h];
        const T* wuv = args.w_uv + static_cast<std::size_t>(h) * args.w_uv_stride_h;

        const int scratch = std::max(k_rows, w);
        float* s_tmp = csa_detail::new_fbuf(scratch);
        float* Os = csa_detail::new_fbuf(dc);
        float* Ow = csa_detail::new_fbuf(dc);
        float* ot = csa_detail::new_fbuf(dc);

        // Selected branch (gathered rows; causal blocks are fully ≤ p_t).
        int k_eff_rows = k_rows;
        if (!args.causal) {  // trim the non-causal tail partial block
          const int last = sel_r[n_sel - 1];
          k_eff_rows = k_rows - (g - std::min(g, skv - last * g));
        }
        const T* gb = gbuf + static_cast<std::size_t>(r) * gbuf_elems;
        const float l_s = csa_detail::row_attn<T>(
            qt_rh, dcat, dc, args.scale, k_eff_rows,
            [&](int r2) {
              return gb + static_cast<std::size_t>(r2) * dcat;
            },
            Os, s_tmp);

        // Window branch: [max(0, p_t − w + 1), p_t] (causal) or the last w
        // tokens of the sequence (non-causal).
        const int w0 = args.causal ? std::max(0, p_t - w + 1) : std::max(0, skv - w);
        const int n_win = (args.causal ? p_t : skv - 1) - w0 + 1;
        const float l_w = csa_detail::row_attn<T>(
            qt_rh, dcat, dc, args.scale, n_win,
            [&](int r2) {
              return cache + static_cast<std::size_t>(w0 + r2) * args.kv_ld;
            },
            Ow, s_tmp);

        // Gates (cmp, slc, win) and the un-absorb.
        const T* gl = args.gate_logits +
                      ((static_cast<std::size_t>(bi) * hq + h) * sq + t) * 3;
        const float g_cmp = 1.0f / (1.0f + std::expf(-static_cast<float>(gl[0])));
        const float g_slc = 1.0f / (1.0f + std::expf(-static_cast<float>(gl[1])));
        const float g_win = 1.0f / (1.0f + std::expf(-static_cast<float>(gl[2])));
        for (int d = 0; d < dc; ++d) {
          const float o_cmp = l_cmp > 0.0f ? oc_rh[d] / l_cmp : 0.0f;
          const float o_slc = l_s > 0.0f ? Os[d] / l_s : 0.0f;
          const float o_win = l_w > 0.0f ? Ow[d] / l_w : 0.0f;
          ot[d] = g_cmp * o_cmp + g_slc * o_slc + g_win * o_win;
        }
        T* out = args.o + static_cast<std::size_t>(bi) * args.o_stride_b +
                 static_cast<std::size_t>(h) * args.o_stride_h +
                 static_cast<std::size_t>(t) * args.o_ld;
        for (int e = 0; e < dn; ++e) {
          out[e] = T(dsa_detail::dot_f32(ot, wuv + static_cast<std::size_t>(e) * dc, dc));
        }
        csa_detail::del_buf(s_tmp);
        csa_detail::del_buf(Os);
        csa_detail::del_buf(Ow);
        csa_detail::del_buf(ot);
      });

      csa_detail::del_buf(qt);
      csa_detail::del_buf(oc);
      csa_detail::del_buf(lh);
      csa_detail::del_buf(csc);
      csa_detail::del_buf(bs);
      ::operator delete(sel, std::align_val_t(64));
      ::operator delete(gbuf, std::align_val_t(64));
    }
    csa_detail::del_buf(centroids);
  }
  return Status::kSuccess;
}

}  // namespace attention
}  // namespace cpu_ops
