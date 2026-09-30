#pragma once

// Fused attention block kernel: one query panel (<= kQBlock rows) of one
// (batch, head) pair, streamed over a kv range with online (single-pass)
// softmax. S = Q·Kᵀ is computed into an L1-sized buffer, scaled and masked,
// turned into P = exp2(scale·log2e·S − m) in place, and consumed by
// O += P·V before the next kv block is loaded — S and P never leave the
// cache. The running row statistics (max M, sum l) and the unnormalized O
// accumulator are owned by the caller, which is what kv-split slices merge
// later (see attention/attention.h).
//
// The Q/K reduction depth (dk) and the V/O width (dv) are independent: MLA's
// absorbed form attends over a 576-deep latent key but accumulates output
// over the 512-deep latent value. dv == dk for plain GQA/MHA/MQA.
//
// Everything below the mma policy is reused as-is: Q/K/V panels go through
// the policy's pack functions (widening f16/bf16 to f32 at pack time), and
// the micro-kernel is the policy's register atom. The softmax statistics use
// simd::hmax/hsum with fixed lane orders, so a row's result depends only on
// its kv block sequence — identical for any thread count when unsplit.

#include <algorithm>
#include <cmath>

#include "cpu_ops/detail/mma_atom.h"
#include "cpu_ops/detail/pack.h"
#include "cpu_ops/detail/simd.h"
#include "cpu_ops/layout.h"

namespace cpu_ops {
namespace attention {
namespace detail {

// Tile geometry. Micro-tiles match the f32 GEMM config (MR=6, NR=16); the
// block sizes bound the working set: packed Q + O + S/P panels stay in the
// L1/L2 range for head dims up to ~256.
constexpr int kAttnMR = 6;
constexpr int kAttnNR = 16;
constexpr int kQBlock = 48;   // query rows per panel (multiple of kAttnMR)
constexpr int kKVBlock = 64;  // kv columns per block (multiple of kAttnNR)

// Sentinel standing in for -infinity in the running row max: finite, so
// exp2f(sentinel - m) flushes to 0 without producing NaNs.
constexpr float kNegInf = -1.0e30f;

// Scratch layout (floats), sized by the two extents:
//   [0, kQBlock*dk)                        packed Q panel (kc = dk)
//   [.  , max(dk, dv_pad)*kKVBlock)        packed B panel (reused for K then V)
//   [.  , kQBlock*kKVBlock)                packed P panel
//   [.  , kQBlock*kKVBlock)                S/P buffer (row stride kKVBlock)
inline std::size_t attention_scratch_elems(int dk, int dv) {
  const int dv_pad = (dv + kAttnNR - 1) / kAttnNR * kAttnNR;
  const int b_elems = std::max(dk, dv_pad) * kKVBlock;
  return static_cast<std::size_t>(kQBlock) * dk +
         static_cast<std::size_t>(b_elems) +
         2u * static_cast<std::size_t>(kQBlock) * kKVBlock;
}

template <typename Policy>
struct AttentionBlockKernel {
  using T = typename Policy::ElemA;
  using Atom = typename Policy::template Atom<kAttnMR, kAttnNR>;
  using VecT = typename Atom::VecT;
  static constexpr int kVL = Atom::kVLEN;

  // q_head/k_head/v_head point at one (batch, head) panel each, row-major
  // over [seq, dim] with leading dimension *_ld (in elements). Processes
  // query rows [q0, q0+rows) against kv positions [kv0, kv1); causal masking
  // aligns the query block to the *end* of the kv range (query row i attends
  // kv j <= q0 + i + (seq_kv - seq_q)). Q and K reduce over dk lanes; V
  // contributes dv output lanes. On return M/l/O hold the running softmax
  // statistics and the unnormalized output (rows x dv) for the range.
  static void run(const T* q_head, int q_ld, const T* k_head, int k_ld,
                  const T* v_head, int v_ld, int rows, int q0, int kv0, int kv1,
                  int seq_q, int seq_kv, int dk, int dv, float scale_log2e,
                  bool causal, float* M, float* l, float* O, float* scratch) {
    const int dv_pad = (dv + kAttnNR - 1) / kAttnNR * kAttnNR;
    const int b_elems = std::max(dk, dv_pad) * kKVBlock;
    const int kv_off = seq_kv - seq_q;
    float* qpk = scratch;
    float* bpk = qpk + static_cast<std::size_t>(kQBlock) * dk;
    float* ppk = bpk + static_cast<std::size_t>(b_elems);
    float* sp = ppk + static_cast<std::size_t>(kQBlock) * kKVBlock;

    // The packed Q panel is built once and reused by every kv block.
    Policy::pack_a(TensorRef<const T, layout::RowMajor>(
                       q_head + static_cast<std::size_t>(q0) * q_ld, q_ld),
                   0, 0, rows, dk, dk, kAttnMR, qpk);
    for (int i = 0; i < rows; ++i) {
      M[i] = kNegInf;
      l[i] = 0.0f;
    }
    for (int i = 0; i < rows * dv; ++i) O[i] = 0.0f;

    for (int c0 = kv0; c0 < kv1; c0 += kKVBlock) {
      const int nb = std::min(kKVBlock, kv1 - c0);
      // Causal blocks are ordered, so once a block starts past the last
      // valid position of the *last* query row, the whole tail is masked.
      if (causal && c0 > q0 + kv_off + rows - 1) break;

      // --- S = Q · Kᵀ into the S buffer (rows x nb, stride kKVBlock) ---
      // Kᵀ as a [dk, seq_kv] column-major matrix: at(k, j) = K[j, k].
      Policy::pack_b(TensorRef<const T, layout::ColumnMajor>(k_head, k_ld), 0, c0, dk,
                     dk, nb, kAttnNR, bpk);
      for (int ir = 0; ir < rows; ir += kAttnMR) {
        for (int jr = 0; jr < nb; jr += kAttnNR) {
          // The atom spills a dense MR x NR tile; scatter it into the
          // kKVBlock-stride S buffer (same pattern as BlockGemm's edge path).
          float tile[kAttnMR][kAttnNR];
          Atom atom;
          atom.run(qpk + (ir / kAttnMR) * dk * kAttnMR,
                   bpk + (jr / kAttnNR) * dk * kAttnNR, dk, &tile[0][0]);
          for (int i = 0; i < kAttnMR && ir + i < rows; ++i)
            for (int j = 0; j < kAttnNR; ++j)
              sp[static_cast<std::size_t>(ir + i) * kKVBlock + jr + j] = tile[i][j];
        }
      }

      // --- online softmax: mask, row max, P = exp2, running sums ---
      for (int i = 0; i < rows; ++i) {
        const int jmax = causal ? q0 + i + kv_off : seq_kv - 1;
        const int vl = std::max(0, std::min(nb, jmax - c0 + 1));
        float* srow = sp + static_cast<std::size_t>(i) * kKVBlock;
        if (vl <= 0) {
          // Whole block masked for this row: the P·V stage below still
          // consumes the full panel, so the row must read as all-zero
          // weights, not as stale scores from a previous block.
          for (int j = 0; j < nb; ++j) srow[j] = 0.0f;
          continue;
        }
        float m_tile = kNegInf;
        int j = 0;
        for (; j + kVL <= vl; j += kVL)
          m_tile = std::max(m_tile, simd::hmax(VecT::load(srow + j)));
        for (; j < vl; ++j) m_tile = std::max(m_tile, srow[j]);
        // The running max lives in the exp2-argument domain (c·S, c > 0), so
        // that P = exp2(c·S − m) <= 1 and the cross-block rescale factor
        // exp2(m_old − m_new) stays consistent with it.
        m_tile *= scale_log2e;

        const float m_new = std::max(M[i], m_tile);
        const float a = std::exp2f(M[i] - m_new);  // rescale factor, 0 on first block
        const VecT cs = VecT::set1(scale_log2e);
        const VecT mn = VecT::set1(-m_new);
        float l_tile = 0.0f;
        j = 0;
        for (; j + kVL <= vl; j += kVL) {
          const VecT p = simd::exp2(simd::fmadd(VecT::load(srow + j), cs, mn));
          p.store(srow + j);
          l_tile += simd::hsum(p);
        }
        for (; j < vl; ++j) {
          const float p = std::exp2f(srow[j] * scale_log2e - m_new);
          srow[j] = p;
          l_tile += p;
        }
        for (; j < nb; ++j) srow[j] = 0.0f;  // masked tail must not reach P·V
        l[i] = l[i] * a + l_tile;
        M[i] = m_new;

        float* orow = O + static_cast<std::size_t>(i) * dv;
        const VecT av = VecT::set1(a);
        int e = 0;
        for (; e + kVL <= dv; e += kVL)
          simd::mul(VecT::load(orow + e), av).store(orow + e);
        for (; e < dv; ++e) orow[e] *= a;
      }

      // --- O += P · V over the value extent ---
      // P is already f32 in the S buffer; pack directly (bypasses the policy,
      // which is typed for the storage element).
      cpu_ops::detail::pack_a(TensorRef<const float, layout::RowMajor>(sp, kKVBlock), 0, 0,
                              rows, nb, kAttnMR, ppk);
      Policy::pack_b(TensorRef<const T, layout::RowMajor>(v_head, v_ld), c0, 0, nb, nb,
                     dv, kAttnNR, bpk);
      for (int dn = 0; dn < dv; dn += kAttnNR) {
        for (int ir = 0; ir < rows; ir += kAttnMR) {
          Atom atom;
          atom.clear();
          atom.mma(ppk + (ir / kAttnMR) * nb * kAttnMR,
                   bpk + (dn / kAttnNR) * nb * kAttnNR, nb);
          const int imax = std::min(kAttnMR, rows - ir);
          for (int i = 0; i < imax; ++i) {
            float* orow = O + static_cast<std::size_t>(ir + i) * dv + dn;
            for (int w = 0; w < Atom::kVecN; ++w)
              simd::add(VecT::load(orow + w * kVL), atom.acc[i][w])
                  .store(orow + w * kVL);
          }
        }
      }
    }
  }
};

}  // namespace detail
}  // namespace attention
}  // namespace cpu_ops
