#pragma once

// Fused multi-head / grouped-query attention for CPU.
//
//   MHA: heads_kv == heads_q          MQA: heads_kv == 1
//   GQA: heads_kv divides heads_q; query head h attends kv head
//        h * heads_kv / heads_q.
//
// One fused kernel per query panel: S = Q·Kᵀ stays in cache, online softmax
// (single pass, running max/sum), O += P·V before the next kv block is
// loaded. Causal masking is applied per block; fully-masked tail blocks are
// skipped, which halves the work for square causal problems. Storage may be
// f32, float16_t or bfloat16_t (compute is f32; narrow operands widen at
// pack time). Results are bit-identical for any thread count when kv-split
// is off, and deterministic across runs when it is on (fixed slice order).
//
// Roadmap — sparse / linear variants reuse this kernel's three seams:
//   * kv iteration: the block loop consumes contiguous [kv0, kv1) ranges;
//     CSA's compressed branch and DSA's top-k selection reduce to iterating
//     gathered kv block lists per query panel instead.
//   * block admission: causal masking is one "is this block live" predicate;
//     CSA/DSA block-diamond masks plug in as additional predicates.
//   * weight transform: P = exp2(x − m) lives in one place (the softmax
//     update in detail/attention_kernel.h); KDA's gated delta-rule
//     intra-chunk correction swaps in a sigmoid-decay transform (no running
//     max) while keeping the same two mma stages.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <new>
#include <type_traits>
#include <vector>

#include "attention_kernel.h"
#include "cpu_ops/gemm/threadblock/mma_policy_fma.h"
#include "cpu_ops/gemm/threadblock/mma_policy_widen.h"
#include "cpu_ops/thread/thread_pool.h"
#include "cpu_ops/numeric_types.h"
#include "cpu_ops/status.h"

namespace cpu_ops {
namespace attention {

namespace detail_storage {

template <typename T>
struct PolicyFor;  // storage type -> mma policy (f32 FMA; narrow operands widen at pack time)
template <>
struct PolicyFor<float> {
  using Policy = mma::FmaPolicy<float>;
};
template <>
struct PolicyFor<float16_t> {
  using Policy = mma::WidenPolicy<float16_t>;
};
template <>
struct PolicyFor<bfloat16_t> {
  using Policy = mma::WidenPolicy<bfloat16_t>;
};

inline float* new_buf(std::size_t elems) {
  return static_cast<float*>(::operator new(elems * sizeof(float), std::align_val_t(64)));
}
inline void free_buf(float* p) { ::operator delete(p, std::align_val_t(64)); }

}  // namespace detail_storage

template <typename T>
struct Attention {
  static_assert(std::is_same<T, float>::value || std::is_same<T, float16_t>::value ||
                    std::is_same<T, bfloat16_t>::value,
                "attention storage must be float, float16_t or bfloat16_t");

  // Per-tensor layout: element (b, h, i, d) at ptr[b*stride_b + h*stride_h +
  // i*ld + d] — the head dim must be contiguous. q has heads_q heads and
  // seq_q rows; k/v have heads_kv heads and seq_kv rows; o mirrors q.
  struct Arguments {
    int batch = 0;
    int heads_q = 0;
    int heads_kv = 0;
    int seq_q = 0;
    int seq_kv = 0;
    int dim = 0;
    // Value/output extent. 0 = same as dim (plain attention). MLA's absorbed
    // form sets dim = latent+rope (e.g. 576) for the Q/K reduction and
    // dim_v = latent (e.g. 512) for the V accumulation/output.
    int dim_v = 0;

    const T* q = nullptr;
    int q_stride_b = 0, q_stride_h = 0, q_ld = 0;
    const T* k = nullptr;
    int k_stride_b = 0, k_stride_h = 0, k_ld = 0;
    const T* v = nullptr;
    int v_stride_b = 0, v_stride_h = 0, v_ld = 0;
    T* o = nullptr;
    int o_stride_b = 0, o_stride_h = 0, o_ld = 0;

    float scale = 1.0f;  // multiplies Q·Kᵀ, typically 1/sqrt(dim)
    bool causal = true;  // query i attends kv j <= i + (seq_kv - seq_q)

    // kv-split slices: 0 = auto (only when the query-panel grid alone cannot
    // feed every thread, e.g. decode), 1 = off, >1 = forced count.
    int split_kv_slices = 0;
  };

  // num_threads <= 0 selects the size of the global thread pool.
  Status operator()(const Arguments& args, int num_threads = 0) const;
};

template <typename T>
Status Attention<T>::operator()(const Arguments& args, int num_threads) const {
  const int b = args.batch, hq = args.heads_q, hkv = args.heads_kv;
  const int sq = args.seq_q, skv = args.seq_kv, d = args.dim;
  const int dv = args.dim_v > 0 ? args.dim_v : d;  // value/output extent
  if (b < 0 || hq < 1 || hkv < 1 || sq < 0 || skv < 0 || d < 1) {
    return Status::kErrorInvalidProblem;
  }
  if (sq == 0) return Status::kSuccess;
  if (skv == 0) return Status::kErrorInvalidProblem;
  if (hq % hkv != 0) return Status::kErrorInvalidArguments;
  if (args.causal && skv < sq) return Status::kErrorInvalidArguments;
  if (args.dim_v < 0) return Status::kErrorInvalidProblem;
  if (!args.q || !args.k || !args.v || !args.o) return Status::kErrorInvalidArguments;
  if (args.q_ld < d || args.k_ld < d || args.v_ld < dv || args.o_ld < dv) {
    return Status::kErrorInvalidArguments;
  }

  using attention::detail::kQBlock;
  using attention::detail::kKVBlock;
  using attention::detail::attention_scratch_elems;
  using Kernel = attention::detail::AttentionBlockKernel<
      typename detail_storage::PolicyFor<T>::Policy>;

  const float scale_log2e = args.scale * 1.4426950408889634f;  // log2(e)

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  const int qblocks = (sq + kQBlock - 1) / kQBlock;
  const int n_tasks = b * hq * qblocks;

  // Runs fn(task, tid) for [0, n) on at most p threads (parallel_for itself
  // always engages the whole pool, so tasks are handed out in p-sized
  // batches). Task results are independent of the batch split.
  const auto for_parallel = [&](int n, const auto& fn) {
    if (n <= 0) return;
    if (p <= 1 || n == 1) {
      for (int t = 0; t < n; ++t) fn(t, 0);
      return;
    }
    for (int base = 0; base < n; base += p) {
      pool.parallel_for(std::min(p, n - base),
                        [&fn, base](int i, int tid) { fn(base + i, tid); });
    }
  };

  // kv-split: same rationale as GEMM split-k — parallelism when the query
  // grid is too small (decode), at the price of a partial-statistics
  // workspace and a deterministic ordered merge pass.
  int slices = 1;
  if (p > 1) {
    const int max_slices = std::max(1, skv / (2 * kKVBlock));
    if (args.split_kv_slices > 1) {
      slices = std::min(args.split_kv_slices, std::max(1, skv / kKVBlock));
    } else if (args.split_kv_slices == 0 && n_tasks < p) {
      slices = std::min(std::max(1, p / n_tasks), max_slices);
    }
  }

  const auto head_q = [&](int bi, int h) {
    return args.q + static_cast<std::size_t>(bi) * args.q_stride_b +
           static_cast<std::size_t>(h) * args.q_stride_h;
  };
  const auto head_kv = [&](int bi, int h) {
    const int g = static_cast<int>(static_cast<long long>(h) * hkv / hq);
    return std::make_pair(
        args.k + static_cast<std::size_t>(bi) * args.k_stride_b +
            static_cast<std::size_t>(g) * args.k_stride_h,
        args.v + static_cast<std::size_t>(bi) * args.v_stride_b +
            static_cast<std::size_t>(g) * args.v_stride_h);
  };
  // Normalized writeback of one finished query panel.
  const auto write_output = [&](int bi, int h, int q0, int rows, const float* O,
                                const float* l) {
    T* out = args.o + static_cast<std::size_t>(bi) * args.o_stride_b +
             static_cast<std::size_t>(h) * args.o_stride_h +
             static_cast<std::size_t>(q0) * args.o_ld;
    for (int i = 0; i < rows; ++i) {
      const float inv = l[i] > 0.0f ? 1.0f / l[i] : 0.0f;
      const float* orow = O + static_cast<std::size_t>(i) * dv;
      T* orow_out = out + static_cast<std::size_t>(i) * args.o_ld;
      for (int j = 0; j < dv; ++j) orow_out[j] = T(orow[j] * inv);
    }
  };

  if (slices == 1) {
    const auto job = [&](int task, int /*thread_index*/) {
      const int qb = task % qblocks;
      const int h = (task / qblocks) % hq;
      const int bi = task / (qblocks * hq);
      const int q0 = qb * kQBlock;
      const int rows = std::min(kQBlock, sq - q0);
      const auto kv = head_kv(bi, h);
      float* scratch = detail_storage::new_buf(attention_scratch_elems(d, dv));
      float* stats = detail_storage::new_buf(2u * kQBlock);  // M then l
      float* O = detail_storage::new_buf(static_cast<std::size_t>(kQBlock) * dv);
      Kernel::run(head_q(bi, h), args.q_ld, kv.first, args.k_ld, kv.second, args.v_ld,
                  rows, q0, 0, skv, sq, skv, d, dv, scale_log2e, args.causal,
                  /*M=*/stats, /*l=*/stats + kQBlock, O, scratch);
      write_output(bi, h, q0, rows, O, stats + kQBlock);
      detail_storage::free_buf(scratch);
      detail_storage::free_buf(stats);
      detail_storage::free_buf(O);
    };
    for_parallel(n_tasks, job);
    return Status::kSuccess;
  }

  // Phase 1: every (query panel, slice) reduces its kv range into a slab of
  // partial statistics: [M: kQBlock | l: kQBlock | O: kQBlock x d] floats.
  const std::size_t slab_elems = static_cast<std::size_t>(kQBlock) * (dv + 2);
  float* ws = detail_storage::new_buf(slab_elems * n_tasks * slices);
  const auto phase1 = [&](int task, int /*thread_index*/) {
    const int slice = task % slices;
    const int t = task / slices;
    const int qb = t % qblocks;
    const int h = (t / qblocks) % hq;
    const int bi = t / (qblocks * hq);
    const int q0 = qb * kQBlock;
    const int rows = std::min(kQBlock, sq - q0);
    const int kv0 = static_cast<int>(static_cast<long long>(skv) * slice / slices);
    const int kv1 = static_cast<int>(static_cast<long long>(skv) * (slice + 1) / slices);
    const auto kv = head_kv(bi, h);
    float* slab = ws + static_cast<std::size_t>(task) * slab_elems;
    float* scratch = detail_storage::new_buf(attention_scratch_elems(d, dv));
    Kernel::run(head_q(bi, h), args.q_ld, kv.first, args.k_ld, kv.second, args.v_ld,
                rows, q0, kv0, kv1, sq, skv, d, dv, scale_log2e, args.causal,
                /*M=*/slab, /*l=*/slab + kQBlock, /*O=*/slab + 2 * kQBlock, scratch);
    detail_storage::free_buf(scratch);
  };
  for_parallel(n_tasks * slices, phase1);

  // Phase 2: merge slices in fixed ascending order (bit-stable across runs),
  // normalize, write out.
  const auto phase2 = [&](int t, int /*thread_index*/) {
    const int qb = t % qblocks;
    const int h = (t / qblocks) % hq;
    const int bi = t / (qblocks * hq);
    const int q0 = qb * kQBlock;
    const int rows = std::min(kQBlock, sq - q0);
    float* mg = detail_storage::new_buf(slab_elems);  // M | l | O, mirrors the slab layout
    float* M = mg, * l = mg + kQBlock, * O = mg + 2 * kQBlock;
    for (int i = 0; i < rows; ++i) {
      M[i] = attention::detail::kNegInf;
      l[i] = 0.0f;
    }
    for (int i = 0; i < rows * dv; ++i) O[i] = 0.0f;
    for (int s = 0; s < slices; ++s) {
      const float* slab = ws + (static_cast<std::size_t>(t) * slices + s) * slab_elems;
      const float* Ms = slab, * ls = slab + kQBlock, * Os = slab + 2 * kQBlock;
      for (int i = 0; i < rows; ++i) {
        const float m = std::max(M[i], Ms[i]);
        const float ag = std::exp2f(M[i] - m);
        const float as = std::exp2f(Ms[i] - m);
        float* orow = O + static_cast<std::size_t>(i) * dv;
        const float* srow = Os + static_cast<std::size_t>(i) * dv;
        for (int j = 0; j < dv; ++j) orow[j] = orow[j] * ag + srow[j] * as;
        l[i] = l[i] * ag + ls[i] * as;
        M[i] = m;
      }
    }
    write_output(bi, h, q0, rows, O, l);
    detail_storage::free_buf(mg);
  };
  for_parallel(n_tasks, phase2);
  detail_storage::free_buf(ws);
  return Status::kSuccess;
}

}  // namespace attention
}  // namespace cpu_ops
