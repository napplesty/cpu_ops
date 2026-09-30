#pragma once

// Multi-head Latent Attention (MLA, DeepSeek-V2/V3) in the weight-absorbed
// form, on top of the fused GQA kernel.
//
// Per layer the cache holds one shared latent per token — c(t) [dim_latent]
// followed by the shared RoPE key k_pe(t) [dim_pe], packed contiguously —
// instead of per-head K/V. With the up-projections absorbed:
//
//   score(i,t,h) = (W_UK_h · q_nope(i,h)) · c(t) + q_pe(i,h) · k_pe(t)
//   o(i,h)       = W_UV_h · Σ_t p(i,t,h) · c(t)
//
// the first line is a single dot over the concatenated [c | k_pe] cache row
// (dim = dim_latent + dim_pe), i.e. exactly an Attention core call with
// heads_kv == 1 (shared K) — q̃ = W_UK_h · q_nope is computed per head by an
// ordinary GEMM, and the value side is the same cache narrowed to the latent
// columns (dim_v = dim_latent < dim). This entry is that orchestration:
//
//   1. per head: q̃ = q_nope · W_UK_h^T      (GEMM, [sq x dn] x [dn x dc])
//   2. per head: q_pe appended after q̃      (copy, [sq x dp])
//   3. fused attention over the shared cache ([sq x (dc+dp)] vs [skv x dc+dp],
//      V = first dc columns, dim_v = dc)
//   4. per head: o = õ · W_UV_h^T            (GEMM, [sq x dc] x [dc x dn])
//
// q_pe / k_pe must already be RoPE-rotated (upstream); the softmax scale —
// 1/sqrt(dim_nope + dim_pe) in DeepSeek — is a single argument. Absorb and
// un-absorb GEMMs run one per head, parallel over heads, single-threaded
// each; the attention core uses the caller's thread count. Determinism
// contracts are inherited from the core (bit-identical across thread counts
// with kv-split off, deterministic across runs with it on).
//
// For narrow storage (f16/bf16) the GEMMs widen to f32 and the intermediate
// q̃ / õ panels are rounded back to storage once — the cache itself is read
// once, in storage precision, which is where the bandwidth win lives.

#include <algorithm>
#include <cstddef>
#include <new>
#include <type_traits>

#include "../02_fused_attention/attention.h"
#include "cpu_ops/detail/thread_pool.h"
#include "cpu_ops/gemm/device/gemm.h"
#include "cpu_ops/status.h"

namespace cpu_ops {
namespace attention {

namespace mla_detail {

// Absorb/un-absorb GEMM: operands in storage type (or f32), output f32.
template <typename T>
struct AbsorbGemmFor;
template <>
struct AbsorbGemmFor<float> {
  using Type = gemm::device::Gemm<float, layout::RowMajor, float, layout::ColumnMajor,
                                  float, layout::RowMajor>;
};
template <>
struct AbsorbGemmFor<float16_t> {
  using Type = gemm::device::GemmF16F32<layout::RowMajor, layout::ColumnMajor,
                                        layout::RowMajor>;
};
template <>
struct AbsorbGemmFor<bfloat16_t> {
  using Type = gemm::device::GemmBF16F32<layout::RowMajor, layout::ColumnMajor,
                                         layout::RowMajor>;
};

// fn(task, tid) over [0, n) with at most p concurrent threads (the pool
// always engages all workers, so tasks go out in p-sized batches).
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

inline float* new_fbuf(std::size_t elems) {
  return static_cast<float*>(::operator new(elems * sizeof(float), std::align_val_t(64)));
}

}  // namespace mla_detail

template <typename T>
struct MlaAttention {
  static_assert(std::is_same<T, float>::value || std::is_same<T, float16_t>::value ||
                    std::is_same<T, bfloat16_t>::value,
                "MLA storage must be float, float16_t or bfloat16_t");

  struct Arguments {
    int batch = 0;
    int heads = 0;  // H query heads; KV is shared (one latent per token)
    int seq_q = 0;
    int seq_kv = 0;

    int dim_nope = 128;    // dn: q_nope width; also the output width
    int dim_pe = 64;       // dp: RoPE width, appended to the latent cache
    int dim_latent = 512;  // dc: latent width (c_kv and the value extent)

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

    T* o = nullptr;  // [b][h][sq][dn]
    int o_stride_b = 0, o_stride_h = 0, o_ld = 0;

    float scale = 1.0f;  // typically 1/sqrt(dim_nope + dim_pe)
    bool causal = true;  // query i attends kv j <= i + (seq_kv - seq_q)
    int split_kv_slices = 0;
  };

  Status operator()(const Arguments& args, int num_threads = 0) const;
};

template <typename T>
Status MlaAttention<T>::operator()(const Arguments& args, int num_threads) const {
  const int b = args.batch, hq = args.heads, sq = args.seq_q, skv = args.seq_kv;
  const int dn = args.dim_nope, dp = args.dim_pe, dc = args.dim_latent;
  if (b < 0 || hq < 1 || sq < 0 || skv < 0 || dn < 1 || dp < 1 || dc < 1) {
    return Status::kErrorInvalidProblem;
  }
  if (sq == 0) return Status::kSuccess;
  if (skv == 0) return Status::kErrorInvalidProblem;
  if (args.causal && skv < sq) return Status::kErrorInvalidArguments;
  if (!args.q_nope || !args.q_pe || !args.kv_cache || !args.w_uk || !args.w_uv ||
      !args.o) {
    return Status::kErrorInvalidArguments;
  }
  if (args.q_nope_ld < dn || args.q_pe_ld < dp || args.kv_ld < dc + dp ||
      args.o_ld < dn || args.w_uk_stride_h < dc * dn || args.w_uv_stride_h < dn * dc) {
    return Status::kErrorInvalidArguments;
  }

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  using AbsorbGemm = typename mla_detail::AbsorbGemmFor<T>::Type;
  AbsorbGemm gemm;

  // Workspaces: q̃‖q_pe panels and unnormalized-value panels, in storage
  // type; plus one f32 panel per absorb task (GEMM output before the single
  // rounding back to storage).
  const int dcat = dc + dp;
  const std::size_t qt_elems = static_cast<std::size_t>(b) * hq * sq * dcat;
  const std::size_t ot_elems = static_cast<std::size_t>(b) * hq * sq * dc;
  T* qt = static_cast<T*>(::operator new(qt_elems * sizeof(T), std::align_val_t(64)));
  T* ot = static_cast<T*>(::operator new(ot_elems * sizeof(T), std::align_val_t(64)));

  // 1 + 2: per head, absorb q_nope into the latent domain and append q_pe.
  mla_detail::run_batched(pool, b * hq, p, [&](int task, int /*tid*/) {
    const int h = task % hq;
    const int bi = task / hq;
    const T* qn = args.q_nope + static_cast<std::size_t>(bi) * args.q_nope_stride_b +
                  static_cast<std::size_t>(h) * args.q_nope_stride_h;
    const T* qp = args.q_pe + static_cast<std::size_t>(bi) * args.q_pe_stride_b +
                  static_cast<std::size_t>(h) * args.q_pe_stride_h;
    const T* wuk = args.w_uk + static_cast<std::size_t>(h) * args.w_uk_stride_h;
    T* qth = qt + static_cast<std::size_t>(task) * sq * dcat;
    float* fbuf = mla_detail::new_fbuf(static_cast<std::size_t>(sq) * dc);

    // W_UK_h is [dc x dn] row-major; as B it is the column-major [dn x dc]
    // matrix W_UK_h^T with ld = dn.
    typename AbsorbGemm::Arguments ga;
    ga.problem_size = {sq, dc, dn};
    ga.ref_A = TensorRef<const T, layout::RowMajor>(qn, args.q_nope_ld);
    ga.ref_B = TensorRef<const T, layout::ColumnMajor>(wuk, dn);
    ga.ref_C = TensorRef<const float, layout::RowMajor>(fbuf, dc);
    ga.ref_D = TensorRef<float, layout::RowMajor>(fbuf, dc);
    ga.epilogue = {1.0f, 0.0f};
    (void)gemm(ga, 1);  // dimensions pre-validated
    for (int i = 0; i < sq; ++i) {
      T* qrow = qth + static_cast<std::size_t>(i) * dcat;
      const float* frow = fbuf + static_cast<std::size_t>(i) * dc;
      for (int j = 0; j < dc; ++j) qrow[j] = T(frow[j]);
      const T* prow = qp + static_cast<std::size_t>(i) * args.q_pe_ld;
      for (int j = 0; j < dp; ++j) qrow[dc + j] = prow[j];
    }
    ::operator delete(fbuf, std::align_val_t(64));
  });

  // 3: fused attention over the shared latent cache; V is the latent prefix
  // of the same buffer, so K and V share one pointer with different extents.
  {
    typename Attention<T>::Arguments aa;
    aa.batch = b;
    aa.heads_q = hq;
    aa.heads_kv = 1;  // shared latent KV
    aa.seq_q = sq;
    aa.seq_kv = skv;
    aa.dim = dcat;
    aa.dim_v = dc;
    aa.q = qt;
    aa.q_stride_b = static_cast<int>(static_cast<std::size_t>(hq) * sq * dcat);
    aa.q_stride_h = static_cast<int>(static_cast<std::size_t>(sq) * dcat);
    aa.q_ld = dcat;
    aa.k = args.kv_cache;
    aa.k_stride_b = args.kv_stride_b;
    aa.k_stride_h = 0;  // g == 0 for heads_kv == 1
    aa.k_ld = args.kv_ld;
    aa.v = args.kv_cache;
    aa.v_stride_b = args.kv_stride_b;
    aa.v_stride_h = 0;
    aa.v_ld = args.kv_ld;
    aa.o = ot;
    aa.o_stride_b = static_cast<int>(static_cast<std::size_t>(hq) * sq * dc);
    aa.o_stride_h = static_cast<int>(static_cast<std::size_t>(sq) * dc);
    aa.o_ld = dc;
    aa.scale = args.scale;
    aa.causal = args.causal;
    aa.split_kv_slices = args.split_kv_slices;
    Attention<T> attn;
    const Status st = attn(aa, num_threads);
    if (st != Status::kSuccess) {
      ::operator delete(qt, std::align_val_t(64));
      ::operator delete(ot, std::align_val_t(64));
      return st;
    }
  }

  // 4: per head, un-absorb the value panel back to the head dim.
  mla_detail::run_batched(pool, b * hq, p, [&](int task, int /*tid*/) {
    const int h = task % hq;
    const int bi = task / hq;
    const T* wuv = args.w_uv + static_cast<std::size_t>(h) * args.w_uv_stride_h;
    const T* oth = ot + static_cast<std::size_t>(task) * sq * dc;
    T* oh = args.o + static_cast<std::size_t>(bi) * args.o_stride_b +
            static_cast<std::size_t>(h) * args.o_stride_h;
    float* fbuf = mla_detail::new_fbuf(static_cast<std::size_t>(sq) * dn);

    // W_UV_h is [dn x dc] row-major; as B it is the column-major [dc x dn]
    // matrix W_UV_h^T with ld = dc.
    typename AbsorbGemm::Arguments ga;
    ga.problem_size = {sq, dn, dc};
    ga.ref_A = TensorRef<const T, layout::RowMajor>(oth, dc);
    ga.ref_B = TensorRef<const T, layout::ColumnMajor>(wuv, dc);
    ga.ref_C = TensorRef<const float, layout::RowMajor>(fbuf, dn);
    ga.ref_D = TensorRef<float, layout::RowMajor>(fbuf, dn);
    ga.epilogue = {1.0f, 0.0f};
    (void)gemm(ga, 1);
    for (int i = 0; i < sq; ++i) {
      const float* frow = fbuf + static_cast<std::size_t>(i) * dn;
      T* orow = oh + static_cast<std::size_t>(i) * args.o_ld;
      for (int j = 0; j < dn; ++j) orow[j] = T(frow[j]);
    }
    ::operator delete(fbuf, std::align_val_t(64));
  });

  ::operator delete(qt, std::align_val_t(64));
  ::operator delete(ot, std::align_val_t(64));
  return Status::kSuccess;
}

}  // namespace attention
}  // namespace cpu_ops
