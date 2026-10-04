#pragma once

// Kernel-layer orchestration behind gemm::device::Gemm and GemmMx: thread
// partition, optional split-k two-phase execution, per-task packing buffers.
// Operand references are forwarded opaquely to the policy's pack functions
// (TensorRef for dense operands, MxTensorRef for block-scaled ones).

#include <algorithm>
#include <cstddef>
#include <new>
#include <vector>

#include "cpu_ops/arch/simd.h"
#include "cpu_ops/epilogue/partial_sum.h"
#include "cpu_ops/gemm/contract.h"
#include "cpu_ops/gemm/threadblock/block_gemm.h"
#include "cpu_ops/gemm/threadblock/threadblock_swizzle.h"
#include "cpu_ops/layout/matrix.h"
#include "cpu_ops/status.h"
#include "cpu_ops/tensor_ref.h"
#include "cpu_ops/thread/thread_pool.h"

namespace cpu_ops {
namespace gemm {
namespace kernel {

namespace detail {

// Degenerate k == 0: fold beta * C into D without touching A or B.
template <typename T, typename LayoutC, typename Epilogue>
void write_zero_k(TensorRef<const T, LayoutC> c, TensorRef<T, LayoutC> d, int m, int n,
                  const Epilogue& epilogue) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) d.at(i, j) = epilogue(T(0), c.at(i, j), true, true, i, j);
  }
}

// Combines split-k partial slabs in slice order (deterministic) and applies
// the user epilogue exactly once per element: D = epilogue(sum_s W_s, C).
template <typename T, typename LayoutC, typename Epilogue>
inline void reduce_splitk(TensorRef<const T, LayoutC> c, TensorRef<T, LayoutC> d,
                          const T* ws, int slices, int m, int n,
                          const Epilogue& epilogue) {
  constexpr int kV = simd::native_width<T>();
  using VecT = simd::Vec<T, kV>;
  const size_t slab = static_cast<size_t>(m) * n;
  if constexpr (LayoutC::kIsRowMajor &&
                threadblock::epilogue_has_apply_vec<Epilogue, VecT>::value) {
    for (int i = 0; i < m; ++i) {
      const T* wrow = ws + static_cast<size_t>(i) * n;
      const T* crow = c.data() + LayoutC::offset(i, 0, c.ld());
      T* drow = d.data() + LayoutC::offset(i, 0, d.ld());
      int j = 0;
      for (; j + kV <= n; j += kV) {
        VecT acc = VecT::load(wrow + j);
        for (int s = 1; s < slices; ++s) acc = simd::add(acc, VecT::load(wrow + s * slab + j));
        epilogue.apply_vec(acc, VecT::load(crow + j), true, true, i, j).store(drow + j);
      }
      for (; j < n; ++j) {
        T acc = wrow[j];
        for (int s = 1; s < slices; ++s) acc += wrow[s * slab + j];
        d.at(i, j) = epilogue(acc, crow[j], true, true, i, j);
      }
    }
  } else {
    for (int i = 0; i < m; ++i) {
      const T* wrow = ws + static_cast<size_t>(i) * n;
      for (int j = 0; j < n; ++j) {
        T acc = wrow[j];
        for (int s = 1; s < slices; ++s) acc += wrow[s * slab + j];
        d.at(i, j) = epilogue(acc, c.at(i, j), true, true, i, j);
      }
    }
  }
}

}  // namespace detail

template <typename Policy, typename LayoutC, typename Epilogue, typename Config,
          typename RefA, typename RefB>
Status run_blocked(const RefA& a, const RefB& b,
                   TensorRef<const typename Policy::AccT, LayoutC> c,
                   TensorRef<typename Policy::AccT, LayoutC> d, int m, int n, int k,
                   const typename Epilogue::Params& epilogue_params,
                   int split_k_slices, int num_threads) {
  using T = typename Policy::AccT;
  static_assert(threadblock::epilogue_params<Epilogue>::value,
                "Epilogue must provide a Params type and be constructible from it");
  const Epilogue epilogue(epilogue_params);

  if (k == 0) {
    detail::write_zero_k(c, d, m, n, epilogue);
    return Status::kSuccess;
  }

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  // Split-k: deep reductions are cut into slices whose partial sums are
  // combined in a second pass. Pays off only when the output grid cannot feed
  // the threads: the extra pass and slab traffic are bounded by a workspace
  // budget and a minimum k depth.
  constexpr std::size_t kSplitkWsBudget = 4ull << 20;
  int k_slices = 1;
  if (p > 1) {
    const int depth_cap = std::max(1, k / Config::kKC);
    if (split_k_slices > 0) {
      k_slices = split_k_slices;
    } else if (k >= 8 * Config::kKC) {
      const std::size_t slab_bytes = static_cast<std::size_t>(m) * n * sizeof(T);
      const int ws_cap = static_cast<int>(std::max<std::size_t>(
          1, kSplitkWsBudget / std::max<std::size_t>(slab_bytes, 1)));
      k_slices = std::min(2 * p, ws_cap);
    }
    k_slices = std::max(1, std::min(k_slices, depth_cap));
  }

  const std::vector<threadblock::GemmRegion> regions =
      threadblock::partition_gemm(m, n, k, Config::kMR, Config::kNR, p, k_slices);

  using Block = threadblock::BlockGemm<Policy, LayoutC, Epilogue, Config>;
  using PackedA = typename Policy::PackedA;
  using PackedB = typename Policy::PackedB;
  constexpr std::size_t buf_a_elems = Block::kPackBufAElems;
  constexpr std::size_t buf_b_elems = Block::kPackBufBElems;

  if (k_slices > 1) {
    // Phase 1 writes raw partial sums per slice; phase 2 reduces them in
    // slice order and applies the real epilogue exactly once.
    using WsEpilogue = epilogue::PartialSum<T>;
    using BlockWs = threadblock::BlockGemm<Policy, layout::RowMajor, WsEpilogue, Config>;
    const std::size_t slab_elems = static_cast<std::size_t>(m) * n;
    T* ws = static_cast<T*>(::operator new(slab_elems * k_slices * sizeof(T),
                                           std::align_val_t(64)));
    auto job = [&](int task, int /*thread_index*/) {
      PackedA* buf_a = static_cast<PackedA*>(
          ::operator new(buf_a_elems * sizeof(PackedA), std::align_val_t(64)));
      PackedB* buf_b = static_cast<PackedB*>(
          ::operator new(buf_b_elems * sizeof(PackedB), std::align_val_t(64)));
      const threadblock::GemmRegion& r = regions[static_cast<size_t>(task)];
      T* slab_ptr = ws + static_cast<size_t>(r.slice) * slab_elems;
      TensorRef<T, layout::RowMajor> slab(slab_ptr, n);
      TensorRef<const T, layout::RowMajor> slab_c(slab_ptr, n);
      BlockWs::run(a, b, slab_c, slab, r.m0, r.m1, r.n0, r.n1, r.k0,
                   r.k1, WsEpilogue{}, buf_a, buf_b);
      ::operator delete(buf_a, std::align_val_t(64));
      ::operator delete(buf_b, std::align_val_t(64));
    };
    pool.parallel_for(static_cast<int>(regions.size()), job);
    detail::reduce_splitk(c, d, ws, k_slices, m, n, epilogue);
    ::operator delete(ws, std::align_val_t(64));
    return Status::kSuccess;
  }

  auto job = [&](int task, int /*thread_index*/) {
    PackedA* buf_a = static_cast<PackedA*>(
        ::operator new(buf_a_elems * sizeof(PackedA), std::align_val_t(64)));
    PackedB* buf_b = static_cast<PackedB*>(
        ::operator new(buf_b_elems * sizeof(PackedB), std::align_val_t(64)));
    const threadblock::GemmRegion& r = regions[static_cast<size_t>(task)];
    Block::run(a, b, c, d, r.m0, r.m1, r.n0, r.n1,
               r.k0, r.k1, epilogue, buf_a, buf_b);
    ::operator delete(buf_a, std::align_val_t(64));
    ::operator delete(buf_b, std::align_val_t(64));
  };

  if (regions.size() == 1) {
    job(0, 0);
  } else {
    pool.parallel_for(static_cast<int>(regions.size()), job);
  }
  return Status::kSuccess;
}

}  // namespace kernel
}  // namespace gemm
}  // namespace cpu_ops
