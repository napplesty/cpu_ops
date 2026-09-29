#pragma once

// Definitions for cpu_ops::gemm::device::Gemm and GemmMx. The library
// instantiates the standard combinations in its own translation units;
// include this header only to instantiate custom epilogues, tile
// configurations, or mma policies.

#include <algorithm>
#include <cstddef>
#include <new>
#include <type_traits>

#include "cpu_ops/detail/block_gemm.h"
#include "cpu_ops/detail/simd.h"
#include "cpu_ops/detail/thread_pool.h"
#include "cpu_ops/detail/thread_swizzle.h"
#include "cpu_ops/epilogue/partial_sum.h"
#include "cpu_ops/gemm/device/gemm.h"
#include "cpu_ops/gemm/device/gemm_mx.h"

namespace cpu_ops {
namespace gemm {
namespace device {

namespace detail {

// Combines split-k partial slabs in slice order (deterministic) and applies
// the user epilogue exactly once per element: D = epilogue(sum_s W_s, C).
// Slabs are row-major m x n with leading dimension n.
template <typename T, typename LayoutC, typename Epilogue>
inline void reduce_splitk(TensorRef<const T, LayoutC> c, TensorRef<T, LayoutC> d,
                          const T* ws, int slices, int m, int n,
                          const Epilogue& epilogue) {
  constexpr int kV = simd::native_width<T>();
  using VecT = simd::Vec<T, kV>;
  const size_t slab = static_cast<size_t>(m) * n;
  if constexpr (LayoutC::kIsRowMajor && gemm::epilogue_has_apply_vec<Epilogue, VecT>::value) {
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

// Shared orchestration behind Gemm and GemmMx: thread partition, optional
// split-k two-phase execution, per-task packing buffers. Operand references
// are forwarded opaquely to the policy's pack functions (TensorRef for dense
// operands, MxTensorRef for block-scaled ones).
template <typename Policy, typename LayoutC, typename Epilogue, typename Config,
          typename RefA, typename RefB>
Status run_blocked(const RefA& a, const RefB& b,
                   TensorRef<const typename Policy::AccT, LayoutC> c,
                   TensorRef<typename Policy::AccT, LayoutC> d, int m, int n, int k,
                   const typename Epilogue::Params& epilogue_params,
                   int split_k_slices, int num_threads) {
  using T = typename Policy::AccT;
  const Epilogue epilogue(epilogue_params);

  if (k == 0) {
    // Degenerate product: fold beta * C into D without touching A or B.
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        d.at(i, j) = epilogue(T(0), c.at(i, j), true, true, i, j);
      }
    }
    return Status::kSuccess;
  }

  thread::ThreadPool& pool = thread::ThreadPool::global();
  int p = num_threads <= 0 ? pool.num_threads() : std::min(num_threads, pool.num_threads());
  if (p < 1) p = 1;

  // Split-k decision. Deep reductions are cut into k slices whose partial
  // products are summed in a second pass. Splitting does two things: it
  // creates tasks when the output grid alone cannot feed all threads, and —
  // because each slice is partitioned across fewer threads — it coarsens the
  // m x n grid, so A and B panels are packed fewer times. The price is a
  // workspace of slices m x n slabs plus a reduction pass over them, so the
  // slice count is bounded by k depth (>= 1 KC block per slice, which also
  // guarantees non-empty slices) and by a workspace budget: measured on
  // AVX2, splitting pays when a slab is around 1 MiB or less, is neutral at
  // 4 MiB, and regresses beyond — the reduce pass and slab traffic outweigh
  // the packing savings once the output grid is large. Below ~8 KC blocks of
  // depth the extra pass does not pay for itself either.
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
      // Aim for roughly two tasks per thread.
      k_slices = std::min(2 * p, ws_cap);
    }
    k_slices = std::max(1, std::min(k_slices, depth_cap));
  }

  const std::vector<GemmRegion> regions =
      partition_gemm(m, n, k, Config::kMR, Config::kNR, p, k_slices);

  using Block = BlockGemm<Policy, LayoutC, Epilogue, Config>;
  using PackedA = typename Policy::PackedA;
  using PackedB = typename Policy::PackedB;
  constexpr int kKCPadded = Policy::pad_kc(Config::kKC);
  constexpr std::size_t buf_a_elems =
      static_cast<std::size_t>(Block::kMCPadded) * kKCPadded;
  constexpr std::size_t buf_b_elems =
      static_cast<std::size_t>(Block::kNCPadded) * kKCPadded;

  if (k_slices > 1) {
    // Phase 1: every region reduces its k slice into its own partial-sum slab
    // (row-major, no epilogue applied yet).
    using WsEpilogue = epilogue::PartialSum<T>;
    using BlockWs = BlockGemm<Policy, layout::RowMajor, WsEpilogue, Config>;
    const std::size_t slab_elems = static_cast<std::size_t>(m) * n;
    T* ws = static_cast<T*>(::operator new(slab_elems * k_slices * sizeof(T),
                                           std::align_val_t(64)));
    auto job = [&](int task, int /*thread_index*/) {
      PackedA* buf_a = static_cast<PackedA*>(
          ::operator new(buf_a_elems * sizeof(PackedA), std::align_val_t(64)));
      PackedB* buf_b = static_cast<PackedB*>(
          ::operator new(buf_b_elems * sizeof(PackedB), std::align_val_t(64)));
      const GemmRegion& r = regions[static_cast<size_t>(task)];
      T* slab_ptr = ws + static_cast<size_t>(r.slice) * slab_elems;
      TensorRef<T, layout::RowMajor> slab(slab_ptr, n);
      TensorRef<const T, layout::RowMajor> slab_c(slab_ptr, n);
      BlockWs::run(a, b, slab_c, slab, r.m0, r.m1, r.n0, r.n1, r.k0,
                   r.k1, WsEpilogue{}, buf_a, buf_b);
      ::operator delete(buf_a, std::align_val_t(64));
      ::operator delete(buf_b, std::align_val_t(64));
    };
    pool.parallel_for(static_cast<int>(regions.size()), job);
    // Phase 2: deterministic ordered reduction + the real epilogue.
    detail::reduce_splitk(c, d, ws, k_slices, m, n, epilogue);
    ::operator delete(ws, std::align_val_t(64));
    return Status::kSuccess;
  }

  auto job = [&](int task, int /*thread_index*/) {
    PackedA* buf_a = static_cast<PackedA*>(
        ::operator new(buf_a_elems * sizeof(PackedA), std::align_val_t(64)));
    PackedB* buf_b = static_cast<PackedB*>(
        ::operator new(buf_b_elems * sizeof(PackedB), std::align_val_t(64)));
    const GemmRegion& r = regions[static_cast<size_t>(task)];
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

}  // namespace detail

template <typename ElementA, typename LayoutA, typename ElementB, typename LayoutB,
          typename ElementC, typename LayoutC, typename ElementAccumulator, typename Epilogue,
          typename Config, typename MmaPolicy>
Status Gemm<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, ElementAccumulator,
            Epilogue, Config, MmaPolicy>::operator()(const Arguments& args,
                                                     int num_threads) const {
  static_assert(std::is_same<ElementA, typename MmaPolicy::ElemA>::value &&
                    std::is_same<ElementB, typename MmaPolicy::ElemB>::value &&
                    std::is_same<ElementAccumulator, typename MmaPolicy::AccT>::value &&
                    std::is_same<ElementC, typename MmaPolicy::AccT>::value,
                "element types must match the mma policy's ElemA/ElemB, and both the "
                "accumulator and C must use the policy's AccT");

  const int m = args.problem_size.m;
  const int n = args.problem_size.n;
  const int k = args.problem_size.k;
  if (m < 0 || n < 0 || k < 0) return Status::kErrorInvalidProblem;
  if (m == 0 || n == 0) return Status::kSuccess;

  if (k == 0) {
    // Degenerate product: fold beta * C into D without touching A or B
    // (which are allowed to be empty in this case).
    const Epilogue epilogue(args.epilogue);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        args.ref_D.at(i, j) = epilogue(ElementC(0), args.ref_C.at(i, j), true, true, i, j);
      }
    }
    return Status::kSuccess;
  }

  if (!args.ref_A.data() || !args.ref_B.data() || !args.ref_C.data() || !args.ref_D.data()) {
    return Status::kErrorInvalidArguments;
  }
  const int lda_min = LayoutA::kIsRowMajor ? k : m;
  const int ldb_min = LayoutB::kIsRowMajor ? n : k;
  const int ldc_min = LayoutC::kIsRowMajor ? n : m;
  const int ldd_min = ldc_min;
  if (args.ref_A.ld() < lda_min || args.ref_B.ld() < ldb_min || args.ref_C.ld() < ldc_min ||
      args.ref_D.ld() < ldd_min) {
    return Status::kErrorInvalidArguments;
  }

  return detail::run_blocked<MmaPolicy, LayoutC, Epilogue, Config>(
      args.ref_A, args.ref_B, args.ref_C, args.ref_D, m, n, k, args.epilogue,
      args.split_k_slices, num_threads);
}

template <typename StorageA, typename StorageB, typename LayoutC, typename Epilogue,
          typename Config>
Status GemmMx<StorageA, StorageB, LayoutC, Epilogue, Config>::operator()(
    const Arguments& args, int num_threads) const {
  const int m = args.problem_size.m;
  const int n = args.problem_size.n;
  const int k = args.problem_size.k;
  if (m < 0 || n < 0 || k < 0) return Status::kErrorInvalidProblem;
  if (m == 0 || n == 0) return Status::kSuccess;

  if (k == 0) {
    // Degenerate product: fold beta * C into D without touching A or B.
    const Epilogue epilogue(args.epilogue);
    for (int i = 0; i < m; ++i) {
      for (int j = 0; j < n; ++j) {
        args.ref_D.at(i, j) = epilogue(0.0f, args.ref_C.at(i, j), true, true, i, j);
      }
    }
    return Status::kSuccess;
  }

  const MxTensorRef<StorageA>& a = args.ref_A;
  const MxTensorRef<StorageB>& b = args.ref_B;
  if (!a.data || !a.scales || !b.data || !b.scales || !args.ref_C.data() ||
      !args.ref_D.data()) {
    return Status::kErrorInvalidArguments;
  }
  // Both operands are contiguous along k: ld counts elements along k for A
  // (m x k row-major) and along k for B (k x n column-major). Scale arrays
  // follow the same layout with ceil(k / 32) entries per row/column.
  const int scale_ld_min = (k + 31) / 32;
  if (a.ld < k || b.ld < k || a.ld_scales < scale_ld_min ||
      b.ld_scales < scale_ld_min) {
    return Status::kErrorInvalidArguments;
  }
  // fp4 packs two elements per byte; a row must start on a byte boundary.
  if ((mma::mx_detail2::traits<StorageA>::kLog2ElemsPerByte > 0 && (a.ld & 1)) ||
      (mma::mx_detail2::traits<StorageB>::kLog2ElemsPerByte > 0 && (b.ld & 1))) {
    return Status::kErrorInvalidArguments;
  }
  const int ldc_min = LayoutC::kIsRowMajor ? n : m;
  if (args.ref_C.ld() < ldc_min || args.ref_D.ld() < ldc_min) {
    return Status::kErrorInvalidArguments;
  }

  using Policy = mma::MxPolicy<StorageA, StorageB>;
  return detail::run_blocked<Policy, LayoutC, Epilogue, Config>(
      a, b, args.ref_C, args.ref_D, m, n, k, args.epilogue, args.split_k_slices,
      num_threads);
}

}  // namespace device
}  // namespace gemm
}  // namespace cpu_ops
