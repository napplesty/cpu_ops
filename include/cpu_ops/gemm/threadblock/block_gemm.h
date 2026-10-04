#pragma once

// Cache-blocked GEMM mainloop over [m0,m1) x [n0,n1) reducing [k0,k1). The
// Policy supplies operand/packed types, panel packers and the micro-kernel
// Atom; the assembly is checked at instantiation (see gemm/contract.h).
// buf_a/buf_b must hold one panel each (kPackBufAElems / kPackBufBElems). The
// epilogue is invoked per output element per k-block with
// (acc, source, first_k_block, last_k_block).

#include <cstddef>
#include <type_traits>

#include "cpu_ops/gemm/contract.h"
#include "cpu_ops/layout/matrix.h"
#include "cpu_ops/tensor_ref.h"

namespace cpu_ops {
namespace gemm {
namespace threadblock {

template <typename Policy, typename LayoutC, typename Epilogue, typename Config>
struct BlockGemm {
  using ElemA = typename Policy::ElemA;
  using ElemB = typename Policy::ElemB;
  using AccT = typename Policy::AccT;
  using PackedA = typename Policy::PackedA;
  using PackedB = typename Policy::PackedB;

  static constexpr int kMR = Config::kMR;
  static constexpr int kNR = Config::kNR;
  static constexpr int kMC = Config::kMC;
  static constexpr int kNC = Config::kNC;
  static constexpr int kKC = Config::kKC;
  static constexpr int kMCPadded = ((kMC + kMR - 1) / kMR) * kMR;
  static constexpr int kNCPadded = ((kNC + kNR - 1) / kNR) * kNR;

  static constexpr int kKCPadded = Policy::pad_kc(kKC);
  static constexpr std::size_t kPackBufAElems =
      static_cast<std::size_t>(kMCPadded / kMR) * panel_stride_a<Policy>::get(kKCPadded, kMR);
  static constexpr std::size_t kPackBufBElems =
      static_cast<std::size_t>(kNCPadded / kNR) * panel_stride_b<Policy>::get(kKCPadded, kNR);

  using Atom = typename Policy::template Atom<kMR, kNR>;

  static_assert(config_shape<Config>::value,
                "GemmConfig must define positive kMR/kNR/kMC/kNC/kKC");
  static_assert(policy_kstep<Policy>::value, "MmaPolicy must define a positive kKStep");
  static_assert(policy_pad_kc<Policy>::value, "MmaPolicy must provide pad_kc(int) -> int");
  static_assert(policy_atom<Policy, kMR, kNR>::value,
                "Policy::Atom<kMR, kNR> must be a register atom (VecT + clear/mma/"
                "store_tile) or provide run(a, b, kc_pad, tile)");
  static_assert(epilogue_functor<Epilogue, AccT>::value,
                "Epilogue must be callable as (AccT, AccT, bool, bool, int, int) -> AccT");
  static_assert(layout_c<LayoutC>::value,
                "LayoutC must provide kIsRowMajor and offset(i, j, ld)");

  template <typename RefA, typename RefB>
  static void run(const RefA& a, const RefB& b, TensorRef<const AccT, LayoutC> c,
                  TensorRef<AccT, LayoutC> d, int m0, int m1, int n0, int n1, int k0,
                  int k1, const Epilogue& epilogue, PackedA* buf_a, PackedB* buf_b) {
    static_assert(pack_a_compatible<Policy, RefA>::value,
                  "Policy::pack_a does not accept the A operand reference type");
    static_assert(pack_b_compatible<Policy, RefB>::value,
                  "Policy::pack_b does not accept the B operand reference type");
    for (int jc = n0; jc < n1; jc += kNC) {
      const int nc = (n1 - jc < kNC) ? (n1 - jc) : kNC;
      for (int pc = k0; pc < k1; pc += kKC) {
        const int kc = (k1 - pc < kKC) ? (k1 - pc) : kKC;
        const int kc_pad = Policy::pad_kc(kc);
        const bool first = (pc == k0);
        const bool last = (pc + kc >= k1);
        Policy::pack_b(b, pc, jc, kc, kc_pad, nc, kNR, buf_b);
        for (int ic = m0; ic < m1; ic += kMC) {
          const int mc = (m1 - ic < kMC) ? (m1 - ic) : kMC;
          Policy::pack_a(a, ic, pc, mc, kc, kc_pad, kMR, buf_a);
          macro_kernel(buf_a, buf_b, c, d, ic, jc, mc, nc, kc_pad, first, last, epilogue);
        }
      }
    }
  }

  // jr is the outer loop so each packed B strip stays hot in L1 while the A
  // panel streams.
  static void macro_kernel(const PackedA* pa, const PackedB* pb,
                           TensorRef<const AccT, LayoutC> c, TensorRef<AccT, LayoutC> d,
                           int i0, int j0, int mc, int nc, int kc_pad, bool first,
                           bool last, const Epilogue& epilogue) {
    constexpr bool kFastStore = can_vector_store<Atom, Epilogue, LayoutC>::value;
    const int stride_a = panel_stride_a<Policy>::get(kc_pad, kMR);
    const int stride_b = panel_stride_b<Policy>::get(kc_pad, kNR);
    for (int jr = 0; jr < nc; jr += kNR) {
      const PackedB* bstrip = pb + static_cast<std::size_t>(jr / kNR) * stride_b;
      const int jmax = (nc - jr < kNR) ? (nc - jr) : kNR;
      for (int ir = 0; ir < mc; ir += kMR) {
        const PackedA* astrip = pa + static_cast<std::size_t>(ir / kMR) * stride_a;
        const int imax = (mc - ir < kMR) ? (mc - ir) : kMR;
        Atom atom;
        if constexpr (kFastStore) {
          atom.clear();
          atom.mma(astrip, bstrip, kc_pad);
          if (imax == kMR && jmax == kNR) {
            store_tile_vec(atom, c, d, i0 + ir, j0 + jr, first, last, epilogue);
          } else {
            AccT tile[kMR][kNR];
            atom.store_tile(&tile[0][0]);
            store_tile_scalar(tile, c, d, i0 + ir, j0 + jr, imax, jmax, first, last,
                              epilogue);
          }
        } else {
          AccT tile[kMR][kNR];
          atom.run(astrip, bstrip, kc_pad, &tile[0][0]);
          store_tile_scalar(tile, c, d, i0 + ir, j0 + jr, imax, jmax, first, last,
                            epilogue);
        }
      }
    }
  }

  static void store_tile_vec(const Atom& atom, TensorRef<const AccT, LayoutC> c,
                             TensorRef<AccT, LayoutC> d, int i0, int j0, bool first,
                             bool last, const Epilogue& epilogue) {
    using VecT = typename Atom::VecT;
    for (int i = 0; i < kMR; ++i) {
      const int row = i0 + i;
      const AccT* srow = &(first ? c.at(row, j0) : d.at(row, j0));
      AccT* drow = &d.at(row, j0);
      for (int w = 0; w < Atom::kVecN; ++w) {
        const VecT out =
            epilogue.apply_vec(atom.acc[i][w], VecT::load(srow + w * Atom::kVLEN), first,
                               last, row, j0 + w * Atom::kVLEN);
        out.store(drow + w * Atom::kVLEN);
      }
    }
  }

  // The first k-block folds in beta * C; later k-blocks accumulate onto the
  // partial result already sitting in D (they differ when C and D don't alias).
  static void store_tile_scalar(const AccT (&tile)[kMR][kNR],
                                TensorRef<const AccT, LayoutC> c, TensorRef<AccT, LayoutC> d,
                                int i0, int j0, int imax, int jmax, bool first, bool last,
                                const Epilogue& epilogue) {
    for (int i = 0; i < imax; ++i) {
      const int row = i0 + i;
      for (int j = 0; j < jmax; ++j) {
        const int col = j0 + j;
        const AccT source = first ? c.at(row, col) : d.at(row, col);
        d.at(row, col) = epilogue(tile[i][j], source, first, last, row, col);
      }
    }
  }
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cpu_ops
