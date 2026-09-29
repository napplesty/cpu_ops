#pragma once

#include <type_traits>

#include "cpu_ops/layout.h"

namespace cpu_ops {
namespace gemm {

// Detects epilogues offering a lane-wise form
// E::apply_vec(Vec, Vec, bool, bool, int, int) (see epilogue/linear_combination.h).
template <typename E, typename V, typename = void>
struct epilogue_has_apply_vec : std::false_type {};
template <typename E, typename V>
struct epilogue_has_apply_vec<
    E, V, std::void_t<decltype(std::declval<const E&>().apply_vec(
              std::declval<const V&>(), std::declval<const V&>(), true, true, 0, 0))>>
    : std::true_type {};

// An atom exposes its accumulator registers (VecT acc[MR][VecN]) unless it
// spills internally (e.g. the SME atom, whose tile lives in ZA storage and is
// only readable as a whole inside run()).
template <typename A, typename = void>
struct atom_has_registers : std::false_type {};
template <typename A>
struct atom_has_registers<A, std::void_t<typename A::VecT>> : std::true_type {};

// Kill-switch for the vectorized store path (debugging).
#if defined(CPU_OPS_NO_VEC_STORE)
constexpr bool kVecStoreEnabled = false;
#else
constexpr bool kVecStoreEnabled = true;
#endif

// The full-tile fast path streams the accumulator straight from registers
// into C/D with vector epilogue application. Requires a row-major C/D, a
// vector-capable epilogue, and a register-exposing atom.
template <typename A, typename E, typename L, typename = void>
struct can_vector_store : std::false_type {};
template <typename A, typename E, typename L>
struct can_vector_store<A, E, L, std::void_t<typename A::VecT>>
    : std::integral_constant<bool, L::kIsRowMajor && kVecStoreEnabled &&
                                       epilogue_has_apply_vec<E, typename A::VecT>::value> {};

// Cache-blocked GEMM mainloop over a rectangular C region [m0, m1) x [n0, n1)
// reducing the k range [k0, k1), parameterized by an MmaPolicy (see
// detail/mma_policy_fma.h):
//
//   for jc over NC:                    // L2-level column block
//     for pc over K step KC:           // L2-level k block
//       pack B panel (kc x nc)
//       for ic over MC:                // L1/L2-level row block
//         pack A panel (mc x kc)
//         macro-kernel over MR x NR micro-tiles
//
// buf_a must hold round_up(MC, MR) * pad_kc(KC) PackedA elements, buf_b must
// hold round_up(NC, NR) * pad_kc(KC) PackedB elements (panels are padded up to
// whole micro-tiles, and k is padded to the policy's k step). The epilogue is
// invoked per output element per k-block with
// (acc, source, first_k_block, last_k_block).
//
// The operand references a/b are whatever the policy's pack_a/pack_b accept:
// TensorRef for dense layouts, MxTensorRef for block-scaled formats.
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

  using Atom = typename Policy::template Atom<kMR, kNR>;

  template <typename RefA, typename RefB>
  static void run(const RefA& a, const RefB& b, TensorRef<const AccT, LayoutC> c,
                  TensorRef<AccT, LayoutC> d, int m0, int m1, int n0, int n1, int k0,
                  int k1, const Epilogue& epilogue, PackedA* buf_a, PackedB* buf_b) {
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

  // Walks the packed panels micro-tile by micro-tile. jr is the outer loop so
  // each packed B strip (kc_pad x NR) stays hot in L1 while the A panel
  // streams.
  static void macro_kernel(const PackedA* pa, const PackedB* pb,
                           TensorRef<const AccT, LayoutC> c, TensorRef<AccT, LayoutC> d,
                           int i0, int j0, int mc, int nc, int kc_pad, bool first,
                           bool last, const Epilogue& epilogue) {
    constexpr bool kFastStore = can_vector_store<Atom, Epilogue, LayoutC>::value;
    for (int jr = 0; jr < nc; jr += kNR) {
      const PackedB* bstrip = pb + (jr / kNR) * kc_pad * kNR;
      const int jmax = (nc - jr < kNR) ? (nc - jr) : kNR;
      for (int ir = 0; ir < mc; ir += kMR) {
        const PackedA* astrip = pa + (ir / kMR) * kc_pad * kMR;
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

  // Full-tile fast path: the accumulator streams straight from registers into
  // C/D — kVecN wide loads/stores per row, no stack round-trip for the tile.
  // Only instantiated when can_vector_store holds.
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

  // Edge tiles (and layouts/atoms/epilogues without a vector path) go element
  // by element. The first k-block folds in beta * C; later k-blocks accumulate
  // onto the partial result already sitting in D. These differ when C and D
  // are not the same buffer.
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

}  // namespace gemm
}  // namespace cpu_ops
