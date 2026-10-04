#pragma once

// Composition-time contract checks for the GEMM component stack. BlockGemm
// and the device layers static_assert these traits, so a bad
// Policy/Config/Epilogue/Layout combination fails here with a named message
// instead of deep inside the mainloop. The first group also drives optional
// fast paths (vector store, per-policy panel strides).

#include <type_traits>

namespace cpu_ops {
namespace gemm {
namespace threadblock {

// --- optional-capability detection ---

// Kill-switch for the vectorized store path (debugging).
#if defined(CPU_OPS_NO_VEC_STORE)
constexpr bool kVecStoreEnabled = false;
#else
constexpr bool kVecStoreEnabled = true;
#endif

// Epilogues offering a lane-wise form E::apply_vec(Vec, Vec, bool, bool, int, int).
template <typename E, typename V, typename = void>
struct epilogue_has_apply_vec : std::false_type {};
template <typename E, typename V>
struct epilogue_has_apply_vec<
    E, V, std::void_t<decltype(std::declval<const E&>().apply_vec(
              std::declval<const V&>(), std::declval<const V&>(), true, true, 0, 0))>>
    : std::true_type {};

// An atom exposes its accumulator registers (VecT acc[MR][VecN]) unless it
// spills internally (the SME/AMX atoms' tiles live in tile storage and are
// only readable as a whole inside run()).
template <typename A, typename = void>
struct atom_has_registers : std::false_type {};
template <typename A>
struct atom_has_registers<A, std::void_t<typename A::VecT>> : std::true_type {};

// Policies whose packed strips carry trailing per-strip metadata (the MX-VNNI
// policy appends per-block f32 scales) override the strip stride; the default
// is kc_pad * tile elements of PackedA/PackedB.
template <typename P, typename = void>
struct panel_stride_a {
  static constexpr int get(int kc_pad, int tile) { return kc_pad * tile; }
};
template <typename P>
struct panel_stride_a<P, std::void_t<decltype(P::panel_stride_a(0, 0))>> {
  static constexpr int get(int kc_pad, int tile) { return P::panel_stride_a(kc_pad, tile); }
};
template <typename P, typename = void>
struct panel_stride_b {
  static constexpr int get(int kc_pad, int tile) { return kc_pad * tile; }
};
template <typename P>
struct panel_stride_b<P, std::void_t<decltype(P::panel_stride_b(0, 0))>> {
  static constexpr int get(int kc_pad, int tile) { return P::panel_stride_b(kc_pad, tile); }
};

// Full-tile fast path: accumulator streams straight from registers into C/D
// with a vector epilogue. Requires row-major C/D, a vector-capable epilogue,
// and a register-exposing atom.
template <typename A, typename E, typename L, typename = void>
struct can_vector_store : std::false_type {};
template <typename A, typename E, typename L>
struct can_vector_store<A, E, L, std::void_t<typename A::VecT>>
    : std::integral_constant<bool, L::kIsRowMajor && kVecStoreEnabled &&
                                       epilogue_has_apply_vec<E, typename A::VecT>::value> {};

// --- contract traits ---

template <typename C, typename = void>
struct config_shape : std::false_type {};
template <typename C>
struct config_shape<C, std::void_t<decltype(C::kMR), decltype(C::kNR), decltype(C::kMC),
                                   decltype(C::kNC), decltype(C::kKC)>>
    : std::integral_constant<bool, (C::kMR > 0 && C::kNR > 0 && C::kMC > 0 && C::kNC > 0 &&
                                    C::kKC > 0)> {};

template <typename P, typename = void>
struct policy_kstep : std::false_type {};
template <typename P>
struct policy_kstep<P, std::void_t<decltype(P::kKStep)>>
    : std::integral_constant<bool, (P::kKStep > 0)> {};

template <typename P, typename = void>
struct policy_pad_kc : std::false_type {};
template <typename P>
struct policy_pad_kc<P, std::void_t<decltype(P::pad_kc(0))>>
    : std::is_convertible<decltype(P::pad_kc(0)), int> {};

// Atom shapes: register form (VecT + clear + mma + store_tile) or run-only
// (spills internally). PA/PB are the policy's packed element types.
template <typename A, typename PA, typename PB, typename AccT, typename = void>
struct atom_register_form : std::false_type {};
template <typename A, typename PA, typename PB, typename AccT>
struct atom_register_form<
    A, PA, PB, AccT,
    std::void_t<typename A::VecT, decltype(std::declval<A&>().clear()),
                decltype(std::declval<A&>().mma(std::declval<const PA*>(),
                                                std::declval<const PB*>(), 0)),
                decltype(std::declval<A&>().store_tile(std::declval<AccT*>()))>>
    : std::true_type {};

template <typename A, typename PA, typename PB, typename AccT, typename = void>
struct atom_run_form : std::false_type {};
template <typename A, typename PA, typename PB, typename AccT>
struct atom_run_form<
    A, PA, PB, AccT,
    std::void_t<decltype(std::declval<A&>().run(std::declval<const PA*>(),
                                                std::declval<const PB*>(), 0,
                                                std::declval<AccT*>()))>>
    : std::true_type {};

template <typename A, typename PA, typename PB, typename AccT, typename = void>
struct atom_contract : std::false_type {};
template <typename A, typename PA, typename PB, typename AccT>
struct atom_contract<A, PA, PB, AccT, std::void_t<decltype(A::kMR), decltype(A::kNR)>>
    : std::integral_constant<bool, atom_register_form<A, PA, PB, AccT>::value ||
                                       atom_run_form<A, PA, PB, AccT>::value> {};

// The policy's Atom alias at a tile shape, checked against the atom contract.
template <typename P, int MR, int NR, typename = void>
struct policy_atom : std::false_type {};
template <typename P, int MR, int NR>
struct policy_atom<P, MR, NR,
                   std::void_t<typename P::template Atom<MR, NR>, typename P::PackedA,
                               typename P::PackedB, typename P::AccT>>
    : atom_contract<typename P::template Atom<MR, NR>, typename P::PackedA,
                    typename P::PackedB, typename P::AccT> {};

// The policy's packers accept the operand reference types the device layer
// forwards (TensorRef for dense, MxTensorRef for block-scaled operands).
template <typename P, typename RefA, typename = void>
struct pack_a_compatible : std::false_type {};
template <typename P, typename RefA>
struct pack_a_compatible<
    P, RefA, std::void_t<decltype(P::pack_a(std::declval<const RefA&>(), 0, 0, 0, 0, 0, 0,
                                            std::declval<typename P::PackedA*>()))>>
    : std::true_type {};
template <typename P, typename RefB, typename = void>
struct pack_b_compatible : std::false_type {};
template <typename P, typename RefB>
struct pack_b_compatible<
    P, RefB, std::void_t<decltype(P::pack_b(std::declval<const RefB&>(), 0, 0, 0, 0, 0, 0,
                                            std::declval<typename P::PackedB*>()))>>
    : std::true_type {};

template <typename E, typename T, typename = void>
struct epilogue_functor : std::false_type {};
template <typename E, typename T>
struct epilogue_functor<
    E, T,
    std::void_t<decltype(std::declval<const E&>()(std::declval<T>(), std::declval<T>(),
                                                  true, true, 0, 0))>>
    : std::is_convertible<decltype(std::declval<const E&>()(
                              std::declval<T>(), std::declval<T>(), true, true, 0, 0)),
                          T> {};

// Device/kernel layers additionally need Epilogue::Params and construction
// from it.
template <typename E, typename = void>
struct epilogue_params : std::false_type {};
template <typename E>
struct epilogue_params<E, std::void_t<typename E::Params>>
    : std::is_constructible<E, const typename E::Params&> {};

template <typename L, typename = void>
struct layout_c : std::false_type {};
template <typename L>
struct layout_c<L, std::void_t<decltype(L::kIsRowMajor), decltype(L::offset(0, 0, 0))>>
    : std::true_type {};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cpu_ops
