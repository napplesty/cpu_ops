#pragma once

// EXPERIMENTAL, opt-in. Not yet validated on AMX hardware.
//
// Mma policy for Intel AMX-INT8 (Sapphire Rapids and later): uint8 x int8 ->
// int32, accumulated by TDPBUSD tile dot products. The micro-tile is a single
// 16x16 s32 C tile fed by a 16x64-byte A tile and a 16x64-byte B tile, i.e.
// 64 k-slices per instruction.
//
// Build requirements: x86-64, GCC 11+ or Clang 12+,
//   -march=sapphirerapids  (or -mamx-tile -mamx-int8)
// The library is header-only, so the AMX kernel is instantiated in the
// consumer's translation unit. Tile instructions are compiled only when
// __AMX_TILE__ && __AMX_INT8__ are defined (CPU_OPS_HAS_AMX_POLICY is then
// defined as well).
//
// Runtime requirement: Linux must grant permission to use the tile data
// (arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA)), which this header
// requests lazily on first use. Without compile-time support or runtime
// permission the atom falls back to a portable scalar loop over the same
// packed layouts, so the policy stays correct (if slow) everywhere.
//
// Packed layouts (kc is padded up to a multiple of 64; padded k-slices are 0):
//   A panel, per strip of 16 rows:  dst[(g * 16 + i) * 64 + t] = A(i, 64g + t)
//   B panel, per strip of 16 cols:  dst[(g * 16 + j) * 4 + t]  = B(4g + t, j)
// The B layout coincides with the VNNI one; each run of 16 consecutive k
// groups is exactly one B tile row block (VNNI interleave).
//
// Usage:
//   #include "cpu_ops/gemm/threadblock/mma_policy_amx.h"
//   using GemmAmx = cpu_ops::gemm::device::Gemm<
//       uint8_t, cpu_ops::layout::RowMajor, int8_t, cpu_ops::layout::RowMajor,
//       int32_t, cpu_ops::layout::RowMajor, int32_t,
//       cpu_ops::epilogue::LinearCombination<int32_t>,
//       cpu_ops::mma::AmxGemmConfig, cpu_ops::mma::AmxPolicy>;

#include <cstdint>
#include <cstring>

#include "cpu_ops/gemm/threadblock/mma_policy_vnni.h"
#include "cpu_ops/tensor_ref.h"

#if defined(__AMX_TILE__) && defined(__AMX_INT8__)
#include <immintrin.h>
#define CPU_OPS_HAS_AMX_POLICY 1
#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#define CPU_OPS_AMX_LINUX_SYSCALL 1
#endif
#endif

namespace cpu_ops {
namespace gemm {
namespace threadblock {

// A panel pack for AMX tiles: 64 consecutive k bytes per tile row.
template <typename LayoutA>
void pack_a_amx(const TensorRef<const uint8_t, LayoutA>& a, int i0, int k0, int mc,
                int kc, int kc_pad, int mr, uint8_t* dst) {
  for (int ib = 0; ib < mc; ib += mr) {
    const int imax = (mc - ib < mr) ? (mc - ib) : mr;
    uint8_t* strip = dst + static_cast<size_t>(ib / mr) * kc_pad * mr;
    for (int g = 0; g < kc_pad / 64; ++g) {
      for (int i = 0; i < mr; ++i) {
        for (int t = 0; t < 64; ++t) {
          const int k = 64 * g + t;
          const bool valid = (i < imax) && (k < kc);
          strip[(g * mr + i) * 64 + t] = valid ? a.at(i0 + ib + i, k0 + k) : uint8_t(0);
        }
      }
    }
  }
}

}  // namespace threadblock
}  // namespace gemm

namespace mma {

#if defined(CPU_OPS_HAS_AMX_POLICY)
namespace amx_detail {

// 64-byte tile configuration: palette 1, tmm0/tmm1/tmm2 each 16 rows of 64
// bytes. The layout matches the Linux XSAVE tileconfig payload.
struct TileConfig {
  uint8_t palette_id;
  uint8_t start_row;
  uint8_t reserved[14];
  uint16_t colsb[16];
  uint8_t rows[16];
};
static_assert(sizeof(TileConfig) == 64, "tileconfig payload must be 64 bytes");

// True when the OS granted tile-data permission (queried once per process).
// Without Linux's arch_prctl interface this reports false, which keeps the
// atom on its scalar fallback.
inline bool runtime_available() {
#if defined(CPU_OPS_AMX_LINUX_SYSCALL)
  static const bool granted = [] {
    // ARCH_REQ_XCOMP_PERM = 0x1023, XFEATURE_XTILEDATA = 18.
    return ::syscall(SYS_arch_prctl, 0x1023L, 18L) == 0;
  }();
  return granted;
#else
  return false;
#endif
}

// Loads the tile configuration into the current thread on first use; returns
// false when tile instructions may not be executed.
inline bool ensure_configured() {
  if (!runtime_available()) return false;
  static thread_local const bool configured = [] {
    alignas(64) TileConfig cfg{};
    cfg.palette_id = 1;
    for (int t = 0; t < 3; ++t) {
      cfg.colsb[t] = 64;
      cfg.rows[t] = 16;
    }
    _tile_loadconfig(&cfg);
    return true;
  }();
  return configured;
}

}  // namespace amx_detail
#endif  // CPU_OPS_HAS_AMX_POLICY

// Micro-kernel: one 16x16 s32 C tile (tmm0) fed by an A tile (tmm1) and a B
// tile (tmm2) per 64 k-slices. Spills the tile to a row-major 16x16 buffer,
// like the SME atom, so it deliberately exposes no register accumulators.
struct MmaAtomAmxInt8 {
  static constexpr int kMR = 16;
  static constexpr int kNR = 16;

  template <int MR_, int NR_>
  struct check {
    static_assert(MR_ == 16 && NR_ == 16, "the AMX-INT8 atom is a single 16x16 tile");
    using type = MmaAtomAmxInt8;
  };

  void run(const uint8_t* a, const int8_t* b, int kc_pad, int32_t* tile) const {
#if defined(CPU_OPS_HAS_AMX_POLICY)
    if (amx_detail::ensure_configured()) {
      const uint8_t* bb = reinterpret_cast<const uint8_t*>(b);
      _tile_zero(0);
      for (int g = 0; g < kc_pad / 64; ++g) {
        _tile_loadd(1, a + static_cast<size_t>(g) * 1024, 64);
        _tile_loadd(2, bb + static_cast<size_t>(g) * 1024, 64);
        _tile_dpbusd(0, 1, 2);
      }
      _tile_stored(0, tile, 64);
      return;
    }
#endif
    run_scalar(a, b, kc_pad, tile);
  }

  // Portable path over the packed layouts; also serves as the reference for
  // the tile path. Note TDPBUSD saturates the s32 accumulate on overflow
  // while this loop wraps, matching the library's scalar dpbusd fallback.
  static void run_scalar(const uint8_t* a, const int8_t* b, int kc_pad,
                         int32_t* tile) {
    for (int i = 0; i < kMR; ++i) {
      for (int j = 0; j < kNR; ++j) {
        int32_t acc = 0;
        for (int k = 0; k < kc_pad; ++k) {
          const uint8_t av = a[(static_cast<size_t>(k / 64) * kMR + i) * 64 + (k % 64)];
          const int8_t bv =
              b[((static_cast<size_t>(k / 64) * 16 + (k % 64) / 4) * kNR + j) * 4 +
                (k % 4)];
          acc += static_cast<int32_t>(av) * static_cast<int32_t>(bv);
        }
        tile[i * kNR + j] = acc;
      }
    }
  }
};

struct AmxPolicy {
  using ElemA = uint8_t;
  using ElemB = int8_t;
  using AccT = int32_t;
  using PackedA = uint8_t;
  using PackedB = int8_t;

  static constexpr int kKStep = 64;
  static constexpr int pad_kc(int kc) { return (kc + 63) & ~63; }

  template <int MR, int NR>
  using Atom = typename MmaAtomAmxInt8::check<MR, NR>::type;
  // (MmaAtomAmxInt8 itself is the atom; check<> enforces the tile shape.)

  template <typename LayoutA>
  static void pack_a(TensorRef<const uint8_t, LayoutA> a, int i0, int k0, int mc, int kc,
                     int kc_pad, int mr, uint8_t* dst) {
    gemm::threadblock::pack_a_amx(a, i0, k0, mc, kc, kc_pad, mr, dst);
  }

  template <typename LayoutB>
  static void pack_b(TensorRef<const int8_t, LayoutB> b, int k0, int j0, int kc,
                     int kc_pad, int nc, int nr, int8_t* dst) {
    gemm::threadblock::pack_b_vnni(b, k0, j0, kc, kc_pad, nc, nr,
                                   reinterpret_cast<uint8_t*>(dst));
  }
};

// Tile config matching the AMX atom: 16x16 micro-tiles.
struct AmxGemmConfig {
  static constexpr int kMR = 16;
  static constexpr int kNR = 16;
  static constexpr int kMC = 128;
  static constexpr int kNC = 128;
  static constexpr int kKC = 256;  // 4 tile k-groups
};

}  // namespace mma
}  // namespace cpu_ops

#if defined(CPU_OPS_AMX_LINUX_SYSCALL)
#undef CPU_OPS_AMX_LINUX_SYSCALL
#endif
#if defined(CPU_OPS_HAS_AMX_POLICY)
#undef CPU_OPS_HAS_AMX_POLICY
#endif
