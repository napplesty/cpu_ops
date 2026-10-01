#pragma once

// Exact top-k index selection: the k indices of the largest values, in
// (value descending, index ascending) order — the same total order a
// partial_sort on (value desc, index asc) produces, so results are
// drop-in identical to the reference formulation.
//
// Algorithm (large n): MSD radix select on the order-preserving transform
// of the f32 bit pattern (positives biased high, negatives inverted, -0
// canonicalized to +0 so float-equal values tie by index). Byte-level
// histogram passes isolate the k-th boundary key; one final scan collects
// everything strictly above the boundary plus the needed boundary
// elements in ascending index order (realizing the tie rule), and a sort
// of the k results restores value-descending order. Small n uses the
// partial_sort formulation directly. NaN inputs are rejected (the total
// order is undefined there); k is clamped to n.
//
// Deterministic by construction: fixed bucket walks, ascending scans.
// Single-threaded — callers parallelize across rows (as DSA does).

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

#include "cpu_ops/status.h"

namespace cpu_ops {
namespace ops {
namespace topk_detail {

inline uint32_t key_of(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  if (b == 0x80000000u) b = 0;  // -0 -> +0: float-equal, tie by index
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

inline int32_t* new_ibuf(std::size_t elems) {
  return static_cast<int32_t*>(
      ::operator new(elems * sizeof(int32_t), std::align_val_t(64)));
}

}  // namespace topk_detail

// Writes min(k, n) indices to `out` in (value desc, index asc) order.
inline Status topk_indices(const float* v, int n, int k, int32_t* out) {
  if (n < 0 || k < 1) return Status::kErrorInvalidProblem;
  if (!v || !out) return Status::kErrorInvalidArguments;
  if (k > n) k = n;
  if (k == 0) return Status::kSuccess;

  for (int i = 0; i < n; ++i) {
    if (std::isnan(v[i])) return Status::kErrorInvalidArguments;
  }

  const auto by_value = [&v](int32_t a, int32_t b) {
    if (v[a] != v[b]) return v[a] > v[b];
    return a < b;
  };

  if (n <= 4096) {
    int32_t* id = topk_detail::new_ibuf(n);
    for (int i = 0; i < n; ++i) id[i] = i;
    std::partial_sort(id, id + k, id + n, by_value);
    std::memcpy(out, id, static_cast<std::size_t>(k) * sizeof(int32_t));
    ::operator delete(id, std::align_val_t(64));
    return Status::kSuccess;
  }

  // MSD radix select: after the level walk, `thresh` is the key of the
  // k-th element and `need` is how many equal-thresh elements (taken in
  // ascending index order) complete the selection; k - need elements are
  // strictly above and all selected.
  uint32_t prefix = 0;
  int need = k;
  for (int level = 3; level >= 0; --level) {
    const int shift = 8 * level;
    // Bytes above the current level; empty at the top level. Written this way
    // because 1u << 32 (the level-3 case of the natural formula) is UB.
    const uint32_t high_mask = (level == 3) ? 0u : ~0u << (8 * level + 8);
    uint32_t counts[256] = {0};
    for (int i = 0; i < n; ++i) {
      const uint32_t key = topk_detail::key_of(v[i]);
      if ((key & high_mask) == prefix) ++counts[(key >> shift) & 0xFF];
    }
    int cum = 0;
    int bv = -1;
    for (int b = 255; b >= 0; --b) {
      cum += static_cast<int>(counts[b]);
      if (cum >= need) {
        bv = b;
        need -= cum - static_cast<int>(counts[b]);
        break;
      }
    }
    if (bv < 0) return Status::kErrorInvalidProblem;  // counts sum < need: impossible
    // The walk's break condition keeps need >= 1: strictly-above counts
    // are always < need when the loop breaks at bucket bv.
    prefix |= static_cast<uint32_t>(bv) << shift;
  }
  const uint32_t thresh = prefix;
  if (need < 0) return Status::kErrorInvalidProblem;

  int filled = 0;
  const int n_eq = need;  // equal-thresh budget
  int used_eq = 0;
  for (int i = 0; i < n && filled < k; ++i) {
    const uint32_t key = topk_detail::key_of(v[i]);
    if (key > thresh) {
      out[filled++] = i;
    } else if (key == thresh && used_eq < n_eq) {
      ++used_eq;
      out[filled++] = i;
    }
  }
  if (filled < k) return Status::kErrorInvalidProblem;  // invariant broken
  std::sort(out, out + k, by_value);
  return Status::kSuccess;
}

}  // namespace ops
}  // namespace cpu_ops
