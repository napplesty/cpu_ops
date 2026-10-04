#pragma once

// Exact top-k selection for f32/f64: writes min(k, n) indices of the largest
// values to `out` in (value desc, index asc) order, bitwise identical to the
// partial_sort formulation; NaN is rejected; deterministic.

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

template <typename T>
struct radix_key;

template <>
struct radix_key<float> {
  using KeyT = uint32_t;
  static KeyT of(float f) {
    KeyT b;
    std::memcpy(&b, &f, sizeof(b));
    if (b == 0x80000000u) b = 0;  // -0 -> +0: float-equal, tie by index
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
  }
};

template <>
struct radix_key<double> {
  using KeyT = uint64_t;
  static KeyT of(double f) {
    KeyT b;
    std::memcpy(&b, &f, sizeof(b));
    if (b == 0x8000000000000000ull) b = 0;  // -0 -> +0: float-equal, tie by index
    return (b & 0x8000000000000000ull) ? ~b : (b | 0x8000000000000000ull);
  }
};

inline int32_t* new_ibuf(std::size_t elems) {
  return static_cast<int32_t*>(
      ::operator new(elems * sizeof(int32_t), std::align_val_t(64)));
}

}  // namespace topk_detail

template <typename T>
Status topk_indices(const T* v, int n, int k, int32_t* out) {
  using KeyT = typename topk_detail::radix_key<T>::KeyT;
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

  constexpr int kLevels = static_cast<int>(sizeof(KeyT));
  KeyT prefix = 0;
  int need = k;
  for (int level = kLevels - 1; level >= 0; --level) {
    const int shift = 8 * level;
    const KeyT high_mask = (level == kLevels - 1)
                               ? KeyT(0)
                               : static_cast<KeyT>(~KeyT(0) << (8 * level + 8));
    uint32_t counts[256] = {0};
    for (int i = 0; i < n; ++i) {
      const KeyT key = topk_detail::radix_key<T>::of(v[i]);
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
    if (bv < 0) return Status::kErrorInvalidProblem;
    prefix |= static_cast<KeyT>(bv) << shift;
  }
  const KeyT thresh = prefix;
  if (need < 0) return Status::kErrorInvalidProblem;

  int filled = 0;
  const int n_eq = need;
  int used_eq = 0;
  for (int i = 0; i < n && filled < k; ++i) {
    const KeyT key = topk_detail::radix_key<T>::of(v[i]);
    if (key > thresh) {
      out[filled++] = i;
    } else if (key == thresh && used_eq < n_eq) {
      ++used_eq;
      out[filled++] = i;
    }
  }
  if (filled < k) return Status::kErrorInvalidProblem;
  std::sort(out, out + k, by_value);
  return Status::kSuccess;
}

}  // namespace ops
}  // namespace cpu_ops
