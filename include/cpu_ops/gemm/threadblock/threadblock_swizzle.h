#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>
#include <utility>
#include <vector>

namespace cpu_ops {
namespace gemm {
namespace threadblock {

// Rectangular tile of the output matrix, half-open [m0, m1) x [n0, n1), plus
// the k range [k0, k1) to reduce (the whole k when the region is unsplit) and
// the index of the k slice (0 when unsplit), which selects the partial-sum
// workspace slab.
struct GemmRegion {
  int m0 = 0;
  int m1 = 0;
  int n0 = 0;
  int n1 = 0;
  int k0 = 0;
  int k1 = 0;
  int slice = 0;
};

namespace detail {

// Returns the [begin, length) of slice `idx` when `count` items are split into
// `parts` balanced contiguous slices (requires 1 <= parts <= count).
inline std::pair<int, int> split_range(int count, int parts, int idx) {
  const int base = count / parts;
  const int rem = count % parts;
  const int begin = idx * base + std::min(idx, rem);
  const int length = base + (idx < rem ? 1 : 0);
  return {begin, length};
}

}  // namespace detail

// Splits an m x n output into up to num_threads rectangular regions aligned to
// mr x nr micro-tiles; the grid fills the thread count first, then minimizes
// panel re-packing (every region packs its own A rows and B columns). With
// k_slices > 1 the k range is also cut into contiguous slices crossed with the
// grid; each slice's regions must write to their own partial-sum slab (indexed
// by slice) that a later reduction pass combines.
inline std::vector<GemmRegion> partition_gemm(int m, int n, int k, int mr, int nr,
                                              int num_threads, int k_slices = 1) {
  if (num_threads < 1) num_threads = 1;
  if (k_slices < 1) k_slices = 1;
  const int mt = (m + mr - 1) / mr;
  const int nt = (n + nr - 1) / nr;

  const int grid_threads = std::max(1, num_threads / k_slices);
  int best_tm = 1, best_tn = 1;
  long best_regions = -1;
  int64_t best_cost = INT64_MAX;
  for (int tm_c = 1; tm_c <= std::min(grid_threads, mt); ++tm_c) {
    const int tn_c = std::min(grid_threads / tm_c, nt);
    const long regions = static_cast<long>(tm_c) * tn_c;
    const int64_t cost = static_cast<int64_t>(tn_c) * m + static_cast<int64_t>(tm_c) * n;
    if (regions > best_regions || (regions == best_regions && cost < best_cost)) {
      best_regions = regions;
      best_cost = cost;
      best_tm = tm_c;
      best_tn = tn_c;
    }
  }
  const int tm = best_tm;
  const int tn = best_tn;

  std::vector<GemmRegion> regions;
  regions.reserve(static_cast<size_t>(tm) * tn * k_slices);
  for (int s = 0; s < k_slices; ++s) {
    const auto ks = detail::split_range(k, k_slices, s);
    const int k0 = ks.first;
    const int k1 = ks.first + ks.second;
    for (int mi = 0; mi < tm; ++mi) {
      const auto ms = detail::split_range(mt, tm, mi);
      const int m0 = ms.first * mr;
      const int m1 = std::min(m, (ms.first + ms.second) * mr);
      for (int ni = 0; ni < tn; ++ni) {
        const auto ns = detail::split_range(nt, tn, ni);
        const int n0 = ns.first * nr;
        const int n1 = std::min(n, (ns.first + ns.second) * nr);
        regions.push_back({m0, m1, n0, n1, k0, k1, s});
      }
    }
  }
  return regions;
}

}  // namespace threadblock
}  // namespace gemm
}  // namespace cpu_ops
