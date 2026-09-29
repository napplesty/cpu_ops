#include "cpu_ops/detail/thread_swizzle.h"

#include <algorithm>
#include <climits>
#include <cstdint>

namespace cpu_ops {
namespace gemm {

namespace {

// Returns the [begin, length) of slice `idx` when `count` items are split into
// `parts` balanced contiguous slices (requires 1 <= parts <= count).
std::pair<int, int> split_range(int count, int parts, int idx) {
  const int base = count / parts;
  const int rem = count % parts;
  const int begin = idx * base + std::min(idx, rem);
  const int length = base + (idx < rem ? 1 : 0);
  return {begin, length};
}

}  // namespace

std::vector<GemmRegion> partition_gemm(int m, int n, int k, int mr, int nr, int num_threads,
                                       int k_slices) {
  if (num_threads < 1) num_threads = 1;
  if (k_slices < 1) k_slices = 1;
  const int mt = (m + mr - 1) / mr;
  const int nt = (n + nr - 1) / nr;

  // Every region packs its own A rows and B columns, so a tm x tn grid costs
  // tn*m*k + tm*k*n packed elements in total. Search the factor pairs of the
  // thread count for the grid that maximizes occupancy first and minimizes
  // that packing traffic second. (Splitting columns only, for example,
  // re-packs the whole of A once per thread.)
  //
  // With k_slices > 1 the k dimension absorbs part of the parallelism, so the
  // m x n grid is sized for num_threads / k_slices threads.
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
    const auto ks = split_range(k, k_slices, s);
    const int k0 = ks.first;
    const int k1 = ks.first + ks.second;
    for (int mi = 0; mi < tm; ++mi) {
      const auto ms = split_range(mt, tm, mi);
      const int m0 = ms.first * mr;
      const int m1 = std::min(m, (ms.first + ms.second) * mr);
      for (int ni = 0; ni < tn; ++ni) {
        const auto ns = split_range(nt, tn, ni);
        const int n0 = ns.first * nr;
        const int n1 = std::min(n, (ns.first + ns.second) * nr);
        regions.push_back({m0, m1, n0, n1, k0, k1, s});
      }
    }
  }
  return regions;
}

}  // namespace gemm
}  // namespace cpu_ops
