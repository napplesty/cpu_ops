#pragma once

#include <vector>

namespace cpu_ops {
namespace gemm {

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

// Splits an m x n output into up to num_threads rectangular regions whose
// boundaries are aligned to mr x nr micro-tiles. The tm x tn grid is chosen to
// fill the thread count first and to minimize panel re-packing traffic
// (tn*m*k + tm*k*n) second, since every region packs its own A rows and B
// columns.
//
// With k_slices > 1 the k range is additionally cut into contiguous slices
// and crossed with the m x n grid, so the result holds grid_regions *
// k_slices regions; each slice's regions must write to their own partial-sum
// slab (indexed by slice) that a later reduction pass combines.
std::vector<GemmRegion> partition_gemm(int m, int n, int k, int mr, int nr, int num_threads,
                                       int k_slices = 1);

}  // namespace gemm
}  // namespace cpu_ops
