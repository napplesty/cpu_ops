#pragma once

// Library-side batching helper for the ops layer. The pool always engages
// all workers on parallel_for, so a thread cap is implemented by handing
// tasks out in p-sized batches (same pattern as the assembled-operator
// examples). Results are independent of the batch split.

#include "cpu_ops/detail/thread_pool.h"

namespace cpu_ops {
namespace ops {
namespace detail {

template <typename F>
void run_batched(thread::ThreadPool& pool, int n, int p, F&& fn) {
  if (n <= 0) return;
  if (p <= 1 || n == 1) {
    for (int t = 0; t < n; ++t) fn(t, 0);
    return;
  }
  for (int base = 0; base < n; base += p) {
    const int b0 = base;
    pool.parallel_for(std::min(p, n - base),
                      [&fn, b0](int i, int tid) { fn(b0 + i, tid); });
  }
}

}  // namespace detail
}  // namespace ops
}  // namespace cpu_ops
