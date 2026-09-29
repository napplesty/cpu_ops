#pragma once

#include <functional>

namespace cpu_ops {
namespace thread {

// Persistent thread pool. The global instance sizes itself to
// std::thread::hardware_concurrency(); worker threads are created lazily on
// first use and joined at process exit.
//
// parallel_for is not re-entrant and not safe to call concurrently from
// multiple user threads; callers get serialized through an internal mutex.
class ThreadPool {
 public:
  static ThreadPool& global();

  // Total number of execution slots: worker threads + the calling thread.
  int num_threads() const;

  // Runs fn(task, thread_index) for every task in [0, n_tasks), potentially in
  // parallel, and blocks until all of them have completed. thread_index 0 is
  // the calling thread.
  void parallel_for(int n_tasks, const std::function<void(int, int)>& fn);

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  ThreadPool();
  ~ThreadPool();

  struct Impl;
  Impl* impl_;
};

}  // namespace thread
}  // namespace cpu_ops
