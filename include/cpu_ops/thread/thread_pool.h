#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace cpu_ops {
namespace thread {

namespace detail {

struct ThreadPoolImpl {
  std::vector<std::thread> workers;
  std::mutex mutex;           // guards fn / n_tasks / generation / stop
  std::condition_variable cv;
  std::function<void(int, int)> fn;
  std::atomic<int> next_task{0};
  std::atomic<int> finished{0};
  int n_tasks = 0;
  unsigned long long generation = 0;
  bool stop = false;
  std::mutex invoke_mutex;  // serializes parallel_for calls

  ThreadPoolImpl() {
    const unsigned hw = std::thread::hardware_concurrency();
    const int n_workers = hw > 1 ? static_cast<int>(hw) - 1 : 0;
    workers.reserve(n_workers);
    for (int i = 0; i < n_workers; ++i) {
      workers.emplace_back([this, i] { worker_loop(i + 1); });
    }
  }

  ~ThreadPoolImpl() {
    {
      std::lock_guard<std::mutex> lk(mutex);
      stop = true;
    }
    cv.notify_all();
    for (std::thread& t : workers) t.join();
  }

  void run_tasks(int tid) {
    for (;;) {
      const int t = next_task.fetch_add(1, std::memory_order_relaxed);
      if (t >= n_tasks) break;
      fn(t, tid);
    }
  }

  void worker_loop(int tid) {
    unsigned long long seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mutex);
        cv.wait(lk, [&] { return stop || generation != seen; });
        if (stop) return;
        seen = generation;
      }
      run_tasks(tid);
      finished.fetch_add(1, std::memory_order_release);
    }
  }

  void parallel_for(int n, const std::function<void(int, int)>& f) {
    if (n <= 0) return;
    if (workers.empty() || n == 1) {
      for (int i = 0; i < n; ++i) f(i, 0);
      return;
    }
    std::lock_guard<std::mutex> invoke_lk(invoke_mutex);
    {
      std::lock_guard<std::mutex> lk(mutex);
      fn = f;
      n_tasks = n;
      next_task.store(0, std::memory_order_relaxed);
      finished.store(0, std::memory_order_relaxed);
      ++generation;
    }
    cv.notify_all();
    run_tasks(0);
    const int total = static_cast<int>(workers.size());
    while (finished.load(std::memory_order_acquire) < total) {
      std::this_thread::yield();
    }
  }
};

}  // namespace detail

// Persistent thread pool. The global instance sizes itself to
// std::thread::hardware_concurrency(); worker threads are created lazily on
// first use and joined at process exit.
//
// parallel_for is not re-entrant and not safe to call concurrently from
// multiple user threads; callers get serialized through an internal mutex.
class ThreadPool {
 public:
  static ThreadPool& global() {
    static ThreadPool pool;
    return pool;
  }

  // Total number of execution slots: worker threads + the calling thread.
  int num_threads() const { return static_cast<int>(impl_->workers.size()) + 1; }

  // Runs fn(task, thread_index) for every task in [0, n_tasks), potentially in
  // parallel, and blocks until all of them have completed. thread_index 0 is
  // the calling thread.
  void parallel_for(int n_tasks, const std::function<void(int, int)>& fn) {
    impl_->parallel_for(n_tasks, fn);
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

 private:
  ThreadPool() : impl_(new detail::ThreadPoolImpl()) {}
  ~ThreadPool() { delete impl_; }

  detail::ThreadPoolImpl* impl_;
};

}  // namespace thread
}  // namespace cpu_ops
