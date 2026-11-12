// SPDX-License-Identifier: MIT
// Persistent worker threads shared by everything that splits one frame's work over the
// CPU's cores.
#pragma once
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "core/backend.h"

namespace gail {

// Fixed set of worker threads; run(n, f) hands out the indices 0..n-1 dynamically to
// the workers and the calling thread, and returns when all of them are done. One pool
// serves every slot of the device: concurrent run() calls take turns, so a slot doing
// rate control, Tier-2 or frame I/O overlaps another slot's encode without
// oversubscribing the machine.
class ThreadPool {
public:
    ThreadPool(int threads, bool lowPriority) : threads_(std::max(threads, 1)) {
        for (int t = 1; t < threads_; ++t)
            workers_.emplace_back([this, lowPriority] {
                if (lowPriority) lowerCurrentThreadPriority();
                loop();
            });
    }
    ~ThreadPool() {
        {
            std::lock_guard<std::mutex> l(m_);
            quit_ = true;
            ++gen_;
        }
        wake_.notify_all();
        for (auto& w : workers_) w.join();
    }
    int threads() const { return threads_; }

    void run(int n, const std::function<void(int)>& f) {
        if (n <= 0) return;
        if (workers_.empty() || n == 1) {
            for (int i = 0; i < n; ++i) f(i);
            return;
        }
        std::lock_guard<std::mutex> turn(gate_);
        {
            std::lock_guard<std::mutex> l(m_);
            job_ = &f;
            n_ = n;
            next_.store(0);
            busy_ = (int)workers_.size();
            error_ = nullptr;
            ++gen_;
        }
        wake_.notify_all();
        work();
        std::unique_lock<std::mutex> l(m_);
        done_.wait(l, [&] { return busy_ == 0; });
        job_ = nullptr;
        if (error_) std::rethrow_exception(error_);
    }

private:
    void work() {
        try {
            for (int i; (i = next_.fetch_add(1)) < n_;) (*job_)(i);
        } catch (...) {
            std::lock_guard<std::mutex> l(m_);
            if (!error_) error_ = std::current_exception();
            next_.store(n_);  // stop handing out work
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            {
                std::unique_lock<std::mutex> l(m_);
                wake_.wait(l, [&] { return gen_ != seen; });
                seen = gen_;
                if (quit_) return;
            }
            work();
            std::lock_guard<std::mutex> l(m_);
            if (--busy_ == 0) done_.notify_one();
        }
    }

    int threads_;
    std::vector<std::thread> workers_;
    std::mutex gate_;  // one run() at a time
    std::mutex m_;
    std::condition_variable wake_, done_;
    uint64_t gen_ = 0;
    bool quit_ = false;
    const std::function<void(int)>* job_ = nullptr;
    int n_ = 0;
    std::atomic<int> next_{0};
    int busy_ = 0;
    std::exception_ptr error_;
};

}  // namespace gail
