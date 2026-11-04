// SPDX-License-Identifier: MIT
// CPU backend: reference implementation of the whole Tier-1 pipeline, parallelised
// over rows / column strips / code-blocks on a persistent thread pool.
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX  // windows.h's min/max macros would break std::min / std::max below
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <sched.h>
#include <sys/resource.h>
#include <unistd.h>
#ifdef __linux__
#include <sys/syscall.h>
#endif
#ifdef __APPLE__
#include <pthread/qos.h>
#endif
#endif

#include "core/backend.h"
#include "core/dwt_cpu.h"
#include "core/quant.h"
#include "core/t1_ref.h"

namespace gail {

namespace {

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

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

class CpuSlot : public EncoderSlot {
public:
    CpuSlot(const FrameLayout& L, const EncodeSettings& s, std::shared_ptr<ThreadPool> pool)
        : L_(L), s_(s), pool_(std::move(pool)), color_(s.color) {
        for (auto& p : planes_) p.resize((size_t)L.width * L.height);
        coeffs_.resize((size_t)L.numCblks() * kCblkArea);
        par_ = [this](int n, const std::function<void(int)>& f) { pool_->run(n, f); };
        // distortion units of the rate control (as the GPU band tables)
        for (const Subband& b : L.bands) bandDistScale_.push_back((float)((double)b.delta * b.delta * b.weight));
    }

    void setSlopeThreshold(double t) override { slopeThresh_ = t; }

    void encode(const void* frame, Tier1Frame& out, FrameTimings* tm, FrameDebug* dbg) override {
        auto t0 = std::chrono::steady_clock::now();
        const uint16_t* rgb = (const uint16_t*)frame;
        const bool planar = s_.input.kind == FrameFormat::Kind::Planar;
        float* pl[3] = {planes_[0].data(), planes_[1].data(), planes_[2].data()};
        const int W = L_.width, H = L_.height;
        if (dbg) dbg->xyz12.resize((size_t)W * H * 3);
        pool_->run(H, [&](int y) {
            float* rows[3] = {pl[0] + (size_t)y * W, pl[1] + (size_t)y * W, pl[2] + (size_t)y * W};
            const uint16_t* src = rgb + (size_t)y * W * 3;
            if (planar) {
                // one row of R'G'B' at a time: the frame never exists as RGB48
                thread_local std::vector<uint16_t> rowBuf;
                if (rowBuf.size() < (size_t)W * 3) rowBuf.resize((size_t)W * 3);
                planarToRgb48Row(s_.input, frame, rowBuf.data(), y);
                src = rowBuf.data();
            }
            color_.convertFrame(src, W, 1, rows, dbg ? dbg->xyz12.data() + (size_t)y * W * 3 : nullptr);
        });
        auto t1 = std::chrono::steady_clock::now();
        fdwt97_2d(pl, 3, W, H, L_.levels, &par_);
        auto t2 = std::chrono::steady_clock::now();
        const float* cpl[3] = {pl[0], pl[1], pl[2]};
        const int n = L_.numCblks();
        out.info.assign(n, CblkInfo{});
        res_.resize(n);
        // Each code-block is quantised and coded by the same thread, straight from cache.
        const float thr = (float)slopeThresh_;
        pool_->run(n, [&](int i) {
            const CodeBlock& cb = L_.cblks[i];
            uint32_t* q = coeffs_.data() + (size_t)i * kCblkArea;
            quantizeCodeBlock(L_, cpl, i, q);
            const T1EarlyStop stop{thr, bandDistScale_[cb.band]};
            t1EncodeRef(q, cb.w, cb.h, L_.bands[cb.band].orient, res_[i], nullptr, kMaxPasses, &stop);
        });
        if (dbg) dbg->coeffs = coeffs_;
        const auto t3 = t2;  // quantisation is part of the Tier-1 stage here
        size_t total = 0, passes = 0;
        for (auto& r : res_) {
            total += r.data.size();
            passes += r.numPasses;
        }
        out.data.resize(total);
        out.passLen.resize(passes);
        out.passDist.resize(passes);
        size_t off = 0, pb = 0;
        for (int i = 0; i < n; ++i) {
            CblkInfo& ci = out.info[i];
            ci.nbps = res_[i].nbps;
            ci.numPasses = res_[i].numPasses;
            ci.dataOffset = (uint32_t)off;
            ci.dataLen = (uint32_t)res_[i].data.size();
            ci.passBase = (uint32_t)pb;
            if (!res_[i].data.empty())  // an empty block has no buffer (memcpy from NULL is undefined)
                std::memcpy(out.data.data() + off, res_[i].data.data(), res_[i].data.size());
            off += res_[i].data.size();
            for (int k = 0; k < res_[i].numPasses; ++k) {
                out.passLen[pb + k] = res_[i].passLen[k];
                out.passDist[pb + k] = res_[i].passDist[k];
            }
            pb += res_[i].numPasses;
        }
        if (tm) {
            tm->codedBytes = (double)total;
            tm->symbols = (double)passes;  // passes coded (the CPU coder does not count symbols)
            tm->color = std::chrono::duration<double, std::milli>(t1 - t0).count();
            tm->dwt = std::chrono::duration<double, std::milli>(t2 - t1).count();
            tm->quant = std::chrono::duration<double, std::milli>(t3 - t2).count();
            tm->t1 = msSince(t3);
            tm->total = msSince(t0);
        }
    }

private:
    const FrameLayout& L_;
    EncodeSettings s_;
    std::shared_ptr<ThreadPool> pool_;
    ParallelFor par_;
    ColorConverter color_;
    std::vector<float> planes_[3];
    std::vector<uint32_t> coeffs_;
    std::vector<T1Result> res_;
    std::vector<float> bandDistScale_;
    double slopeThresh_ = 0.0;
};

class CpuDevice : public Device {
public:
    CpuDevice(int threads, bool lowPriority)
        : threads_(threads > 0 ? threads : usableCpuCount()), lowPriority_(lowPriority) {}
    std::string name() const override { return "CPU (" + std::to_string(threads_) + " threads)"; }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override {
        if (!pool_) pool_ = std::make_shared<ThreadPool>(threads_, lowPriority_);
        return std::make_unique<CpuSlot>(L, s, pool_);
    }

private:
    int threads_;
    bool lowPriority_;
    std::shared_ptr<ThreadPool> pool_;  // created with the first slot, shared by all of them
};

}  // namespace

std::unique_ptr<Device> createCpuDevice(int threads, bool lowPriority) {
    return std::make_unique<CpuDevice>(threads, lowPriority);
}

DeviceInfo cpuDeviceInfo(int threads) {
    DeviceInfo d;
    d.name = "CPU (" + std::to_string(threads > 0 ? threads : usableCpuCount()) + " threads)";
    return d;
}

int usableCpuCount() {
    int n = 0;
#ifdef _WIN32
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys))
        for (; proc; proc &= proc - 1) ++n;
#elif defined(__linux__)
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) n = CPU_COUNT(&set);
#endif
    if (n <= 0) n = (int)std::thread::hardware_concurrency();
    return std::max(n, 1);
}

void lowerCurrentThreadPriority() {
#ifdef _WIN32
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
#elif defined(__linux__)
    // Linux nice values are per thread: this touches only the caller.
    setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), 10);
#elif defined(__APPLE__)
    // Utility QoS: below the default (user-initiated) class, above background.
    pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
#endif
}

std::string formatGpuUuid(const uint8_t b[16]) {
    char s[48];
    std::snprintf(s, sizeof s, "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", b[0], b[1],
                  b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]);
    return s;
}

}  // namespace gail
