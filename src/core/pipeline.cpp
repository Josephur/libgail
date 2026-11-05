// SPDX-License-Identifier: MIT
#include "core/pipeline.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "core/t2.h"

namespace gail {

uint64_t memoryHeadroom(uint64_t total) { return std::max<uint64_t>(512ull << 20, total * 6 / 100); }

int fitSlots(uint64_t available, uint64_t total, uint64_t perSlot, int requested, uint64_t limit) {
    requested = std::max(requested, 1);
    if (!perSlot) return requested;
    uint64_t budget = limit;
    if (available) {
        const uint64_t head = memoryHeadroom(total);
        const uint64_t usable = available > head ? available - head : 0;
        budget = limit ? std::min(limit, usable) : usable;
    } else if (!limit) {
        return requested;
    }
    const uint64_t per = perSlot + perSlot / 10;
    return (int)std::max<uint64_t>(1, std::min<uint64_t>(budget / per, (uint64_t)requested));
}

namespace {

std::string mib(uint64_t bytes) { return std::to_string((bytes + (1u << 19)) >> 20) + " MiB"; }

}  // namespace

PipelineSummary runPipeline(FrameSource& src, std::vector<std::unique_ptr<Device>>& devices, const FrameLayout& L,
                            const EncodeSettings& s, const PipelineOptions& o) {
    const size_t frameBytes = src.format().bytes();
    if (src.width() != L.width || src.height() != L.height) throw std::runtime_error("source size mismatch");
    if (s.input.kind != src.format().kind || s.input.bytes() != frameBytes)
        throw std::runtime_error("encoder input format does not match the source");
    // A budget the headers alone exceed can never be met: refuse it here rather than
    // producing frames that fail the DCI check (the rate control cannot drop headers).
    const size_t headerBytes = emptyCodestreamBytes(L);
    if (o.limits.maxTotal < headerBytes)
        throw std::runtime_error("the frame budget of " + std::to_string(o.limits.maxTotal) +
                                 " bytes is below the " + std::to_string(headerBytes) +
                                 " bytes the codestream headers take: raise the bitrate or lower the frame rate");

    auto note = [&](const std::string& m) {
        if (o.onLog) o.onLog(m);
    };

    // Create slots up front (device memory allocation happens here): as many as requested
    // and as fit the device memory that is free now, leaving headroom for other users of
    // the GPU. A frame whose memory the driver refuses anyway is dropped, not fatal.
    struct SlotRef {
        int device;
        std::unique_ptr<EncoderSlot> slot;
    };
    std::vector<SlotRef> slots;
    std::vector<int> made;
    auto addSlots = [&](int d, int want) {
        Device& dev = *devices[d];
        const uint64_t per = dev.slotBytes(L, s);
        const DeviceMemory mem = dev.memory();
        const int n = fitSlots(mem.budgeted ? mem.available : 0, mem.total, per, want, o.deviceMemoryLimit);
        if (n < want)
            note(dev.name() + ": " + std::to_string(n) + " of " + std::to_string(want) + " frames in flight fit (" +
                 (mem.budgeted ? mib(mem.available) + " free" : "limit " + mib(o.deviceMemoryLimit)) + ", about " +
                 mib(per) + " per frame)");
        int got = 0;
        for (int k = 0; k < n; ++k) {
            try {
                slots.push_back({d, dev.createSlot(L, s)});
                ++got;
            } catch (const DeviceOutOfMemory& e) {
                note(dev.name() + ": " + e.what() +
                     (got ? "; running with " + std::to_string(got) + " frame(s) in flight" : "; device skipped"));
                break;
            }
        }
        made.push_back(got);
    };
    for (int d = 0; d < (int)devices.size(); ++d) addSlots(d, o.slotsPerDevice);
    if (slots.empty()) {
        if (!o.cpuFallback)
            throw DeviceOutOfMemory("no device has memory for a frame: close other GPU applications, or lower the "
                                    "frames in flight");
        note("no device has memory for a frame: encoding on the CPU");
        devices.push_back(createCpuDevice(o.cpuThreads, o.lowPriority));
        addSlots((int)devices.size() - 1, kCpuSlots);
    }

    PipelineSummary sum;
    sum.framesPerDevice.assign(devices.size(), 0);
    sum.slotsPerDevice = made;
    sum.slotsRetired.assign(devices.size(), 0);
    if (o.onReady) o.onReady();  // every device set-up error has been thrown by now

    // Memory pressure during the run: twice a second one of a device's slots asks the driver
    // how much memory is left; when another process has taken so much that less than half
    // the headroom remains, the slot that noticed retires (frees its buffers) so that process
    // is not the one to run out. One frame in flight per device always stays.
    struct DevState {
        std::atomic<int> slots{0};
        std::atomic<int> retired{0};
        std::atomic<int64_t> nextCheck{0};
        std::mutex m;
    };
    std::deque<DevState> dstate(devices.size());
    for (size_t i = 0; i < made.size(); ++i) dstate[i].slots = made[i];
    const auto t0 = std::chrono::steady_clock::now();
    auto nowMs = [&] { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count(); };
    auto giveUpSlot = [&](int d) -> bool {
        DevState& ds = dstate[d];
        if (ds.slots.load() <= 1) return false;
        const int64_t now = nowMs();
        int64_t due = ds.nextCheck.load();
        if (now < due || !ds.nextCheck.compare_exchange_strong(due, now + 500)) return false;
        const DeviceMemory mem = devices[d]->memory();
        if (!mem.budgeted || mem.available >= memoryHeadroom(mem.total) / 2) return false;
        std::lock_guard<std::mutex> l(ds.m);
        const int cur = ds.slots.load();
        if (cur <= 1) return false;
        ds.slots.store(cur - 1);
        ds.nextCheck.store(now + 1000);  // let the freed memory show before judging again
        ds.retired++;
        note(devices[d]->name() + ": device memory is running low (" + mib(mem.available) + " free): giving up one of " +
             std::to_string(cur) + " frames in flight");
        return true;
    };
    std::mutex srcMutex, outMutex;
    std::condition_variable outCv;
    std::map<long long, std::pair<FrameStats, std::vector<uint8_t>>> pending;
    long long nextIn = 0, nextOut = 0, active = (long long)slots.size();
    bool eof = false;
    std::exception_ptr error;
    std::atomic<double> lastLambda{0.0};
    auto cancelled = [&] { return o.cancel && o.cancel->load(); };

    auto fail = [&](std::exception_ptr e) {
        bool first = false;
        {
            std::lock_guard<std::mutex> l(outMutex);
            if (!error) {
                error = e;
                first = true;
            }
        }
        // Wake the source first: a worker may be blocked in src.read() holding srcMutex
        if (first && o.onError) o.onError();
        {
            std::lock_guard<std::mutex> l(srcMutex);
            eof = true;
        }
        outCv.notify_all();
    };

    // Every slot pulls the next frame itself, decoding straight into its (pinned) input
    // buffer when the backend offers one; faster devices naturally take more frames.
    std::vector<std::thread> workers;
    for (auto& sr : slots) {
        workers.emplace_back([&, ref = &sr] {
            if (o.lowPriority) lowerCurrentThreadPriority();
            Tier1Frame t1;
            std::vector<uint8_t> cs;
            std::vector<uint8_t> own;
            void* in = ref->slot->inputBuffer();
            if (!in) {
                own.resize(frameBytes);
                in = own.data();
            }
            for (;;) {
                FrameStats st;
                st.device = ref->device;
                try {
                    {
                        std::lock_guard<std::mutex> l(srcMutex);
                        if (eof || cancelled()) break;
                        if (!src.read(in)) {
                            eof = true;
                            break;
                        }
                        st.index = nextIn++;
                    }
                    const double thr = s.slopeFactor > 0 ? lastLambda.load() * s.slopeFactor : 0.0;
                    if (s.slopeFactor > 0) ref->slot->setSlopeThreshold(thr);
                    FrameDebug dbg;
                    dbg.full = false;
                    const bool dump = o.dumpReferenceEvery > 0 && st.index % o.dumpReferenceEvery == 0;
                    ref->slot->encode(in, t1, &st.tier1, dump ? &dbg : nullptr);
                    if (cancelled()) break;
                    if (dump && !o.outDir.empty()) {
                        char name[256];
                        snprintf(name, sizeof name, "%s/frame_%06lld.xyz", o.outDir.c_str(), st.index);
                        if (FILE* f = fopen(name, "wb")) {
                            fwrite(dbg.xyz12.data(), 2, dbg.xyz12.size(), f);
                            fclose(f);
                        }
                    }
                    const auto r0 = std::chrono::steady_clock::now();
                    RateResult rr = allocateRate(L, t1, o.limits);
                    if (thr > 0 && std::min({rr.lambdaComp[0], rr.lambdaComp[1], rr.lambdaComp[2]}) < thr) {
                        // The frame is easier than the previous one: passes the rate control would
                        // now keep may have been skipped. Re-code it without early termination.
                        ref->slot->setSlopeThreshold(0.0);
                        ref->slot->encode(in, t1, nullptr, nullptr);
                        rr = allocateRate(L, t1, o.limits);
                        st.recoded = true;
                    }
                    buildCodestream(L, t1, rr.passes.data(), &cs, nullptr);
                    st.rateMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - r0).count();
                    st.bytes = cs.size();
                    st.lambda = std::min({rr.lambdaComp[0], rr.lambdaComp[1], rr.lambdaComp[2]});
                    lastLambda.store(st.lambda);
                    st.mseEstimate = rr.distortion / ((double)L.width * L.height * 3);
                    if (o.check) {
                        CheckReport rep = checkDciCodestream(cs.data(), cs.size(), o.fps);
                        st.compliant = rep.ok;
                        if (!rep.ok) st.checkSummary = formatReport(rep, false);
                    }
                } catch (...) {
                    fail(std::current_exception());
                    break;
                }
                {
                    std::lock_guard<std::mutex> l(outMutex);
                    pending.emplace(st.index, std::make_pair(st, cs));
                }
                outCv.notify_all();
                if (giveUpSlot(ref->device)) {
                    ref->slot.reset();  // frees the device memory; this thread is done
                    break;
                }
            }
            {
                std::lock_guard<std::mutex> l(outMutex);
                --active;
            }
            outCv.notify_all();
        });
    }

    // Writer (this thread)
    for (;;) {
        std::pair<FrameStats, std::vector<uint8_t>> item;
        {
            std::unique_lock<std::mutex> l(outMutex);
            outCv.wait(l, [&] { return pending.count(nextOut) || active == 0 || error || cancelled(); });
            if (error || cancelled()) break;
            if (!pending.count(nextOut)) break;  // all workers finished
            item = std::move(pending[nextOut]);
            pending.erase(nextOut);
        }
        const FrameStats& st = item.first;
        try {
            if (!o.outDir.empty()) {
                char name[256];
                snprintf(name, sizeof name, o.namePattern.c_str(), (int)st.index);
                const std::string path = o.outDir + "/" + name;
                FILE* f = fopen(path.c_str(), "wb");
                if (!f) throw std::runtime_error("cannot write " + path);
                const bool ok = fwrite(item.second.data(), 1, item.second.size(), f) == item.second.size();
                if (fclose(f) != 0 || !ok) throw std::runtime_error("cannot write " + path);
            }
            sum.frames++;
            sum.bytes += st.bytes;
            sum.maxFrameBytes = std::max(sum.maxFrameBytes, st.bytes);
            if (!st.compliant) sum.nonCompliant++;
            sum.framesPerDevice[st.device]++;
            if (o.onFrame) o.onFrame(st, item.second);
        } catch (...) {
            // workers must be stopped and joined before the error leaves this function
            fail(std::current_exception());
            break;
        }
        ++nextOut;
    }
    for (auto& w : workers) w.join();
    for (size_t d = 0; d < dstate.size(); ++d) sum.slotsRetired[d] = dstate[d].retired.load();
    if (error) std::rethrow_exception(error);
    sum.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return sum;
}

}  // namespace gail
