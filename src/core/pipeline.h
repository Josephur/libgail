// SPDX-License-Identifier: MIT
// Multi-device frame pipeline: a reader thread feeds frames to every encoder slot
// on every device (pull model, so faster GPUs take more frames); each slot runs
// Tier-1 on its device, then rate control + Tier-2 on the CPU; a writer emits the
// codestreams in order.
#pragma once
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/backend.h"
#include "core/codestream_check.h"
#include "core/frame_input.h"
#include "core/rate_control.h"

namespace gail {

struct FrameStats {
    long long index = 0;
    int device = 0;
    size_t bytes = 0;
    double lambda = 0;
    double mseEstimate = 0;   // estimated remaining MSE (12-bit code values^2)
    FrameTimings tier1;
    double rateMs = 0;        // rate control + Tier-2
    bool compliant = true;
    bool recoded = false;     // Tier-1 re-run without early termination (scene got easier)
    std::string checkSummary;
};

struct PipelineOptions {
    int slotsPerDevice = 2;                // frames in flight per device: the most to use
    // Device memory the frames in flight of one device may take, in bytes. 0 = as much as
    // the driver reports free when the pipeline starts, less a headroom for other users of
    // the GPU (memoryHeadroom()). Fewer frames are used when they would not fit, and a
    // frame whose memory the driver still refuses is dropped rather than failing the run.
    uint64_t deviceMemoryLimit = 0;
    // When no device can hold even one frame, encode on the CPU instead of failing.
    bool cpuFallback = false;
    int cpuThreads = 0;                    // for that fallback (0 = usable cores)
    // Run the per-frame worker threads (rate control, Tier-2; the whole encode on the CPU
    // backend) below normal priority, for encodes in the background of a desktop.
    bool lowPriority = false;
    // Notes about frames in flight being reduced (planning, allocation failures, memory
    // pressure during the run).
    std::function<void(const std::string&)> onLog;
    std::string outDir;                    // empty: don't write files
    std::string namePattern = "frame_%06d.j2c";
    double fps = 24.0;
    RateLimits limits;
    bool check = true;                     // run the DCI checker on every frame
    int dumpReferenceEvery = 0;            // also write frame_NNNNNN.xyz (12-bit X'Y'Z' source, raw LE)
                                           // for every Nth frame (0 = never)
    std::function<void(const FrameStats&, const std::vector<uint8_t>&)> onFrame;
    // Called once, from the failing thread, as soon as an error stops the pipeline, so a
    // source that blocks in read() (e.g. one fed by an app) can be woken up.
    std::function<void()> onError;
    // Called once, from the calling thread, when the frames in flight are set up (or the
    // CPU fallback taken) and the first frame is about to be read. An embedding that runs
    // the pipeline on its own thread waits for this, so that a device with no room for a
    // frame, or one that fails to set up, is reported by its create call and not by the
    // first frame. Not called when set-up fails: the error leaves runPipeline() instead.
    std::function<void()> onReady;
    // Set by another thread to stop early: frames not yet written are dropped, and only
    // the device work already under way is waited for.
    const std::atomic<bool>* cancel = nullptr;
};

struct PipelineSummary {
    long long frames = 0;
    double seconds = 0;
    size_t bytes = 0;
    size_t maxFrameBytes = 0;
    long long nonCompliant = 0;
    std::vector<long long> framesPerDevice;
    std::vector<int> slotsPerDevice;  // frames in flight each device started with
    std::vector<int> slotsRetired;    // ... and gave up to memory pressure during the run
};

// Device memory left to other users of a GPU with `total` bytes: 512 MiB or 6 %, whichever
// is more. The pipeline plans frames in flight below it and gives frames up when the free
// memory falls under half of it.
uint64_t memoryHeadroom(uint64_t total);

// How many frames in flight to start on a device: the `requested` number, or fewer when
// `perSlot` bytes each would not fit into `available` bytes (what the driver says this
// process may still allocate) less the headroom, or into `limit` (bytes; 0 = no limit).
// Unknown sizes (0) do not limit. At least 1: that frame is still attempted and may fail
// softly. An estimate, so a tenth is added to each slot for allocator rounding.
int fitSlots(uint64_t available, uint64_t total, uint64_t perSlot, int requested, uint64_t limit);

PipelineSummary runPipeline(FrameSource& src, std::vector<std::unique_ptr<Device>>& devices, const FrameLayout& L,
                            const EncodeSettings& s, const PipelineOptions& o);

}  // namespace gail
