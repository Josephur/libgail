// SPDX-License-Identifier: MIT
// Compute backend interface. A backend turns an RGB48 frame into Tier-1 output
// (colour conversion, DWT, quantisation, EBCOT Tier-1). Rate control and Tier-2
// run on the CPU and are shared by all backends.
#pragma once
#include <cstdint>
#include <cstdio>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/color.h"
#include "core/layout.h"
#include "core/tier1_types.h"

namespace gail {

struct EncodeSettings {
    Profile profile = Profile::Cinema2K;
    int width = 2048, height = 1080;
    double baseStep = 1.0;
    ColorParams color;
    FrameFormat input;  // layout of the frames passed to encode() (width/height = container)
    // Tier-1 early termination: a code-block stops after the first pass whose estimated
    // RD slope is below slopeFactor * (lambda of the most recent frame). 0 = code every pass.
    double slopeFactor = 0.0;
    double bitsPerSymbol = 0.5;  // rate estimate used for that slope
};

struct FrameTimings {
    // milliseconds; GPU backends report device time per stage
    double upload = 0, color = 0, dwt = 0, quant = 0, t1 = 0, t1cm = 0, t1mq = 0, download = 0, gpu = 0, total = 0;
    double symbols = 0, codedBytes = 0;  // Tier-1 work: MQ symbols coded, codeword bytes produced
};

// Optional debug taps (filled when non-null).
struct FrameDebug {
    bool full = true;                // false: only the X'Y'Z' reference image
    std::vector<uint16_t> xyz12;     // interleaved integer X'Y'Z'
    std::vector<uint32_t> coeffs;    // quantised code-block words
    std::vector<float> wavelet;      // 3 Mallat-layout planes after the DWT (GPU backends)
};

class EncoderSlot {
public:
    virtual ~EncoderSlot() = default;
    // Optional pinned input buffer (input.bytes() bytes). Frames decoded straight into it
    // skip one copy; encode() detects the pointer.
    virtual void* inputBuffer() { return nullptr; }
    // Rate-distortion slope below which Tier-1 may stop coding finer passes (0 = off).
    virtual void setSlopeThreshold(double) {}
    virtual void encode(const void* frame, Tier1Frame& out, FrameTimings* timings = nullptr,
                        FrameDebug* debug = nullptr) = 0;
};

// Device memory as the driver sees it right now. `available` is what this process may
// still allocate (free memory, or the driver's budget less our own use), so it shrinks
// when other processes take memory. Both 0 when the backend cannot tell (CPU).
struct DeviceMemory {
    uint64_t total = 0;
    uint64_t available = 0;
    bool budgeted = false;  // available is meaningful
};

// createSlot() throws this when the device (or pinned host) memory for a slot cannot be
// allocated; the device stays usable and the slots created so far keep working.
struct DeviceOutOfMemory : std::runtime_error {
    using std::runtime_error::runtime_error;
};

class Device {
public:
    virtual ~Device() = default;
    virtual std::string name() const = 0;
    virtual std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& layout, const EncodeSettings& s) = 0;
    // Device memory one slot for this layout allocates (0 = none / unknown), and what is
    // left now; the pipeline sizes the number of frames in flight from these.
    virtual uint64_t slotBytes(const FrameLayout&, const EncodeSettings&) const { return 0; }
    virtual DeviceMemory memory() const { return {}; }
};

// What a device is, without setting it up (for device pickers and selection by ID).
struct DeviceInfo {
    int index = 0;           // as passed to create*Devices(select)
    std::string name;        // "NVIDIA GeForce RTX 3090"
    std::string uuid;        // "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" (nvidia-smi style); "" if unknown
    std::string pciBusId;    // "0000:01:00.0"; "" if unknown
    uint64_t memoryBytes = 0;
};
// 16 UUID bytes as nvidia-smi prints them.
std::string formatGpuUuid(const uint8_t bytes[16]);

// threads: worker threads of the CPU backend (0 = usableCpuCount()). lowPriority: run them
// below normal priority so an encode in the background leaves the desktop responsive.
std::unique_ptr<Device> createCpuDevice(int threads, bool lowPriority = false);
DeviceInfo cpuDeviceInfo(int threads);
// Logical CPUs this process may run on: the affinity mask (taskset, containers, job
// objects) rather than the machine's core count.
int usableCpuCount();
// Lowers the calling thread's scheduling priority (nice +10 / below normal); best effort.
void lowerCurrentThreadPriority();
// Frames in flight on the CPU device: its slots share one thread pool, so a second slot
// only overlaps rate control, Tier-2 and frame I/O with the other slot's encode.
constexpr int kCpuSlots = 2;
// What a GPU backend has to say: a device it could not set up or a frame it had to redo
// on the CPU (Warning), and what each device is (Info). Empty: nothing is said. The
// library hands the app's log callback in here; the tools print to stderr.
enum class LogLevel { Warning = 1, Info = 2 };
using Logger = std::function<void(LogLevel, const std::string&)>;
inline Logger stderrLogger(bool verbose) {
    return [verbose](LogLevel level, const std::string& m) {
        if (verbose || level == LogLevel::Warning) std::fprintf(stderr, "%s\n", m.c_str());
    };
}
#ifdef GAIL_HAVE_VULKAN
// Discrete/integrated GPUs exposed through Vulkan (index = order of enumeration).
std::vector<std::unique_ptr<Device>> createVulkanDevices(const std::vector<int>& select, const Logger& log);
std::vector<DeviceInfo> listVulkanDevices();
#endif
#ifdef GAIL_HAVE_CUDA
std::vector<std::unique_ptr<Device>> createCudaDevices(const std::vector<int>& select, const Logger& log);
std::vector<DeviceInfo> listCudaDevices();
#endif

}  // namespace gail
