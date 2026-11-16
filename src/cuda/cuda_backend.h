// SPDX-License-Identifier: MIT
// What the CUDA backend offers the code around it: its devices, whether its kernels can
// run on one, and the Tier-1 output of a slot without a copy. cuda/cuda_plugin.cpp puts
// this behind the plugin interface.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/backend.h"

namespace gail {

struct CudaDeviceDesc {
    DeviceInfo info;               // index: as the CUDA runtime numbers the device
    int smMajor = 0, smMinor = 0;  // compute capability
};

// The devices this build's CUDA runtime sees. Throws when the runtime cannot start: no
// NVIDIA driver, or one older than the toolkit needs.
std::vector<CudaDeviceDesc> cudaEnumerate();

// Whether this build's kernels can run on a device: loads them with the device current,
// as a launch would, which also covers code the driver compiles from PTX. Returns ""
// when they can, else the reason.
std::string cudaKernelsUnusable(int device, bool* noCode = nullptr);

std::unique_ptr<Device> createCudaDevice(int device, const Logger& log);

// A slot's Tier-1 output where the slot left it (pinned memory, or its own frame after a
// CPU fallback): valid until the slot encodes again.
struct Tier1View {
    const CblkInfo* info = nullptr;
    const uint32_t* passLen = nullptr;
    const int32_t* passDist = nullptr;
    const uint8_t* data = nullptr;
    size_t numInfo = 0, numPasses = 0, numData = 0;
};

// What createSlot() of a CUDA device returns.
class CudaEncoderSlot : public EncoderSlot {
public:
    virtual Tier1View encodeView(const void* frame, FrameTimings* timings, FrameDebug* debug) = 0;
    void encode(const void* frame, Tier1Frame& out, FrameTimings* timings, FrameDebug* debug) final {
        const Tier1View v = encodeView(frame, timings, debug);
        out.info.assign(v.info, v.info + v.numInfo);
        out.passLen.assign(v.passLen, v.passLen + v.numPasses);
        out.passDist.assign(v.passDist, v.passDist + v.numPasses);
        out.data.assign(v.data, v.data + v.numData);
    }
};

}  // namespace gail
