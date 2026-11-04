// SPDX-License-Identifier: MIT
// Tier-1 output for one frame, produced by any backend (CPU, Vulkan, CUDA) and
// consumed by rate control and Tier-2.
#pragma once
#include <cstdint>
#include <vector>

#include "core/layout.h"

namespace gail {

struct CblkInfo {
    uint32_t nbps;        // significant bitplanes (0: nothing coded)
    uint32_t numPasses;   // coding passes available
    uint32_t dataOffset;  // byte offset of the codeword in Tier1Frame::data
    uint32_t dataLen;     // codeword length (all passes)
    uint32_t passBase;    // index of this block's first entry in passLen / passDist
};

struct Tier1Frame {
    std::vector<CblkInfo> info;      // one per code-block
    std::vector<uint32_t> passLen;   // truncation length after each pass (compact, see passBase)
    std::vector<int32_t> passDist;   // distortion reduction of each pass (x4096, plane units)
    std::vector<uint8_t> data;

    uint32_t len(int cb, int k) const { return passLen[info[cb].passBase + k]; }
    int32_t dist(int cb, int k) const { return passDist[info[cb].passBase + k]; }
};

}  // namespace gail
