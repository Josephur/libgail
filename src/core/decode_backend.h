// SPDX-License-Identifier: MIT
// Decode backend interface: a slot turns one frame's located code-blocks (the output of
// parsePackets) into X'Y'Z' pixels. Header parsing and Tier-2 run on the host for every
// backend; Tier-1, dequantization, the inverse wavelet and the inverse color transform
// run on the slot's device.
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "core/layout.h"
#include "core/t2_parse.h"

namespace gail {

class ThreadPool;

struct DecodeSettings {
    int reduce = 0;        // resolution levels to leave out: the picture is 1/2^reduce size
    bool highBits = true;  // 12-bit codes in the high bits of each 16-bit word, else the low
};

struct DecodeTimings {
    double t1 = 0, dwt = 0, color = 0, total = 0;  // ms; t1 includes dequantization
};

// Size of the decoded picture for a layout and a reduction.
inline void decodedSize(const FrameLayout& L, int reduce, int& w, int& h) {
    w = L.width;
    h = L.height;
    for (int i = 0; i < reduce; ++i) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

class DecoderSlot {
public:
    virtual ~DecoderSlot() = default;
    // frame: this frame's layout: the geometry the slot was created for, with the step
    // sizes its own headers signal. data, size: the codestream the blocks point into.
    // dst: decodedSize() pixels of three interleaved 16-bit words (X', Y', Z'), rows
    // strideBytes apart; bytes of a row past the picture are not written.
    virtual void decode(const FrameLayout& frame, const uint8_t* data, size_t size, const std::vector<CblkCode>& blocks, void* dst,
                        ptrdiff_t strideBytes, DecodeTimings* tm) = 0;
};

// CPU reference backend. The slot keeps the settings by value and the geometry of L.
std::unique_ptr<DecoderSlot> createCpuDecoderSlot(const FrameLayout& L, const DecodeSettings& s, std::shared_ptr<ThreadPool> pool);

// Whether a backend ("cuda", "vulkan", "cpu") of this build has decode slots.
bool backendDecodes(const std::string& backend);

// Whether two layouts describe the same frame structure (size, levels, bands, blocks),
// whatever their step sizes.
inline bool sameGeometry(const FrameLayout& a, const FrameLayout& b) {
    return a.width == b.width && a.height == b.height && a.levels == b.levels && a.bands.size() == b.bands.size() &&
           a.cblks.size() == b.cblks.size();
}

}  // namespace gail
