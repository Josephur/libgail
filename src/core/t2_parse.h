// SPDX-License-Identifier: MIT
// Tier-2 decoding: reads the packet headers of a codestream (ISO/IEC 15444-1 B.9, B.10)
// and reports, for every code-block, how many bit-planes and coding passes it has and
// where its compressed bytes lie. The input is untrusted: every length is checked against
// the tile-part it sits in.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/codestream_parse.h"
#include "core/layout.h"

namespace gail {

struct CblkCode {
    uint32_t offset = 0;  // of its bytes within the codestream
    uint32_t len = 0;
    uint8_t passes = 0;   // coding passes present (0: nothing coded for this block)
    uint8_t nbps = 0;     // magnitude bit-planes, counted from the most significant coded one
};

// Fills out[i] for every code-block i of the layout that belongs to resolutions
// 0..maxRes; blocks of higher resolutions are left with no passes. Returns false with a
// message when the packets are damaged.
bool parsePackets(const uint8_t* data, const StreamHeader& h, const FrameLayout& L, int maxRes, std::vector<CblkCode>& out,
                  std::string& error);

}  // namespace gail
