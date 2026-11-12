// SPDX-License-Identifier: MIT
// Reads the headers of a JPEG 2000 codestream (ISO/IEC 15444-1 Annex A): what the image
// is, how it was coded, and where each tile-part's packets lie. The input is untrusted:
// nothing here reads outside [data, data + size), whatever the lengths in it claim.
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/layout.h"
#include "core/progression.h"

namespace gail {

constexpr int kMaxLevels = 32;                   // A.6.1: 0..32 decomposition levels
constexpr int kMaxBands = 3 * kMaxLevels + 1;

struct TilePartSpan {
    int part = 0;               // TPsot
    size_t begin = 0, end = 0;  // the packets: [begin, end) within the codestream
};

struct StreamHeader {
    // SIZ
    int rsiz = 0;
    int width = 0, height = 0;  // image area on the reference grid
    int components = 0;
    int bitDepth = 0;           // of component 0
    int tiles = 0;
    // COD
    int levels = 0, layers = 0, progression = 0, mct = 0;
    int cblkLog2W = 0, cblkLog2H = 0, cblkStyle = 0;
    bool reversible = false;    // 5/3 wavelet
    bool sop = false, eph = false;
    bool highThroughput = false;  // Part 15 block coder
    // QCD / QCC, per component
    int guardBits = 0;
    int eps[kNumComps][kMaxBands] = {};
    int mu[kNumComps][kMaxBands] = {};
    // Packet order: one volume, or the ranges of resolutions a POC marker gives in turn
    std::vector<ProgressionVolume> volumes;
    std::vector<TilePartSpan> tileParts;

    // Whether this library decodes the stream, and if not, the first reason why.
    bool supported = true;
    std::string unsupported;
};

// Parses the main header and, unless headerOnly, every tile-part header. Returns false
// with a message for a codestream that is damaged or not JPEG 2000 at all. A well-formed
// stream outside what the decoder handles returns true with h.supported == false.
bool parseCodestream(const uint8_t* data, size_t size, bool headerOnly, StreamHeader& h, std::string& error);

// Geometry and step sizes of a supported stream. Throws std::runtime_error when the
// signaled step sizes need more bit-planes than the block coder carries.
FrameLayout layoutFor(const StreamHeader& h);

}  // namespace gail
