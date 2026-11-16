// SPDX-License-Identifier: MIT
// Tier-2: packet headers (tag trees), CPRL progression with the DCI 4K POC,
// tile-parts and the DCI main header (SIZ/COD/QCD/POC/COM/TLM).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "core/layout.h"
#include "core/tier1_types.h"

namespace gail {

struct CodestreamSizes {
    size_t total = 0;
    size_t comp[3] = {0, 0, 0};       // bytes per component (its tile-parts incl. SOT/SOD)
    int numTileParts = 0;
    size_t tilePart[6] = {};
};

// The label in every codestream's COM marker, e.g. "GAIL 1.1.0+3.gabc1234": it identifies
// frames GAIL encoded, and the build that encoded them.
const char* codestreamLabel();

// What follows the label, after a space, to name the kind of device that encoded the
// frame: "vulk" for Vulkan, "cuda" and the toolkit's major.minor for CUDA ("cuda13.4").
// The CPU has none, so its comment is the label alone.
constexpr size_t kMaxDeviceTagBytes = 9;  // "cuda99.99"
// The COM text of a frame encoded by a device with this tag ("" for none).
std::string codestreamComment(const std::string& tag);

// Assemble (out != nullptr) or just measure (out == nullptr) the codestream.
// passes[i] is the number of coding passes of code-block i to include. tag names the
// device that encoded the frame (codestreamComment()); at most kMaxDeviceTagBytes.
//
// The sizes count the COM marker at its longest: the label, a space and
// kMaxDeviceTagBytes. They are what the rate control and the frame budget work with, so
// the coded data does not depend on the tag, and a codestream written with a shorter tag
// is that many bytes smaller than sizes->total.
void buildCodestream(const FrameLayout& L, const Tier1Frame& t1, const uint8_t* passes, std::vector<uint8_t>* out,
                     CodestreamSizes* sizes, const std::string& tag = std::string());

// Size of the codestream with no coding pass included: the main header, the tile-part
// headers and one byte per empty packet. The smallest frame this layout can produce, so
// no frame budget below it can be met (the rate control cannot drop anything else).
size_t emptyCodestreamBytes(const FrameLayout& L);

}  // namespace gail
