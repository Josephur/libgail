// SPDX-License-Identifier: MIT
// Tier-2: packet headers (tag trees), CPRL progression with the DCI 4K POC,
// tile-parts and the DCI main header (SIZ/COD/QCD/POC/COM/TLM).
#pragma once
#include <cstddef>
#include <cstdint>
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

// Assemble (out != nullptr) or just measure (out == nullptr) the codestream.
// passes[i] is the number of coding passes of code-block i to include.
void buildCodestream(const FrameLayout& L, const Tier1Frame& t1, const uint8_t* passes, std::vector<uint8_t>* out,
                     CodestreamSizes* sizes);

}  // namespace gail
