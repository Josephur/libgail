// SPDX-License-Identifier: MIT
// Post-compression rate-distortion optimisation (PCRD-opt) with DCI limits:
// whole codestream <= maxTotal bytes and every component <= maxComp bytes.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "core/layout.h"
#include "core/t2.h"
#include "core/tier1_types.h"

namespace gail {

struct RateLimits {
    size_t maxTotal = 0;  // bytes per frame (whole codestream)
    size_t maxComp = 0;   // bytes per colour component (sum of its tile-parts)
};

struct RateResult {
    std::vector<uint8_t> passes;  // passes included per code-block
    double lambda = 0;            // final global slope threshold
    double lambdaComp[3] = {0, 0, 0};
    double distortion = 0;        // estimated remaining weighted MSE * pixels
    CodestreamSizes sizes;
    int iterations = 0;
};

// DCI limits for a frame rate (bits/s for the whole image and per component).
RateLimits dciLimits(double fps, double maxMbps = 230.0, double compMbps = 200.0);

RateResult allocateRate(const FrameLayout& L, const Tier1Frame& t1, const RateLimits& lim);

}  // namespace gail
