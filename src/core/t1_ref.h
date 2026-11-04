// SPDX-License-Identifier: MIT
// CPU reference EBCOT Tier-1 encoder (straightforward sequential implementation of
// ISO/IEC 15444-1 Annex D, no mode switches). Used as the golden model for the GPU
// kernels and as the CPU backend.
#pragma once
#include <cstdint>
#include <vector>

#include "core/layout.h"

namespace gail {

struct T1Result {
    int nbps = 0;        // significant magnitude bitplanes (0 = block not coded)
    int numPasses = 0;
    std::vector<uint8_t> data;              // full codeword
    uint32_t passLen[kMaxPasses] = {};      // truncation length after each pass
    int32_t passDist[kMaxPasses] = {};      // distortion reduction of each pass (x4096, plane units)
};

struct T1Trace {
    std::vector<uint8_t> symbols;           // (cx << 1) | d
    std::vector<uint32_t> passEnd;          // symbol count at the end of each pass
};

// Early termination, the same rule as the GPU kernels (t1_cm.comp / cuda_backend.cu):
// after each bitplane p > 0 the block stops once (a) the plane's estimated
// rate-distortion slope is below slopeThresh and (b) even removing all remaining error at
// the cost of one more such plane could not beat it. The plane rate is an empirical-entropy
// estimate of its symbols, so sparse planes do not stop a block too early. distScale is the
// band's delta^2 * weight (distortion units of the rate control); slopeThresh 0 codes
// every pass.
struct T1EarlyStop {
    float slopeThresh = 0.f;
    float distScale = 1.f;
};

// coeffs: 32x32 row-major sign-magnitude words (bit 31 sign, magnitude with
// kFracBits fractional bits). maxPasses limits the coded passes.
void t1EncodeRef(const uint32_t* coeffs, int w, int h, int orient, T1Result& out, T1Trace* trace = nullptr,
                 int maxPasses = kMaxPasses, const T1EarlyStop* stop = nullptr);

// Distortion of pass k in absolute units (multiply by band step^2 * weight).
inline double passPlaneScale(int nbps, int k) {
    const int plane = k == 0 ? nbps - 1 : nbps - 2 - (k - 1) / 3;
    return (double)(1ull << (2 * plane)) / 4096.0;
}

}  // namespace gail
