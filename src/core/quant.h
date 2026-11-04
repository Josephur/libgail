// SPDX-License-Identifier: MIT
// Dead-zone scalar quantisation of wavelet coefficients into the code-block-major
// layout consumed by Tier-1 (1024 sign-magnitude words per code-block).
#pragma once
#include <cstdint>

#include "core/layout.h"

namespace gail {

// planes[c] are Mallat-layout DWT outputs (width x height floats).
// out receives layout.numCblks() * 1024 words.
void quantizeFrame(const FrameLayout& layout, const float* const planes[3], uint32_t* out);
// One code-block (index into layout.cblks) into its 1024 words at out; the blocks are
// independent, so callers may spread them over threads.
void quantizeCodeBlock(const FrameLayout& layout, const float* const planes[3], int cblk, uint32_t* out);

// Quantise one coefficient: sign-magnitude with kFracBits fractional bits, clamped
// so the integer part fits the band's Mb magnitude bitplanes.
inline uint32_t quantizeCoeff(float v, float invDelta, uint32_t maxMag) {
    float a = v < 0 ? -v : v;
    float q = a * invDelta * (float)(1 << kFracBits);
    uint32_t m = q >= (float)maxMag ? maxMag : (uint32_t)q;
    return m | (v < 0 && m >= (1u << kFracBits) ? 0x80000000u : 0u);
}

}  // namespace gail
