// SPDX-License-Identifier: MIT
#include "core/quant.h"

#include <cstring>

namespace gail {

void quantizeCodeBlock(const FrameLayout& L, const float* const planes[3], int idx, uint32_t* out) {
    const CodeBlock& cb = L.cblks[idx];
    const Subband& b = L.bands[cb.band];
    const float* plane = planes[b.comp];
    const float inv = 1.0f / b.delta;
    const uint32_t maxMag = ((1u << b.Mb) << kFracBits) - 1;
    std::memset(out, 0, kCblkArea * sizeof(uint32_t));
    for (int y = 0; y < cb.h; ++y) {
        const float* src = plane + (size_t)(b.py0 + cb.y0 + y) * L.width + b.px0 + cb.x0;
        uint32_t* dst = out + y * kCblkSize;
        for (int x = 0; x < cb.w; ++x) dst[x] = quantizeCoeff(src[x], inv, maxMag);
    }
}

void quantizeFrame(const FrameLayout& L, const float* const planes[3], uint32_t* out) {
    const int n = L.numCblks();
    for (int i = 0; i < n; ++i) quantizeCodeBlock(L, planes, i, out + (size_t)i * kCblkArea);
}

}  // namespace gail
