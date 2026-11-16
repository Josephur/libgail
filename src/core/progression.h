// SPDX-License-Identifier: MIT
// The order of packets in a DCI codestream: component - position - resolution - layer
// (CPRL, ISO/IEC 15444-1 B.12.1.5), in one progression volume for 2K and two for 4K,
// where the POC marker puts the 2K resolutions before the 4K one. Shared by the code
// that writes packets and the code that reads them, so the two cannot disagree.
#pragma once
#include <algorithm>

#include "core/layout.h"

namespace gail {

struct ProgressionVolume {
    int rs = 0, re = 0;  // resolutions [rs, re)
};

// Fills out[] and returns how many volumes the frame has (1 or 2).
inline int progressionVolumes(const FrameLayout& L, ProgressionVolume out[2]) {
    const bool fourk = L.profile == Profile::Cinema4K;
    const int NL = L.levels;
    out[0] = {0, fourk ? NL : NL + 1};
    out[1] = {NL, NL + 1};
    return fourk ? 2 : 1;
}

// Calls f(res, precinct) for every packet of component comp within volume v, in order.
// With one layer, a packet is one precinct of one resolution.
template <class F> inline void forEachPacket(const FrameLayout& L, ProgressionVolume v, int comp, F&& f) {
    const int NL = L.levels;
    int ystep = 1 << 30, xstep = 1 << 30;
    for (int r = v.rs; r < v.re; ++r) {
        const Resolution& R = L.resolution(0, r);
        xstep = std::min(xstep, 1 << (R.ppx + NL - r));
        ystep = std::min(ystep, 1 << (R.ppy + NL - r));
    }
    for (int y = 0; y < L.height; y += ystep)
        for (int x = 0; x < L.width; x += xstep)
            for (int r = v.rs; r < v.re; ++r) {
                const Resolution& R = L.resolution(comp, r);
                const int xr = 1 << (R.ppx + NL - r), yr = 1 << (R.ppy + NL - r);
                if ((x & (xr - 1)) || (y & (yr - 1))) continue;
                const int px = x / xr, py = y / yr;
                if (px >= R.npx || py >= R.npy) continue;
                f(r, py * R.npx + px);
            }
}

}  // namespace gail
