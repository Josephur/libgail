// SPDX-License-Identifier: MIT
// Shared end of the colour kernels: R'G'B' (0..1) -> X'Y'Z' codes via kernels/color_math.h,
// optional X'Y'Z' tap, DC shift and ICT into the three wavelet input planes.
F32Buf g_cp;
F32Buf g_eotf;
F32Buf g_lut3;
#define CP(i) g_cp.v[i]
#define EOTF(i) g_eotf.v[i]
#define LUT3(i) g_lut3.v[i]
#define GAIL_GLSL 1
#include "kernels/color_math.h"

void finishPixel(uint i, vec3 rgb, uint64_t planes, uint64_t xyz, uint numPixels) {
    float cx, cy, cz;
    cm_rgb_to_xyz12(rgb.r, rgb.g, rgb.b, cx, cy, cz);
    if (xyz != 0ul) {
        U16Buf o = U16Buf(xyz);
        o.v[3 * i] = uint16_t(cx);
        o.v[3 * i + 1] = uint16_t(cy);
        o.v[3 * i + 2] = uint16_t(cz);
    }
    vec3 s = vec3(cx, cy, cz) - 2048.0;
    F32Buf pl = F32Buf(planes);
    pl.v[i] = 0.299 * s.x + 0.587 * s.y + 0.114 * s.z;
    pl.v[numPixels + i] = -0.16875 * s.x - 0.33126 * s.y + 0.5 * s.z;
    pl.v[2u * numPixels + i] = 0.5 * s.x - 0.41869 * s.y - 0.08131 * s.z;
}
