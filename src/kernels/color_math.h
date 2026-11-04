// SPDX-License-Identifier: MIT
// Per-pixel colour handling shared by the CPU reference, the Vulkan (GLSL) and the CUDA
// backends: source R'G'B' -> 12-bit DCI X'Y'Z' (SMPTE ST 428-1).
//
// Two routes:
//  * 3D LUT (CP_LUT_SIZE > 0): an app-supplied LUT maps source R'G'B' (0..1) straight to
//    X'Y'Z' (0..1, i.e. code / 4095); everything else is skipped.
//  * managed: EOTF table (code -> cd/m^2) -> primaries matrix to CIE XYZ -> optional tone
//    map on luminance, max(R,G,B) or each channel -> optional gamut compression into the target projector gamut
//    -> X' = (X / 52.37f)^(1/2.6f).
//
// The including code defines CP(i) (float parameter i), EOTF(i) (float, i in 0..65535)
// and LUT3(i) (float, LUT entry i, three per node, red fastest). Where those tables are
// reached through pointers (CUDA), it also defines CM_ARGS (extra leading parameters,
// with a trailing comma) and CM_PASS (the matching arguments).
#ifndef CM_ARGS
#define CM_ARGS
#define CM_PASS
#endif
#ifndef GAIL_COLOR_MATH_H
#define GAIL_COLOR_MATH_H

#include "kernels/portable.h"

#include "kernels/color_params.h"

GAIL_FN float cm_pq_eotf(float e) {
    const float m1 = 2610.0f / 16384.0f, m2 = 2523.0f / 4096.0f * 128.0f, c1 = 3424.0f / 4096.0f,
                c2 = 2413.0f / 4096.0f * 32.0f, c3 = 2392.0f / 4096.0f * 32.0f;
    float p = FPOW(FMIN(FMAX(e, 0.0f), 1.0f), 1.0f / m2);
    return 10000.0f * FPOW(FMAX(p - c1, 0.0f) / (c2 - c3 * p), 1.0f / m1);
}

GAIL_FN float cm_pq_inv(float nits) {
    const float m1 = 2610.0f / 16384.0f, m2 = 2523.0f / 4096.0f * 128.0f, c1 = 3424.0f / 4096.0f,
                c2 = 2413.0f / 4096.0f * 32.0f, c3 = 2392.0f / 4096.0f * 32.0f;
    float y = FPOW(FMIN(FMAX(nits / 10000.0f, 0.0f), 1.0f), m1);
    return FPOW((c1 + c2 * y) / (1.0f + c3 * y), m2);
}

// Tone map: returns the target luminance (cd/m^2) for source luminance y (cd/m^2).
GAIL_FN float cm_tone(CM_ARGS float y) {
    if (y <= 0.0f) return 0.0f;
    if (CP(CP_TONE) > 1.5f) {  // BT.2390 EETF in the PQ domain (absolute)
        float srcMax = CP(CP_PQ_SRCMAX), ks = CP(CP_PQ_KS);
        float e1 = cm_pq_inv(y) / srcMax;
        // identity below the knee; returning y (not a PQ round trip) avoids the float
        // cancellation in cm_pq_eotf near black, where GPU and CPU pow() would disagree
        if (e1 < ks) return y;
        float t = (FMIN(e1, 1.0f) - ks) / (1.0f - ks);
        float t2 = t * t, t3 = t2 * t;
        float e2 = (2.0f * t3 - 3.0f * t2 + 1.0f) * ks + (t3 - 2.0f * t2 + t) * (1.0f - ks) +
                   (-2.0f * t3 + 3.0f * t2) * CP(CP_PQ_MAXLUM);
        return cm_pq_eotf(e2 * srcMax);
    }
    // Scene-relative: reference white -> GAIN * peak, linear below the knee, then an
    // extended-Reinhard shoulder (C1 at the knee) reaching peak exactly at the source peak.
    float v = CP(CP_GAIN) * y / CP(CP_SRC_REF);
    float k = CP(CP_KNEE);
    if (v > k) {
        float t = (v - k) / (1.0f - k);
        float T = CP(CP_SHOULDER_T);
        v = k + (1.0f - k) * FMIN(t * (1.0f + t / (T * T)) / (1.0f + t), 1.0f);
    }
    return v * CP(CP_PEAK);
}

// Pull a colour toward neutral grey of the same luminance until every target-gamut
// channel lies in [0, 1] (no channel clips, so texture inside saturated highlights
// survives); colours beyond THRESH of the way to the boundary are compressed smoothly.
GAIL_FN void cm_gamut(CM_ARGS INOUT(float) X, INOUT(float) Y, INOUT(float) Z) {
    float r = CP(CP_XYZ2TGT + 0) * X + CP(CP_XYZ2TGT + 1) * Y + CP(CP_XYZ2TGT + 2) * Z;
    float g = CP(CP_XYZ2TGT + 3) * X + CP(CP_XYZ2TGT + 4) * Y + CP(CP_XYZ2TGT + 5) * Z;
    float b = CP(CP_XYZ2TGT + 6) * X + CP(CP_XYZ2TGT + 7) * Y + CP(CP_XYZ2TGT + 8) * Z;
    float n = Y / CP(CP_PEAK);  // neutral at the same luminance (target RGB rows sum to luminance)
    if (n >= 1.0f) {
        r = 1.0f;
        g = 1.0f;
        b = 1.0f;
    } else {
        // smallest alpha in (0, 1] keeping n + alpha * (c - n) inside [0, 1] for all channels
        float amax = 1.0f;
        float d0 = r - n, d1 = g - n, d2 = b - n;
        if (d0 > 0.0f) amax = FMIN(amax, (1.0f - n) / d0);
        if (d0 < 0.0f) amax = FMIN(amax, -n / d0);
        if (d1 > 0.0f) amax = FMIN(amax, (1.0f - n) / d1);
        if (d1 < 0.0f) amax = FMIN(amax, -n / d1);
        if (d2 > 0.0f) amax = FMIN(amax, (1.0f - n) / d2);
        if (d2 < 0.0f) amax = FMIN(amax, -n / d2);
        // distance relative to the boundary along this hue: dist = 1 / amax
        float dist = 1.0f / FMAX(amax, 1e-6f);
        float th = CP(CP_GAMUT_THRESH);
        float alpha = 1.0f;
        if (dist > th) {
            // map (th, inf) -> (th, 1) with a smooth roll-off, unchanged below th
            float x = (dist - th) / (1.0f - th);
            float y = x / (1.0f + x);
            alpha = (th + (1.0f - th) * y) / dist;
        }
        r = n + alpha * d0;
        g = n + alpha * d1;
        b = n + alpha * d2;
    }
    X = CP(CP_TGT2XYZ + 0) * r + CP(CP_TGT2XYZ + 1) * g + CP(CP_TGT2XYZ + 2) * b;
    Y = CP(CP_TGT2XYZ + 3) * r + CP(CP_TGT2XYZ + 4) * g + CP(CP_TGT2XYZ + 5) * b;
    Z = CP(CP_TGT2XYZ + 6) * r + CP(CP_TGT2XYZ + 7) * g + CP(CP_TGT2XYZ + 8) * b;
}

GAIL_FN float cm_encode(float v) {  // cd/m^2 -> 12-bit code (as float)
    float n = FMIN(FMAX(v / 52.37f, 0.0f), 1.0f);
    return FFLOOR(4095.0f * FPOW(n, 1.0f / 2.6f) + 0.5f);
}

GAIL_FN float cm_lut_at(CM_ARGS I32 n, I32 i, I32 j, I32 k, I32 c) { return LUT3(((k * n + j) * n + i) * 3 + c); }

// Source R'G'B' (0..1) -> X'Y'Z' codes.
GAIL_FN void cm_rgb_to_xyz12(CM_ARGS float r, float g, float b, INOUT(float) cx, INOUT(float) cy, INOUT(float) cz) {
    I32 n = TO_I32(CP(CP_LUT_SIZE));
    if (n > 1) {
        float fr = FMIN(FMAX(r, 0.0f), 1.0f) * TO_F32(n - 1);
        float fg = FMIN(FMAX(g, 0.0f), 1.0f) * TO_F32(n - 1);
        float fb = FMIN(FMAX(b, 0.0f), 1.0f) * TO_F32(n - 1);
        I32 i0 = IMIN(TO_I32(fr), n - 2), j0 = IMIN(TO_I32(fg), n - 2), k0 = IMIN(TO_I32(fb), n - 2);
        float ar = fr - TO_F32(i0), ag = fg - TO_F32(j0), ab = fb - TO_F32(k0);
        float o[3];
        for (I32 c = 0; c < 3; ++c) {
            float c00 = cm_lut_at(CM_PASS n, i0, j0, k0, c) * (1.0f - ar) + cm_lut_at(CM_PASS n, i0 + 1, j0, k0, c) * ar;
            float c10 = cm_lut_at(CM_PASS n, i0, j0 + 1, k0, c) * (1.0f - ar) + cm_lut_at(CM_PASS n, i0 + 1, j0 + 1, k0, c) * ar;
            float c01 = cm_lut_at(CM_PASS n, i0, j0, k0 + 1, c) * (1.0f - ar) + cm_lut_at(CM_PASS n, i0 + 1, j0, k0 + 1, c) * ar;
            float c11 = cm_lut_at(CM_PASS n, i0, j0 + 1, k0 + 1, c) * (1.0f - ar) + cm_lut_at(CM_PASS n, i0 + 1, j0 + 1, k0 + 1, c) * ar;
            float v = (c00 * (1.0f - ag) + c10 * ag) * (1.0f - ab) + (c01 * (1.0f - ag) + c11 * ag) * ab;
            o[c] = FFLOOR(FMIN(FMAX(v, 0.0f), 1.0f) * 4095.0f + 0.5f);
        }
        cx = o[0];
        cy = o[1];
        cz = o[2];
        return;
    }
    float R = EOTF(TO_I32(FFLOOR(FMIN(FMAX(r, 0.0f), 1.0f) * 65535.0f + 0.5f)));
    float G = EOTF(TO_I32(FFLOOR(FMIN(FMAX(g, 0.0f), 1.0f) * 65535.0f + 0.5f)));
    float B = EOTF(TO_I32(FFLOOR(FMIN(FMAX(b, 0.0f), 1.0f) * 65535.0f + 0.5f)));
    float X = CP(0) * R + CP(1) * G + CP(2) * B;
    float Y = CP(4) * R + CP(5) * G + CP(6) * B;
    float Z = CP(8) * R + CP(9) * G + CP(10) * B;
    if (CP(CP_TONE) > 0.5f && Y > 0.0f) {
        float tr = CP(CP_XYZ2TGT + 0) * X + CP(CP_XYZ2TGT + 1) * Y + CP(CP_XYZ2TGT + 2) * Z;
        float tg = CP(CP_XYZ2TGT + 3) * X + CP(CP_XYZ2TGT + 4) * Y + CP(CP_XYZ2TGT + 5) * Z;
        float tb = CP(CP_XYZ2TGT + 6) * X + CP(CP_XYZ2TGT + 7) * Y + CP(CP_XYZ2TGT + 8) * Z;
        float pk = CP(CP_PEAK);
        if (CP(CP_TONE_NORM) > 1.5f) {
            // per channel in the target gamut: bright saturated colours drift toward
            // white as their channels approach the peak, the way film and SDR grades
            // render them. Channels below 0 (outside the gamut) are scaled like the
            // brightest channel, so an out-of-gamut colour shrinks in proportion instead
            // of being dragged toward black by an uncompressed negative channel.
            float mx = FMAX(tr, FMAX(tg, tb)) * pk;
            float sneg = mx > 0.0f ? cm_tone(CM_PASS mx) / mx : 1.0f;
            tr = tr > 0.0f ? cm_tone(CM_PASS tr * pk) / pk : tr * sneg;
            tg = tg > 0.0f ? cm_tone(CM_PASS tg * pk) / pk : tg * sneg;
            tb = tb > 0.0f ? cm_tone(CM_PASS tb * pk) / pk : tb * sneg;
            X = CP(CP_TGT2XYZ + 0) * tr + CP(CP_TGT2XYZ + 1) * tg + CP(CP_TGT2XYZ + 2) * tb;
            Y = CP(CP_TGT2XYZ + 3) * tr + CP(CP_TGT2XYZ + 4) * tg + CP(CP_TGT2XYZ + 5) * tb;
            Z = CP(CP_TGT2XYZ + 6) * tr + CP(CP_TGT2XYZ + 7) * tg + CP(CP_TGT2XYZ + 8) * tb;
        } else {
            // max(R,G,B) >= Y, so a saturated highlight is brought down until its brightest
            // channel fits under the peak and keeps its hue instead of being bleached to
            // white by the gamut stage. Luminance keeps Y instead.
            float m = CP(CP_TONE_NORM) > 0.5f ? FMAX(Y, FMAX(tr, FMAX(tg, tb)) * pk) : Y;
            float s = cm_tone(CM_PASS m) / m;
            X *= s;
            Y *= s;
            Z *= s;
        }
    }
    if (CP(CP_GAMUT) > 0.5f) cm_gamut(CM_PASS X, Y, Z);
    cx = cm_encode(X);
    cy = cm_encode(Y);
    cz = cm_encode(Z);
}

#endif
