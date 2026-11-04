// SPDX-License-Identifier: MIT
// EBCOT Tier-1 context tables and distortion estimates (ISO/IEC 15444-1 Annex D),
// shared by the CPU reference and GPU kernels.
#ifndef GAIL_T1_COMMON_H
#define GAIL_T1_COMMON_H

#include "kernels/portable.h"

#define T1_FRAC_BITS 6  // fractional magnitude bits carried with each coefficient
#define T1_SIGN_BIT 0x80000000u
#define T1_CTX_SC 9
#define T1_CTX_MR 14
#ifndef MQ_CTX_RL
#define MQ_CTX_RL 17
#define MQ_CTX_UNI 18
#endif

// Significance (zero-coding) context, Table D.1. h, v: significant horizontal /
// vertical neighbours (0..2), d: diagonal (0..4). orient: 0 LL, 1 HL, 2 LH, 3 HH.
GAIL_FN U32 t1_zc_ctx(I32 h, I32 v, I32 d, I32 orient) {
    if (orient == 1) {
        I32 t = h;
        h = v;
        v = t;
    }
    if (orient != 3) {
        if (h == 2) return 8u;
        if (h == 1) return v >= 1 ? 7u : (d >= 1 ? 6u : 5u);
        if (v == 2) return 4u;
        if (v == 1) return 3u;
        return d >= 2 ? 2u : TO_U32(d);
    }
    I32 hv = h + v;
    if (d >= 3) return 8u;
    if (d == 2) return hv >= 1 ? 7u : 6u;
    if (d == 1) return hv >= 2 ? 5u : (hv == 1 ? 4u : 3u);
    return hv >= 2 ? 2u : TO_U32(hv);
}

// Sign coding, Table D.3. hc, vc in {-1,0,1}. Returns ctx | xorbit << 8.
GAIL_FN U32 t1_sc_ctx(I32 hc, I32 vc) {
    if (hc == 0 && vc == 0) return 9u;
    if (hc == 0) return vc > 0 ? 10u : (10u | 256u);
    U32 x = hc < 0 ? 256u : 0u;
    I32 vv = hc < 0 ? -vc : vc;
    return (vv > 0 ? 13u : (vv == 0 ? 12u : 11u)) | x;
}

// Distortion reduction (x4096, in units of (2^p * delta)^2) when a coefficient
// becomes significant at bitplane p. w = 7 magnitude bits from bit p downwards.
GAIL_FN I32 t1_dist_sig(U32 w) { return 192 * TO_I32(w) - 9216; }
// Distortion reduction for a magnitude refinement at bitplane p.
GAIL_FN I32 t1_dist_ref(U32 w) { return w >= 64u ? 64 * TO_I32(w) - 5120 : 3072 - 64 * TO_I32(w); }

#endif
