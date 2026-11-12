// SPDX-License-Identifier: MIT
// EBCOT Tier-1 block decoder for the GPU backends (ISO/IEC 15444-1 Annex D, no mode
// switches), one thread per code-block. Shared source for CUDA and GLSL, and compiled for
// the host by the tests. The CPU backend has its own decoder (core/t1_dec.cpp), written
// differently, which this one is checked against.
//
// A block's state is four sets of 32-bit row masks (bit x = column x): significance,
// sign, "coded in this plane's significance pass" and "refined before". A stripe of four
// rows is worked on in registers together with the row above and the row below, and
// contexts come from two small tables indexed by neighborhood bits.
//
// The including code defines, besides what mq_decoder.h needs:
//   T1D_OUT_T            type of the handle to the block's output
//   T1D_WR(out, i, v)    store word v for sample i = 32 * row + column
//   T1D_RD(out, i)       read it back
//   T1D_ZC(i)            zero-coding context for orientation o and a 3x3 neighborhood n:
//                        i = 512 * o + n, where bits 0..2 of n are columns x-1, x, x+1 of
//                        the row above, bits 3..5 the sample's own row, bits 6..8 the row below
//   T1D_SC(i)            sign context | xor bit << 7, for i = (significant, negative) pairs
//                        of the left, right, upper and lower neighbor, two bits each from bit 0
// The words written are those of core/t1_dec.h: bit 31 the sign, the low bits twice the
// magnitude plus the mid-point of the bit-planes not decoded. The output must be zero
// beforehand; samples that stay insignificant are not written.
#ifndef GAIL_T1_DECODER_H
#define GAIL_T1_DECODER_H

#include "kernels/mq_decoder.h"
#include "kernels/portable.h"

#define T1D_ZC_SIZE 2048
#define T1D_SC_SIZE 256

struct T1dState {
    U32 sig[34];  // rows -1 .. 32: the rows outside the block stay zero
    U32 sgn[34];
    U32 vis[32];
    U32 ref[32];
};

// Columns x-1, x, x+1 of a row mask in bits 0, 1, 2.
#define T1D_NB3(row, x) (((((row) >> (x)) & 3u) << 1) | ((((row) << 1) >> (x)) & 1u))
#define T1D_HS(r) ((r) | ((r) << 1) | ((r) >> 1))
#define T1D_ZCIDX(Su, Sm, Sd) (T1D_NB3(Su, x) | (T1D_NB3(Sm, x) << 3) | (T1D_NB3(Sd, x) << 6))

// The sample at (row k of the stripe, column x) has its most significant bit in plane p:
// read its sign (D.3.2) and give it that bit plus the mid-point of the planes below.
#define T1D_BECOME_SIG(k, Su, Sm, Sd, Gu, Gm, Gd)                                                      \
    {                                                                                                  \
        U32 hs = T1D_NB3(Sm, x), hg = T1D_NB3(Gm, x);                                                  \
        U32 sci = (hs & 1u) | ((hg & 1u) << 1) | (((hs >> 2) & 1u) << 2) | (((hg >> 2) & 1u) << 3) |    \
                  (((Su >> x) & 1u) << 4) | (((Gu >> x) & 1u) << 5) | (((Sd >> x) & 1u) << 6) |        \
                  (((Gd >> x) & 1u) << 7);                                                             \
        U32 sc = T1D_SC(sci);                                                                          \
        U32 neg = mqd_decode(q, sc & 15u, in) ^ (sc >> 7);                                             \
        Sm |= bit;                                                                                     \
        if (neg != 0u) Gm |= bit;                                                                      \
        T1D_WR(out, (y0 + TO_U32(k)) * 32u + x, (neg << 31) | (3u << p));                              \
    }

// Significance propagation (D.3.1): not yet significant, with a significant neighbor.
#define T1D_SPP_ROW(k, rm, Su, Sm, Sd, Gu, Gm, Gd, Vk)                                                 \
    if ((rm & bit) != 0u && (Sm & bit) == 0u) {                                                        \
        U32 zi = T1D_ZCIDX(Su, Sm, Sd);                                                                \
        if (zi != 0u) {                                                                                \
            U32 dbit = mqd_decode(q, T1D_ZC(zcBase + zi), in);                                         \
            Vk |= bit;                                                                                 \
            if (dbit != 0u) {                                                                          \
                T1D_BECOME_SIG(k, Su, Sm, Sd, Gu, Gm, Gd)                                              \
                colmask |= (bit << 1) & wmask;                                                         \
            }                                                                                          \
        }                                                                                              \
    }

// Magnitude refinement (D.3.3): significant before this plane's significance pass. The
// bit replaces the mid-point guess made one plane up: 2^p either way in the doubled value.
#define T1D_MRP_ROW(k, Su, Sm, Sd, Vk, Rk)                                                             \
    if ((Sm & ~Vk & bit) != 0u) {                                                                      \
        U32 nb = T1D_NB3(Su, x) | T1D_NB3(Sd, x) | (T1D_NB3(Sm, x) & 5u);                              \
        U32 cx = (Rk & bit) != 0u ? 16u : (nb != 0u ? 15u : 14u);                                      \
        U32 dbit = mqd_decode(q, cx, in);                                                              \
        U32 oi = (y0 + TO_U32(k)) * 32u + x;                                                           \
        U32 word = T1D_RD(out, oi);                                                                    \
        T1D_WR(out, oi, dbit != 0u ? word + (1u << p) : word - (1u << p));                             \
        Rk |= bit;                                                                                     \
    }

// Cleanup (D.3.4): everything not yet significant and not coded in the significance pass.
#define T1D_CUP_ROW(k, uk, Su, Sm, Sd, Gu, Gm, Gd)                                                     \
    if (TO_U32(k) >= k0 && (uk & bit) != 0u) {                                                         \
        U32 dbit = 1u;                                                                                 \
        if (TO_U32(k) != forced) dbit = mqd_decode(q, T1D_ZC(zcBase + T1D_ZCIDX(Su, Sm, Sd)), in);     \
        if (dbit != 0u) {                                                                              \
            T1D_BECOME_SIG(k, Su, Sm, Sd, Gu, Gm, Gd)                                                  \
            any |= bit;                                                                                \
        }                                                                                              \
    }

// Decodes the first npasses coding passes of a w x h block (1..32 each) with nbps
// bit-planes (1..30); q must have been set up with mqd_init() for the block's bytes.
// Whatever the arguments, nothing outside the block's own 32 x 32 words is written.
GAIL_FN void t1d_decode_block(INOUT(T1dState) st, INOUT(MqDec) q, MQD_IN_T in, U32 w, U32 h, U32 orient, U32 nbps,
                              U32 npasses, T1D_OUT_T out) {
    if (w == 0u || h == 0u || w > 32u || h > 32u || nbps == 0u || nbps > 30u || npasses == 0u) return;
    if (npasses > 3u * nbps - 2u) npasses = 3u * nbps - 2u;
    U32 zcBase = (orient & 3u) * 512u;
    for (int i = 0; i < 34; ++i) {
        st.sig[i] = 0u;
        st.sgn[i] = 0u;
    }
    for (int i = 0; i < 32; ++i) {
        st.vis[i] = 0u;
        st.ref[i] = 0u;
    }
    U32 wmask = w == 32u ? 0xFFFFFFFFu : (1u << w) - 1u;
    U32 p = nbps - 1u;
    U32 type = 2u;  // 0 significance propagation, 1 magnitude refinement, 2 cleanup
    for (U32 pass = 0u; pass < npasses; ++pass) {
        for (U32 y0 = 0u; y0 < h; y0 += 4u) {
            U32 ylen = h - y0 < 4u ? h - y0 : 4u;
            U32 rm0 = wmask, rm1 = ylen > 1u ? wmask : 0u, rm2 = ylen > 2u ? wmask : 0u, rm3 = ylen > 3u ? wmask : 0u;
            // rows y0 - 1 .. y0 + 4 (the state arrays are offset by one row)
            U32 S0 = st.sig[y0], S5 = st.sig[y0 + 5u];
            U32 S1 = st.sig[y0 + 1u], S2 = st.sig[y0 + 2u], S3 = st.sig[y0 + 3u], S4 = st.sig[y0 + 4u];
            if (type == 0u) {
                U32 G0 = st.sgn[y0], G5 = st.sgn[y0 + 5u];
                U32 G1 = st.sgn[y0 + 1u], G2 = st.sgn[y0 + 2u], G3 = st.sgn[y0 + 3u], G4 = st.sgn[y0 + 4u];
                U32 V1 = 0u, V2 = 0u, V3 = 0u, V4 = 0u;
                U32 h0 = T1D_HS(S0), h1 = T1D_HS(S1), h2 = T1D_HS(S2), h3 = T1D_HS(S3), h4 = T1D_HS(S4), h5 = T1D_HS(S5);
                // columns with a sample to code; one more to the right joins when a sample
                // becomes significant
                U32 colmask = ((h0 | h1 | h2) & ~S1 & rm0) | ((h1 | h2 | h3) & ~S2 & rm1) | ((h2 | h3 | h4) & ~S3 & rm2) |
                              ((h3 | h4 | h5) & ~S4 & rm3);
                while (colmask != 0u) {
                    U32 x = TO_U32(FIND_LSB(colmask));
                    U32 bit = 1u << x;
                    colmask &= ~bit;
                    T1D_SPP_ROW(0, rm0, S0, S1, S2, G0, G1, G2, V1)
                    T1D_SPP_ROW(1, rm1, S1, S2, S3, G1, G2, G3, V2)
                    T1D_SPP_ROW(2, rm2, S2, S3, S4, G2, G3, G4, V3)
                    T1D_SPP_ROW(3, rm3, S3, S4, S5, G3, G4, G5, V4)
                }
                st.sig[y0 + 1u] = S1;
                st.sig[y0 + 2u] = S2;
                st.sig[y0 + 3u] = S3;
                st.sig[y0 + 4u] = S4;
                st.sgn[y0 + 1u] = G1;
                st.sgn[y0 + 2u] = G2;
                st.sgn[y0 + 3u] = G3;
                st.sgn[y0 + 4u] = G4;
                st.vis[y0] = V1;
                st.vis[y0 + 1u] = V2;
                st.vis[y0 + 2u] = V3;
                st.vis[y0 + 3u] = V4;
            } else if (type == 1u) {
                U32 V1 = st.vis[y0], V2 = st.vis[y0 + 1u], V3 = st.vis[y0 + 2u], V4 = st.vis[y0 + 3u];
                U32 R1 = st.ref[y0], R2 = st.ref[y0 + 1u], R3 = st.ref[y0 + 2u], R4 = st.ref[y0 + 3u];
                U32 colmask = (S1 & ~V1) | (S2 & ~V2) | (S3 & ~V3) | (S4 & ~V4);
                while (colmask != 0u) {
                    U32 x = TO_U32(FIND_LSB(colmask));
                    U32 bit = 1u << x;
                    colmask &= ~bit;
                    T1D_MRP_ROW(0, S0, S1, S2, V1, R1)
                    T1D_MRP_ROW(1, S1, S2, S3, V2, R2)
                    T1D_MRP_ROW(2, S2, S3, S4, V3, R3)
                    T1D_MRP_ROW(3, S3, S4, S5, V4, R4)
                }
                st.ref[y0] = R1;
                st.ref[y0 + 1u] = R2;
                st.ref[y0 + 2u] = R3;
                st.ref[y0 + 3u] = R4;
            } else {
                U32 G0 = st.sgn[y0], G5 = st.sgn[y0 + 5u];
                U32 G1 = st.sgn[y0 + 1u], G2 = st.sgn[y0 + 2u], G3 = st.sgn[y0 + 3u], G4 = st.sgn[y0 + 4u];
                U32 u0 = ~S1 & ~st.vis[y0] & rm0, u1 = ~S2 & ~st.vis[y0 + 1u] & rm1, u2 = ~S3 & ~st.vis[y0 + 2u] & rm2,
                    u3 = ~S4 & ~st.vis[y0 + 3u] & rm3;
                U32 full = ylen == 4u ? (u0 & u1 & u2 & u3) : 0u;
                U32 any = S0 | S1 | S2 | S3 | S4 | S5;
                U32 colmask = u0 | u1 | u2 | u3;
                while (colmask != 0u) {
                    U32 x = TO_U32(FIND_LSB(colmask));
                    U32 bit = 1u << x;
                    colmask &= ~bit;
                    U32 k0 = 0u, forced = 99u;
                    if ((full & bit) != 0u && T1D_NB3(any, x) == 0u) {
                        // run-length mode: four untouched samples with no significant neighbor
                        if (mqd_decode(q, TO_U32(MQ_CTX_RL), in) == 0u) continue;
                        k0 = mqd_decode(q, TO_U32(MQ_CTX_UNI), in) << 1;
                        k0 |= mqd_decode(q, TO_U32(MQ_CTX_UNI), in);
                        forced = k0;
                    }
                    T1D_CUP_ROW(0, u0, S0, S1, S2, G0, G1, G2)
                    T1D_CUP_ROW(1, u1, S1, S2, S3, G1, G2, G3)
                    T1D_CUP_ROW(2, u2, S2, S3, S4, G2, G3, G4)
                    T1D_CUP_ROW(3, u3, S3, S4, S5, G3, G4, G5)
                }
                st.sig[y0 + 1u] = S1;
                st.sig[y0 + 2u] = S2;
                st.sig[y0 + 3u] = S3;
                st.sig[y0 + 4u] = S4;
                st.sgn[y0 + 1u] = G1;
                st.sgn[y0 + 2u] = G2;
                st.sgn[y0 + 3u] = G3;
                st.sgn[y0 + 4u] = G4;
                // the plane is finished for these rows
                st.vis[y0] = 0u;
                st.vis[y0 + 1u] = 0u;
                st.vis[y0 + 2u] = 0u;
                st.vis[y0 + 3u] = 0u;
            }
        }
        if (type == 2u) {
            type = 0u;
            p -= 1u;
        } else {
            type += 1u;
        }
    }
}

#endif
