// SPDX-License-Identifier: MIT
// MQ arithmetic decoder (ISO/IEC 15444-1 Annex C.3), the counterpart of mq_encoder.h.
// Shared source for C++, CUDA and GLSL.
//
// The including code must define:
//   MQD_IN_T           type of the handle to the compressed bytes (pointer in C++/CUDA,
//                      base index in GLSL)
//   MQD_BYTE(in, i)    the byte at index i (as U32); only called with i below the end
//                      given to mqd_init()
// and may define:
//   MQD_CTX(d, cx)     lvalue holding context cx's state (default: d.ctx[cx])
//   MQD_TAB(i)         probability table lookup (default: the constant MQ_TABLE)
//
// The data is untrusted. Bytes are only read below the end of the codeword; beyond it the
// decoder is fed 0xFF, as C.3.4 specifies for a codeword that has run out, so a damaged
// or truncated codeword yields wrong symbols and nothing worse.
#ifndef GAIL_MQ_DECODER_H
#define GAIL_MQ_DECODER_H

#include "kernels/mq_table.h"
#include "kernels/portable.h"

#ifndef MQD_CTX
#define MQD_CTX(d, cx) d.ctx[cx]
#define MQD_CTX_IN_STATE 1
#endif
#ifndef MQD_TAB
#define MQD_TAB(i) MQ_TABLE[i]
#endif

struct MqDec {
    U32 A;
    U32 C;
    I32 CT;
    U32 bp;   // index of the byte last taken into C
    U32 end;  // one past the codeword's last byte
#ifdef MQD_CTX_IN_STATE
    U32 ctx[MQ_NUM_CTX];  // context state: its MQ_TABLE entry | mps << 31
#endif
};

#define MQD_AT(d, in, i) ((i) < d.end ? MQD_BYTE(in, i) : 0xFFu)

// BYTEIN (C.3.4): after a 0xFF the next byte brings 7 bits, and a byte above 0x8F there
// is a marker, which is not consumed and feeds ones from then on.
GAIL_FN void mqd_bytein(INOUT(MqDec) d, MQD_IN_T in) {
    U32 b = MQD_AT(d, in, d.bp);
    U32 b1 = MQD_AT(d, in, d.bp + 1u);
    if (b == 0xFFu) {
        if (b1 > 0x8Fu) {
            d.C += 0xFF00u;
            d.CT = 8;
        } else {
            d.bp += 1u;
            d.C += b1 << 9;
            d.CT = 7;
        }
    } else {
        d.bp += 1u;
        d.C += b1 << 8;
        d.CT = 8;
    }
}

// INITDEC (C.3.5) for the codeword in bytes [start, start + len), with the context
// states Annex D starts every code-block from (Table D.7).
GAIL_FN void mqd_init(INOUT(MqDec) d, MQD_IN_T in, U32 start, U32 len) {
    d.bp = start;
    d.end = start + len;
    d.C = MQD_AT(d, in, d.bp) << 16;
    mqd_bytein(d, in);
    d.C <<= 7;
    d.CT -= 7;
    d.A = 0x8000u;
    for (int i = 0; i < MQ_NUM_CTX; ++i) MQD_CTX(d, i) = MQD_TAB(0);
    MQD_CTX(d, 0) = MQD_TAB(4);
    MQD_CTX(d, MQ_CTX_RL) = MQD_TAB(3);
    MQD_CTX(d, MQ_CTX_UNI) = MQD_TAB(46);
}

// RENORMD (C.3.3)
GAIL_FN void mqd_renorm(INOUT(MqDec) d, MQD_IN_T in) {
    do {
        if (d.CT == 0) mqd_bytein(d, in);
        d.A <<= 1;
        d.C <<= 1;
        d.CT -= 1;
    } while ((d.A & 0x8000u) == 0u);
}

// DECODE (C.3.2): the next decision in context cx.
GAIL_FN U32 mqd_decode(INOUT(MqDec) d, U32 cx, MQD_IN_T in) {
    U32 st = MQD_CTX(d, cx);
    U32 qe = st & 0xFFFFu;
    U32 mps = st >> 31;
    U32 nmps = MQD_TAB((st >> 16) & 63u) | (mps << 31);
    U32 nlps = MQD_TAB((st >> 22) & 63u) | ((mps ^ ((st >> 28) & 1u)) << 31);
    U32 bit = mps;
    d.A -= qe;
    if ((d.C >> 16) < qe) {
        // the lower sub-interval: the less probable symbol unless the intervals are exchanged
        if (d.A < qe) {
            MQD_CTX(d, cx) = nmps;
        } else {
            bit = mps ^ 1u;
            MQD_CTX(d, cx) = nlps;
        }
        d.A = qe;
        mqd_renorm(d, in);
    } else {
        d.C -= qe << 16;
        if ((d.A & 0x8000u) == 0u) {
            if (d.A < qe) {
                bit = mps ^ 1u;
                MQD_CTX(d, cx) = nlps;
            } else {
                MQD_CTX(d, cx) = nmps;
            }
            mqd_renorm(d, in);
        }
    }
    return bit;
}

#endif
