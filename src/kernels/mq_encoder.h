// SPDX-License-Identifier: MIT
// MQ arithmetic encoder (ISO/IEC 15444-1 Annex C) plus exact truncation-length
// computation for non-terminated coding passes. Shared by the CPU reference,
// the Vulkan (GLSL) and the CUDA backends.
//
// The including code must define:
//   MQ_OUT_T          type of the output handle (pointer in C++/CUDA, base index in GLSL)
//   MQ_PUT(o, i, v)   store byte v at index i
//   MQ_GET(o, i)      load byte at index i (as U32)
// and may define:
//   MQ_CTX(e, cx)     lvalue holding context cx's state (default: e.ctx[cx]); GPU kernels
//                     keep contexts in shared memory so the coder state stays in registers
//   MQ_TAB(i)         probability table lookup (default: the constant MQ_TABLE)
#ifndef GAIL_MQ_ENCODER_H
#define GAIL_MQ_ENCODER_H

#include "kernels/portable.h"

#define MQ_NUM_CTX 19
#ifndef MQ_CTX_RL
#define MQ_CTX_RL 17
#define MQ_CTX_UNI 18
#endif

// Packed state table: Qe | NMPS << 16 | NLPS << 22 | SWITCH << 28 (Table C.2)
GAIL_CONST U32 MQ_TABLE[47] = ARRAY_BEGIN(U32, 47)
    0x5601u | (1u << 16) | (1u << 22) | (1u << 28),
    0x3401u | (2u << 16) | (6u << 22),
    0x1801u | (3u << 16) | (9u << 22),
    0x0AC1u | (4u << 16) | (12u << 22),
    0x0521u | (5u << 16) | (29u << 22),
    0x0221u | (38u << 16) | (33u << 22),
    0x5601u | (7u << 16) | (6u << 22) | (1u << 28),
    0x5401u | (8u << 16) | (14u << 22),
    0x4801u | (9u << 16) | (14u << 22),
    0x3801u | (10u << 16) | (14u << 22),
    0x3001u | (11u << 16) | (17u << 22),
    0x2401u | (12u << 16) | (18u << 22),
    0x1C01u | (13u << 16) | (20u << 22),
    0x1601u | (29u << 16) | (21u << 22),
    0x5601u | (15u << 16) | (14u << 22) | (1u << 28),
    0x5401u | (16u << 16) | (14u << 22),
    0x5101u | (17u << 16) | (15u << 22),
    0x4801u | (18u << 16) | (16u << 22),
    0x3801u | (19u << 16) | (17u << 22),
    0x3401u | (20u << 16) | (18u << 22),
    0x3001u | (21u << 16) | (19u << 22),
    0x2801u | (22u << 16) | (19u << 22),
    0x2401u | (23u << 16) | (20u << 22),
    0x2201u | (24u << 16) | (21u << 22),
    0x1C01u | (25u << 16) | (22u << 22),
    0x1801u | (26u << 16) | (23u << 22),
    0x1601u | (27u << 16) | (24u << 22),
    0x1401u | (28u << 16) | (25u << 22),
    0x1201u | (29u << 16) | (26u << 22),
    0x1101u | (30u << 16) | (27u << 22),
    0x0AC1u | (31u << 16) | (28u << 22),
    0x09C1u | (32u << 16) | (29u << 22),
    0x08A1u | (33u << 16) | (30u << 22),
    0x0521u | (34u << 16) | (31u << 22),
    0x0441u | (35u << 16) | (32u << 22),
    0x02A1u | (36u << 16) | (33u << 22),
    0x0221u | (37u << 16) | (34u << 22),
    0x0141u | (38u << 16) | (35u << 22),
    0x0111u | (39u << 16) | (36u << 22),
    0x0085u | (40u << 16) | (37u << 22),
    0x0049u | (41u << 16) | (38u << 22),
    0x0025u | (42u << 16) | (39u << 22),
    0x0015u | (43u << 16) | (40u << 22),
    0x0009u | (44u << 16) | (41u << 22),
    0x0005u | (45u << 16) | (42u << 22),
    0x0001u | (45u << 16) | (43u << 22),
    0x5601u | (46u << 16) | (46u << 22)
ARRAY_END;

#ifndef MQ_CTX
#define MQ_CTX(e, cx) e.ctx[cx]
#define MQ_CTX_IN_STATE 1
#endif
#ifndef MQ_TAB
#define MQ_TAB(i) MQ_TABLE[i]
#endif

struct MqEnc {
    U32 A;
    U32 C;
    I32 CT;
    I32 bp;   // index of the pending byte B (-1 before the first BYTEOUT)
    U32 B;
#ifdef MQ_CTX_IN_STATE
    U32 ctx[MQ_NUM_CTX];  // context state: its MQ_TABLE entry | mps << 31
#endif
};

// Snapshot of the coder state at the end of a coding pass.
struct MqSnap {
    I32 bp;
    U32 B;
    U32 C;
    U32 A;
    I32 CT;
};

GAIL_FN void mq_init(INOUT(MqEnc) e) {
    e.A = 0x8000u;
    e.C = 0u;
    e.CT = 12;
    e.bp = -1;
    e.B = 0u;
    // Contexts hold their probability-table entry (plus the MPS in bit 31) rather than
    // a state index, so the common path needs a single lookup.
    for (int i = 0; i < MQ_NUM_CTX; ++i) MQ_CTX(e, i) = MQ_TAB(0);
    MQ_CTX(e, 0) = MQ_TAB(4);
    MQ_CTX(e, MQ_CTX_RL) = MQ_TAB(3);
    MQ_CTX(e, MQ_CTX_UNI) = MQ_TAB(46);
}

GAIL_FN void mq_byteout(INOUT(MqEnc) e, MQ_OUT_T ob) {
    bool stuff = e.B == 0xFFu;
    if (!stuff && e.C >= 0x8000000u) {
        // carry into B
        e.B += 1u;
        if (e.B == 0xFFu) {
            e.C &= 0x7FFFFFFu;
            stuff = true;
        }
    }
    if (e.bp >= 0) MQ_PUT(ob, e.bp, e.B);
    e.bp += 1;
    if (stuff) {
        e.B = e.C >> 20;
        e.C &= 0xFFFFFu;
        e.CT = 7;
    } else {
        e.B = (e.C >> 19) & 0xFFu;  // bit 27 (if set) is the carry already added to the previous byte
        e.C &= 0x7FFFFu;
        e.CT = 8;
    }
}

GAIL_FN void mq_renorm(INOUT(MqEnc) e, MQ_OUT_T ob) {
    I32 n = 15 - FIND_MSB(e.A);  // shifts needed to bring A back to >= 0x8000
    while (n > 0) {
        I32 s = n < e.CT ? n : e.CT;
        e.A <<= TO_U32(s);
        e.C <<= TO_U32(s);
        e.CT -= s;
        n -= s;
        if (e.CT == 0) mq_byteout(e, ob);
    }
}

// CODEMPS / CODELPS (Annex C.2.5/C.2.6) in branch-free form. With a = A - Qe and
// exch = (a < Qe) (conditional exchange), both procedures reduce to
//   C += Qe   iff  isMps != exch,   else  A = Qe   (otherwise A = a)
// and the context moves to NMPS / NLPS only when renormalisation follows. GPU lanes
// coding different code-blocks therefore no longer diverge on the MPS/LPS decision.
GAIL_FN void mq_encode(INOUT(MqEnc) e, U32 cx, U32 d, MQ_OUT_T ob) {
    U32 t = MQ_CTX(e, cx);
    U32 mps = t >> 31;
    U32 qe = t & 0xFFFFu;
    U32 a = e.A - qe;
    bool isMps = d == mps;
    bool renorm = !isMps || (a & 0x8000u) == 0u;
    bool addC = isMps != (a < qe);
    e.C += addC ? qe : 0u;
    e.A = addC ? a : qe;
    U32 nidx = isMps ? ((t >> 16) & 63u) : ((t >> 22) & 63u);
    U32 nmps = isMps ? mps : (mps ^ ((t >> 28) & 1u));
    if (renorm) {
        MQ_CTX(e, cx) = MQ_TAB(nidx) | (nmps << 31);
        mq_renorm(e, ob);
    }
}

GAIL_FN MqSnap mq_snapshot(MqEnc e) {
    MqSnap s;
    s.bp = e.bp;
    s.B = e.B;
    s.C = e.C;
    s.A = e.A;
    s.CT = e.CT;
    return s;
}

// Terminate the codeword (Annex C.2.9). Returns the codeword length in bytes.
GAIL_FN U32 mq_flush(INOUT(MqEnc) e, MQ_OUT_T ob) {
    U32 tempc = e.C + e.A;
    e.C |= 0xFFFFu;
    if (e.C >= tempc) e.C -= 0x8000u;
    e.C <<= TO_U32(e.CT);
    mq_byteout(e, ob);
    e.C <<= TO_U32(e.CT);
    mq_byteout(e, ob);
    if (e.B != 0xFFu) {
        MQ_PUT(ob, e.bp, e.B);
        return TO_U32(e.bp + 1);
    }
    return TO_U32(e.bp);
}

// Smallest prefix length L >= minLen of the final codeword (len bytes) such that a
// standard decoder, which feeds 1-bits once the segment is exhausted, decodes every
// symbol coded before snapshot s exactly as encoded. The prefix never ends in 0xFF.
//
// Arithmetic is relative to the start of byte s.bp, in units of 2^-16 of the C
// register LSB at snapshot time. The pending byte B's LSB aligns with C bit (27-CT).
GAIL_FN U32 mq_trunc_len(MqSnap s, MQ_OUT_T ob, U32 len, U32 minLen) {
    I32 base = 43 - s.CT;  // shift that places byte s.bp at its weight (B LSB = 2^(27-CT+16))
    U64 low = (TO_U64(s.B) << TO_U32(base)) + (TO_U64(s.C) << 16u);
    U64 high = low + (TO_U64(s.A) << 16u);
    U64 P = 0u;
    I32 rel = 0;  // bit position of byte i relative to byte s.bp
    for (I32 i = s.bp; i < TO_I32(len); ++i) {
        U32 bv = i >= 0 ? MQ_GET(ob, i) : 0u;
        I32 sh = base - rel;
        if (sh < 0) break;
        P += TO_U64(bv) << TO_U32(sh);
        U64 T = TO_U64(1u) << TO_U32(sh);
        U32 L = TO_U32(i + 1);
        if (L >= minLen && bv != 0xFFu && P + T <= high && P + T > low) return L;
        rel += bv == 0xFFu ? 7 : 8;
    }
    return len;
}

#endif
