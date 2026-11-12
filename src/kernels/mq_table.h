// SPDX-License-Identifier: MIT
// The MQ coder's probability state table and context numbering (ISO/IEC 15444-1 Annex C
// and D.3), shared by everything that codes or decodes with it. Compiles as C++, CUDA
// and GLSL.
#ifndef GAIL_MQ_TABLE_H
#define GAIL_MQ_TABLE_H

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

#endif
