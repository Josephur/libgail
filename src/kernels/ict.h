// SPDX-License-Identifier: MIT
// The irreversible multiple component transform (ISO/IEC 15444-1 G.3), forward and
// inverse. Plain literals: C++ and CUDA code casts them to the type it computes in.
//   Y0 =  ICT_Y0_R * R + ICT_Y0_G * G + ICT_Y0_B * B         R = Y0 + ICT_R_Y2 * Y2
//   Y1 =  ICT_Y1_R * R + ICT_Y1_G * G + ICT_Y1_B * B         G = Y0 + ICT_G_Y1 * Y1 + ICT_G_Y2 * Y2
//   Y2 =  ICT_Y2_R * R + ICT_Y2_G * G + ICT_Y2_B * B         B = Y0 + ICT_B_Y1 * Y1
// (in a DCI frame the three inputs are X', Y' and Z').
#ifndef GAIL_ICT_H
#define GAIL_ICT_H

#define ICT_Y0_R 0.299
#define ICT_Y0_G 0.587
#define ICT_Y0_B 0.114
#define ICT_Y1_R (-0.16875)
#define ICT_Y1_G (-0.33126)
#define ICT_Y1_B 0.5
#define ICT_Y2_R 0.5
#define ICT_Y2_G (-0.41869)
#define ICT_Y2_B (-0.08131)

#define ICT_R_Y2 1.402
#define ICT_G_Y1 (-0.34413)
#define ICT_G_Y2 (-0.71414)
#define ICT_B_Y1 1.772

#endif
