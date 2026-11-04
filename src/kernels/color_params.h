// SPDX-License-Identifier: MIT
// Layout of the colour parameter block consumed by kernels/color_math.h (one float each).
#ifndef GAIL_COLOR_PARAMS_H
#define GAIL_COLOR_PARAMS_H

// Parameter block layout (floats)
#define CP_RGB2XYZ 0        // 3x4 row-major, linear source RGB (cd/m^2) -> XYZ (cd/m^2)
#define CP_TONE 12          // 0 none, 1 shoulder (scene-relative), 2 BT.2390 EETF (absolute PQ)
#define CP_SRC_REF 13       // shoulder: source reference white, cd/m^2 (203 for PQ/HLG)
#define CP_GAIN 14          // shoulder: target reference white / target peak
#define CP_KNEE 15          // shoulder: start of the shoulder, fraction of target peak
#define CP_SHOULDER_T 16    // shoulder: normalised distance from knee to the source peak
#define CP_PEAK 17          // target peak white, cd/m^2 (48)
#define CP_GAMUT 18         // 0 off, 1 compress into target gamut
#define CP_GAMUT_THRESH 19  // compression starts at this fraction of the gamut boundary
#define CP_XYZ2TGT 20       // 3x3: XYZ (cd/m^2) -> target RGB, 1.0f = peak white
#define CP_TGT2XYZ 29       // 3x3: target RGB -> XYZ (cd/m^2)
#define CP_LUT_SIZE 38      // 3D LUT nodes per axis (0 = no LUT)
#define CP_PQ_SRCMAX 39     // BT.2390: PQ(source peak)
#define CP_PQ_MAXLUM 40     // BT.2390: target peak in normalised PQ
#define CP_PQ_KS 41         // BT.2390: knee start
#define CP_TONE_NORM 42     // tone map applied to: 0 luminance, 1 max(R,G,B), 2 each channel (target gamut)
#define CP_COUNT 48

#endif
