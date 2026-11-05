// SPDX-License-Identifier: MIT
// Frame input: decodes any FFmpeg-readable video to RGB48 frames fitted into a
// DCI container (scale + letterbox/pillarbox), or synthesises test patterns.
#pragma once
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "core/color.h"
#include "core/frame_input.h"
#include "core/layout.h"

namespace gail {

struct SourceInfo {
    int width = 0, height = 0;
    double fps = 24.0;
    std::string transfer, primaries, matrix, range, pixFmt, chromaLocation;
    double masteringPeak = 0;  // cd/m^2 (0 = unknown)
    double maxCLL = 0;
    long long frames = 0;
};

SourceInfo probeSource(const std::string& path);

// The Y'CbCr matrix a source is decoded with, as swscale names it: "bt709", "bt601" or
// "bt2020". From the stream's colour-space tag; when that is missing, BT.2020 for BT.2020
// primaries, else BT.709 for HD (720 lines or more) and BT.601 for SD. Both input paths
// (native planar and RGB48 through swscale) use this one decision.
std::string sourceMatrix(const SourceInfo& info);


struct SourceOptions {
    std::string path;
    int width = 0, height = 0;   // output (container) size
    double fps = 0;              // output frame rate (0 = keep)
    double start = 0;            // seconds
    long long maxFrames = -1;
    bool fitToContainer = true;  // keep aspect ratio and pad
    bool yuv = true;             // hand over native 4:2:0 Y'CbCr when the source is 4:2:0
};

std::unique_ptr<FrameSource> openFfmpegSource(const SourceOptions& o, const SourceInfo& info);
// Frames preloaded from another source, replayed `loops` times (benchmarks without decode cost).
std::unique_ptr<FrameSource> openMemorySource(FrameSource& from, int frames, int loops);
// Directory of raw 12-bit X'Y'Z' frames (*.xyz: interleaved little-endian uint16, container
// size), e.g. from a DI pipeline or `gail-enc --dump-xyz`. Encode with transfer "gamma26"
// and primaries "xyz", which pass the code values through unchanged.
std::unique_ptr<FrameSource> openXyzDirSource(const std::string& dir, int width, int height, long long maxFrames);
// Moving colour-bar / gradient / noise pattern for benchmarks and tests.
std::unique_ptr<FrameSource> openPatternSource(int width, int height, long long frames, unsigned seed);

}  // namespace gail
