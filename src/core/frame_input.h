// SPDX-License-Identifier: MIT
// Where the pipeline gets its frames: an FFmpeg decoder or test pattern in the tools
// (core/frame_source.h), or frames handed over by an app through the embedding API.
#pragma once
#include "core/color.h"

namespace gail {

class FrameSource {
public:
    virtual ~FrameSource() = default;
    virtual const FrameFormat& format() const = 0;
    int width() const { return format().width; }
    int height() const { return format().height; }
    // Reads the next frame (format().bytes() bytes). Returns false at end.
    virtual bool read(void* dst) = 0;
};

}  // namespace gail
