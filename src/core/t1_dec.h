// SPDX-License-Identifier: MIT
// CPU EBCOT Tier-1 decoder (ISO/IEC 15444-1 Annex D, no mode switches): the reference
// the GPU decoders are checked against, and the CPU backend's block decoder.
#pragma once
#include <cstdint>

#include "core/layout.h"

namespace gail {

// Decodes the first `passes` coding passes of one code-block from its `len` bytes.
// out receives 32x32 row-major words (w x h of them used, the rest untouched): bit 31 is
// the sign, and the low bits are twice the magnitude with the mid-point of the bit-planes
// that were not decoded added in, so a coefficient is (out & 0x7FFFFFFF) * step / 2.
// The data is untrusted: any bytes decode to something, without reading outside them.
// nbps is at most 30 and passes at most 3 * nbps - 2.
void t1DecodeBlock(const uint8_t* data, uint32_t len, int w, int h, int orient, int nbps, int passes, uint32_t* out);

}  // namespace gail
