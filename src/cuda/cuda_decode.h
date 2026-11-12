// SPDX-License-Identifier: MIT
// CUDA decode slot (see core/decode_backend.h): Tier-1, dequantization, the inverse
// wavelet and the inverse color transform on one CUDA device.
#pragma once
#include <memory>

#include "core/decode_backend.h"

namespace gail {

std::unique_ptr<DecoderSlot> createCudaDecoderSlot(int device, const FrameLayout& L, const DecodeSettings& s);

}  // namespace gail
