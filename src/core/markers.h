// SPDX-License-Identifier: MIT
// JPEG 2000 codestream marker codes (ISO/IEC 15444-1 Table A.2).
#pragma once
#include <cstdint>

namespace gail::marker {

constexpr uint16_t SOC = 0xFF4F;  // start of codestream
constexpr uint16_t SIZ = 0xFF51;  // image and tile size
constexpr uint16_t COD = 0xFF52;  // coding style default
constexpr uint16_t COC = 0xFF53;  // coding style component
constexpr uint16_t TLM = 0xFF55;  // tile-part lengths
constexpr uint16_t PLM = 0xFF57;  // packet lengths, main header
constexpr uint16_t PLT = 0xFF58;  // packet lengths, tile-part header
constexpr uint16_t QCD = 0xFF5C;  // quantization default
constexpr uint16_t QCC = 0xFF5D;  // quantization component
constexpr uint16_t RGN = 0xFF5E;  // region of interest
constexpr uint16_t POC = 0xFF5F;  // progression order change
constexpr uint16_t PPM = 0xFF60;  // packed packet headers, main header
constexpr uint16_t PPT = 0xFF61;  // packed packet headers, tile-part header
constexpr uint16_t CRG = 0xFF63;  // component registration
constexpr uint16_t COM = 0xFF64;  // comment
constexpr uint16_t SOT = 0xFF90;  // start of tile-part
constexpr uint16_t SOP = 0xFF91;  // start of packet
constexpr uint16_t EPH = 0xFF92;  // end of packet header
constexpr uint16_t SOD = 0xFF93;  // start of data
constexpr uint16_t EOC = 0xFFD9;  // end of codestream

}  // namespace gail::marker
