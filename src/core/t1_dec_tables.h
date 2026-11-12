// SPDX-License-Identifier: MIT
// The two context tables kernels/t1_decode.h indexes (Tables D.1 and D.3 by
// neighborhood bits), built on the host and handed to the GPU decoders.
#pragma once
#include <cstdint>

namespace gail {

constexpr int kT1dZcSize = 2048;  // T1D_ZC_SIZE
constexpr int kT1dScSize = 256;   // T1D_SC_SIZE

// zc: kT1dZcSize bytes, sc: kT1dScSize bytes.
void buildT1DecodeTables(uint8_t* zc, uint8_t* sc);

}  // namespace gail
