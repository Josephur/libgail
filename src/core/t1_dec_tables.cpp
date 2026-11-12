// SPDX-License-Identifier: MIT
#include "core/t1_dec_tables.h"

#include <algorithm>

#include "kernels/t1_common.h"

namespace gail {

void buildT1DecodeTables(uint8_t* zc, uint8_t* sc) {
    for (int o = 0; o < 4; ++o)
        for (int n = 0; n < 512; ++n) {
            auto b = [&](int i) { return (n >> i) & 1; };
            const int h = b(3) + b(5), v = b(1) + b(7), d = b(0) + b(2) + b(6) + b(8);
            zc[o * 512 + n] = (uint8_t)t1_zc_ctx(h, v, d, o);
        }
    for (int n = 0; n < 256; ++n) {
        // contribution of a neighbor: 0 when insignificant, else +1 or -1 by its sign
        auto chi = [&](int i) { return ((n >> (2 * i)) & 1) ? (((n >> (2 * i + 1)) & 1) ? -1 : 1) : 0; };
        const int hc = std::clamp(chi(0) + chi(1), -1, 1), vc = std::clamp(chi(2) + chi(3), -1, 1);
        const uint32_t s = t1_sc_ctx(hc, vc);
        sc[n] = (uint8_t)((s & 0xFF) | ((s >> 8) << 7));
    }
}

}  // namespace gail
