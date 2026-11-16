// SPDX-License-Identifier: MIT
#include "core/t1_dec.h"

#include <algorithm>
#include <vector>

#define MQD_IN_T const uint8_t*
#define MQD_BYTE(in, i) ((uint32_t)(in)[(i)])
#include "kernels/mq_decoder.h"
#include "kernels/portable.h"
#include "kernels/t1_common.h"

namespace gail {

namespace {

// Per-sample state, as in the reference encoder: flags plus the counts of significant
// neighbors (horizontal 0..2, vertical 0..2, diagonal 0..4), so the zero-coding context
// is one table lookup and "no significant neighbor" one mask test.
enum : uint16_t {
    F_SIG = 1,
    F_SGN = 2,
    F_VIS = 4,
    F_REF = 8,
    F_H1 = 1 << 4,
    F_V1 = 1 << 6,
    F_D1 = 1 << 8,
    F_NBR = 0x7F0,
};

struct ZcTable {
    uint8_t ctx[4][128];
    ZcTable() {
        for (int o = 0; o < 4; ++o)
            for (int n = 0; n < 128; ++n) {
                const int h = std::min(n & 3, 2), v = std::min((n >> 2) & 3, 2), d = std::min(n >> 4, 4);
                ctx[o][n] = (uint8_t)t1_zc_ctx(h, v, d, o);
            }
    }
};
const ZcTable g_zc;

thread_local std::vector<uint16_t> g_flags;

class Decoder {
public:
    Decoder(const uint8_t* data, uint32_t len, int w, int h, int orient, uint32_t* out)
        : in_(data), w_(w), h_(h), stride_(w + 2), zc_(g_zc.ctx[orient]), flags_(g_flags), out_(out) {
        flags_.assign((size_t)(w + 2) * (h + 2), 0);
        for (int y = 0; y < h; ++y) std::fill(out + y * kCblkSize, out + y * kCblkSize + w, 0u);
        mqd_init(mq_, in_, 0, len);
    }

    void run(int nbps, int passes) {
        int done = 0;
        for (int p = nbps - 1; p >= 0 && done < passes; --p) {
            for (int type = (p == nbps - 1 ? 2 : 0); type < 3 && done < passes; ++type, ++done) {
                if (type == 0) spp(p);
                else if (type == 1) mrp(p);
                else cup(p);
            }
            for (auto& f : flags_) f &= (uint16_t)~F_VIS;
        }
    }

private:
    uint16_t& at(int x, int y) { return flags_[(size_t)(y + 1) * stride_ + x + 1]; }
    uint16_t fl(int x, int y) const { return flags_[(size_t)(y + 1) * stride_ + x + 1]; }
    int chi(int x, int y) const {
        const uint16_t f = fl(x, y);
        return (f & F_SIG) ? ((f & F_SGN) ? -1 : 1) : 0;
    }
    uint32_t decode(uint32_t cx) { return mqd_decode(mq_, cx, in_); }
    uint32_t zc(uint16_t f) const { return zc_[(f >> 4) & 0x7F]; }

    // The sample's most significant bit is in plane p: read its sign (D.3.2), tell its
    // neighbors, and set its value to that bit plus the mid-point of the planes below.
    void becomeSig(int x, int y, int p) {
        const int hc = std::clamp(chi(x - 1, y) + chi(x + 1, y), -1, 1);
        const int vc = std::clamp(chi(x, y - 1) + chi(x, y + 1), -1, 1);
        const uint32_t sc = t1_sc_ctx(hc, vc);
        const uint32_t s = decode(sc & 0xFF) ^ (sc >> 8);
        uint16_t* f = &at(x, y);
        f[0] |= (uint16_t)(F_SIG | (s ? F_SGN : 0));
        f[-1] += F_H1;
        f[1] += F_H1;
        f[-stride_] += F_V1;
        f[stride_] += F_V1;
        f[-stride_ - 1] += F_D1;
        f[-stride_ + 1] += F_D1;
        f[stride_ - 1] += F_D1;
        f[stride_ + 1] += F_D1;
        out_[y * kCblkSize + x] = (s << 31) | (3u << p);
    }

    uint16_t stripeAny(int x, int y0, int ylen) const {
        const uint16_t* f = &flags_[(size_t)(y0 + 1) * stride_ + x + 1];
        uint16_t any = f[0];
        for (int k = 1; k < ylen; ++k) any |= f[k * stride_];
        return any;
    }

    // Significance propagation pass (D.3.1)
    void spp(int p) {
        for (int y0 = 0; y0 < h_; y0 += 4) {
            const int y1 = std::min(y0 + 4, h_);
            for (int x = 0; x < w_; ++x) {
                if (!(stripeAny(x, y0, y1 - y0) & F_NBR)) continue;
                for (int y = y0; y < y1; ++y) {
                    const uint16_t f = fl(x, y);
                    if ((f & F_SIG) || !(f & F_NBR)) continue;
                    const uint32_t b = decode(zc(f));
                    at(x, y) |= F_VIS;
                    if (b) becomeSig(x, y, p);
                }
            }
        }
    }

    // Magnitude refinement pass (D.3.3): the bit of plane p replaces the mid-point guess
    // made one plane up, which in the doubled value is a step of 2^p either way.
    void mrp(int p) {
        for (int y0 = 0; y0 < h_; y0 += 4) {
            const int y1 = std::min(y0 + 4, h_);
            for (int x = 0; x < w_; ++x) {
                if (!(stripeAny(x, y0, y1 - y0) & F_SIG)) continue;
                for (int y = y0; y < y1; ++y) {
                    const uint16_t f = fl(x, y);
                    if (!(f & F_SIG) || (f & F_VIS)) continue;
                    const uint32_t cx = (f & F_REF) ? 16 : ((f & F_NBR) ? 15 : 14);
                    const uint32_t b = decode(cx);
                    at(x, y) |= F_REF;
                    out_[y * kCblkSize + x] += b ? (1u << p) : (0u - (1u << p));
                }
            }
        }
    }

    // Cleanup pass (D.3.4), with its run-length mode for columns of four untouched samples
    void cup(int p) {
        for (int y0 = 0; y0 < h_; y0 += 4) {
            const int ylen = std::min(4, h_ - y0);
            for (int x = 0; x < w_; ++x) {
                int r = 0;
                if (ylen == 4) {
                    const uint16_t* f = &at(x, y0);
                    const uint16_t any = f[0] | f[stride_] | f[2 * stride_] | f[3 * stride_];
                    if (!(any & (F_SIG | F_VIS | F_NBR))) {
                        if (!decode(MQ_CTX_RL)) continue;
                        int first = (int)decode(MQ_CTX_UNI) << 1;
                        first |= (int)decode(MQ_CTX_UNI);
                        becomeSig(x, y0 + first, p);
                        r = first + 1;
                    }
                }
                for (; r < ylen; ++r) {
                    const int y = y0 + r;
                    const uint16_t f = fl(x, y);
                    if (f & (F_SIG | F_VIS)) continue;
                    if (decode(zc(f))) becomeSig(x, y, p);
                }
            }
        }
    }

    const uint8_t* in_;
    int w_, h_, stride_;
    const uint8_t* zc_;
    std::vector<uint16_t>& flags_;
    uint32_t* out_;
    MqDec mq_;
};

}  // namespace

void t1DecodeBlock(const uint8_t* data, uint32_t len, int w, int h, int orient, int nbps, int passes, uint32_t* out) {
    Decoder d(data, len, w, h, orient, out);
    d.run(nbps, passes);
}

}  // namespace gail
