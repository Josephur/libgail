// SPDX-License-Identifier: MIT
#include "core/t1_ref.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#define MQ_OUT_T uint8_t*
#define MQ_PUT(o, i, v) ((o)[(i)] = (uint8_t)(v))
#define MQ_GET(o, i) ((uint32_t)(o)[(i)])
#include "kernels/portable.h"
#include "kernels/mq_encoder.h"
#include "kernels/t1_common.h"

namespace gail {

namespace {

// Per-sample state word. Besides the usual flags it carries the counts of significant
// neighbours (horizontal 0..2, vertical 0..2, diagonal 0..4), maintained when a sample
// becomes significant; the zero-coding context is then one table lookup instead of eight
// neighbour reads, and "no significant neighbour" is a single mask test.
enum : uint16_t {
    F_SIG = 1,
    F_SGN = 2,
    F_VIS = 4,
    F_REF = 8,
    F_H1 = 1 << 4,   // + one horizontal neighbour
    F_V1 = 1 << 6,   // + one vertical neighbour
    F_D1 = 1 << 8,   // + one diagonal neighbour
    F_NBR = 0x7F0,   // all neighbour counts
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

struct Scratch {
    std::vector<uint16_t> flags;
    std::vector<uint8_t> buf;
};
thread_local Scratch g_scratch;

class Coder {
public:
    Coder(const uint32_t* coeffs, int w, int h, int orient, T1Trace* trace, const T1EarlyStop* stop, Scratch& s)
        : c_(coeffs), w_(w), h_(h), stride_(w + 2), zc_(g_zc.ctx[orient]), flags_(s.flags), buf_(s.buf), trace_(trace) {
        if (stop && stop->slopeThresh > 0.f) {
            track_ = true;
            slopeThresh_ = stop->slopeThresh;
            distScale_ = stop->distScale;
        }
        flags_.assign((size_t)(w + 2) * (h + 2), 0);
        maxMag_ = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const uint32_t v = c_[y * kCblkSize + x];
                if (v & T1_SIGN_BIT) at(x, y) = F_SGN;
                maxMag_ = std::max(maxMag_, (v & ~T1_SIGN_BIT) >> T1_FRAC_BITS);
            }
    }

    void run(T1Result& out, int maxPasses) {
        out.nbps = FIND_MSB(maxMag_) + 1;
        out.numPasses = 0;
        out.data.clear();
        if (!out.nbps) return;
        if (3 * out.nbps - 2 > kMaxPasses) throw std::runtime_error("too many bitplanes in code-block");

        // Every byte below the final length is written by the coder before it is read.
        const size_t need = (size_t)kCblkArea * 4 * out.nbps + 64;
        if (buf_.size() < need) buf_.resize(need);
        mq_init(mq_);
        MqSnap snaps[kMaxPasses];
        int np = 0;
        for (int p = out.nbps - 1; p >= 0 && np < maxPasses; --p) {
            int32_t planeDist = 0;
            for (int type = (p == out.nbps - 1 ? 2 : 0); type < 3 && np < maxPasses; ++type) {
                dist_ = 0;
                if (type == 0) spp(p);
                else if (type == 1) mrp(p);
                else cup(p);
                snaps[np] = mq_snapshot(mq_);
                out.passDist[np] = dist_;
                planeDist += dist_;
                if (trace_) trace_->passEnd.push_back((uint32_t)trace_->symbols.size());
                ++np;
            }
            for (auto& f : flags_) f &= (uint16_t)~F_VIS;
            if (track_ && stopAfterPlane(p, planeDist)) break;
        }
        const uint32_t len = mq_flush(mq_, buf_.data());
        out.numPasses = np;
        uint32_t prev = 1;
        for (int k = 0; k < np; ++k) {
            uint32_t L = k == np - 1 ? len : mq_trunc_len(snaps[k], buf_.data(), len, prev);
            out.passLen[k] = L;
            prev = std::max(prev, L);
        }
        out.data.assign(buf_.begin(), buf_.begin() + len);
    }

private:
    uint16_t& at(int x, int y) { return flags_[(size_t)(y + 1) * stride_ + x + 1]; }
    uint16_t fl(int x, int y) const { return flags_[(size_t)(y + 1) * stride_ + x + 1]; }
    uint32_t mag(int x, int y) const { return c_[y * kCblkSize + x] & ~T1_SIGN_BIT; }
    int chi(int x, int y) const {
        const uint16_t f = fl(x, y);
        return (f & F_SIG) ? ((f & F_SGN) ? -1 : 1) : 0;
    }

    void encode(uint32_t cx, uint32_t d) {
        mq_encode(mq_, cx, d, buf_.data());
        if (track_) hist_[cx] += d ? 0x10000u : 1u;
        if (trace_) trace_->symbols.push_back((uint8_t)((cx << 1) | d));
    }

    // Empirical-entropy estimate (bytes) of the plane's symbols: an adaptive arithmetic
    // coder gets close to sum_ctx n * H(n1 / n), plus a little per active context.
    float planeBytesEstimate() {
        float bits = 0.f;
        for (int cx = 0; cx < MQ_NUM_CTX; ++cx) {
            const float n0 = (float)(hist_[cx] & 0xFFFFu), n1 = (float)(hist_[cx] >> 16), n = n0 + n1;
            if (n > 0.f)
                bits += -(n0 * std::log2(std::max(n0, 0.5f) / n) + n1 * std::log2(std::max(n1, 0.5f) / n)) + 2.f;
            hist_[cx] = 0;
        }
        return bits * 0.125f + 1.f;
    }

    // Remaining squared error (in units of delta^2) once bitplanes >= p are known, with
    // mid-point reconstruction.
    float residualEnergy(int p) const {
        float e = 0.f;
        const float step = std::exp2((float)p);
        for (int y = 0; y < h_; ++y)
            for (int x = 0; x < w_; ++x) {
                const uint32_t m = mag(x, y);
                const float v = (float)m * (1.f / 64.f);
                const uint32_t q = m >> (p + T1_FRAC_BITS);
                const float rec = q == 0 ? 0.f : ((float)q + 0.5f) * step;
                e += (v - rec) * (v - rec);
            }
        return e;
    }

    // Early termination after bitplane p (see T1EarlyStop). Resets the symbol histogram.
    bool stopAfterPlane(int p, int32_t planeDist) {
        const float bytes = planeBytesEstimate();
        if (p <= 0) return false;
        const float dAbs = (float)planeDist * std::exp2((float)(2 * p)) * (1.f / 4096.f) * distScale_;
        if (!(dAbs < slopeThresh_ * bytes)) return false;
        return residualEnergy(p) * distScale_ < slopeThresh_ * bytes;
    }

    uint32_t zc(uint16_t f) const { return zc_[(f >> 4) & 0x7F]; }

    void codeSign(int x, int y) {
        const int hc = std::clamp(chi(x - 1, y) + chi(x + 1, y), -1, 1);
        const int vc = std::clamp(chi(x, y - 1) + chi(x, y + 1), -1, 1);
        const uint32_t sc = t1_sc_ctx(hc, vc);
        const uint32_t s = (fl(x, y) & F_SGN) ? 1 : 0;
        encode(sc & 0xFF, s ^ (sc >> 8));
    }

    uint32_t bit(int x, int y, int p) const { return (mag(x, y) >> (p + T1_FRAC_BITS)) & 1; }
    uint32_t wbits(int x, int y, int p) const { return (mag(x, y) >> p) & 0x7F; }

    void becomeSig(int x, int y, int p) {
        codeSign(x, y);
        uint16_t* f = &at(x, y);
        f[0] |= F_SIG;
        f[-1] += F_H1;
        f[1] += F_H1;
        f[-stride_] += F_V1;
        f[stride_] += F_V1;
        f[-stride_ - 1] += F_D1;
        f[-stride_ + 1] += F_D1;
        f[stride_ - 1] += F_D1;
        f[stride_ + 1] += F_D1;
        dist_ += t1_dist_sig(wbits(x, y, p));
    }

    // OR of the flags of stripe column x (rows y0 .. y0+3, or fewer at the bottom).
    uint16_t stripeAny(int x, int y0, int ylen) const {
        const uint16_t* f = &flags_[(size_t)(y0 + 1) * stride_ + x + 1];
        uint16_t any = f[0];
        for (int k = 1; k < ylen; ++k) any |= f[k * stride_];
        return any;
    }

    void spp(int p) {
        for (int y0 = 0; y0 < h_; y0 += 4) {
            const int y1 = std::min(y0 + 4, h_);
            for (int x = 0; x < w_; ++x) {
                if (!(stripeAny(x, y0, y1 - y0) & F_NBR)) continue;  // nothing to code in this column
                for (int y = y0; y < y1; ++y) {
                    const uint16_t f = fl(x, y);
                    if ((f & F_SIG) || !(f & F_NBR)) continue;
                    const uint32_t b = bit(x, y, p);
                    encode(zc(f), b);
                    at(x, y) |= F_VIS;
                    if (b) becomeSig(x, y, p);
                }
            }
        }
    }

    void mrp(int p) {
        for (int y0 = 0; y0 < h_; y0 += 4) {
            const int y1 = std::min(y0 + 4, h_);
            for (int x = 0; x < w_; ++x) {
                if (!(stripeAny(x, y0, y1 - y0) & F_SIG)) continue;  // no significant sample
                for (int y = y0; y < y1; ++y) {
                    const uint16_t f = fl(x, y);
                    if (!(f & F_SIG) || (f & F_VIS)) continue;
                    const uint32_t cx = (f & F_REF) ? 16 : ((f & F_NBR) ? 15 : 14);
                    encode(cx, bit(x, y, p));
                    at(x, y) |= F_REF;
                    dist_ += t1_dist_ref(wbits(x, y, p));
                }
            }
        }
    }

    void cup(int p) {
        for (int y0 = 0; y0 < h_; y0 += 4) {
            const int ylen = std::min(4, h_ - y0);
            for (int x = 0; x < w_; ++x) {
                int r = 0;
                if (ylen == 4) {
                    const uint16_t* f = &at(x, y0);
                    const uint16_t any = f[0] | f[stride_] | f[2 * stride_] | f[3 * stride_];
                    if (!(any & (F_SIG | F_VIS | F_NBR))) {
                        int first = -1;
                        for (int k = 0; k < 4; ++k)
                            if (bit(x, y0 + k, p)) {
                                first = k;
                                break;
                            }
                        if (first < 0) {
                            encode(MQ_CTX_RL, 0);
                            continue;
                        }
                        encode(MQ_CTX_RL, 1);
                        encode(MQ_CTX_UNI, (uint32_t)first >> 1);
                        encode(MQ_CTX_UNI, (uint32_t)first & 1);
                        becomeSig(x, y0 + first, p);
                        r = first + 1;
                    }
                }
                for (; r < ylen; ++r) {
                    const int y = y0 + r;
                    const uint16_t f = fl(x, y);
                    if (f & (F_SIG | F_VIS)) continue;
                    const uint32_t b = bit(x, y, p);
                    encode(zc(f), b);
                    if (b) becomeSig(x, y, p);
                }
            }
        }
    }

    const uint32_t* c_;
    int w_, h_, stride_;
    const uint8_t* zc_;
    std::vector<uint16_t>& flags_;
    std::vector<uint8_t>& buf_;
    uint32_t maxMag_ = 0;
    MqEnc mq_;
    int32_t dist_ = 0;
    T1Trace* trace_;
    bool track_ = false;
    float slopeThresh_ = 0.f, distScale_ = 1.f;
    uint32_t hist_[MQ_NUM_CTX] = {};  // per-context symbol counts of the current plane: zeros | ones << 16
};

}  // namespace

void t1EncodeRef(const uint32_t* coeffs, int w, int h, int orient, T1Result& out, T1Trace* trace, int maxPasses,
                 const T1EarlyStop* stop) {
    Coder coder(coeffs, w, h, orient, trace, stop, g_scratch);
    coder.run(out, maxPasses);
}

}  // namespace gail
