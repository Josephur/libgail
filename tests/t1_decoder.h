// SPDX-License-Identifier: MIT
// Minimal EBCOT Tier-1 decoder used only for testing the encoder: decodes the
// first `passes` coding passes of a code-block codeword (padded with 0xFF like a
// standard decoder) and returns the reconstructed magnitude bits and signs.
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

#include "kernels/mq_encoder.h"
#include "kernels/t1_common.h"

namespace gail::test {

class MqDec {
public:
    MqDec(const uint8_t* d, size_t len) : buf_(d, d + len) {
        buf_.push_back(0xFF);
        buf_.push_back(0xFF);
        bp_ = 0;
        c_ = (uint32_t)buf_[0] << 16;
        bytein();
        c_ <<= 7;
        ct_ -= 7;
        a_ = 0x8000;
        for (int i = 0; i < MQ_NUM_CTX; ++i) ctx_[i] = 0;
        ctx_[0] = 4;
        ctx_[MQ_CTX_RL] = 3;
        ctx_[MQ_CTX_UNI] = 46;
    }
    uint32_t decode(uint32_t cx) {
        uint32_t st = ctx_[cx], idx = st & 63, mps = st >> 6;
        uint32_t t = MQ_TABLE[idx], qe = t & 0xFFFF;
        uint32_t d;
        a_ -= qe;
        if ((c_ >> 16) < qe) {
            if (a_ < qe) {
                a_ = qe;
                d = mps;
                ctx_[cx] = ((t >> 16) & 63) | (mps << 6);
            } else {
                a_ = qe;
                d = 1 - mps;
                ctx_[cx] = ((t >> 22) & 63) | ((mps ^ (t >> 28)) << 6);
            }
            renorm();
        } else {
            c_ -= qe << 16;
            if ((a_ & 0x8000) == 0) {
                if (a_ < qe) {
                    d = 1 - mps;
                    ctx_[cx] = ((t >> 22) & 63) | ((mps ^ (t >> 28)) << 6);
                } else {
                    d = mps;
                    ctx_[cx] = ((t >> 16) & 63) | (mps << 6);
                }
                renorm();
            } else {
                d = mps;
            }
        }
        return d;
    }

private:
    void bytein() {
        if (buf_[bp_] == 0xFF) {
            if (bp_ + 1 >= buf_.size() || buf_[bp_ + 1] > 0x8F) {
                c_ += 0xFF00;
                ct_ = 8;
            } else {
                ++bp_;
                c_ += (uint32_t)buf_[bp_] << 9;
                ct_ = 7;
            }
        } else {
            ++bp_;
            c_ += (uint32_t)(bp_ < buf_.size() ? buf_[bp_] : 0xFF) << 8;
            ct_ = 8;
        }
    }
    void renorm() {
        do {
            if (ct_ == 0) bytein();
            a_ <<= 1;
            c_ <<= 1;
            --ct_;
        } while ((a_ & 0x8000) == 0);
    }
    std::vector<uint8_t> buf_;
    size_t bp_;
    uint32_t a_, c_;
    int ct_ = 0;
    uint32_t ctx_[MQ_NUM_CTX];
};

// Decoded state: per coefficient, magnitude bits (relative to bit 0 of q) and sign.
struct T1Decoded {
    std::vector<uint32_t> mag;
    std::vector<uint8_t> sign, sig;
};

inline T1Decoded t1Decode(const uint8_t* data, size_t len, int w, int h, int orient, int nbps, int passes) {
    MqDec mq(data, len);
    T1Decoded r;
    r.mag.assign(w * h, 0);
    r.sign.assign(w * h, 0);
    r.sig.assign(w * h, 0);
    std::vector<uint8_t> vis(w * h, 0), ref(w * h, 0);
    auto S = [&](int x, int y) -> int { return (x >= 0 && y >= 0 && x < w && y < h) ? r.sig[y * w + x] : 0; };
    auto chi = [&](int x, int y) -> int { return S(x, y) ? (r.sign[y * w + x] ? -1 : 1) : 0; };
    auto zc = [&](int x, int y) {
        return t1_zc_ctx(S(x - 1, y) + S(x + 1, y), S(x, y - 1) + S(x, y + 1),
                         S(x - 1, y - 1) + S(x + 1, y - 1) + S(x - 1, y + 1) + S(x + 1, y + 1), orient);
    };
    auto sign = [&](int x, int y) {
        int hc = std::clamp(chi(x - 1, y) + chi(x + 1, y), -1, 1);
        int vc = std::clamp(chi(x, y - 1) + chi(x, y + 1), -1, 1);
        uint32_t sc = t1_sc_ctx(hc, vc);
        r.sign[y * w + x] = (uint8_t)(mq.decode(sc & 0xFF) ^ (sc >> 8));
    };
    int done = 0;
    for (int p = nbps - 1; p >= 0 && done < passes; --p) {
        for (int type = (p == nbps - 1 ? 2 : 0); type < 3 && done < passes; ++type, ++done) {
            for (int y0 = 0; y0 < h; y0 += 4)
                for (int x = 0; x < w; ++x) {
                    const int ylen = std::min(4, h - y0);
                    int k = 0;
                    if (type == 2 && ylen == 4) {
                        bool run = true;
                        for (int j = 0; j < 4; ++j) {
                            const int i = (y0 + j) * w + x;
                            if (r.sig[i] || vis[i] || zc(x, y0 + j)) run = false;
                        }
                        if (run) {
                            if (!mq.decode(MQ_CTX_RL)) continue;
                            int first = (int)mq.decode(MQ_CTX_UNI) << 1;
                            first |= (int)mq.decode(MQ_CTX_UNI);
                            const int y = y0 + first;
                            r.mag[y * w + x] |= 1u << p;
                            sign(x, y);
                            r.sig[y * w + x] = 1;
                            k = first + 1;
                        }
                    }
                    for (; k < ylen; ++k) {
                        const int y = y0 + k, i = y * w + x;
                        if (type == 0) {
                            if (r.sig[i]) continue;
                            const uint32_t cx = zc(x, y);
                            if (!cx) continue;
                            vis[i] = 1;
                            if (mq.decode(cx)) {
                                r.mag[i] |= 1u << p;
                                sign(x, y);
                                r.sig[i] = 1;
                            }
                        } else if (type == 1) {
                            if (!r.sig[i] || vis[i]) continue;
                            const uint32_t cx = ref[i] ? 16 : (zc(x, y) ? 15 : 14);
                            if (mq.decode(cx)) r.mag[i] |= 1u << p;
                            ref[i] = 1;
                        } else {
                            if (r.sig[i] || vis[i]) continue;
                            if (mq.decode(zc(x, y))) {
                                r.mag[i] |= 1u << p;
                                sign(x, y);
                                r.sig[i] = 1;
                            }
                        }
                    }
                }
        }
        std::fill(vis.begin(), vis.end(), 0);
    }
    return r;
}

}  // namespace gail::test
