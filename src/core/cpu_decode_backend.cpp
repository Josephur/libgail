// SPDX-License-Identifier: MIT
// CPU decode backend: the reference for the GPU decoders. Code-blocks, wavelet rows and
// column strips, and picture rows are spread over the shared thread pool.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>

#include "core/decode_backend.h"
#include "core/dwt_cpu.h"
#include "core/t1_dec.h"
#include "core/thread_pool.h"
#include "kernels/ict.h"

namespace gail {

namespace {

double msBetween(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

class CpuDecoderSlot : public DecoderSlot {
public:
    CpuDecoderSlot(const FrameLayout& L, const DecodeSettings& s, std::shared_ptr<ThreadPool> pool)
        : L_(L), s_(s), pool_(std::move(pool)) {
        if (s.reduce < 0 || s.reduce > L.levels) throw std::runtime_error("more resolution levels left out than the stream has");
        for (auto& p : planes_) p.resize((size_t)L.width * L.height);
        par_ = [this](int n, const std::function<void(int)>& f) { pool_->run(n, f); };
        decodedSize(L_, s_.reduce, outW_, outH_);
        // the code-blocks of the resolutions that are decoded
        for (int i = 0; i < L_.numCblks(); ++i)
            if (L_.bands[L_.cblks[i].band].res <= L_.levels - s_.reduce) wanted_.push_back(i);
    }

    void decode(const FrameLayout& frame, const uint8_t* data, size_t size, const std::vector<CblkCode>& blocks, void* dst,
                ptrdiff_t strideBytes, DecodeTimings* tm) override {
        if (!sameGeometry(frame, L_) || (int)blocks.size() != L_.numCblks())
            throw std::logic_error("frame does not match the layout the slot was made for");
        const auto t0 = std::chrono::steady_clock::now();
        float* pl[3] = {planes_[0].data(), planes_[1].data(), planes_[2].data()};
        const int W = L_.width;

        // Tier-1 and dequantization: each block's coefficients go straight to their place in
        // the wavelet plane (E.1.1.2, mid-point reconstruction is already in the decoded word)
        pool_->run((int)wanted_.size(), [&](int k) {
            const int i = wanted_[k];
            const CodeBlock& cb = L_.cblks[i];
            const Subband& b = L_.bands[cb.band];
            const CblkCode& code = blocks[i];
            float* plane = pl[b.comp] + (size_t)(b.py0 + cb.y0) * W + b.px0 + cb.x0;
            if (!code.passes || (size_t)code.offset + code.len > size) {
                for (int y = 0; y < cb.h; ++y) std::fill(plane + (size_t)y * W, plane + (size_t)y * W + cb.w, 0.f);
                return;
            }
            uint32_t q[kCblkArea];
            t1DecodeBlock(data + code.offset, code.len, cb.w, cb.h, b.orient, code.nbps, code.passes, q);
            const float half = 0.5f * frame.bands[cb.band].delta;
            for (int y = 0; y < cb.h; ++y) {
                const uint32_t* src = q + y * kCblkSize;
                float* row = plane + (size_t)y * W;
                for (int x = 0; x < cb.w; ++x) {
                    const float v = (float)(src[x] & 0x7FFFFFFFu) * half;
                    row[x] = (src[x] & 0x80000000u) ? -v : v;
                }
            }
        });
        const auto t1 = std::chrono::steady_clock::now();

        idwt97_2d(pl, 3, W, L_.height, L_.levels, s_.reduce, &par_);
        const auto t2 = std::chrono::steady_clock::now();

        // Inverse ICT (G.3), level shift, rounding and clamping to 12 bits
        const int shift = s_.highBits ? 4 : 0;
        pool_->run(outH_, [&](int y) {
            const float* a = pl[0] + (size_t)y * W;
            const float* bb = pl[1] + (size_t)y * W;
            const float* c = pl[2] + (size_t)y * W;
            uint16_t* out = (uint16_t*)((uint8_t*)dst + (ptrdiff_t)y * strideBytes);
            for (int x = 0; x < outW_; ++x) {
                const float y0 = a[x], y1 = bb[x], y2 = c[x];
                const float v[3] = {y0 + (float)ICT_R_Y2 * y2, y0 + (float)ICT_G_Y1 * y1 + (float)ICT_G_Y2 * y2,
                                    y0 + (float)ICT_B_Y1 * y1};
                for (int k = 0; k < 3; ++k) {
                    const int code = (int)std::floor(v[k] + 2048.5f);
                    out[3 * x + k] = (uint16_t)(std::clamp(code, 0, 4095) << shift);
                }
            }
        });
        if (tm) {
            const auto t3 = std::chrono::steady_clock::now();
            tm->t1 = msBetween(t0, t1);
            tm->dwt = msBetween(t1, t2);
            tm->color = msBetween(t2, t3);
            tm->total = msBetween(t0, t3);
        }
    }

private:
    const FrameLayout L_;
    const DecodeSettings s_;
    std::shared_ptr<ThreadPool> pool_;
    ParallelFor par_;
    std::vector<float> planes_[3];
    std::vector<int> wanted_;
    int outW_ = 0, outH_ = 0;
};

}  // namespace

bool backendDecodes(const std::string& backend) { return backend == "cpu"; }

std::unique_ptr<DecoderSlot> createCpuDecoderSlot(const FrameLayout& L, const DecodeSettings& s, std::shared_ptr<ThreadPool> pool) {
    return std::make_unique<CpuDecoderSlot>(L, s, std::move(pool));
}

}  // namespace gail
