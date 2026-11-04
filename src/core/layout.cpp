// SPDX-License-Identifier: MIT
#include "core/layout.h"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dwt_cpu.h"

namespace gail {

const char* profileName(Profile p) { return p == Profile::Cinema4K ? "CINEMA4K" : "CINEMA2K"; }

namespace {

// 1-D synthesis basis norm for a low (high=false) or high band at `level`.
double synth1d(int level, bool high) {
    const int n = 64 << level;
    std::vector<double> sig(n, 0.0), tmp(n);
    const int bandLen = n >> level;
    const int pos = high ? bandLen + bandLen / 2 : bandLen / 2;
    sig[pos] = 1.0;
    for (int l = level; l >= 1; --l) {
        const int len = n >> (l - 1);
        idwt97_1d(sig.data(), tmp.data(), len);
        std::copy(tmp.begin(), tmp.begin() + len, sig.begin());
    }
    double e = 0;
    for (double v : sig) e += v * v;
    return std::sqrt(e);
}

}  // namespace

double synthesisNorm(int level, int orient) {
    static std::mutex m;
    static std::map<std::pair<int, int>, double> cache;
    std::lock_guard<std::mutex> lock(m);
    auto key = std::make_pair(level, orient);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const bool hx = orient == 1 || orient == 3;
    const bool hy = orient == 2 || orient == 3;
    const double v = synth1d(level, hx) * synth1d(level, hy);
    cache[key] = v;
    return v;
}

double ictSynthesisNorm(int comp) {
    // Inverse ICT (Annex G.3): R = Y + 1.402 Cr; G = Y - 0.34413 Cb - 0.71414 Cr; B = Y + 1.772 Cb
    static const double cols[3][3] = {{1, 1, 1}, {0, -0.34413, 1.772}, {1.402, -0.71414, 0}};
    double e = 0;
    for (int i = 0; i < 3; ++i) e += cols[comp][i] * cols[comp][i];
    return std::sqrt(e);
}

FrameLayout makeLayout(int width, int height, Profile profile, double baseStep) {
    FrameLayout L;
    L.profile = profile;
    L.width = width;
    L.height = height;
    L.baseStep = baseStep;
    const bool fourk = profile == Profile::Cinema4K;
    if (fourk ? (width > 4096 || height > 2160) : (width > 2048 || height > 1080))
        throw std::runtime_error("image size exceeds the DCI container for this profile");
    if (width < 64 || height < 64) throw std::runtime_error("image too small");
    L.levels = fourk ? 6 : 5;
    L.guardBits = fourk ? 2 : 1;

    const int NL = L.levels;
    std::vector<int> ws(NL + 1), hs(NL + 1);
    ws[0] = width;
    hs[0] = height;
    for (int l = 1; l <= NL; ++l) {
        ws[l] = (ws[l - 1] + 1) / 2;
        hs[l] = (hs[l - 1] + 1) / 2;
    }

    int cblkCount = 0;
    for (int c = 0; c < kNumComps; ++c) {
        for (int r = 0; r <= NL; ++r) {
            const int norients = r == 0 ? 1 : 3;
            for (int o = 0; o < norients; ++o) {
                Subband b;
                b.comp = c;
                b.res = r;
                b.orient = r == 0 ? 0 : o + 1;
                b.level = r == 0 ? NL : NL - r + 1;
                const int n = b.level;
                const bool hx = b.orient == 1 || b.orient == 3;
                const bool hy = b.orient == 2 || b.orient == 3;
                b.px0 = hx ? ws[n] : 0;
                b.py0 = hy ? hs[n] : 0;
                b.w = hx ? ws[n - 1] - ws[n] : ws[n];
                b.h = hy ? hs[n - 1] - hs[n] : hs[n];
                b.ncbx = (b.w + kCblkSize - 1) / kCblkSize;
                b.ncby = (b.h + kCblkSize - 1) / kCblkSize;
                b.cblkBase = cblkCount;
                cblkCount += b.ncbx * b.ncby;

                // Quantization: equal image-domain MSE contribution per coefficient.
                const double norm = synthesisNorm(n, b.orient);
                const int gain = b.orient == 0 ? 0 : (b.orient == 3 ? 2 : 1);
                const int Rb = kBitDepth + gain;
                double target = baseStep / norm;
                double x = target / std::ldexp(1.0, Rb);  // = 2^-eps (1 + mu/2^11)
                int eps = (int)std::ceil(-std::log2(x));
                if (std::ldexp(x, eps) < 1.0) ++eps;
                if (std::ldexp(x, eps) >= 2.0) --eps;
                int mu = (int)std::lround((std::ldexp(x, eps) - 1.0) * 2048.0);
                if (mu >= 2048) {
                    mu = 0;
                    --eps;
                }
                if (eps < 0 || eps > 31) throw std::runtime_error("quantization step out of range");
                b.eps = eps;
                b.mu = mu;
                b.delta = (float)(std::ldexp(1.0, Rb - eps) * (1.0 + mu / 2048.0));
                b.Mb = L.guardBits + eps - 1;
                if (b.Mb > 22) throw std::runtime_error("quantisation step too fine (more than 22 bitplanes)");
                const double wc = ictSynthesisNorm(c);
                b.weight = norm * norm * wc * wc;
                L.bands.push_back(b);
            }
        }
    }

    // QCD is shared: per-band steps must be identical across components (they are, since
    // they depend on level/orient only). Component weights only affect rate control.
    L.cblks.resize(cblkCount);
    for (int bi = 0; bi < (int)L.bands.size(); ++bi) {
        const Subband& b = L.bands[bi];
        for (int cy = 0; cy < b.ncby; ++cy)
            for (int cx = 0; cx < b.ncbx; ++cx) {
                CodeBlock& cb = L.cblks[b.cblkBase + cy * b.ncbx + cx];
                cb.band = bi;
                cb.x0 = cx * kCblkSize;
                cb.y0 = cy * kCblkSize;
                cb.w = std::min(kCblkSize, b.w - cb.x0);
                cb.h = std::min(kCblkSize, b.h - cb.y0);
            }
    }

    // Precincts: 2^7 at r=0, 2^8 elsewhere (DCI). In band coordinates every precinct is
    // 128x128, i.e. a 4x4 grid of code-blocks.
    for (int c = 0; c < kNumComps; ++c) {
        for (int r = 0; r <= NL; ++r) {
            Resolution R;
            R.comp = c;
            R.res = r;
            R.w = ws[NL - r];
            R.h = hs[NL - r];
            R.ppx = R.ppy = r == 0 ? 7 : 8;
            R.npx = (R.w + (1 << R.ppx) - 1) >> R.ppx;
            R.npy = (R.h + (1 << R.ppy) - 1) >> R.ppy;
            const int bandPrecLog2 = r == 0 ? R.ppx : R.ppx - 1;  // precinct size in band coords
            const int cbPerPrec = 1 << (bandPrecLog2 - kCblkLog2);
            R.precincts.resize((size_t)R.npx * R.npy);
            for (int py = 0; py < R.npy; ++py)
                for (int px = 0; px < R.npx; ++px) {
                    Precinct& P = R.precincts[(size_t)py * R.npx + px];
                    P.nb = r == 0 ? 1 : 3;
                    for (int k = 0; k < P.nb; ++k) {
                        const int bi = L.bandIndex(c, r, r == 0 ? 0 : k + 1);
                        const Subband& b = L.bands[bi];
                        PrecinctBand& pb = P.bands[k];
                        pb.band = bi;
                        pb.cbx0 = px * cbPerPrec;
                        pb.cby0 = py * cbPerPrec;
                        pb.ncbx = std::max(0, std::min(cbPerPrec, b.ncbx - pb.cbx0));
                        pb.ncby = std::max(0, std::min(cbPerPrec, b.ncby - pb.cby0));
                        if (pb.ncbx == 0 || pb.ncby == 0) pb.ncbx = pb.ncby = 0;
                    }
                }
            L.resolutions.push_back(std::move(R));
        }
    }
    return L;
}

Container parseContainer(const std::string& s) {
    if (s == "flat") return Container::Flat;
    if (s == "scope") return Container::Scope;
    if (s == "full") return Container::Full;
    throw std::runtime_error("unknown container: " + s);
}

void containerSize(Profile p, Container c, int& w, int& h) {
    const bool k4 = p == Profile::Cinema4K;
    switch (c) {
    case Container::Flat: w = k4 ? 3996 : 1998; h = k4 ? 2160 : 1080; break;
    case Container::Scope: w = k4 ? 4096 : 2048; h = k4 ? 1716 : 858; break;
    case Container::Full: w = k4 ? 4096 : 2048; h = k4 ? 2160 : 1080; break;
    }
}

}  // namespace gail
