// SPDX-License-Identifier: MIT
#include "core/layout.h"

#include <cmath>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/dwt_cpu.h"
#include "kernels/ict.h"

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
    // Columns of the inverse ICT (kernels/ict.h): what one unit of each component adds to R, G, B
    static const double cols[3][3] = {{1, 1, 1}, {0, ICT_G_Y1, ICT_B_Y1}, {ICT_R_Y2, ICT_G_Y2, 0}};
    double e = 0;
    for (int i = 0; i < 3; ++i) e += cols[comp][i] * cols[comp][i];
    return std::sqrt(e);
}

namespace {

// Rb of E.1: the component's bit depth plus the subband's gain (0 LL, 1 HL / LH, 2 HH).
int nominalRange(const Subband& b) { return kBitDepth + (b.orient == 0 ? 0 : (b.orient == 3 ? 2 : 1)); }

// Step size and bit-plane count from the signaled exponent and mantissa (E.1, E.2).
void applyStep(Subband& b, int guardBits, int eps, int mu) {
    b.eps = eps;
    b.mu = mu;
    b.delta = (float)(std::ldexp(1.0, nominalRange(b) - eps) * (1.0 + mu / 2048.0));
    b.Mb = guardBits + eps - 1;
    if (b.Mb > 22) throw std::runtime_error("quantization step too fine (more than 22 bitplanes)");
}

}  // namespace

FrameLayout makeGeometry(int width, int height, Profile profile, int levels, int guardBits) {
    FrameLayout L;
    L.profile = profile;
    L.width = width;
    L.height = height;
    L.levels = levels;
    L.guardBits = guardBits;

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
                L.bands.push_back(b);
            }
        }
    }

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

void setSignaledSteps(FrameLayout& L, const int* eps, const int* mu) {
    const int nb = L.bandsPerComp();
    for (int bi = 0; bi < (int)L.bands.size(); ++bi) {
        const int i = bi % nb;
        if (eps[i] < 0 || eps[i] > 31 || mu[i] < 0 || mu[i] > 2047) throw std::runtime_error("quantization step out of range");
        applyStep(L.bands[bi], L.guardBits, eps[i], mu[i]);
    }
}

FrameLayout makeLayout(int width, int height, Profile profile, double baseStep) {
    const bool fourk = profile == Profile::Cinema4K;
    if (fourk ? (width > 4096 || height > 2160) : (width > 2048 || height > 1080))
        throw std::runtime_error("image size exceeds the DCI container for this profile");
    if (width < 64 || height < 64) throw std::runtime_error("image too small");
    FrameLayout L = makeGeometry(width, height, profile, fourk ? 6 : 5, fourk ? 2 : 1);
    L.baseStep = baseStep;

    // QCD is shared: per-band steps must be identical across components (they are, since
    // they depend on level/orient only). Component weights only affect rate control.
    for (Subband& b : L.bands) {
        // Quantization: equal image-domain MSE contribution per coefficient.
        const double norm = synthesisNorm(b.level, b.orient);
        double target = baseStep / norm;
        double x = target / std::ldexp(1.0, nominalRange(b));  // = 2^-eps (1 + mu/2^11)
        int eps = (int)std::ceil(-std::log2(x));
        if (std::ldexp(x, eps) < 1.0) ++eps;
        if (std::ldexp(x, eps) >= 2.0) --eps;
        int mu = (int)std::lround((std::ldexp(x, eps) - 1.0) * 2048.0);
        if (mu >= 2048) {
            mu = 0;
            --eps;
        }
        if (eps < 0 || eps > 31) throw std::runtime_error("quantization step out of range");
        applyStep(b, L.guardBits, eps, mu);
        const double wc = ictSynthesisNorm(b.comp);
        b.weight = norm * norm * wc * wc;
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
