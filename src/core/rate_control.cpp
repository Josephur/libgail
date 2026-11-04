// SPDX-License-Identifier: MIT
#include "core/rate_control.h"

#include <algorithm>
#include <cmath>

#include "core/t1_ref.h"

namespace gail {

RateLimits dciLimits(double fps, double maxMbps, double compMbps) {
    RateLimits l;
    // Stay strictly below the limits (libdcp flags size > rint(Mbps*1e6/(8*fps))).
    l.maxTotal = (size_t)std::floor(maxMbps * 1e6 / (8.0 * fps)) - 16;
    l.maxComp = (size_t)std::floor(compMbps * 1e6 / (8.0 * fps)) - 16;
    return l;
}

namespace {

struct Hull {
    std::vector<uint32_t> start;   // per code-block offset into the arrays below
    std::vector<uint8_t> passes;   // passes at each hull point
    std::vector<uint32_t> rate;
    std::vector<double> slope;
    std::vector<double> dgain;     // distortion reduction from the previous hull point
};

Hull buildHull(const FrameLayout& L, const Tier1Frame& t1) {
    const int n = L.numCblks();
    Hull H;
    H.start.resize(n + 1);
    std::vector<uint32_t> R;
    std::vector<double> D;
    std::vector<int> pk;
    for (int i = 0; i < n; ++i) {
        H.start[i] = (uint32_t)H.passes.size();
        const CblkInfo& ci = t1.info[i];
        const int np = (int)ci.numPasses;
        if (np == 0) continue;
        const Subband& b = L.bands[L.cblks[i].band];
        const double scale = (double)b.delta * b.delta * b.weight;
        R.assign(1, 0);
        D.assign(1, 0.0);
        pk.assign(1, 0);
        double acc = 0;
        for (int k = 0; k < np; ++k) {
            acc += t1.dist(i, k) * passPlaneScale((int)ci.nbps, k) * scale;
            const uint32_t r = t1.len(i, k);
            if (acc <= D.back()) continue;
            while (pk.size() >= 2) {
                const size_t m = pk.size() - 1;
                const double sNew = r == R[m] ? INFINITY : (acc - D[m]) / (double)(r - R[m]);
                const double sLast = R[m] == R[m - 1] ? INFINITY : (D[m] - D[m - 1]) / (double)(R[m] - R[m - 1]);
                if (sNew >= sLast) {
                    R.pop_back();
                    D.pop_back();
                    pk.pop_back();
                } else {
                    break;
                }
            }
            R.push_back(r);
            D.push_back(acc);
            pk.push_back(k + 1);
        }
        for (size_t m = 1; m < pk.size(); ++m) {
            H.passes.push_back((uint8_t)pk[m]);
            H.rate.push_back(R[m]);
            const double dr = (double)R[m] - (double)R[m - 1];
            H.slope.push_back(dr <= 0 ? INFINITY : (D[m] - D[m - 1]) / dr);
            H.dgain.push_back(D[m] - D[m - 1]);
        }
    }
    H.start[n] = (uint32_t)H.passes.size();
    return H;
}

void select(const FrameLayout& L, const Hull& H, const double lambda[3], std::vector<uint8_t>& passes, int compMask) {
    const int n = L.numCblks();
    for (int i = 0; i < n; ++i) {
        const int c = L.bands[L.cblks[i].band].comp;
        if (!(compMask & (1 << c))) continue;
        uint8_t np = 0;
        for (uint32_t h = H.start[i]; h < H.start[i + 1]; ++h) {
            if (H.slope[h] < lambda[c]) break;
            np = H.passes[h];
        }
        passes[i] = np;
    }
}

}  // namespace

// Fast size model: data bytes come straight from the convex hulls; packet-header bytes
// are modelled as (calibrated header bytes) + beta * (change in included blocks) and
// re-calibrated against an exact Tier-2 measurement after each search. A handful of
// exact measurements replaces a full bisection of exact ones; the result is always
// verified exactly before it is returned.
namespace {
struct FastEval {
    double data[3] = {0, 0, 0};
    double incl[3] = {0, 0, 0};
};

// Hull points binned by log2(slope): data(l) and included-block counts become prefix
// sums, so the model costs O(1) per evaluation (resolution 1/1024 octave).
class SlopeHistogram {
public:
    SlopeHistogram(const FrameLayout& L, const Hull& H, double loLog, double hiLog) : lo_(loLog) {
        nb_ = std::max(16, (int)std::ceil((hiLog - loLog) * kPerOctave) + 2);
        for (int c = 0; c < 3; ++c) {
            rate_[c].assign(nb_ + 1, 0.0);
            incl_[c].assign(nb_ + 1, 0.0);
        }
        for (int i = 0; i < L.numCblks(); ++i) {
            const int c = L.bands[L.cblks[i].band].comp;
            uint32_t prev = 0;
            for (uint32_t h = H.start[i]; h < H.start[i + 1]; ++h) {
                const int b = bin(H.slope[h]);
                rate_[c][b] += (double)H.rate[h] - prev;
                if (h == H.start[i]) incl_[c][b] += 1;
                prev = H.rate[h];
            }
        }
        // suffix sums: totals over all points with bin >= b (slope >= bin start)
        for (int c = 0; c < 3; ++c)
            for (int b = nb_ - 1; b >= 0; --b) {
                rate_[c][b] += rate_[c][b + 1];
                incl_[c][b] += incl_[c][b + 1];
            }
    }
    FastEval eval(const double lam[3]) const {
        FastEval f;
        for (int c = 0; c < 3; ++c) {
            const int b = lam[c] <= 0 ? 0 : std::min(nb_, bin(lam[c]));
            f.data[c] = rate_[c][b];
            f.incl[c] = incl_[c][b];
        }
        return f;
    }

private:
    static constexpr double kPerOctave = 1024;
    int bin(double slope) const {
        if (!std::isfinite(slope)) return nb_ - 1;
        const int b = (int)std::floor((std::log2(slope) - lo_) * kPerOctave);
        return std::clamp(b, 0, nb_ - 1);
    }
    double lo_;
    int nb_;
    std::vector<double> rate_[3], incl_[3];
};
}  // namespace

RateResult allocateRate(const FrameLayout& L, const Tier1Frame& t1, const RateLimits& lim) {
    Hull H = buildHull(L, t1);
    RateResult res;
    res.passes.assign(L.numCblks(), 0);
    double lam[3] = {0, 0, 0};

    auto measure = [&](const double l[3]) {
        select(L, H, l, res.passes, 7);
        buildCodestream(L, t1, res.passes.data(), nullptr, &res.sizes);
        ++res.iterations;
        return res.sizes;
    };

    double maxSlope = 0, minSlope = INFINITY;
    for (double s : H.slope)
        if (std::isfinite(s) && s > 0) {
            maxSlope = std::max(maxSlope, s);
            minSlope = std::min(minSlope, s);
        }
    if (maxSlope == 0) maxSlope = minSlope = 1;
    const double loBound = std::log2(minSlope) - 1, hiBound = std::log2(maxSlope) + 1;
    const SlopeHistogram hist(L, H, loBound, hiBound);

    CodestreamSizes sz = measure(lam);
    // Header model, per component: hdr(l) ~ hdrCal + beta * (incl(l) - inclCal)
    double hdrCal[3], inclCal[3], beta[3];
    auto calibrate = [&](const double l[3], const CodestreamSizes& ex) {
        FastEval f = hist.eval(l);
        for (int c = 0; c < 3; ++c) {
            hdrCal[c] = (double)ex.comp[c] - f.data[c];
            inclCal[c] = f.incl[c];
            beta[c] = f.incl[c] > 0 ? std::max(1.0, std::min(8.0, hdrCal[c] / f.incl[c])) : 3.0;
        }
    };
    auto model = [&](const double l[3], double comp[3]) {
        FastEval f = hist.eval(l);
        double tot = (double)(sz.total - sz.comp[0] - sz.comp[1] - sz.comp[2]);  // main header + EOC
        for (int c = 0; c < 3; ++c) {
            comp[c] = f.data[c] + hdrCal[c] + beta[c] * (f.incl[c] - inclCal[c]);
            tot += comp[c];
        }
        return tot;
    };
    // Smallest threshold (log2) for which pred(l) holds, by bisection on the model.
    auto search = [&](auto pred, double lo, double hi) {
        for (int it = 0; it < 48 && hi - lo > 1e-5; ++it) {
            const double mid = 0.5 * (lo + hi);
            if (pred(mid)) hi = mid;
            else lo = mid;
        }
        return hi;
    };

    if (sz.total > lim.maxTotal) {
        calibrate(lam, sz);
        double target = (double)lim.maxTotal;
        for (int round = 0; round < 6; ++round) {
            const double lg = search(
                [&](double m) {
                    const double l = std::exp2(m), ll[3] = {l, l, l};
                    double comp[3];
                    return model(ll, comp) <= target;
                },
                loBound, hiBound);
            lam[0] = lam[1] = lam[2] = std::exp2(lg);
            sz = measure(lam);
            if (sz.total <= lim.maxTotal && sz.total + lim.maxTotal / 500 >= lim.maxTotal) break;
            calibrate(lam, sz);
        }
        // guarantee: nudge upwards until the exact size fits
        while (sz.total > lim.maxTotal) {
            for (double& l : lam) l *= 1.002;
            sz = measure(lam);
        }
    }

    // Per-component cap: raise that component's threshold until it fits.
    for (int c = 0; c < kNumComps; ++c) {
        if (sz.comp[c] <= lim.maxComp) continue;
        calibrate(lam, sz);
        for (int round = 0; round < 6; ++round) {
            const double lg = search(
                [&](double m) {
                    double ll[3] = {lam[0], lam[1], lam[2]};
                    ll[c] = std::exp2(m);
                    double comp[3];
                    model(ll, comp);
                    return comp[c] <= (double)lim.maxComp;
                },
                std::log2(std::max(lam[c], minSlope)), hiBound);
            lam[c] = std::exp2(lg);
            sz = measure(lam);
            if (sz.comp[c] <= lim.maxComp && sz.comp[c] + lim.maxComp / 500 >= lim.maxComp) break;
            calibrate(lam, sz);
        }
        while (sz.comp[c] > lim.maxComp) {
            lam[c] *= 1.002;
            sz = measure(lam);
        }
    }
    res.lambda = std::max({lam[0], lam[1], lam[2]});
    for (int c = 0; c < 3; ++c) res.lambdaComp[c] = lam[c];

    // Remaining distortion estimate (sum of excluded gains).
    double dist = 0;
    for (int i = 0; i < L.numCblks(); ++i)
        for (uint32_t h = H.start[i]; h < H.start[i + 1]; ++h)
            if (H.passes[h] > res.passes[i]) dist += H.dgain[h];
    res.distortion = dist;
    res.sizes = sz;
    return res;
}

}  // namespace gail
