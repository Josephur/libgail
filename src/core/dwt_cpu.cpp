// SPDX-License-Identifier: MIT
#include "core/dwt_cpu.h"

#include <algorithm>
#include <cstring>
#include <vector>

namespace gail {

namespace {

// Whole-sample symmetric extension index for a signal of length n (n >= 2).
inline int mirror(int i, int n) {
    const int period = 2 * (n - 1);
    i %= period;
    if (i < 0) i += period;
    return i < n ? i : period - i;
}

// x[i] += coef * (x[i-1] + x[i+1]) for all i with i % 2 == parity (n >= 2). The interior
// runs without index arithmetic; only the two ends use the symmetric extension, where the
// mirrored neighbour is x[1] (left) and x[n-2] (right). Same operations in the same order
// as the plain mirrored loop, so the results are bit-identical.
template <typename T> inline void lift(T* x, int n, int parity, T coef) {
    int i = parity;
    if (i == 0) {
        x[0] += coef * (x[1] + x[1]);
        i = 2;
    }
    const int last = n - 1;
    for (; i < last; i += 2) x[i] += coef * (x[i - 1] + x[i + 1]);
    if (i == last) x[last] += coef * (x[last - 1] + x[last - 1]);
}

// Lifting on a strip of S interleaved columns: x[i*S + k] for sample i, column k.
template <int S> inline void liftStrip(float* x, int n, int parity, float coef) {
    auto step = [&](int i, int a, int b) {
        float* xi = x + (size_t)i * S;
        const float* xa = x + (size_t)a * S;
        const float* xb = x + (size_t)b * S;
        for (int k = 0; k < S; ++k) xi[k] += coef * (xa[k] + xb[k]);
    };
    int i = parity;
    if (i == 0) {
        step(0, 1, 1);
        i = 2;
    }
    const int last = n - 1;
    for (; i < last; i += 2) step(i, i - 1, i + 1);
    if (i == last) step(last, last - 1, last - 1);
}

// Forward transform of one row in place (tmp: w floats of scratch).
void fdwtRow(float* row, int w, float* tmp) {
    if (w < 2) return;
    lift<float>(row, w, 1, (float)kAlpha);
    lift<float>(row, w, 0, (float)kBeta);
    lift<float>(row, w, 1, (float)kGamma);
    lift<float>(row, w, 0, (float)kDelta);
    const int sn = (w + 1) / 2;
    for (int i = 0; i < sn; ++i) tmp[i] = row[2 * i] * (float)(1.0 / kK);
    for (int i = 0; i < w / 2; ++i) tmp[sn + i] = row[2 * i + 1] * (float)kK;
    std::memcpy(row, tmp, sizeof(float) * (size_t)w);
}

constexpr int kStrip = 16;  // columns transformed together (one cache line per row)

// Forward transform of columns [x0, x0 + nx) of the top-left w x h region, nx <= kStrip
// (strip: h * kStrip floats of scratch). Columns are gathered into the strip, lifted
// together and scattered back de-interleaved.
void fdwtColumns(float* plane, int width, int h, int x0, int nx, float* strip) {
    if (h < 2) return;
    for (int y = 0; y < h; ++y) {
        const float* src = plane + (size_t)y * width + x0;
        float* dst = strip + (size_t)y * kStrip;
        for (int k = 0; k < nx; ++k) dst[k] = src[k];
    }
    liftStrip<kStrip>(strip, h, 1, (float)kAlpha);
    liftStrip<kStrip>(strip, h, 0, (float)kBeta);
    liftStrip<kStrip>(strip, h, 1, (float)kGamma);
    liftStrip<kStrip>(strip, h, 0, (float)kDelta);
    const int sn = (h + 1) / 2;
    for (int i = 0; i < sn; ++i) {
        const float* src = strip + (size_t)(2 * i) * kStrip;
        float* dst = plane + (size_t)i * width + x0;
        for (int k = 0; k < nx; ++k) dst[k] = src[k] * (float)(1.0 / kK);
    }
    for (int i = 0; i < h / 2; ++i) {
        const float* src = strip + (size_t)(2 * i + 1) * kStrip;
        float* dst = plane + (size_t)(sn + i) * width + x0;
        for (int k = 0; k < nx; ++k) dst[k] = src[k] * (float)kK;
    }
}

// Per-thread scratch: a row buffer and a column strip.
struct DwtScratch {
    std::vector<float> row, strip;
    void reserve(int w, int h) {
        if ((int)row.size() < w) row.resize(w);
        if (strip.size() < (size_t)h * kStrip) strip.resize((size_t)h * kStrip);
    }
};
thread_local DwtScratch g_scratch;

}  // namespace

template <typename T> void fdwt97_1d(const T* in, T* out, int n) {
    if (n <= 0) return;
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    std::vector<T> x(in, in + n);
    lift<T>(x.data(), n, 1, (T)kAlpha);
    lift<T>(x.data(), n, 0, (T)kBeta);
    lift<T>(x.data(), n, 1, (T)kGamma);
    lift<T>(x.data(), n, 0, (T)kDelta);
    const int sn = (n + 1) / 2;
    for (int i = 0; i < sn; ++i) out[i] = x[2 * i] * (T)(1.0 / kK);
    for (int i = 0; i < n / 2; ++i) out[sn + i] = x[2 * i + 1] * (T)kK;
}

template <typename T> void idwt97_1d(const T* in, T* out, int n) {
    if (n <= 0) return;
    if (n == 1) {
        out[0] = in[0];
        return;
    }
    const int sn = (n + 1) / 2;
    std::vector<T> x(n);
    for (int i = 0; i < sn; ++i) x[2 * i] = in[i] * (T)kK;
    for (int i = 0; i < n / 2; ++i) x[2 * i + 1] = in[sn + i] * (T)(1.0 / kK);
    lift<T>(x.data(), n, 0, (T)-kDelta);
    lift<T>(x.data(), n, 1, (T)-kGamma);
    lift<T>(x.data(), n, 0, (T)-kBeta);
    lift<T>(x.data(), n, 1, (T)-kAlpha);
    std::copy(x.begin(), x.end(), out);
}

template void fdwt97_1d<float>(const float*, float*, int);
template void fdwt97_1d<double>(const double*, double*, int);
template void idwt97_1d<float>(const float*, float*, int);
template void idwt97_1d<double>(const double*, double*, int);

void fdwt97_2d(float* const* planes, int nplanes, int width, int height, int levels, const ParallelFor* par) {
    const ParallelFor serial = [](int n, const std::function<void(int)>& f) {
        for (int i = 0; i < n; ++i) f(i);
    };
    const ParallelFor& run = par ? *par : serial;
    int w = width, h = height;
    for (int l = 0; l < levels; ++l) {
        // vertical (columns) then horizontal (rows), Annex F.4.2
        const int strips = (w + kStrip - 1) / kStrip;
        run(strips * nplanes, [&](int i) {
            const int c = i / strips, x0 = (i % strips) * kStrip;
            g_scratch.reserve(w, h);
            fdwtColumns(planes[c], width, h, x0, std::min(kStrip, w - x0), g_scratch.strip.data());
        });
        run(h * nplanes, [&](int i) {
            const int c = i / h, y = i % h;
            g_scratch.reserve(w, h);
            fdwtRow(planes[c] + (size_t)y * width, w, g_scratch.row.data());
        });
        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }
}

void fdwt97_2d(float* plane, int width, int height, int levels) {
    float* planes[1] = {plane};
    fdwt97_2d(planes, 1, width, height, levels, nullptr);
}

void idwt97_2d(double* plane, int width, int height, int levels) {
    std::vector<int> ws(levels + 1), hs(levels + 1);
    ws[0] = width;
    hs[0] = height;
    for (int l = 1; l <= levels; ++l) {
        ws[l] = (ws[l - 1] + 1) / 2;
        hs[l] = (hs[l - 1] + 1) / 2;
    }
    std::vector<double> line(std::max(width, height)), tmp(std::max(width, height));
    for (int l = levels; l >= 1; --l) {
        const int w = ws[l - 1], h = hs[l - 1];
        for (int y = 0; y < h; ++y) {
            double* row = plane + (size_t)y * width;
            idwt97_1d(row, tmp.data(), w);
            std::copy(tmp.begin(), tmp.begin() + w, row);
        }
        for (int x = 0; x < w; ++x) {
            for (int y = 0; y < h; ++y) line[y] = plane[(size_t)y * width + x];
            idwt97_1d(line.data(), tmp.data(), h);
            for (int y = 0; y < h; ++y) plane[(size_t)y * width + x] = tmp[y];
        }
    }
}

}  // namespace gail
