// SPDX-License-Identifier: MIT
// CUDA backend: the same pipeline as the Vulkan backend (colour, 9/7 DWT,
// quantisation, EBCOT Tier-1 context modelling + MQ coding), written with CUDA
// warp intrinsics. Each slot owns a stream, so frames in flight overlap on the GPU;
// input is decoded straight into pinned memory and results are compacted into
// mapped host memory by the GPU.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/backend.h"
#include "core/t1_ref.h"

// ---- shared Tier-1 kernels (portable headers) -------------------------------------
// MQ output sink with capacity check; the coder's contexts and probability table live
// in shared memory so the coder state itself stays in registers.
struct MqSink {
    uint8_t* base;
    uint32_t cap;
    bool overflow;
};
__device__ __forceinline__ void mqSinkPut(MqSink& o, int i, uint32_t v) {
    if ((uint32_t)i < o.cap) o.base[i] = (uint8_t)v;
    else o.overflow = true;
}
__shared__ uint32_t s_ctx[19][64];
__shared__ uint32_t s_tab[47];
#define MQ_OUT_T MqSink&
#define MQ_PUT(o, i, v) mqSinkPut(o, i, v)
#define MQ_GET(o, i) ((uint32_t)(o).base[(i)])
#define MQ_CTX(e, cx) s_ctx[cx][threadIdx.x]
#define MQ_TAB(i) s_tab[i]
#include "kernels/mq_encoder.h"
#include "kernels/t1_common.h"
// Colour maths shared with the CPU and Vulkan paths; tables reached through kernel pointers.
#define CM_ARGS const float* __restrict__ cp, const float* __restrict__ eotf, const float* __restrict__ lut3,
#define CM_PASS cp, eotf, lut3,
#define CP(i) cp[i]
#define EOTF(i) eotf[i]
#define LUT3(i) lut3[i]
#include "kernels/color_math.h"

namespace gail {

namespace {

#define CUCHECK(x)                                                                                        \
    do {                                                                                                  \
        cudaError_t e_ = (x);                                                                             \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " #x); \
    } while (0)

constexpr int MAX_PASSES = 64;
constexpr int MAX_PLANES = 24;
constexpr unsigned FULL = 0xFFFFFFFFu;

// ---------------------------------------------------------------- colour
struct YuvParams {
    int width, aw, ah, ox, oy, bits, fullRange, chroma, sx, sy, rgb;
    float crR, cbG, crG, cbB;
};

__device__ __forceinline__ float yuvSample(const void* src, int bits, uint32_t i) {
    return bits > 8 ? (float)((const uint16_t*)src)[i] : (float)((const uint8_t*)src)[i];
}

// R'G'B' for container pixel i (see color_yuv.comp): native planar Y'CbCr or G'B'R' input,
// black outside the active picture.
__device__ void yuvToRgb(const void* src, const YuvParams& p, uint32_t i, uint32_t idx[3]) {
    const int x = (int)(i % p.width) - p.ox, y = (int)(i / p.width) - p.oy;
    float rgb[3] = {0.f, 0.f, 0.f};
    if (x >= 0 && y >= 0 && x < p.aw && y < p.ah) {
        const int cw = (p.aw + p.sx) >> p.sx, ch = (p.ah + p.sy) >> p.sy;
        float cxp = !p.sx ? (float)x : p.chroma == 1 ? (x - 0.5f) * 0.5f : x * 0.5f;
        float cyp = !p.sy ? (float)y : p.chroma == 2 ? y * 0.5f : (y - 0.5f) * 0.5f;
        cxp = fminf(fmaxf(cxp, 0.f), (float)(cw - 1));
        cyp = fminf(fmaxf(cyp, 0.f), (float)(ch - 1));
        const int x0 = (int)cxp, y0 = (int)cyp, x1 = min(x0 + 1, cw - 1), y1 = min(y0 + 1, ch - 1);
        const float fx = cxp - x0, fy = cyp - y0;
        const uint32_t uOff = (uint32_t)(p.aw * p.ah), vOff = uOff + (uint32_t)(cw * ch);
        auto bil = [&](uint32_t off) {
            const float a = yuvSample(src, p.bits, off + y0 * cw + x0), b = yuvSample(src, p.bits, off + y0 * cw + x1);
            const float c = yuvSample(src, p.bits, off + y1 * cw + x0), d = yuvSample(src, p.bits, off + y1 * cw + x1);
            return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
        };
        float Y = yuvSample(src, p.bits, y * p.aw + x), U = bil(uOff), V = bil(vOff);
        const float maxv = (float)((1 << p.bits) - 1), sc = (float)(1 << (p.bits - 8));
        if (p.rgb) {  // planes G', B', R'
            rgb[0] = V / maxv;
            rgb[1] = Y / maxv;
            rgb[2] = U / maxv;
        } else if (p.fullRange) {
            Y /= maxv;
            U = (U - 128.f * sc) / maxv;
            V = (V - 128.f * sc) / maxv;
        } else {
            Y = (Y - 16.f * sc) / (219.f * sc);
            U = (U - 128.f * sc) / (224.f * sc);
            V = (V - 128.f * sc) / (224.f * sc);
        }
        if (!p.rgb) {
            rgb[0] = Y + p.crR * V;
            rgb[1] = Y - p.cbG * U - p.crG * V;
            rgb[2] = Y + p.cbB * U;
        }
    }
    for (int c = 0; c < 3; ++c) idx[c] = (uint32_t)floorf(fminf(fmaxf(rgb[c], 0.f), 1.f) * 65535.f + 0.5f);
}

__global__ void colorKernel(const void* __restrict__ src, const float* __restrict__ eotf,
                            const float* __restrict__ cp, const float* __restrict__ lut3, float* __restrict__ planes,
                            uint16_t* __restrict__ xyz, uint32_t npix, bool yuv, YuvParams yp) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= npix) return;
    uint32_t idx[3];
    if (yuv) {
        yuvToRgb(src, yp, i, idx);
    } else {
        const uint16_t* rgb = (const uint16_t*)src;
        idx[0] = rgb[3 * i];
        idx[1] = rgb[3 * i + 1];
        idx[2] = rgb[3 * i + 2];
    }
    float cx, cy, cz;
    cm_rgb_to_xyz12(CM_PASS idx[0] / 65535.f, idx[1] / 65535.f, idx[2] / 65535.f, cx, cy, cz);
    if (xyz) {
        xyz[3 * i] = (uint16_t)cx;
        xyz[3 * i + 1] = (uint16_t)cy;
        xyz[3 * i + 2] = (uint16_t)cz;
    }
    const float x = cx - 2048.f, y = cy - 2048.f, z = cz - 2048.f;
    planes[i] = 0.299f * x + 0.587f * y + 0.114f * z;
    planes[npix + i] = -0.16875f * x - 0.33126f * y + 0.5f * z;
    planes[2 * npix + i] = 0.5f * x - 0.41869f * y - 0.08131f * z;
}

// ---------------------------------------------------------------- DWT
__constant__ float kLift[4] = {-1.586134342059924f, -0.052980118572961f, 0.882911075530934f, 0.443506852043971f};
constexpr float kK = 1.230174104914001f;

// Whole-sample symmetric extension of a signal of length n >= 2 (only halo samples
// outside the signal take the loop; the lifting loops never call this).
__device__ __forceinline__ int mirrorIdx(int i, int n) {
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

__global__ void dwtRowsKernel(const float* __restrict__ src, float* __restrict__ dst, int stride, int planeSize, int w) {
    __shared__ float row[4096];
    const int y = blockIdx.x, comp = blockIdx.y;
    const size_t base = (size_t)comp * planeSize + (size_t)y * stride;
    for (int i = threadIdx.x; i < w; i += blockDim.x) row[i] = src[base + i];
    __syncthreads();
    if (w == 1) {
        if (threadIdx.x == 0) dst[base] = row[0];
        return;
    }
    for (int s = 0; s < 4; ++s) {
        const int parity = (s & 1) ? 0 : 1;
        const float c = kLift[s];
        for (int i = threadIdx.x * 2 + parity; i < w; i += 2 * blockDim.x) {
            // symmetric extension at the ends: the mirrored neighbour is row[1] / row[w-2]
            const int im = i > 0 ? i - 1 : 1, ip = i + 1 < w ? i + 1 : w - 2;
            row[i] += c * (row[im] + row[ip]);
        }
        __syncthreads();
    }
    const int sn = (w + 1) / 2;
    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        if (i & 1) dst[base + sn + i / 2] = row[i] * kK;
        else dst[base + i / 2] = row[i] * (1.f / kK);
    }
}

constexpr int TILE = 64, HALO = 8, ROWS = TILE + 2 * HALO;

__global__ void dwtColsKernel(const float* __restrict__ src, float* __restrict__ dst, int stride, int planeSize, int w,
                              int h) {
    __shared__ float tile[ROWS][33];
    const int lx = threadIdx.x, ly = threadIdx.y;
    const int x = blockIdx.x * 32 + lx;
    const int y0 = blockIdx.y * TILE;
    const size_t base = (size_t)blockIdx.z * planeSize + x;
    const bool col = x < w;
    if (h == 1) {
        if (col && ly == 0 && y0 == 0) dst[base] = src[base];
        return;
    }
    for (int r = ly; r < ROWS; r += 8) tile[r][lx] = col ? src[base + (size_t)mirrorIdx(y0 - HALO + r, h) * stride] : 0.f;
    __syncthreads();
    for (int s = 0; s < 4; ++s) {
        const int parity = (s & 1) ? 0 : 1;
        const float c = kLift[s];
        for (int r = parity + 2 * ly; r < ROWS - 1; r += 16)
            if (r > 0) tile[r][lx] += c * (tile[r - 1][lx] + tile[r + 1][lx]);
        __syncthreads();
    }
    if (!col) return;
    const int sn = (h + 1) / 2;
    for (int r = HALO + ly; r < HALO + TILE; r += 8) {
        const int y = y0 - HALO + r;
        if (y >= h) break;
        const float v = tile[r][lx];
        if (y & 1) dst[base + (size_t)(sn + y / 2) * stride] = v * kK;
        else dst[base + (size_t)(y / 2) * stride] = v * (1.f / kK);
    }
}

// ---------------------------------------------------------------- quantisation
struct CblkEntry {
    uint32_t band, ax, ay, packed;  // packed: w | h << 8 | orient << 16 | comp << 20
};
struct BandEntry {
    float invDelta;
    uint32_t maxMag;
    float distScale;
    uint32_t pad;
};

__global__ void quantKernel(const float* __restrict__ planes, const CblkEntry* __restrict__ cblks,
                            const BandEntry* __restrict__ bands, uint32_t* __restrict__ coeffs, uint32_t n, int stride,
                            int planeSize) {
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t cb = gid >> 10;
    if (cb >= n) return;
    const int k = gid & 1023, r = k >> 5, x = k & 31;
    const CblkEntry e = cblks[cb];
    uint32_t out = 0;
    if (x < (int)(e.packed & 0xFF) && r < (int)((e.packed >> 8) & 0xFF)) {
        const BandEntry b = bands[e.band];
        const float v = planes[(size_t)((e.packed >> 20) & 0xF) * planeSize + (size_t)(e.ay + r) * stride + e.ax + x];
        const float q = fabsf(v) * b.invDelta * 64.0f;
        const uint32_t m = q >= (float)b.maxMag ? b.maxMag : (uint32_t)q;
        out = m | ((v < 0.f && m >= 64u) ? 0x80000000u : 0u);
    }
    coeffs[gid] = out;
}

// ---------------------------------------------------------------- Tier-1 context modelling
constexpr int SUBGROUPS = 4, SYM_CAP = 2560;

struct T1cmParams {
    const float* planes;  // wavelet planes: the kernel quantises its block itself
    int stride, planeSize;
    const CblkEntry* cblks;
    const BandEntry* bands;
    uint32_t* info;      // 4 words per block
    uint32_t* passEnd;
    int32_t* passDist;
    uint32_t* chunks;
    uint8_t* symbols;
    uint32_t* counters;  // [0] symbol allocator, [2] error flags
    const float* frameParams;
    uint32_t n, symCapacity, maxPasses;
};

__device__ __forceinline__ uint32_t bitAt(uint32_t m, int i) { return (i < 0 || i > 31) ? 0u : ((m >> i) & 1u); }
__device__ __forceinline__ int chiAt(uint32_t s, uint32_t g, int i) { return bitAt(s, i) ? (bitAt(g, i) ? -1 : 1) : 0; }

struct LaneState {
    uint64_t acc;
    uint32_t accN;
};

__global__ void __launch_bounds__(128) t1cmKernel(T1cmParams P) {
    __shared__ uint32_t s_mag[SUBGROUPS][32][32];
    __shared__ uint8_t s_sym[SUBGROUPS][SYM_CAP];
    __shared__ uint32_t s_hist[SUBGROUPS][19];
    const int lane = threadIdx.x & 31, sg = threadIdx.x >> 5;
    const uint32_t cb = blockIdx.x * SUBGROUPS + sg;
    if (cb >= P.n) return;  // warp-uniform

    const CblkEntry ce = P.cblks[cb];
    const int w = ce.packed & 0xFF, h = (ce.packed >> 8) & 0xFF, orient = (ce.packed >> 16) & 0xF;
    uint32_t sgn = 0, colMax = 0;
    {
        // dead-zone quantisation of this block (same arithmetic as quantKernel)
        const BandEntry b = P.bands[ce.band];
        const float* src = P.planes + (size_t)((ce.packed >> 20) & 0xF) * P.planeSize + (size_t)ce.ay * P.stride + ce.ax + lane;
        for (int r = 0; r < 32; ++r) {
            uint32_t c = 0;
            if (lane < w && r < h) {
                const float v = src[(size_t)r * P.stride];
                const float q = fabsf(v) * b.invDelta * 64.0f;
                const uint32_t m = q >= (float)b.maxMag ? b.maxMag : (uint32_t)q;
                c = m | ((v < 0.f && m >= 64u) ? 0x80000000u : 0u);
            }
            s_mag[sg][r][lane] = c & 0x7FFFFFFFu;
            sgn |= (c >> 31) << r;
            colMax = max(colMax, c & 0x7FFFFFFFu);
        }
    }
    __syncwarp();
    uint32_t mx = colMax;
    for (int o = 16; o > 0; o >>= 1) mx = max(mx, __shfl_xor_sync(FULL, mx, o));
    const uint32_t maxMag = mx >> 6;
    const int nbps = maxMag ? 32 - __clz(maxMag) : 0;
    if (nbps == 0 || nbps > MAX_PLANES) {
        if (lane == 0) {
            P.info[4 * cb] = 0;
            P.info[4 * cb + 1] = 0;
            if (nbps > MAX_PLANES) atomicOr(&P.counters[2], 4u);
        }
        return;
    }
    const float slopeThresh = P.frameParams[0];
    const bool track = slopeThresh > 0.f;
    const float distScale = P.bands[ce.band].distScale;
    if (lane < 19) s_hist[sg][lane] = 0;
    __syncwarp();

    auto shUp = [&](uint32_t v) { const uint32_t r = __shfl_up_sync(FULL, v, 1); return lane == 0 ? 0u : r; };
    auto shDown = [&](uint32_t v) { const uint32_t r = __shfl_down_sync(FULL, v, 1); return lane == 31 ? 0u : r; };

    const uint32_t sgnL = shUp(sgn), sgnR = shDown(sgn);
    const bool laneValid = lane < w;
    const uint32_t rowsValid = h >= 32 ? 0xFFFFFFFFu : ((1u << h) - 1u);
    const uint32_t colMask = laneValid ? rowsValid : 0u;

    LaneState ls{0, 0};
    auto emit = [&](uint32_t cx, uint32_t d) {
        ls.acc |= (uint64_t)((cx << 1) | d) << (6 * ls.accN);
        ++ls.accN;
        if (track) atomicAdd(&s_hist[sg][cx], d ? 0x10000u : 1u);
    };
    auto flush = [&](uint32_t planeCount) {
        // Exclusive prefix of the per-lane symbol counts (at most 10 per stripe and pass)
        // from four independent ballots instead of a five-step dependent shuffle scan.
        const uint32_t n = ls.accN;
        const uint32_t b0 = __ballot_sync(FULL, n & 1u), b1 = __ballot_sync(FULL, n & 2u);
        const uint32_t b2 = __ballot_sync(FULL, n & 4u), b3 = __ballot_sync(FULL, n & 8u);
        const uint32_t lt = (1u << lane) - 1u;
        const uint32_t excl = __popc(b0 & lt) + 2 * __popc(b1 & lt) + 4 * __popc(b2 & lt) + 8 * __popc(b3 & lt);
        const uint32_t tot = __popc(b0) + 2 * __popc(b1) + 4 * __popc(b2) + 8 * __popc(b3);
        const uint32_t dst = planeCount + excl;
        for (uint32_t j = 0; j < ls.accN; ++j)
            if (dst + j < SYM_CAP) s_sym[sg][dst + j] = (uint8_t)((ls.acc >> (6 * j)) & 63u);
        ls.acc = 0;
        ls.accN = 0;
        return planeCount + tot;
    };
    auto codeZcSign = [&](uint32_t CL, uint32_t CR, uint32_t CU, uint32_t CD, uint32_t Bp, int r, bool zc, int& dist,
                          int p) {
        if (zc) {
            const int hh = bitAt(CL, r) + bitAt(CR, r);
            const int vv = bitAt(CU, r - 1) + bitAt(CD, r + 1);
            const int dd = bitAt(CL, r - 1) + bitAt(CL, r + 1) + bitAt(CR, r - 1) + bitAt(CR, r + 1);
            emit(t1_zc_ctx(hh, vv, dd, orient), bitAt(Bp, r));
        }
        if (bitAt(Bp, r)) {
            const int hc = max(-1, min(1, chiAt(CL, sgnL, r) + chiAt(CR, sgnR, r)));
            const int vc = max(-1, min(1, chiAt(CU, sgn, r - 1) + chiAt(CD, sgn, r + 1)));
            const uint32_t sc = t1_sc_ctx(hc, vc);
            emit(sc & 0xFF, bitAt(sgn, r) ^ (sc >> 8));
            dist += t1_dist_sig((s_mag[sg][r][lane] >> p) & 0x7Fu);
        }
    };

    uint32_t sig = 0, refd = 0, symTotal = 0, np = 0, planesCoded = 0;
    const int nstripes = (h + 3) / 4;
    bool stop = false;
    for (int p = nbps - 1; p >= 0 && np < P.maxPasses; --p) {
        uint32_t Bp = 0;
        for (int r = 0; r < h; ++r) Bp |= ((s_mag[sg][r][lane] >> (p + 6)) & 1u) << r;
        uint32_t planeCount = 0, vis = 0;
        int planeDist = 0;
        const uint32_t sig0 = sig;
        for (int type = (p == nbps - 1 ? 2 : 0); type < 3 && np < P.maxPasses; ++type) {
            int dist = 0;
            if (type == 0) {
                for (int s = 0; s < nstripes; ++s) {
                    const int r0 = 4 * s;
                    const uint32_t SM = (0xFu << r0) & colMask;
                    const uint32_t S = sig, SL = shUp(S), SR = shDown(S);
                    const uint32_t cand = ~S & SM;
                    const uint32_t nS = (S << 1) | (S >> 1) | SL | (SL << 1) | (SL >> 1) | SR | (SR << 1) | (SR >> 1);
                    if (!__any_sync(FULL, cand & nS)) continue;  // no sample with a significant neighbour
                    uint32_t NEW = 0, E = 0, NL = 0;
                    for (;;) {
                        NL = shUp(NEW);
                        uint32_t e = cand & (nS | NL | (NL << 1) | (NL >> 1));
                        uint32_t nn = e & Bp;
                        e |= cand & (nn << 1);
                        nn = e & Bp;
                        e |= cand & (nn << 1);
                        nn = e & Bp;
                        e |= cand & (nn << 1);
                        nn = e & Bp;
                        const bool changed = nn != NEW;
                        NEW = nn;
                        E = e;
                        if (!__any_sync(FULL, changed)) break;
                    }
                    NL = shUp(NEW);
                    const uint32_t CL = SL | NL, CR = SR, CU = S | NEW, CD = S;
                    for (int r = r0; r < r0 + 4; ++r)
                        if (bitAt(E, r)) codeZcSign(CL, CR, CU, CD, Bp, r, true, dist, p);
                    planeCount = flush(planeCount);
                    sig = S | NEW;
                    vis |= E;
                }
            } else if (type == 1) {
                const uint32_t S = sig, SL = shUp(S), SR = shDown(S);
                const uint32_t nAny = (S << 1) | (S >> 1) | SL | (SL << 1) | (SL >> 1) | SR | (SR << 1) | (SR >> 1);
                for (int s = 0; s < nstripes; ++s) {
                    const int r0 = 4 * s;
                    const uint32_t coded = sig0 & (0xFu << r0) & colMask;
                    if (!__any_sync(FULL, coded)) continue;  // nothing to refine in this stripe
                    for (int r = r0; r < r0 + 4; ++r) {
                        if (!bitAt(coded, r)) continue;
                        emit(bitAt(refd, r) ? 16u : (bitAt(nAny, r) ? 15u : 14u), bitAt(Bp, r));
                        dist += t1_dist_ref((s_mag[sg][r][lane] >> p) & 0x7Fu);
                    }
                    planeCount = flush(planeCount);
                }
                refd |= sig0;
            } else {
                for (int s = 0; s < nstripes; ++s) {
                    const int r0 = 4 * s;
                    const uint32_t SM = (0xFu << r0) & colMask;
                    const uint32_t S = sig, SL = shUp(S), SR = shDown(S);
                    const uint32_t Cc = SM & ~S & ~vis;
                    if (!__any_sync(FULL, Cc)) continue;  // every sample already coded this plane
                    const uint32_t NEWc = Cc & Bp;
                    const uint32_t NLc = shUp(NEWc);
                    const uint32_t CL = SL | NLc, CR = SR, CU = S | NEWc, CD = S;
                    const uint32_t four = 0xFu << r0;
                    bool run = laneValid && (r0 + 3 < h) && (Cc & four) == four;
                    if (run) {
                        const uint32_t win = r0 == 0 ? 0x1Fu : (0x3Fu << (r0 - 1));
                        run = (CL & win) == 0 && (CR & win) == 0 && bitAt(S, r0 - 1) == 0 && bitAt(S, r0 + 4) == 0;
                    }
                    int rs = r0;
                    if (run) {
                        const uint32_t nb = (NEWc >> r0) & 0xFu;
                        if (nb == 0) {
                            emit(MQ_CTX_RL, 0);
                            rs = r0 + 4;
                        } else {
                            const int first = __ffs(nb) - 1;
                            emit(MQ_CTX_RL, 1);
                            emit(MQ_CTX_UNI, (uint32_t)first >> 1);
                            emit(MQ_CTX_UNI, (uint32_t)first & 1);
                            codeZcSign(CL, CR, CU, CD, Bp, r0 + first, false, dist, p);
                            rs = r0 + first + 1;
                        }
                    }
                    for (int r = rs; r < r0 + 4; ++r)
                        if (bitAt(Cc, r)) codeZcSign(CL, CR, CU, CD, Bp, r, true, dist, p);
                    planeCount = flush(planeCount);
                    sig = S | NEWc;
                }
            }
            int total = dist;
            for (int o = 16; o > 0; o >>= 1) total += __shfl_xor_sync(FULL, total, o);
            if (lane == 0) {
                P.passEnd[cb * MAX_PASSES + np] = symTotal + planeCount;
                P.passDist[cb * MAX_PASSES + np] = total;
            }
            ++np;
            planeDist += total;
        }
        if (track) {
            // see t1_cm.comp: entropy-based plane rate estimate + residual bound
            __syncwarp();
            float bits = 0.f;
            if (lane < 19) {
                const uint32_t hv = s_hist[sg][lane];
                const float n0 = (float)(hv & 0xFFFF), n1 = (float)(hv >> 16), nn = n0 + n1;
                if (nn > 0.f) bits = -(n0 * log2f(fmaxf(n0, 0.5f) / nn) + n1 * log2f(fmaxf(n1, 0.5f) / nn)) + 2.f;
                s_hist[sg][lane] = 0;
            }
            for (int o = 16; o > 0; o >>= 1) bits += __shfl_xor_sync(FULL, bits, o);
            const float bytes = bits * 0.125f + 1.f;
            if (p > 0) {
                float e = 0.f;
                const float step = exp2f((float)p);
                for (int r = 0; r < h; ++r) {
                    const uint32_t m = s_mag[sg][r][lane];
                    const float v = (float)m * (1.f / 64.f);
                    const uint32_t q = m >> (p + 6);
                    const float rec = q == 0 ? 0.f : ((float)q + 0.5f) * step;
                    e += (v - rec) * (v - rec);
                }
                for (int o = 16; o > 0; o >>= 1) e += __shfl_xor_sync(FULL, e, o);
                const float dAbs = (float)planeDist * exp2f((float)(2 * p)) * (1.f / 4096.f) * distScale;
                if (dAbs < slopeThresh * bytes && e * distScale < slopeThresh * bytes) stop = true;
            }
        }
        __syncwarp();
        uint32_t chunkOff = 0;
        if (lane == 0) {
            chunkOff = atomicAdd(&P.counters[0], (planeCount + 15u) & ~15u);
            if (planeCount > (uint32_t)SYM_CAP || chunkOff + planeCount > P.symCapacity) atomicOr(&P.counters[2], 1u);
        }
        chunkOff = __shfl_sync(FULL, chunkOff, 0);
        if (chunkOff + planeCount <= P.symCapacity && planeCount <= (uint32_t)SYM_CAP)
            for (uint32_t i = lane; i < planeCount; i += 32) P.symbols[chunkOff + i] = s_sym[sg][i];
        if (lane == 0) {
            const uint32_t j = nbps - 1 - p;
            P.chunks[cb * 2 * MAX_PLANES + 2 * j] = chunkOff;
            P.chunks[cb * 2 * MAX_PLANES + 2 * j + 1] = planeCount;
        }
        symTotal += planeCount;
        ++planesCoded;
        __syncwarp();
        if (stop) break;
    }
    if (lane == 0) {
        P.info[4 * cb] = nbps;
        P.info[4 * cb + 1] = np;
        P.info[4 * cb + 2] = planesCoded;
    }
}

// ---------------------------------------------------------------- ordering
__global__ void orderKernel(const uint32_t* __restrict__ info, const uint32_t* __restrict__ passEnd,
                            uint32_t* __restrict__ perm, uint32_t n) {
    constexpr int NB = 128;
    __shared__ uint32_t hist[NB];
    auto bucketOf = [&](uint32_t cb) -> uint32_t {
        const uint32_t np = info[4 * cb + 1];
        const uint32_t s = np ? passEnd[cb * MAX_PASSES + np - 1] : 0;
        if (s < 4) return NB - 1;
        const int l = 31 - __clz(s);
        const uint32_t sub = (s >> (l - 2)) & 3u;
        return NB - 1 - min((uint32_t)l * 4 + sub, (uint32_t)NB - 1);
    };
    const int t = threadIdx.x;
    if (t < NB) hist[t] = 0;
    __syncthreads();
    for (uint32_t i = t; i < n; i += blockDim.x) atomicAdd(&hist[bucketOf(i)], 1u);
    __syncthreads();
    if (t == 0) {
        uint32_t run = 0;
        for (int b = 0; b < NB; ++b) {
            const uint32_t c = hist[b];
            hist[b] = run;
            run += c;
        }
    }
    __syncthreads();
    for (uint32_t i = t; i < n; i += blockDim.x) perm[atomicAdd(&hist[bucketOf(i)], 1u)] = i;
}


// ---------------------------------------------------------------- MQ coding
struct T1mqParams {
    uint32_t* info;
    const uint32_t* passEnd;
    uint32_t* passLen;
    const uint32_t* chunks;
    const uint8_t* symbols;
    uint8_t* out;
    uint32_t* counters;  // [1] output allocator, [2] error flags
    const uint32_t* perm;
    uint32_t n, outCapacity;
};

// mq_encode (kernels/mq_encoder.h) with the context state t passed in and the new state
// returned, so the caller can prefetch the next symbol's state and keep the shared-memory
// load out of the per-symbol dependency chain.
__device__ __forceinline__ uint32_t mqEncodeT(MqEnc& e, uint32_t cx, uint32_t t, uint32_t d, MqSink& ob) {
    const uint32_t mps = t >> 31, qe = t & 0xFFFFu;
    const uint32_t a = e.A - qe;
    const bool isMps = d == mps;
    const bool renorm = !isMps || (a & 0x8000u) == 0u;
    const bool addC = isMps != (a < qe);
    e.C += addC ? qe : 0u;
    e.A = addC ? a : qe;
    if (renorm) {
        const uint32_t nidx = isMps ? ((t >> 16) & 63u) : ((t >> 22) & 63u);
        const uint32_t nmps = isMps ? mps : (mps ^ ((t >> 28) & 1u));
        t = s_tab[nidx] | (nmps << 31);
        s_ctx[cx][threadIdx.x] = t;
        mq_renorm(e, ob);
    }
    return t;
}

__global__ void __launch_bounds__(64) t1mqKernel(T1mqParams P) {
    if (threadIdx.x < 47) s_tab[threadIdx.x] = MQ_TABLE[threadIdx.x];
    __syncthreads();
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= P.n) return;
    // Symbols did not fit (pathological content): some chunks were never written, and the
    // host redoes this frame on the CPU
    if (P.counters[2] & 1u) return;
    const uint32_t cb = P.perm[gid];
    const uint32_t np = P.info[4 * cb + 1];
    if (np == 0) {
        P.info[4 * cb + 2] = 0;
        P.info[4 * cb + 3] = 0;
        return;
    }
    const uint32_t total = P.passEnd[cb * MAX_PASSES + np - 1];
    const uint32_t cap = (total / 2 + 64 + 3) & ~3u;
    const uint32_t outBase = atomicAdd(&P.counters[1], cap);
    if (outBase + cap > P.outCapacity) {
        atomicOr(&P.counters[2], 2u);
        P.info[4 * cb + 1] = 0;
        P.info[4 * cb + 3] = 0;
        return;
    }
    MqSink sink{P.out + outBase, cap, false};
    MqEnc e;
    mq_init(e);
    MqSnap snaps[MAX_PASSES];
    uint32_t k = 0, pos = 0;
    uint32_t nextEnd = P.passEnd[cb * MAX_PASSES];  // 0xFFFFFFFF once every pass end is recorded
    auto passEnds = [&] {
        do {
            snaps[k] = mq_snapshot(e);
            ++k;
            nextEnd = k < np ? P.passEnd[cb * MAX_PASSES + k] : 0xFFFFFFFFu;
        } while (nextEnd == pos);
    };
    const uint32_t planes = P.info[4 * cb + 2];
    for (uint32_t j = 0; j < planes; ++j) {
        const uint32_t off = P.chunks[cb * 2 * MAX_PLANES + 2 * j];
        const uint32_t cnt = P.chunks[cb * 2 * MAX_PLANES + 2 * j + 1];
        const uint4* src = reinterpret_cast<const uint4*>(P.symbols + off);  // 16-byte aligned chunks
        uint4 next = cnt ? src[0] : make_uint4(0, 0, 0, 0);
        for (uint32_t i = 0; i < cnt; i += 16) {
            // Symbols are coded 16 at a time (one chunk, fully unrolled so word and byte
            // selection are constants), with the next chunk loaded one chunk ahead.
            const uint4 cur = next;
            if (i + 16 < cnt) next = src[(i >> 4) + 1];
            const uint32_t m = min(16u, cnt - i);
            const uint32_t words[4] = {cur.x, cur.y, cur.z, cur.w};
            uint32_t sym = words[0] & 0xFFu;
            uint32_t t = s_ctx[sym >> 1][threadIdx.x];
#pragma unroll
            for (uint32_t b = 0; b < 16; ++b) {
                if (b >= m) break;
                const uint32_t symNext = b + 1 < 16 ? (words[(b + 1) >> 2] >> (8 * ((b + 1) & 3))) & 0xFFu : 0u;
                const uint32_t tNext = s_ctx[symNext >> 1][threadIdx.x];  // prefetch
                if (pos == nextEnd) passEnds();
                const uint32_t cx = sym >> 1;
                const uint32_t tNew = mqEncodeT(e, cx, t, sym & 1u, sink);
                t = cx == (symNext >> 1) ? tNew : tNext;
                sym = symNext;
                ++pos;
            }
        }
    }
    if (pos == nextEnd) passEnds();
    const uint32_t len = mq_flush(e, sink);
    if (sink.overflow) {
        atomicOr(&P.counters[2], 2u);
        P.info[4 * cb + 1] = 0;
        P.info[4 * cb + 3] = 0;
        return;
    }
    uint32_t prev = 1;
    for (uint32_t q = 0; q < np; ++q) {
        const uint32_t L = q == np - 1 ? len : mq_trunc_len(snaps[q], sink, len, prev);
        P.passLen[cb * MAX_PASSES + q] = L;
        prev = max(prev, L);
    }
    P.info[4 * cb + 2] = outBase;
    P.info[4 * cb + 3] = len;
}

// ---------------------------------------------------------------- compaction
__global__ void scanKernel(const uint32_t* __restrict__ info, uint32_t* __restrict__ offsets, uint32_t n) {
    __shared__ uint2 part[1024];
    const uint32_t t = threadIdx.x, per = (n + 1023) / 1024;
    const uint32_t b = t * per, e = min(b + per, n);
    auto item = [&](uint32_t i) { return make_uint2((info[4 * i + 3] + 3) & ~3u, info[4 * i + 1]); };
    uint2 sum = make_uint2(0, 0);
    for (uint32_t i = b; i < e; ++i) {
        const uint2 v = item(i);
        sum.x += v.x;
        sum.y += v.y;
    }
    part[t] = sum;
    __syncthreads();
    for (uint32_t d = 1; d < 1024; d <<= 1) {
        const uint2 v = t >= d ? part[t - d] : make_uint2(0, 0);
        __syncthreads();
        part[t].x += v.x;
        part[t].y += v.y;
        __syncthreads();
    }
    uint2 run = make_uint2(part[t].x - sum.x, part[t].y - sum.y);
    for (uint32_t i = b; i < e; ++i) {
        offsets[2 * i] = run.x;
        offsets[2 * i + 1] = run.y;
        const uint2 v = item(i);
        run.x += v.x;
        run.y += v.y;
    }
    if (t == 1023) {
        offsets[2 * n] = part[1023].x;
        offsets[2 * n + 1] = part[1023].y;
    }
}

struct CompactParams {
    const uint32_t* info;
    const uint32_t* offsets;
    const uint8_t* src;
    const uint32_t* passLen;
    const int32_t* passDist;
    uint32_t* hostInfo;   // 5 words per block (device staging, copied to the host afterwards)
    uint32_t* hostPass;   // lengths, then distortions at +passCap
    uint8_t* hostData;
    uint32_t n, dataCap, passCap;
};

__global__ void compactKernel(CompactParams P) {
    const uint32_t cb = blockIdx.x, t = threadIdx.x;
    if (cb >= P.n) return;
    const uint32_t np = P.info[4 * cb + 1], len = P.info[4 * cb + 3];
    const uint32_t dOff = P.offsets[2 * cb], pOff = P.offsets[2 * cb + 1];
    if (t == 0) {
        uint32_t* hi = P.hostInfo + 5 * cb;
        hi[0] = P.info[4 * cb];
        hi[1] = np;
        hi[2] = dOff;
        hi[3] = len;
        hi[4] = pOff;
    }
    if (pOff + np <= P.passCap && t < np) {
        P.hostPass[pOff + t] = P.passLen[cb * MAX_PASSES + t];
        P.hostPass[P.passCap + pOff + t] = (uint32_t)P.passDist[cb * MAX_PASSES + t];
    }
    const uint32_t words = (len + 3) >> 2;
    if (len == 0 || dOff + words * 4 > P.dataCap) return;
    const uint32_t* s = reinterpret_cast<const uint32_t*>(P.src + P.info[4 * cb + 2]);
    uint32_t* d = reinterpret_cast<uint32_t*>(P.hostData + dOff);
    for (uint32_t i = t; i < words; i += blockDim.x) d[i] = s[i];
}

// ---------------------------------------------------------------- device / slot
// Buffer capacities of one slot, and the device memory they add up to. Kept in one place
// so the pipeline can size the number of frames in flight before anything is allocated.
struct SlotSizes {
    size_t npix = 0, n = 0, bands = 0, inBytes = 0, symCap = 0, outCap = 0, passCap = 0, dataCap = 0, lut3d = 0;
    // cudaMalloc hands out whole 2 MiB pages
    static size_t page(size_t b) { return (b + (2u << 20) - 1) & ~(size_t)((2u << 20) - 1); }
    size_t deviceBytes() const {
        const size_t sizes[] = {n * 20,      passCap * 8,  dataCap,          inBytes,       65536 * 4,  npix * 12,
                                npix * 12,   n * 4096,     n * 16,           bands * 16,    n * 16,     passCap * 4,
                                passCap * 4, passCap * 4,  n * 2 * MAX_PLANES * 4, symCap, outCap,     64,
                                (n + 1) * 8, n * 4,        npix * 6,         CP_COUNT * 4,  lut3d};
        size_t total = 0;
        for (size_t b : sizes) total += page(b);
        return total;
    }
};

SlotSizes slotSizes(const FrameLayout& L, const EncodeSettings& s) {
    SlotSizes z;
    z.npix = (size_t)L.width * L.height;
    z.n = (size_t)L.numCblks();
    z.bands = L.bands.size();
    z.inBytes = s.input.bytes();
    // Generous capacities; overflow is detected and falls back to the CPU coder.
    z.symCap = std::max<size_t>(64ull << 20, z.n * 6144);
    z.outCap = z.symCap / 2 + z.n * 80;
    z.passCap = z.n * MAX_PASSES;
    z.dataCap = std::max<size_t>(32ull << 20, z.n * 2048);
    ColorConverter probe(s.color);  // sizes the optional 3D LUT buffer
    z.lut3d = std::max<size_t>(16, probe.lut3d().size() * 4);
    return z;
}

// cudaMalloc / cudaHostAlloc failing for lack of memory becomes DeviceOutOfMemory (and the
// error is cleared, so the device stays usable); anything else is a plain CUDA error.
void cuAlloc(cudaError_t e, const char* what, size_t bytes) {
    if (e == cudaSuccess) return;
    if (e == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        throw DeviceOutOfMemory(std::string("cannot allocate ") + std::to_string(bytes >> 20) + " MiB of " + what);
    }
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) + " allocating " + what);
}

class CudaSlot : public EncoderSlot {
public:
    CudaSlot(int dev, const FrameLayout& L, const EncodeSettings& s, const Logger& log)
        : dev_(dev), L_(L), s_(s), log_(log) {
        try {
            init(L, s);
        } catch (...) {
            release();
            throw;
        }
    }
    ~CudaSlot() override { release(); }

private:
    void init(const FrameLayout& L, const EncodeSettings& s) {
        CUCHECK(cudaSetDevice(dev_));
        const SlotSizes z = slotSizes(L, s);
        const size_t npix = z.npix;
        n_ = (uint32_t)z.n;
        symCap_ = z.symCap;
        outCap_ = z.outCap;
        passCap_ = z.passCap;
        dataCap_ = z.dataCap;
        inBytes_ = z.inBytes;
        CUCHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        hostAlloc(hostIn_, inBytes_, cudaHostAllocWriteCombined);
        // Results are compacted into device staging buffers and copied to pinned host
        // memory in three exact-size transfers: far fewer PCIe transactions than the GPU
        // writing scattered small records straight into mapped host memory.
        hostAlloc(hostInfo_, (size_t)n_ * 20, 0);
        hostAlloc(hostPass_, passCap_ * 8, 0);
        hostAlloc(hostData_, dataCap_, 0);
        hostAlloc(hostSmall_, 128, 0);
        hostAlloc(frameParams_, 64, cudaHostAllocMapped);
        alloc(dInfoOut_, (size_t)n_ * 20);
        alloc(dPassOut_, passCap_ * 8);
        alloc(dDataOut_, dataCap_);
        CUCHECK(cudaHostGetDevicePointer(&dFrameParams_, frameParams_, 0));
        alloc(rgb_, inBytes_);
        alloc(lut_, 65536 * 4);
        alloc(planesA_, npix * 12);
        alloc(planesB_, npix * 12);
        alloc(coeffs_, (size_t)n_ * 4096);
        alloc(cblkTab_, (size_t)n_ * 16);
        alloc(bandTab_, L.bands.size() * 16);
        alloc(info_, (size_t)n_ * 16);
        alloc(passEnd_, passCap_ * 4);
        alloc(passDist_, passCap_ * 4);
        alloc(passLen_, passCap_ * 4);
        alloc(chunks_, (size_t)n_ * 2 * MAX_PLANES * 4);
        alloc(symbols_, symCap_);
        alloc(out_, outCap_);
        alloc(counters_, 64);
        alloc(offsets_, ((size_t)n_ + 1) * 8);
        alloc(perm_, (size_t)n_ * 4);
        alloc(xyz_, npix * 6);
        for (auto& ev : ev_) CUCHECK(cudaEventCreate(&ev));
        setSlopeThreshold(0.0);
        uploadStatic();
    }
    // Frees whatever init() got to (also after a failed construction).
    void release() {
        cudaSetDevice(dev_);
        if (stream_) cudaStreamSynchronize(stream_);
        for (void* p : allocs_) cudaFree(p);
        allocs_.clear();
        for (void** p : {&hostIn_, (void**)&hostInfo_, (void**)&hostPass_, (void**)&hostData_, (void**)&hostSmall_,
                         (void**)&frameParams_})
            if (*p) {
                cudaFreeHost(*p);
                *p = nullptr;
            }
        for (auto& ev : ev_)
            if (ev) {
                cudaEventDestroy(ev);
                ev = nullptr;
            }
        if (stream_) cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    template <typename T> void hostAlloc(T*& p, size_t bytes, unsigned flags) {
        cuAlloc(cudaHostAlloc((void**)&p, bytes, flags), "pinned host memory", bytes);
    }

public:

    void* inputBuffer() override { return hostIn_; }
    void setSlopeThreshold(double t) override {
        frameParams_[0] = (float)t;
        frameParams_[1] = (float)s_.bitsPerSymbol;
    }

    void encode(const void* frame, Tier1Frame& out, FrameTimings* tm, FrameDebug* dbg) override {
        const auto t0 = std::chrono::steady_clock::now();
        CUCHECK(cudaSetDevice(dev_));
        const uint32_t W = L_.width, H = L_.height, npix = W * H;
        if (frame != hostIn_) std::memcpy(hostIn_, frame, inBytes_);
        cudaStream_t st = stream_;
        CUCHECK(cudaEventRecord(ev_[0], st));
        CUCHECK(cudaMemcpyAsync(rgb_, hostIn_, inBytes_, cudaMemcpyHostToDevice, st));
        CUCHECK(cudaMemsetAsync(counters_, 0, 64, st));
        CUCHECK(cudaEventRecord(ev_[1], st));
        const FrameFormat& in = s_.input;
        YuvParams yp{(int)W, in.activeW, in.activeH, in.ox, in.oy, in.bits, in.fullRange ? 1 : 0, (int)in.chroma,
                     in.sx, in.sy, in.rgb ? 1 : 0, 0, 0, 0, 0};
        in.matrix(yp.crR, yp.cbG, yp.crG, yp.cbB);
        colorKernel<<<(npix + 255) / 256, 256, 0, st>>>(rgb_, (float*)lut_, (float*)colorParams_, (float*)lut3d_,
                                                         (float*)planesA_,
                                                         dbg ? (uint16_t*)xyz_ : nullptr, npix,
                                                         in.kind == FrameFormat::Kind::Planar, yp);
        CUCHECK(cudaEventRecord(ev_[2], st));
        uint32_t w = W, h = H;
        for (int l = 0; l < L_.levels; ++l) {
            dwtRowsKernel<<<dim3(h, 3), 256, 0, st>>>((float*)planesA_, (float*)planesB_, W, npix, w);
            dwtColsKernel<<<dim3((w + 31) / 32, (h + TILE - 1) / TILE, 3), dim3(32, 8), 0, st>>>(
                (float*)planesB_, (float*)planesA_, W, npix, w, h);
            w = (w + 1) / 2;
            h = (h + 1) / 2;
        }
        CUCHECK(cudaEventRecord(ev_[3], st));
        // Tier-1 quantises each block itself; the coefficient buffer is only filled for the
        // debug taps (and by the CPU fallback).
        if (dbg && dbg->full) runQuant(st);
        CUCHECK(cudaEventRecord(ev_[4], st));
        T1cmParams cm{(float*)planesA_,    (int)W,              (int)npix,           (CblkEntry*)cblkTab_,
                      (BandEntry*)bandTab_, (uint32_t*)info_,   (uint32_t*)passEnd_, (int32_t*)passDist_,
                      (uint32_t*)chunks_,   (uint8_t*)symbols_, (uint32_t*)counters_, dFrameParams_,
                      n_,                   (uint32_t)symCap_,  (uint32_t)MAX_PASSES};
        t1cmKernel<<<(n_ + SUBGROUPS - 1) / SUBGROUPS, 128, 0, st>>>(cm);
        CUCHECK(cudaEventRecord(ev_[5], st));
        orderKernel<<<1, 1024, 0, st>>>((uint32_t*)info_, (uint32_t*)passEnd_, (uint32_t*)perm_, n_);
        T1mqParams mq{(uint32_t*)info_,    (uint32_t*)passEnd_, (uint32_t*)passLen_, (uint32_t*)chunks_,
                      (uint8_t*)symbols_,  (uint8_t*)out_,      (uint32_t*)counters_, (uint32_t*)perm_,
                      n_,                  (uint32_t)outCap_};
        t1mqKernel<<<(n_ + 63) / 64, 64, 0, st>>>(mq);
        CUCHECK(cudaEventRecord(ev_[6], st));
        scanKernel<<<1, 1024, 0, st>>>((uint32_t*)info_, (uint32_t*)offsets_, n_);
        CompactParams cp{(uint32_t*)info_,    (uint32_t*)offsets_,  (uint8_t*)out_,      (uint32_t*)passLen_,
                         (int32_t*)passDist_,  (uint32_t*)dInfoOut_, (uint32_t*)dPassOut_, (uint8_t*)dDataOut_,
                         n_,                   (uint32_t)dataCap_,   (uint32_t)passCap_};
        compactKernel<<<n_, 64, 0, st>>>(cp);
        CUCHECK(cudaMemcpyAsync(hostSmall_, counters_, 64, cudaMemcpyDeviceToHost, st));
        CUCHECK(cudaMemcpyAsync(hostSmall_ + 16, (uint32_t*)offsets_ + 2 * n_, 8, cudaMemcpyDeviceToHost, st));
        CUCHECK(cudaStreamSynchronize(st));
        CUCHECK(cudaGetLastError());

        const uint32_t flags = hostSmall_[2], totalData = hostSmall_[16], totalPasses = hostSmall_[17];
        const bool ok = !flags && totalData <= dataCap_ && totalPasses <= passCap_;
        if (ok) {
            CUCHECK(cudaMemcpyAsync(hostInfo_, dInfoOut_, (size_t)n_ * 20, cudaMemcpyDeviceToHost, st));
            CUCHECK(cudaMemcpyAsync(hostPass_, dPassOut_, (size_t)totalPasses * 4, cudaMemcpyDeviceToHost, st));
            CUCHECK(cudaMemcpyAsync(hostPass_ + passCap_, (uint32_t*)dPassOut_ + passCap_, (size_t)totalPasses * 4,
                                    cudaMemcpyDeviceToHost, st));
            CUCHECK(cudaMemcpyAsync(hostData_, dDataOut_, totalData, cudaMemcpyDeviceToHost, st));
        }
        CUCHECK(cudaEventRecord(ev_[7], st));
        CUCHECK(cudaStreamSynchronize(st));
        if (dbg) {
            dbg->xyz12.resize((size_t)npix * 3);
            CUCHECK(cudaMemcpy(dbg->xyz12.data(), xyz_, (size_t)npix * 6, cudaMemcpyDeviceToHost));
            if (dbg->full) {
                dbg->coeffs.resize((size_t)n_ * 1024);
                CUCHECK(cudaMemcpy(dbg->coeffs.data(), coeffs_, (size_t)n_ * 4096, cudaMemcpyDeviceToHost));
                dbg->wavelet.resize((size_t)npix * 3);
                CUCHECK(cudaMemcpy(dbg->wavelet.data(), planesA_, (size_t)npix * 12, cudaMemcpyDeviceToHost));
            }
        }
        if (!ok) {
            if (log_)
                log_(LogLevel::Warning, "cuda: Tier-1 buffer overflow (flags " + std::to_string(flags) +
                                            "), using CPU fallback for this frame");
            cpuFallback(out);
        } else {
            out.info.resize(n_);
            std::memcpy(out.info.data(), hostInfo_, (size_t)n_ * sizeof(CblkInfo));
            out.passLen.assign(hostPass_, hostPass_ + totalPasses);
            out.passDist.assign((const int32_t*)hostPass_ + passCap_, (const int32_t*)hostPass_ + passCap_ + totalPasses);
            out.data.assign(hostData_, hostData_ + totalData);
        }
        if (tm) {
            float ms[7];
            for (int i = 0; i < 7; ++i) cudaEventElapsedTime(&ms[i], ev_[i], ev_[i + 1]);
            tm->upload = ms[0];
            tm->color = ms[1];
            tm->dwt = ms[2];
            tm->quant = ms[3];
            tm->t1cm = ms[4];
            tm->t1mq = ms[5];
            tm->t1 = ms[4] + ms[5];
            tm->download = ms[6];
            float g = 0;
            cudaEventElapsedTime(&g, ev_[0], ev_[7]);
            tm->gpu = g;
            tm->symbols = hostSmall_[0];
            tm->codedBytes = totalData;
            tm->total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        }
    }

private:
    void alloc(void*& p, size_t bytes) {
        cuAlloc(cudaMalloc(&p, bytes), "device memory", bytes);
        allocs_.push_back(p);
    }
    void uploadStatic() {
        ColorConverter cc(s_.color);
        CUCHECK(cudaMemcpy(lut_, cc.lut().data(), 65536 * 4, cudaMemcpyHostToDevice));
        alloc(colorParams_, CP_COUNT * 4);
        CUCHECK(cudaMemcpy(colorParams_, cc.params().data(), CP_COUNT * 4, cudaMemcpyHostToDevice));
        alloc(lut3d_, std::max<size_t>(16, cc.lut3d().size() * 4));
        if (!cc.lut3d().empty())
            CUCHECK(cudaMemcpy(lut3d_, cc.lut3d().data(), cc.lut3d().size() * 4, cudaMemcpyHostToDevice));
        std::vector<CblkEntry> tab(n_);
        for (uint32_t i = 0; i < n_; ++i) {
            const CodeBlock& cb = L_.cblks[i];
            const Subband& b = L_.bands[cb.band];
            tab[i] = {(uint32_t)cb.band, (uint32_t)(b.px0 + cb.x0), (uint32_t)(b.py0 + cb.y0),
                      (uint32_t)cb.w | ((uint32_t)cb.h << 8) | ((uint32_t)b.orient << 16) | ((uint32_t)b.comp << 20)};
        }
        CUCHECK(cudaMemcpy(cblkTab_, tab.data(), tab.size() * sizeof(CblkEntry), cudaMemcpyHostToDevice));
        std::vector<BandEntry> bt(L_.bands.size());
        for (size_t i = 0; i < bt.size(); ++i) {
            const Subband& b = L_.bands[i];
            bt[i] = {1.0f / b.delta, ((1u << b.Mb) << kFracBits) - 1, (float)((double)b.delta * b.delta * b.weight), 0};
        }
        CUCHECK(cudaMemcpy(bandTab_, bt.data(), bt.size() * sizeof(BandEntry), cudaMemcpyHostToDevice));
    }
    void runQuant(cudaStream_t st) {
        quantKernel<<<n_ * 1024 / 256, 256, 0, st>>>((float*)planesA_, (CblkEntry*)cblkTab_, (BandEntry*)bandTab_,
                                                     (uint32_t*)coeffs_, n_, L_.width, L_.width * L_.height);
    }
    void cpuFallback(Tier1Frame& out) {
        std::vector<uint32_t> coeffs((size_t)n_ * 1024);
        runQuant(stream_);
        CUCHECK(cudaStreamSynchronize(stream_));
        CUCHECK(cudaMemcpy(coeffs.data(), coeffs_, coeffs.size() * 4, cudaMemcpyDeviceToHost));
        out.info.assign(n_, CblkInfo{});
        out.data.clear();
        out.passLen.clear();
        out.passDist.clear();
        T1Result r;
        for (uint32_t i = 0; i < n_; ++i) {
            const CodeBlock& cb = L_.cblks[i];
            t1EncodeRef(coeffs.data() + (size_t)i * 1024, cb.w, cb.h, L_.bands[cb.band].orient, r);
            CblkInfo& ci = out.info[i];
            ci = {(uint32_t)r.nbps, (uint32_t)r.numPasses, (uint32_t)out.data.size(), (uint32_t)r.data.size(),
                  (uint32_t)out.passLen.size()};
            out.data.insert(out.data.end(), r.data.begin(), r.data.end());
            out.passLen.insert(out.passLen.end(), r.passLen, r.passLen + r.numPasses);
            out.passDist.insert(out.passDist.end(), r.passDist, r.passDist + r.numPasses);
        }
    }

    int dev_;
    const FrameLayout& L_;
    EncodeSettings s_;
    Logger log_;
    uint32_t n_ = 0;
    size_t symCap_ = 0, outCap_ = 0, passCap_ = 0, dataCap_ = 0, inBytes_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_[8] = {};
    std::vector<void*> allocs_;
    void *rgb_ = nullptr, *lut_ = nullptr, *colorParams_ = nullptr, *lut3d_ = nullptr, *planesA_ = nullptr, *planesB_ = nullptr, *coeffs_ = nullptr,
         *cblkTab_ = nullptr, *bandTab_ = nullptr, *info_ = nullptr, *passEnd_ = nullptr, *passDist_ = nullptr,
         *passLen_ = nullptr, *chunks_ = nullptr, *symbols_ = nullptr, *out_ = nullptr, *counters_ = nullptr,
         *offsets_ = nullptr, *perm_ = nullptr, *xyz_ = nullptr, *dInfoOut_ = nullptr, *dPassOut_ = nullptr,
         *dDataOut_ = nullptr;
    void* hostIn_ = nullptr;
    uint32_t* hostInfo_ = nullptr;
    uint32_t* hostPass_ = nullptr;
    uint8_t* hostData_ = nullptr;
    uint32_t* hostSmall_ = nullptr;
    float* frameParams_ = nullptr;
    float* dFrameParams_ = nullptr;
};

class CudaDevice : public Device {
public:
    CudaDevice(int dev, const Logger& log) : dev_(dev), log_(log) {
        cudaDeviceProp p;
        CUCHECK(cudaGetDeviceProperties(&p, dev));
        name_ = p.name;
        CUCHECK(cudaSetDevice(dev));
        // Slot threads sleep in cudaStreamSynchronize instead of spinning: a frame takes
        // milliseconds, and the cores are wanted by the rate control of the other slots.
        CUCHECK(cudaSetDeviceFlags(cudaDeviceMapHost | cudaDeviceScheduleBlockingSync));
        if (log_) {
            char buf[512];
            std::snprintf(buf, sizeof buf, "cuda: device %d %s (sm_%d%d, %d SMs, %.1f GB)", dev, p.name, p.major, p.minor,
                          p.multiProcessorCount, p.totalGlobalMem / 1e9);
            log_(LogLevel::Info, buf);
        }
    }
    std::string name() const override { return name_ + " (CUDA)"; }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override {
        return std::make_unique<CudaSlot>(dev_, L, s, log_);
    }
    uint64_t slotBytes(const FrameLayout& L, const EncodeSettings& s) const override {
        return slotSizes(L, s).deviceBytes();
    }
    DeviceMemory memory() const override {
        DeviceMemory m;
        size_t f = 0, t = 0;
        if (cudaSetDevice(dev_) == cudaSuccess && cudaMemGetInfo(&f, &t) == cudaSuccess) {
            m.total = t;
            m.available = f;
            m.budgeted = true;
        } else {
            cudaGetLastError();
        }
        return m;
    }

private:
    int dev_;
    std::string name_;
    Logger log_;
};

}  // namespace

std::vector<DeviceInfo> listCudaDevices() {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return {};
    std::vector<DeviceInfo> out;
    for (int i = 0; i < n; ++i) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, i) != cudaSuccess) continue;
        DeviceInfo d;
        d.index = i;
        d.name = p.name;
        d.uuid = formatGpuUuid(reinterpret_cast<const uint8_t*>(p.uuid.bytes));
        char pci[32];
        std::snprintf(pci, sizeof pci, "%04x:%02x:%02x.0", p.pciDomainID, p.pciBusID, p.pciDeviceID);
        d.pciBusId = pci;
        d.memoryBytes = p.totalGlobalMem;
        out.push_back(d);
    }
    return out;
}

std::vector<std::unique_ptr<Device>> createCudaDevices(const std::vector<int>& select, const Logger& log) {
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) return {};
    std::vector<std::unique_ptr<Device>> out;
    for (int i = 0; i < n; ++i) {
        if (!select.empty() && std::find(select.begin(), select.end(), i) == select.end()) continue;
        try {
            out.push_back(std::make_unique<CudaDevice>(i, log));
        } catch (const std::exception& e) {
            if (log) log(LogLevel::Warning, "cuda: skipping device " + std::to_string(i) + ": " + e.what());
        }
    }
    return out;
}

}  // namespace gail
