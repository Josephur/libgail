// SPDX-License-Identifier: MIT
// CUDA decode slot. The host has located every code-block's bytes (core/t2_parse); here
// the frame's codestream and a table of block descriptors go to the device, one GPU
// thread decodes each block straight into its place in the wavelet planes, and the
// inverse wavelet and inverse color transform turn those into the picture, which comes
// back through pinned memory. Each slot owns a stream, so frames in flight overlap.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

#include "core/t1_dec_tables.h"
#include "cuda/cuda_common.h"
#include "cuda/cuda_decode.h"
#include "kernels/dwt97.h"
#include "kernels/ict.h"

// ---- shared Tier-1 decode kernel (portable headers) ------------------------------------
// A block writes its words where its coefficients belong in the wavelet planes.
struct T1Out {
    uint32_t* p;
    uint32_t stride;
};
__constant__ uint8_t c_zc[gail::kT1dZcSize];
__constant__ uint8_t c_sc[gail::kT1dScSize];
#define MQD_IN_T const uint8_t*
#define MQD_BYTE(in, i) ((uint32_t)(in)[(i)])
#define T1D_OUT_T const T1Out&
#define T1D_WR(out, i, v) ((out).p[((i) >> 5) * (out).stride + ((i) & 31u)] = (v))
#define T1D_RD(out, i) ((out).p[((i) >> 5) * (out).stride + ((i) & 31u)])
#define T1D_ZC(i) ((uint32_t)c_zc[(i)])
#define T1D_SC(i) ((uint32_t)c_sc[(i)])
#include "kernels/t1_decode.h"

namespace gail {

namespace {

// ---------------------------------------------------------------- Tier-1 + dequantization
// One code-block as the kernel sees it. Slots of a warp left empty have dst == kNoBlock.
struct BlockDesc {
    uint32_t off, len;  // its bytes within the frame's codestream
    uint32_t geom;      // (w - 1) | (h - 1) << 5 | orient << 10 | nbps << 12 | passes << 18
    uint32_t dst;       // index of its top-left sample in the planes
    float half;         // half the quantization step of its band
    uint32_t pad;
};
constexpr uint32_t kNoBlock = 0xFFFFFFFFu;

// One thread per block. The planes are zero beforehand; the thread leaves floats in its
// block: each decoded word (sign, twice the magnitude plus mid-point) times half a step.
__global__ void t1DecodeKernel(const uint8_t* __restrict__ bytes, uint32_t nbytes, const BlockDesc* __restrict__ descs,
                               uint32_t ndesc, uint32_t* __restrict__ planes, uint32_t planeWords, uint32_t stride) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= ndesc) return;
    const BlockDesc d = descs[i];
    if (d.dst == kNoBlock) return;
    const uint32_t w = (d.geom & 31u) + 1u, h = ((d.geom >> 5) & 31u) + 1u, orient = (d.geom >> 10) & 3u;
    const uint32_t nbps = (d.geom >> 12) & 63u, passes = (d.geom >> 18) & 255u;
    // The host builds the table from the frame's layout; these keep a wrong table from
    // reading or writing outside the buffers all the same.
    if (d.dst >= planeWords || planeWords - d.dst < (h - 1u) * stride + w) return;
    uint32_t len = d.len;
    if (d.off > nbytes || len > nbytes - d.off) len = 0u;
    const uint8_t* in = bytes + d.off;
    T1dState st;
    MqDec q;
    mqd_init(q, in, 0u, len);
    const T1Out out{planes + d.dst, stride};
    t1d_decode_block(st, q, in, w, h, orient, nbps, passes, out);
    float* f = (float*)(planes + d.dst);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < w; ++x) {
            const uint32_t v = out.p[y * stride + x];
            const float a = (float)(v & 0x7FFFFFFFu) * d.half;
            f[y * stride + x] = (v & 0x80000000u) ? -a : a;
        }
}

// ---------------------------------------------------------------- inverse wavelet
// The forward kernels of the encoder run backwards: per level rows, then columns
// (F.3, with the scaling of the two bands first and the lifting steps in reverse order).
__constant__ float kUnlift[4] = {(float)-DWT97_DELTA, (float)-DWT97_GAMMA, (float)-DWT97_BETA, (float)-DWT97_ALPHA};
constexpr float kK = (float)DWT97_K;

__device__ __forceinline__ int mirrorIdx(int i, int n) {
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * (n - 1) - i;
    }
    return i;
}

__global__ void idwtRowsKernel(const float* __restrict__ src, float* __restrict__ dst, int stride, int planeSize, int w) {
    __shared__ float row[4096];
    const int y = blockIdx.x, comp = blockIdx.y;
    const size_t base = (size_t)comp * planeSize + (size_t)y * stride;
    if (w == 1) {
        if (threadIdx.x == 0) dst[base] = src[base];
        return;
    }
    const int sn = (w + 1) / 2;
    for (int i = threadIdx.x; i < w; i += blockDim.x) row[i] = (i & 1) ? src[base + sn + i / 2] * (1.f / kK) : src[base + i / 2] * kK;
    __syncthreads();
    for (int s = 0; s < 4; ++s) {
        const int parity = s & 1;
        const float c = kUnlift[s];
        for (int i = threadIdx.x * 2 + parity; i < w; i += 2 * blockDim.x) {
            // symmetric extension at the ends: the mirrored neighbor is row[1] / row[w-2]
            const int im = i > 0 ? i - 1 : 1, ip = i + 1 < w ? i + 1 : w - 2;
            row[i] += c * (row[im] + row[ip]);
        }
        __syncthreads();
    }
    for (int i = threadIdx.x; i < w; i += blockDim.x) dst[base + i] = row[i];
}

constexpr int TILE = 64, HALO = 8, ROWS = TILE + 2 * HALO;

__global__ void idwtColsKernel(const float* __restrict__ src, float* __restrict__ dst, int stride, int planeSize, int w,
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
    const int sn = (h + 1) / 2;
    for (int r = ly; r < ROWS; r += 8) {
        // sample y of the interleaved column: even ones from the low band, odd from the high
        const int y = mirrorIdx(y0 - HALO + r, h);
        tile[r][lx] = !col ? 0.f : (y & 1) ? src[base + (size_t)(sn + y / 2) * stride] * (1.f / kK) : src[base + (size_t)(y / 2) * stride] * kK;
    }
    __syncthreads();
    for (int s = 0; s < 4; ++s) {
        const int parity = s & 1;
        const float c = kUnlift[s];
        // every step makes one more row at each edge of the tile stale: the halo takes them
        for (int r = parity + 2 * ly; r < ROWS - 1; r += 16)
            if (r > 0) tile[r][lx] += c * (tile[r - 1][lx] + tile[r + 1][lx]);
        __syncthreads();
    }
    if (!col) return;
    for (int r = HALO + ly; r < HALO + TILE; r += 8) {
        const int y = y0 - HALO + r;
        if (y >= h) break;
        dst[base + (size_t)y * stride] = tile[r][lx];
    }
}

// ---------------------------------------------------------------- inverse color transform
// Inverse ICT (G.3), level shift, rounding and clamping to 12 bits, three words per pixel.
__global__ void colorOutKernel(const float* __restrict__ planes, uint16_t* __restrict__ out, int stride, int planeSize, int w,
                               int h, int shift) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    const size_t i = (size_t)y * stride + x;
    const float y0 = planes[i], y1 = planes[planeSize + i], y2 = planes[2 * (size_t)planeSize + i];
    const float v[3] = {y0 + (float)ICT_R_Y2 * y2, y0 + (float)ICT_G_Y1 * y1 + (float)ICT_G_Y2 * y2, y0 + (float)ICT_B_Y1 * y1};
    uint16_t* o = out + ((size_t)y * w + x) * 3;
    for (int k = 0; k < 3; ++k) {
        const int code = (int)floorf(v[k] + 2048.5f);
        o[k] = (uint16_t)(min(max(code, 0), 4095) << shift);
    }
}

double msSince(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

class CudaDecoderSlot : public DecoderSlot {
public:
    CudaDecoderSlot(int dev, const FrameLayout& L, const DecodeSettings& s) : dev_(dev), L_(L), s_(s) {
        try {
            init();
        } catch (...) {
            release();
            throw;
        }
    }
    ~CudaDecoderSlot() override { release(); }

    void decode(const FrameLayout& frame, const uint8_t* data, size_t size, const std::vector<CblkCode>& blocks, void* dst,
                ptrdiff_t strideBytes, DecodeTimings* tm) override {
        if (!sameGeometry(frame, L_) || (int)blocks.size() != L_.numCblks())
            throw std::logic_error("frame does not match the layout the slot was made for");
        if (size > 0xFFFFFFFFu) throw std::runtime_error("codestream too large");
        const auto t0 = std::chrono::steady_clock::now();
        CUCHECK(cudaSetDevice(dev_));
        const cudaStream_t st = stream_;
        const uint32_t W = (uint32_t)L_.width, H = (uint32_t)L_.height, npix = W * H;

        // The coded blocks, longest first, packed into warps by a byte budget: a warp runs
        // as long as its longest thread, so a long block gets a warp to itself and short
        // ones share. Slots left over are marked empty.
        order_.clear();
        for (int i : wanted_)
            if (blocks[i].passes && (size_t)blocks[i].offset + blocks[i].len <= size) order_.push_back(i);
        std::sort(order_.begin(), order_.end(), [&](int a, int b) { return blocks[a].len > blocks[b].len; });
        ensureBytes(size);
        ensureDescs(order_.size() * 32 + 32);
        uint32_t n = 0, inWarp = 0, warpBytes = 0;
        const BlockDesc empty{0, 0, 0, kNoBlock, 0.f, 0};
        for (int i : order_) {
            const CblkCode& c = blocks[i];
            if (inWarp && (inWarp == 32 || warpBytes + c.len > budget_)) {
                for (; inWarp < 32; ++inWarp) hDescs_[n++] = empty;
                inWarp = 0;
                warpBytes = 0;
            }
            const CodeBlock& cb = L_.cblks[i];
            const Subband& b = frame.bands[cb.band];
            BlockDesc d;
            d.off = c.offset;
            d.len = c.len;
            d.geom = (uint32_t)(cb.w - 1) | ((uint32_t)(cb.h - 1) << 5) | ((uint32_t)b.orient << 10) | ((uint32_t)c.nbps << 12) |
                     ((uint32_t)c.passes << 18);
            d.dst = (uint32_t)b.comp * npix + (uint32_t)(b.py0 + cb.y0) * W + (uint32_t)(b.px0 + cb.x0);
            d.half = 0.5f * b.delta;
            d.pad = 0;
            hDescs_[n++] = d;
            ++inWarp;
            warpBytes += c.len;
        }
        std::memcpy(hBytes_, data, size);

        CUCHECK(cudaEventRecord(ev_[0], st));
        CUCHECK(cudaMemcpyAsync(dBytes_, hBytes_, size, cudaMemcpyHostToDevice, st));
        if (n) CUCHECK(cudaMemcpyAsync(dDescs_, hDescs_, (size_t)n * sizeof(BlockDesc), cudaMemcpyHostToDevice, st));
        CUCHECK(cudaMemsetAsync(planesA_, 0, (size_t)npix * 3 * sizeof(float), st));
        if (n)
            t1DecodeKernel<<<(n + 63) / 64, 64, 0, st>>>((const uint8_t*)dBytes_, (uint32_t)size, (const BlockDesc*)dDescs_, n,
                                                          (uint32_t*)planesA_, npix * 3, W);
        CUCHECK(cudaEventRecord(ev_[1], st));
        if (tm && tm->wavelet) {  // tests: the dequantized coefficients, before the wavelet
            CUCHECK(cudaStreamSynchronize(st));
            for (int c = 0; c < 3; ++c) {
                tm->wavelet[c].resize(npix);
                CUCHECK(cudaMemcpy(tm->wavelet[c].data(), (float*)planesA_ + (size_t)c * npix, (size_t)npix * sizeof(float),
                                   cudaMemcpyDeviceToHost));
            }
        }
        for (int l = L_.levels; l > s_.reduce; --l) {
            const int w = ws_[l - 1], h = hs_[l - 1];
            idwtRowsKernel<<<dim3(h, 3), 256, 0, st>>>((float*)planesA_, (float*)planesB_, W, npix, w);
            idwtColsKernel<<<dim3((w + 31) / 32, (h + TILE - 1) / TILE, 3), dim3(32, 8), 0, st>>>((float*)planesB_, (float*)planesA_,
                                                                                               W, npix, w, h);
        }
        CUCHECK(cudaEventRecord(ev_[2], st));
        colorOutKernel<<<dim3((outW_ + 255) / 256, outH_), 256, 0, st>>>((float*)planesA_, (uint16_t*)dOut_, W, npix, outW_, outH_,
                                                                         s_.highBits ? 4 : 0);
        const size_t outBytes = (size_t)outW_ * outH_ * 6;
        CUCHECK(cudaMemcpyAsync(hOut_, dOut_, outBytes, cudaMemcpyDeviceToHost, st));
        CUCHECK(cudaEventRecord(ev_[3], st));
        CUCHECK(cudaGetLastError());
        CUCHECK(cudaStreamSynchronize(st));
        const size_t rowBytes = (size_t)outW_ * 6;
        if (strideBytes == (ptrdiff_t)rowBytes) {
            std::memcpy(dst, hOut_, outBytes);
        } else {
            for (int y = 0; y < outH_; ++y) std::memcpy((uint8_t*)dst + (ptrdiff_t)y * strideBytes, (uint8_t*)hOut_ + (size_t)y * rowBytes, rowBytes);
        }
        if (tm) {
            float a = 0, b = 0, c = 0;
            cudaEventElapsedTime(&a, ev_[0], ev_[1]);
            cudaEventElapsedTime(&b, ev_[1], ev_[2]);
            cudaEventElapsedTime(&c, ev_[2], ev_[3]);
            tm->t1 = a;
            tm->dwt = b;
            tm->color = c;
            tm->total = msSince(t0);
        }
    }

private:
    void init() {
        CUCHECK(cudaSetDevice(dev_));
        cudaDeviceProp p;
        CUCHECK(cudaGetDeviceProperties(&p, dev_));
        // Many small warps suit a GPU with many multiprocessors; a smaller one does better
        // with fewer, fuller warps (measured on an 82-SM and a 36-SM device).
        budget_ = p.multiProcessorCount >= 64 ? 1000u : 8000u;
        if (s_.reduce < 0 || s_.reduce > L_.levels) throw std::runtime_error("more resolution levels left out than the stream has");
        uint8_t zc[kT1dZcSize], sc[kT1dScSize];
        buildT1DecodeTables(zc, sc);
        CUCHECK(cudaMemcpyToSymbol(c_zc, zc, sizeof zc));
        CUCHECK(cudaMemcpyToSymbol(c_sc, sc, sizeof sc));
        CUCHECK(cudaStreamCreate(&stream_));
        for (auto& e : ev_) CUCHECK(cudaEventCreate(&e));
        ws_.resize(L_.levels + 1);
        hs_.resize(L_.levels + 1);
        ws_[0] = L_.width;
        hs_[0] = L_.height;
        for (int l = 1; l <= L_.levels; ++l) {
            ws_[l] = (ws_[l - 1] + 1) / 2;
            hs_[l] = (hs_[l - 1] + 1) / 2;
        }
        decodedSize(L_, s_.reduce, outW_, outH_);
        for (int i = 0; i < L_.numCblks(); ++i)
            if (L_.bands[L_.cblks[i].band].res <= L_.levels - s_.reduce) wanted_.push_back(i);
        const size_t planeBytes = (size_t)L_.width * L_.height * 3 * sizeof(float);
        const size_t outBytes = (size_t)outW_ * outH_ * 6;
        cuAlloc(cudaMalloc(&planesA_, planeBytes), "device memory", planeBytes);
        cuAlloc(cudaMalloc(&planesB_, planeBytes), "device memory", planeBytes);
        cuAlloc(cudaMalloc(&dOut_, outBytes), "device memory", outBytes);
        cuAlloc(cudaHostAlloc(&hOut_, outBytes, cudaHostAllocDefault), "pinned host memory", outBytes);
        ensureBytes(1500000);  // a DCI frame is at most 250 Mbit/s over 24 frames
        ensureDescs((size_t)wanted_.size() + 32);
    }

    void ensureBytes(size_t n) {
        if (n <= capBytes_) return;
        if (dBytes_) cudaFree(dBytes_);
        if (hBytes_) cudaFreeHost(hBytes_);
        dBytes_ = nullptr;
        hBytes_ = nullptr;
        capBytes_ = 0;
        const size_t cap = n + n / 2;
        cuAlloc(cudaMalloc(&dBytes_, cap), "device memory", cap);
        cuAlloc(cudaHostAlloc((void**)&hBytes_, cap, cudaHostAllocDefault), "pinned host memory", cap);
        capBytes_ = cap;
    }
    void ensureDescs(size_t n) {
        if (n <= capDescs_) return;
        if (dDescs_) cudaFree(dDescs_);
        if (hDescs_) cudaFreeHost(hDescs_);
        dDescs_ = nullptr;
        hDescs_ = nullptr;
        capDescs_ = 0;
        const size_t cap = n + n / 2, bytes = cap * sizeof(BlockDesc);
        cuAlloc(cudaMalloc(&dDescs_, bytes), "device memory", bytes);
        cuAlloc(cudaHostAlloc((void**)&hDescs_, bytes, cudaHostAllocDefault), "pinned host memory", bytes);
        capDescs_ = cap;
    }

    void release() {
        cudaSetDevice(dev_);
        if (stream_) cudaStreamSynchronize(stream_);
        for (void* p : {planesA_, planesB_, dOut_, dBytes_, (void*)dDescs_})
            if (p) cudaFree(p);
        for (void* p : {hOut_, (void*)hBytes_, (void*)hDescs_})
            if (p) cudaFreeHost(p);
        for (auto& e : ev_)
            if (e) cudaEventDestroy(e);
        if (stream_) cudaStreamDestroy(stream_);
        cudaGetLastError();
    }

    int dev_;
    const FrameLayout L_;
    const DecodeSettings s_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t ev_[4] = {};
    uint32_t budget_ = 1000;
    std::vector<int> ws_, hs_, wanted_, order_;
    int outW_ = 0, outH_ = 0;
    void *planesA_ = nullptr, *planesB_ = nullptr, *dOut_ = nullptr, *hOut_ = nullptr, *dBytes_ = nullptr;
    uint8_t* hBytes_ = nullptr;
    BlockDesc* dDescs_ = nullptr;
    BlockDesc* hDescs_ = nullptr;
    size_t capBytes_ = 0, capDescs_ = 0;
};

}  // namespace

std::unique_ptr<DecoderSlot> createCudaDecoderSlot(int device, const FrameLayout& L, const DecodeSettings& s) {
    return std::make_unique<CudaDecoderSlot>(device, L, s);
}

}  // namespace gail
