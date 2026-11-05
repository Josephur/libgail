// SPDX-License-Identifier: MIT
// Unit / integration tests for the Gail encoder.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <algorithm>
#include <atomic>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "core/backend.h"
#include "core/codestream_check.h"
#include "core/dwt_cpu.h"
#include "core/frame_source.h"
#include "core/pipeline.h"
#include "core/quant.h"
#include "core/rate_control.h"
#include "core/t1_ref.h"
#include "core/t2.h"

#define MQ_OUT_T uint8_t*
#define MQ_PUT(o, i, v) ((o)[(i)] = (uint8_t)(v))
#define MQ_GET(o, i) ((uint32_t)(o)[(i)])
#include "t1_decoder.h"

using namespace gail;

static int g_fail = 0;
#define CHECK(cond, ...)                                  \
    do {                                                  \
        if (!(cond)) {                                    \
            std::printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            std::printf(__VA_ARGS__);                     \
            std::printf("\n");                            \
            ++g_fail;                                     \
        }                                                 \
    } while (0)

static void testDwtRoundTrip() {
    const int W = 301, H = 157, NL = 5;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> u(-2048, 2047);
    std::vector<float> img(W * H);
    for (auto& v : img) v = u(rng);
    std::vector<float> f = img;
    fdwt97_2d(f.data(), W, H, NL);
    std::vector<double> d(f.begin(), f.end());
    idwt97_2d(d.data(), W, H, NL);
    double maxErr = 0;
    for (int i = 0; i < W * H; ++i) maxErr = std::max(maxErr, std::fabs(d[i] - img[i]));
    std::printf("dwt round trip max error %.3g\n", maxErr);
    CHECK(maxErr < 0.05, "round trip error too large");
}

static void testNorms() {
    for (int l = 1; l <= 3; ++l)
        std::printf("norms level %d: LL %.4f HL %.4f HH %.4f\n", l, synthesisNorm(l, 0), synthesisNorm(l, 1),
                    synthesisNorm(l, 3));
}

// Encode random code-blocks and verify every truncation point decodes exactly.
static void testT1Truncation() {
    std::mt19937 rng(42);
    int blocks = 0, checks = 0;
    long long bytesUsed = 0, bytesOpj = 0;
    for (int trial = 0; trial < 400; ++trial) {
        const int w = trial % 7 == 0 ? 1 + rng() % 32 : 32;
        const int h = trial % 5 == 0 ? 1 + rng() % 32 : 32;
        const int orient = trial % 4;
        std::vector<uint32_t> c(kCblkArea, 0);
        const int mode = trial % 3;
        std::normal_distribution<double> g(0, mode == 0 ? 20 : (mode == 1 ? 300 : 3000));
        std::uniform_real_distribution<double> sp(0, 1);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                double v = g(rng);
                if (mode == 1 && sp(rng) < 0.7) v *= 0.05;  // sparse
                c[y * 32 + x] = quantizeCoeff((float)v, 1.0f, (1u << 26) - 1);
            }
        T1Result res;
        T1Trace tr;
        t1EncodeRef(c.data(), w, h, orient, res, &tr);
        ++blocks;
        if (!res.nbps) continue;
        // full decode must reproduce all magnitude bits and signs
        auto full = test::t1Decode(res.data.data(), res.data.size(), w, h, orient, res.nbps, res.numPasses);
        bool okFull = true;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const uint32_t m = (c[y * 32 + x] & 0x7FFFFFFF) >> kFracBits;
                if (full.mag[y * w + x] != m) okFull = false;
                if (m && full.sign[y * w + x] != (c[y * 32 + x] >> 31)) okFull = false;
            }
        CHECK(okFull, "full decode mismatch trial %d", trial);
        for (int k = 1; k <= res.numPasses; ++k) {
            const uint32_t L = res.passLen[k - 1];
            auto ref = test::t1Decode(res.data.data(), res.data.size(), w, h, orient, res.nbps, k);
            auto tru = test::t1Decode(res.data.data(), L, w, h, orient, res.nbps, k);
            const bool same = ref.mag == tru.mag && ref.sig == tru.sig;
            bool signs = true;
            for (int i = 0; i < w * h; ++i)
                if (ref.sig[i] && ref.sign[i] != tru.sign[i]) signs = false;
            CHECK(same && signs, "truncation after pass %d (len %u of %zu) decodes wrong, trial %d", k, L,
                  res.data.size(), trial);
            CHECK(L == res.data.size() || res.data[L - 1] != 0xFF, "truncation ends in 0xFF");
            ++checks;
            if (k < res.numPasses) {
                ++bytesOpj;
                if (L == res.data.size()) ++bytesUsed;  // fell back to the whole codeword
            }
        }
    }
    std::printf("t1 truncation: %d blocks, %d truncation points verified, %lld of %lld intermediate points fell back "
                "to the full codeword\n", blocks, checks, bytesUsed, bytesOpj);
}

static void testFrameCpu(Profile prof) {
    int W, H;
    containerSize(prof, Container::Flat, W, H);
    FrameLayout L = makeLayout(W, H, prof, 1.0);
    auto src = openPatternSource(W, H, 1, 3);
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    src->read(rgb.data());
    EncodeSettings es;
    es.profile = prof;
    es.width = W;
    es.height = H;
    es.input = src->format();
    auto dev = createCpuDevice(0);
    auto slot = dev->createSlot(L, es);
    Tier1Frame t1;
    FrameTimings tm;
    FrameDebug dbg;
    slot->encode(rgb.data(), t1, &tm, &dbg);
    if (const char* lim = std::getenv("GAIL_TEST_LIMIT")) {
        // Re-code every block with a pass limit (emulates Tier-1 early termination).
        const int maxP = std::atoi(lim);
        Tier1Frame t2f;
        t2f.info.assign(L.numCblks(), CblkInfo{});
        for (int i = 0; i < L.numCblks(); ++i) {
            const CodeBlock& cb = L.cblks[i];
            T1Result r;
            t1EncodeRef(dbg.coeffs.data() + (size_t)i * kCblkArea, cb.w, cb.h, L.bands[cb.band].orient, r, nullptr, maxP);
            CblkInfo& ci = t2f.info[i];
            ci.nbps = r.nbps;
            ci.numPasses = r.numPasses;
            ci.dataOffset = (uint32_t)t2f.data.size();
            ci.dataLen = (uint32_t)r.data.size();
            ci.passBase = (uint32_t)t2f.passLen.size();
            t2f.data.insert(t2f.data.end(), r.data.begin(), r.data.end());
            t2f.passLen.insert(t2f.passLen.end(), r.passLen, r.passLen + r.numPasses);
            t2f.passDist.insert(t2f.passDist.end(), r.passDist, r.passDist + r.numPasses);
        }
        t1 = std::move(t2f);
    }
    auto tq0 = std::chrono::steady_clock::now();
    RateResult rr = allocateRate(L, t1, dciLimits(24));
    auto tq1 = std::chrono::steady_clock::now();
    std::vector<uint8_t> cs;
    buildCodestream(L, t1, rr.passes.data(), &cs, nullptr);
    auto tq2 = std::chrono::steady_clock::now();
    CodestreamSizes szm;
    buildCodestream(L, t1, rr.passes.data(), nullptr, &szm);
    auto tq3 = std::chrono::steady_clock::now();
    CheckReport rep = checkDciCodestream(cs.data(), cs.size(), 24);
    auto tq4 = std::chrono::steady_clock::now();
    auto msd = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
    std::printf("  timing: allocateRate %.2f ms, build %.2f ms, measure-only %.2f ms, check %.2f ms\n", msd(tq0, tq1),
                msd(tq1, tq2), msd(tq2, tq3), msd(tq3, tq4));
    std::printf("cpu frame %s: %zu bytes, T1 %.0f ms, RC iterations %d, est PSNR %.2f dB\n  %s\n", profileName(prof),
                cs.size(), tm.total, rr.iterations,
                10 * std::log10(4095.0 * 4095 / (rr.distortion / ((double)W * H * 3))),
                formatReport(rep, true).c_str());
    CHECK(rep.ok, "frame not DCI compliant");
    CHECK(rep.comment == codestreamLabel(), "frame not labelled as GAIL's");
    const std::string path = std::string("/tmp/gail_test_") + (prof == Profile::Cinema4K ? "4k" : "2k") + ".j2c";
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f) {
        std::fwrite(cs.data(), 1, cs.size(), f);
        std::fclose(f);
    }
    f = std::fopen((path.substr(0, path.size() - 4) + ".xyz").c_str(), "wb");
    if (f) {
        std::fwrite(dbg.xyz12.data(), 2, dbg.xyz12.size(), f);
        std::fclose(f);
    }
}

#ifdef GAIL_HAVE_VULKAN
static void testGpu(Profile prof, bool cuda) {
    int W, H;
    containerSize(prof, Container::Flat, W, H);
    FrameLayout L = makeLayout(W, H, prof, 1.0);
    auto src = openPatternSource(W, H, 1, 3);
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    src->read(rgb.data());
    EncodeSettings es;
    es.profile = prof;
    es.width = W;
    es.height = H;
    es.input = src->format();
    std::vector<std::unique_ptr<Device>> devs;
#ifdef GAIL_HAVE_CUDA
    if (cuda) devs = createCudaDevices({}, stderrLogger(true));
#endif
    if (!cuda) devs = createVulkanDevices({}, stderrLogger(true));
    if (devs.empty()) {
        std::printf("  (skipped: no %s devices on this machine)\n", cuda ? "CUDA" : "Vulkan");
        return;
    }
    // CPU reference for the colour/DWT stages
    auto cpu = createCpuDevice(0)->createSlot(L, es);
    Tier1Frame tc;
    FrameDebug dc;
    cpu->encode(rgb.data(), tc, nullptr, &dc);
    for (auto& d : devs) {
        auto slot = d->createSlot(L, es);
        Tier1Frame tg;
        FrameDebug dg;
        FrameTimings tm;
        if (const char* st = std::getenv("GAIL_TEST_SLOPE")) {
            // derive a realistic threshold from a full encode of the same frame
            slot->encode(rgb.data(), tg, &tm, nullptr);
            RateResult r0 = allocateRate(L, tg, dciLimits(24));
            const double thr = r0.lambdaComp[0] * std::atof(st);
            std::printf("  early termination threshold %.4g (lambda %.4g)\n", thr, r0.lambdaComp[0]);
            slot->setSlopeThreshold(thr);
        }
        slot->encode(rgb.data(), tg, &tm, &dg);
        {
            // Compare the GPU wavelet against the CPU transform of the GPU's own colour output.
            ColorConverter cc(es.color);
            std::vector<float> pl[3];
            for (auto& p : pl) p.resize((size_t)W * H);
            for (size_t i = 0; i < (size_t)W * H; ++i) {
                const uint16_t* x = &dg.xyz12[3 * i];
                ictForward(x[0] - 2048.f, x[1] - 2048.f, x[2] - 2048.f, pl[0][i], pl[1][i], pl[2][i]);
            }
            double maxErr = 0;
            size_t where = 0;
            for (int c = 0; c < 3; ++c) {
                fdwt97_2d(pl[c].data(), W, H, L.levels);
                for (size_t i = 0; i < (size_t)W * H; ++i) {
                    const double e = std::fabs(pl[c][i] - dg.wavelet[(size_t)c * W * H + i]);
                    if (e > maxErr) {
                        maxErr = e;
                        where = (size_t)c * W * H + i;
                    }
                }
            }
            const size_t wi = where % ((size_t)W * H);
            std::printf("  DWT max abs diff vs CPU %.4g at comp %zu x %zu y %zu\n", maxErr, where / ((size_t)W * H),
                        wi % W, wi / W);
            CHECK(maxErr < 0.05, "GPU DWT differs from CPU");
        }
        size_t xyzDiff = 0, coefDiff = 0;
        for (size_t i = 0; i < dg.xyz12.size(); ++i) xyzDiff += dg.xyz12[i] != dc.xyz12[i];
        for (size_t i = 0; i < dg.coeffs.size(); ++i) {
            const int64_t a = dg.coeffs[i] & 0x7FFFFFFF, b = dc.coeffs[i] & 0x7FFFFFFF;
            coefDiff += std::llabs(a - b) > 64;  // more than one quantisation step apart
        }
        std::printf("%s: xyz codes differing %zu / %zu, coefficients off by >1 step %zu / %zu\n", d->name().c_str(),
                    xyzDiff, dg.xyz12.size(), coefDiff, dg.coeffs.size());
        // Tier-1 must be bit exact on identical coefficients.
        int bad = 0;
        const int n = L.numCblks();
        for (int i = 0; i < n && bad < 5; ++i) {
            const CodeBlock& cb = L.cblks[i];
            T1Result r;
            const CblkInfo& gi = tg.info[i];
            t1EncodeRef(dg.coeffs.data() + (size_t)i * kCblkArea, cb.w, cb.h, L.bands[cb.band].orient, r, nullptr,
                        gi.numPasses ? (int)gi.numPasses : kMaxPasses);
            bool ok = (int)gi.nbps == r.nbps && (int)gi.numPasses == r.numPasses && gi.dataLen == r.data.size() &&
                      (r.data.empty() || std::memcmp(tg.data.data() + gi.dataOffset, r.data.data(), r.data.size()) == 0);
            for (int k = 0; ok && k < r.numPasses; ++k)
                ok = tg.len(i, k) == r.passLen[k] && tg.dist(i, k) == r.passDist[k];
            if (!ok) {
                ++bad;
                std::printf("  cblk %d (band %d %dx%d orient %d): gpu nbps %u passes %u len %u | cpu nbps %d passes %d len %zu\n",
                            i, cb.band, cb.w, cb.h, L.bands[cb.band].orient, gi.nbps, gi.numPasses, gi.dataLen, r.nbps,
                            r.numPasses, r.data.size());
            }
        }
        CHECK(bad == 0, "GPU Tier-1 differs from CPU reference");
        {
            // SIMT efficiency of one-block-per-thread MQ coding: natural vs size-sorted order
            std::vector<uint32_t> cnt(n);
            for (int i = 0; i < n; ++i) {
                T1Trace tr;
                T1Result r;
                const CodeBlock& cb = L.cblks[i];
                t1EncodeRef(dg.coeffs.data() + (size_t)i * kCblkArea, cb.w, cb.h, L.bands[cb.band].orient, r, &tr);
                cnt[i] = (uint32_t)tr.symbols.size();
            }
            auto eff = [&](const std::vector<uint32_t>& v) {
                double sum = 0, work = 0;
                for (size_t w0 = 0; w0 < v.size(); w0 += 32) {
                    uint32_t mx = 0;
                    for (size_t k = w0; k < std::min(v.size(), w0 + 32); ++k) {
                        mx = std::max(mx, v[k]);
                        sum += v[k];
                    }
                    work += 32.0 * mx;
                }
                return sum / work;
            };
            std::vector<uint32_t> sorted = cnt;
            std::sort(sorted.begin(), sorted.end(), std::greater<uint32_t>());
            std::printf("  MQ lane efficiency: natural order %.1f%%, sorted %.1f%%, max block %u symbols, mean %.0f\n",
                        100 * eff(cnt), 100 * eff(sorted), sorted[0],
                        std::accumulate(cnt.begin(), cnt.end(), 0.0) / n);
        }
        if (std::getenv("GAIL_TEST_SLOPE")) {
            int early = 0, shown = 0;
            for (int i = 0; i < n; ++i) {
                const CodeBlock& cb = L.cblks[i];
                const Subband& b = L.bands[cb.band];
                T1Result r;
                t1EncodeRef(dg.coeffs.data() + (size_t)i * kCblkArea, cb.w, cb.h, b.orient, r);
                const int ng = (int)tg.info[i].numPasses;
                if (ng >= r.numPasses) continue;
                ++early;
                if (shown++ < 8) {
                    const double sc = (double)b.delta * b.delta * b.weight;
                    std::printf("  cblk %d band %d (lvl %d orient %d comp %d) nbps %d passes %d/%d:", i, cb.band, b.level,
                                b.orient, b.comp, r.nbps, ng, r.numPasses);
                    uint32_t prev = 0, prevSym = 0;
                    T1Trace tr;
                    T1Result r2;
                    t1EncodeRef(dg.coeffs.data() + (size_t)i * kCblkArea, cb.w, cb.h, b.orient, r2, &tr);
                    for (int k = 0; k < std::min(r.numPasses, ng + 2); ++k) {
                        const double dist = r.passDist[k] * passPlaneScale(r.nbps, k) * sc;
                        const uint32_t syms = tr.passEnd[k] - prevSym;
                        std::printf(" [k%d D=%.3g R=%u syms=%u est=%.3g act=%.3g]", k, dist, r.passLen[k] - prev, syms,
                                    dist / (syms * 0.5 / 8 + 1), dist / std::max(1u, r.passLen[k] - prev));
                        prev = r.passLen[k];
                        prevSym = tr.passEnd[k];
                    }
                    std::printf("\n");
                }
            }
            std::printf("  %d of %d blocks stopped early\n", early, n);
        }
        std::printf("  GPU time %.2f ms (upload %.2f color %.2f dwt %.2f quant %.2f t1cm %.2f t1mq %.2f download %.2f), "
                    "wall %.2f ms, data %zu bytes\n",
                    tm.gpu, tm.upload, tm.color, tm.dwt, tm.quant, tm.t1cm, tm.t1mq, tm.download, tm.total,
                    tg.data.size());
        RateResult rr = allocateRate(L, tg, dciLimits(24));
        std::vector<uint8_t> cs;
        buildCodestream(L, tg, rr.passes.data(), &cs, nullptr);
        CheckReport rep = checkDciCodestream(cs.data(), cs.size(), 24);
        CHECK(rep.ok, "GPU frame not compliant: %s", formatReport(rep, false).c_str());
        const std::string path = std::string("/tmp/gail_gpu_") + (prof == Profile::Cinema4K ? "4k" : "2k") + ".j2c";
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f) {
            std::fwrite(cs.data(), 1, cs.size(), f);
            std::fclose(f);
        }
        f = std::fopen((path.substr(0, path.size() - 4) + ".xyz").c_str(), "wb");
        if (f) {
            std::fwrite(dg.xyz12.data(), 2, dg.xyz12.size(), f);
            std::fclose(f);
        }
    }
}
#endif

// Code-blocks coded with a pass limit (early termination) must still decode exactly.
static void testT1PassLimit() {
    std::mt19937 rng(9);
    int checks = 0;
    for (int trial = 0; trial < 300; ++trial) {
        const int w = 32, h = trial % 3 == 0 ? 17 : 32, orient = trial % 4;
        std::vector<uint32_t> c(kCblkArea, 0);
        std::normal_distribution<double> g(0, 500);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) c[y * 32 + x] = quantizeCoeff((float)g(rng), 1.0f, (1u << 26) - 1);
        T1Result full;
        t1EncodeRef(c.data(), w, h, orient, full);
        const int limit = 1 + trial % std::max(1, full.numPasses - 1);
        T1Result lim;
        t1EncodeRef(c.data(), w, h, orient, lim, nullptr, limit);
        CHECK(lim.numPasses == limit, "pass limit not honoured");
        for (int k = 1; k <= lim.numPasses; ++k) {
            auto ref = test::t1Decode(full.data.data(), full.data.size(), w, h, orient, full.nbps, k);
            auto got = test::t1Decode(lim.data.data(), lim.passLen[k - 1], w, h, orient, lim.nbps, k);
            CHECK(ref.mag == got.mag && ref.sig == got.sig, "limited codeword decodes wrong at pass %d/%d trial %d", k,
                  limit, trial);
            ++checks;
        }
    }
    std::printf("pass-limited code-blocks: %d truncation points verified\n", checks);
}

// Native Y'CbCr input: GPU conversion vs the CPU reference, and vs FFmpeg's RGB48 path.
static void testYuv(Profile prof) {
    const char* video = "test-media/real_4k_hdr_30s.mkv";
    if (FILE* f = std::fopen(video, "rb")) std::fclose(f);
    else {
        std::printf("  (skipped: %s not found)\n", video);
        return;
    }
    int W, H;
    containerSize(prof, Container::Flat, W, H);
    FrameLayout L = makeLayout(W, H, prof, 1.0);
    SourceInfo info;
    try {
        info = probeSource(video);
    } catch (const std::exception& e) {
        std::printf("  (skipped: %s)\n", e.what());  // no ffprobe on this machine
        return;
    }
    EncodeSettings es;
    es.profile = prof;
    es.width = W;
    es.height = H;
    es.color.transfer = parseTransfer(info.transfer);
    es.color.primaries = parsePrimaries(info.primaries);
    es.color.hdrPeak = info.masteringPeak > 0 ? info.masteringPeak : 1000;
    auto grab = [&](bool yuv, std::vector<uint8_t>& frame) {
        SourceOptions so;
        so.path = video;
        so.width = W;
        so.height = H;
        so.start = 12;
        so.maxFrames = 1;
        so.yuv = yuv;
        auto src = openFfmpegSource(so, info);
        frame.resize(src->format().bytes());
        src->read(frame.data());
        return src->format();
    };
    std::vector<uint8_t> fy, fr;
    const FrameFormat yf = grab(true, fy), rf = grab(false, fr);
    CHECK(yf.kind == FrameFormat::Kind::Planar && !yf.rgb && yf.sx && yf.sy, "source not delivered as Y'CbCr 4:2:0");
    std::printf("  yuv frame: %dx%d at (%d,%d) in %dx%d, %d-bit, %s range, Kr %.4f\n", yf.activeW, yf.activeH, yf.ox,
                yf.oy, yf.width, yf.height, yf.bits, yf.fullRange ? "full" : "limited", yf.kr);
    auto xyzOf = [&](Device& d, const FrameFormat& f, const std::vector<uint8_t>& frame) {
        EncodeSettings e = es;
        e.input = f;
        auto slot = d.createSlot(L, e);
        Tier1Frame t1;
        FrameDebug dbg;
        dbg.full = false;
        slot->encode(frame.data(), t1, nullptr, &dbg);
        return dbg.xyz12;
    };
    auto cpu = createCpuDevice(0);
    const auto ref = xyzOf(*cpu, yf, fy);
    const auto viaRgb = xyzOf(*cpu, rf, fr);
    auto compare = [&](const char* what, const std::vector<uint16_t>& a, double minPsnr) {
        double mse = 0;
        int maxd = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            const int d = (int)a[i] - (int)ref[i];
            mse += (double)d * d;
            maxd = std::max(maxd, std::abs(d));
        }
        mse /= a.size();
        const double p = mse > 0 ? 10 * std::log10(4095.0 * 4095.0 / mse) : 99;
        std::printf("  %-34s vs CPU Y'CbCr reference: PSNR %.2f dB, max diff %d\n", what, p, maxd);
        CHECK(p >= minPsnr, "%s deviates too much", what);
    };
    compare("FFmpeg RGB48 conversion", viaRgb, 40.0);
#ifdef GAIL_HAVE_VULKAN
    for (auto& d : createVulkanDevices({}, stderrLogger(false))) compare(d->name().c_str(), xyzOf(*d, yf, fy), 70.0);
#endif
#ifdef GAIL_HAVE_CUDA
    for (auto& d : createCudaDevices({}, stderrLogger(false))) compare(d->name().c_str(), xyzOf(*d, yf, fy), 70.0);
#endif
}

// Planar input layouts without media files: Y'CbCr 4:2:0 / 4:2:2 / 4:4:4 and G'B'R' at every
// depth are built from a known R'G'B' picture, the CPU reference must give that picture
// back, and every GPU must convert like the CPU. The picture is odd-sized, so the last
// chroma column and row cover a single luma sample.
static void testPlanar() {
    const Profile prof = Profile::Cinema2K;
    int W, H;
    containerSize(prof, Container::Flat, W, H);
    const int aw = 1917, ah = 803;
    // linear ramps: chroma interpolation reproduces them, so only quantisation remains
    auto rgbAt = [&](int x, int y, double out[3]) {
        out[0] = 0.05 + 0.9 * x / (aw - 1);
        out[1] = 0.05 + 0.9 * y / (ah - 1);
        out[2] = 0.05 + 0.9 * (x + y) / (aw + ah - 2);
    };
    struct Layout {
        const char* name;
        int sx, sy;
        bool rgb;
        int bits;
    };
    const Layout layouts[] = {{"yuv420p", 1, 1, false, 8},  {"yuv420p10", 1, 1, false, 10}, {"yuv420p12", 1, 1, false, 12},
                              {"yuv422p", 1, 0, false, 8},  {"yuv422p10", 1, 0, false, 10}, {"yuv422p12", 1, 0, false, 12},
                              {"yuv444p", 0, 0, false, 8},  {"yuv444p10", 0, 0, false, 10}, {"yuv444p12", 0, 0, false, 12},
                              {"gbrp", 0, 0, true, 8},      {"gbrp10", 0, 0, true, 10},     {"gbrp12", 0, 0, true, 12},
                              {"gbrp16", 0, 0, true, 16}};
    EncodeSettings es;
    es.profile = prof;
    es.width = W;
    es.height = H;
    const FrameLayout L = makeLayout(W, H, prof, 1.0);
    auto xyzOf = [&](Device& d, const FrameFormat& f, const std::vector<uint8_t>& frame) {
        EncodeSettings e = es;
        e.input = f;
        auto slot = d.createSlot(L, e);
        Tier1Frame t1;
        FrameDebug dbg;
        dbg.full = false;
        slot->encode(frame.data(), t1, nullptr, &dbg);
        return dbg.xyz12;
    };
    auto cpu = createCpuDevice(0);
    std::vector<std::unique_ptr<Device>> gpus;
#ifdef GAIL_HAVE_VULKAN
    for (auto& d : createVulkanDevices({}, stderrLogger(false))) gpus.push_back(std::move(d));
#endif
#ifdef GAIL_HAVE_CUDA
    for (auto& d : createCudaDevices({}, stderrLogger(false))) gpus.push_back(std::move(d));
#endif
    for (const Layout& l : layouts) {
        FrameFormat f;
        f.kind = FrameFormat::Kind::Planar;
        f.width = W;
        f.height = H;
        f.activeW = aw;
        f.activeH = ah;
        f.ox = (W - aw) / 2;
        f.oy = (H - ah) / 2;
        f.sx = l.sx;
        f.sy = l.sy;
        f.rgb = l.rgb;
        f.bits = l.bits;
        f.kr = 0.2126, f.kb = 0.0722;  // BT.709, limited range
        const int cw = f.chromaW(), ch = f.chromaH();
        const double maxv = (1 << l.bits) - 1, sc = 1 << (l.bits - 8), kg = 1 - f.kr - f.kb;
        std::vector<uint8_t> frame(f.bytes());
        auto put = [&](size_t i, double v) {
            const long q = std::lround(std::clamp(v, 0.0, maxv));
            if (l.bits > 8) reinterpret_cast<uint16_t*>(frame.data())[i] = (uint16_t)q;
            else frame[i] = (uint8_t)q;
        };
        const size_t uOff = (size_t)aw * ah, vOff = uOff + (size_t)cw * ch;
        for (int y = 0; y < ah; ++y)
            for (int x = 0; x < aw; ++x) {
                double c[3];
                rgbAt(x, y, c);
                if (l.rgb) {  // planes G', B', R'
                    put((size_t)y * aw + x, c[1] * maxv);
                    put(uOff + (size_t)y * aw + x, c[2] * maxv);
                    put(vOff + (size_t)y * aw + x, c[0] * maxv);
                } else {
                    put((size_t)y * aw + x, (16 + 219 * (f.kr * c[0] + kg * c[1] + f.kb * c[2])) * sc);
                }
            }
        if (!l.rgb)
            for (int j = 0; j < ch; ++j)
                for (int i = 0; i < cw; ++i) {
                    // chroma sits on even luma columns (left), and between the two luma rows
                    // it covers for 4:2:0
                    double c[3] = {0, 0, 0}, t[3];
                    const int x = i << l.sx, y0 = j << l.sy, y1 = std::min(y0 + l.sy, ah - 1);
                    for (int yy : {y0, y1}) {
                        rgbAt(x, yy, t);
                        for (int k = 0; k < 3; ++k) c[k] += t[k] / 2;
                    }
                    const double Y = f.kr * c[0] + kg * c[1] + f.kb * c[2];
                    put(uOff + (size_t)j * cw + i, (128 + 224 * (c[2] - Y) / (2 * (1 - f.kb))) * sc);
                    put(vOff + (size_t)j * cw + i, (128 + 224 * (c[0] - Y) / (2 * (1 - f.kr))) * sc);
                }
        std::vector<uint16_t> rgb((size_t)W * H * 3);
        planarToRgb48(f, frame.data(), rgb.data());
        double maxErr = 0;
        for (int y = 0; y < ah; ++y)
            for (int x = 0; x < aw; ++x) {
                double c[3];
                rgbAt(x, y, c);
                const uint16_t* o = &rgb[((size_t)(y + f.oy) * W + x + f.ox) * 3];
                for (int k = 0; k < 3; ++k) maxErr = std::max(maxErr, std::fabs(o[k] / 65535.0 - c[k]));
            }
        const bool black = rgb[0] == 0 && rgb.back() == 0 && rgb[((size_t)(f.oy + 1) * W + f.ox - 1) * 3] == 0;
        // quantisation (8-bit Y'CbCr: about 2.5 code values through the matrix) plus the
        // half-row chroma offset at the top and bottom edges of 4:2:0
        const double tol = (l.rgb ? 0.6 : 3.0) / maxv + (l.sy ? 0.0015 : 0) + 1.0 / 65535;
        std::printf("  %-10s R'G'B' max error %.5f (tolerance %.5f)%s", l.name, maxErr, tol, black ? "" : " BORDER NOT BLACK");
        CHECK(maxErr <= tol && black, "%s: planar conversion off", l.name);
        const auto ref = xyzOf(*cpu, f, frame);
        for (auto& g : gpus) {
            const auto x = xyzOf(*g, f, frame);
            int maxd = 0;
            for (size_t i = 0; i < x.size(); ++i) maxd = std::max(maxd, std::abs((int)x[i] - (int)ref[i]));
            std::printf(", %s %d", g->name().c_str(), maxd);
            CHECK(maxd <= 2, "%s on %s: X'Y'Z' differs from the CPU by up to %d", l.name, g->name().c_str(), maxd);
        }
        std::printf("\n");
    }
}

// Colour handling on the CPU reference: exact X'Y'Z' pass-through, in-memory 3D LUTs,
// the tone-map variants on neutrals and the cinema peak.
static void testColor() {
    {
        ColorParams p;
        p.transfer = Transfer::Gamma26;
        p.primaries = Primaries::XYZ;
        ColorConverter cc(p);
        int bad = 0;
        for (int c = 0; c < 4096; ++c) {
            const int in[3] = {c, 4095 - c, c / 2};
            int out[3];
            cc.toXYZ12((uint16_t)((in[0] * 65535u + 2047u) / 4095u), (uint16_t)((in[1] * 65535u + 2047u) / 4095u),
                       (uint16_t)((in[2] * 65535u + 2047u) / 4095u), out);
            bad += out[0] != in[0] || out[1] != in[1] || out[2] != in[2];
        }
        std::printf("  X'Y'Z' pass-through: %d of 4096 codes changed\n", bad);
        CHECK(bad == 0, "X'Y'Z' pass-through is not exact");
    }
    {
        ColorParams p;
        const int n = 17;
        p.lut3dSize = n;
        for (int b = 0; b < n; ++b)
            for (int g = 0; g < n; ++g)
                for (int r = 0; r < n; ++r)
                    for (float v : {r, g, b}) p.lut3dData.push_back(v / (n - 1));
        ColorConverter cc(p);
        int maxd = 0;
        for (int i = 0; i < 65536; i += 97) {
            int out[3];
            cc.toXYZ12((uint16_t)i, (uint16_t)(65535 - i), (uint16_t)(i / 3), out);
            const int want[3] = {(int)std::lround(i / 65535.0 * 4095), (int)std::lround((65535 - i) / 65535.0 * 4095),
                                 (int)std::lround(i / 3 / 65535.0 * 4095)};
            for (int c = 0; c < 3; ++c) maxd = std::max(maxd, std::abs(out[c] - want[c]));
        }
        std::printf("  identity 17^3 LUT: max deviation %d codes\n", maxd);
        CHECK(maxd <= 1, "identity LUT deviates by %d", maxd);
        p.lut3dData.pop_back();
        bool threw = false;
        try {
            ColorConverter bad(p);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw, "truncated LUT data accepted");
    }
    // PQ BT.2020: every tone-map variant leaves neutrals alone below the knee (same curve),
    // is monotonic, and lands the source peak on the cinema peak.
    const ToneNorm norms[] = {ToneNorm::Channel, ToneNorm::MaxRGB, ToneNorm::Luminance};
    const char* names[] = {"channel", "maxrgb", "luminance"};
    std::vector<int> greys[3];
    for (int k = 0; k < 3; ++k) {
        ColorParams p;
        p.transfer = Transfer::PQ;
        p.primaries = Primaries::BT2020;
        p.hdrPeak = 1000;
        p.toneMap = ToneMap::Shoulder;
        p.toneNorm = norms[k];
        ColorConverter cc(p);
        int prev = -1, nonMono = 0;
        for (int i = 0; i < 65536; i += 64) {
            int out[3];
            cc.toXYZ12((uint16_t)i, (uint16_t)i, (uint16_t)i, out);
            nonMono += out[1] < prev;
            prev = out[1];
            greys[k].push_back(out[1]);
        }
        int peak[3];
        cc.toXYZ12((uint16_t)std::lround(pqInverseEotf(1000) * 65535), (uint16_t)std::lround(pqInverseEotf(1000) * 65535),
                   (uint16_t)std::lround(pqInverseEotf(1000) * 65535), peak);
        const int want = (int)std::floor(4095 * std::pow(48 / 52.37, 1 / 2.6) + 0.5);
        std::printf("  %-9s: source peak -> Y' %d (cinema peak %d), non-monotonic steps %d\n", names[k], peak[1], want,
                    nonMono);
        CHECK(nonMono == 0, "%s tone map is not monotonic", names[k]);
        CHECK(std::abs(peak[1] - want) <= 2, "%s: source peak lands on %d, not %d", names[k], peak[1], want);
    }
    int diff = 0;
    for (size_t i = 0; i < greys[0].size(); ++i)
        diff = std::max({diff, std::abs(greys[0][i] - greys[1][i]), std::abs(greys[0][i] - greys[2][i])});
    CHECK(diff <= 1, "tone-map variants disagree on neutrals by %d codes", diff);
}

// GPU colour kernels against the CPU reference for every colour mode (an HDR source
// exercises the tone map and gamut stages; the frame tests use SDR patterns).
static void testGpuColor(bool cuda) {
    const int W = 1998, H = 1080;
    FrameLayout L = makeLayout(W, H, Profile::Cinema2K, 1.0);
    auto src = openPatternSource(W, H, 1, 5);
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    src->read(rgb.data());
    std::vector<std::unique_ptr<Device>> devs;
#ifdef GAIL_HAVE_CUDA
    if (cuda) devs = createCudaDevices({}, stderrLogger(false));
#endif
#ifdef GAIL_HAVE_VULKAN
    if (!cuda) devs = createVulkanDevices({}, stderrLogger(false));
#endif
    if (devs.empty()) return;
    struct Mode {
        const char* name;
        ColorParams p;
    };
    std::vector<Mode> modes;
    for (auto [name, norm, tone] : {std::tuple{"pq channel", ToneNorm::Channel, ToneMap::Shoulder},
                                    std::tuple{"pq maxrgb", ToneNorm::MaxRGB, ToneMap::Shoulder},
                                    std::tuple{"pq luminance", ToneNorm::Luminance, ToneMap::Shoulder},
                                    std::tuple{"pq bt2390", ToneNorm::Channel, ToneMap::BT2390}}) {
        ColorParams p;
        p.transfer = Transfer::PQ;
        p.primaries = Primaries::BT2020;
        p.hdrPeak = 4000;
        p.toneNorm = norm;
        p.toneMap = tone;
        p.gamut = TargetGamut::P3D65;
        modes.push_back({name, p});
    }
    {
        ColorParams p;  // the defaults: HDR signal as gamma 2.4, no gamut compression
        p.transfer = Transfer::PQ;
        p.primaries = Primaries::BT2020;
        modes.push_back({"pq signal", p});
    }
    {
        ColorParams p;
        p.transfer = Transfer::Gamma26;
        p.primaries = Primaries::XYZ;
        modes.push_back({"xyz", p});
        p = ColorParams();
        p.lut3dSize = 5;
        for (int i = 0; i < 125; ++i)
            for (int c = 0; c < 3; ++c) p.lut3dData.push_back(0.5f + 0.4f * std::sin(i * 0.7f + c));
        modes.push_back({"lut3d", p});
    }
    for (auto& m : modes) {
        EncodeSettings es;
        es.width = W;
        es.height = H;
        es.input = src->format();
        es.color = m.p;
        ColorConverter cc(m.p);
        for (auto& d : devs) {
            auto slot = d->createSlot(L, es);
            Tier1Frame t;
            FrameDebug dbg;
            dbg.full = false;
            slot->encode(rgb.data(), t, nullptr, &dbg);
            size_t differ = 0, worst = 0;
            int maxd = 0;
            for (size_t i = 0; i < (size_t)W * H; ++i) {
                int ref[3];
                cc.toXYZ12(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], ref);
                for (int c = 0; c < 3; ++c) {
                    const int dd = std::abs((int)dbg.xyz12[3 * i + c] - ref[c]);
                    differ += dd != 0;
                    if (dd > maxd) maxd = dd, worst = i;
                }
            }
            std::printf("  %-12s %s: max diff %d, %zu of %zu codes differ\n", m.name, d->name().c_str(), maxd, differ,
                        (size_t)W * H * 3);
            if (maxd > 1) {
                int ref[3];
                cc.toXYZ12(rgb[3 * worst], rgb[3 * worst + 1], rgb[3 * worst + 2], ref);
                std::printf("    worst: R'G'B' %u %u %u -> CPU %d %d %d, GPU %u %u %u\n", rgb[3 * worst],
                            rgb[3 * worst + 1], rgb[3 * worst + 2], ref[0], ref[1], ref[2], dbg.xyz12[3 * worst],
                            dbg.xyz12[3 * worst + 1], dbg.xyz12[3 * worst + 2]);
            }
            // off-by-one rounding only; BT.2390 goes through PQ and back (two chained pow()
            // in float, fast-math on the GPU), which rounds differently on ~1% of codes
            CHECK(maxd <= 1 && differ * 50 < (size_t)W * H * 3, "%s colour on %s deviates from the CPU", m.name,
                  d->name().c_str());
        }
    }
}

static void testCpuCount() {
    const int n = usableCpuCount(), hw = (int)std::thread::hardware_concurrency();
    CHECK(n >= 1 && n <= std::max(hw, 1), "usable CPUs %d of %d", n, hw);
    CHECK(createCpuDevice(0)->name() == "CPU (" + std::to_string(n) + " threads)", "default thread count");
    CHECK(createCpuDevice(3, true)->name() == "CPU (3 threads)", "explicit thread count");
    std::printf("  %d usable of %d logical CPUs\n", n, hw);
}

static void testFitSlots() {
    const uint64_t G = 1ull << 30, M = 1ull << 20;
    CHECK(memoryHeadroom(8 * G) == 512 * M && memoryHeadroom(24 * G) == 24 * G * 6 / 100, "headroom");
    CHECK(fitSlots(0, 0, 700 * M, 4, 0) == 4, "unknown free memory does not limit");
    CHECK(fitSlots(6 * G, 8 * G, 0, 4, 0) == 4, "unknown slot size does not limit");
    CHECK(fitSlots(6 * G, 8 * G, 700 * M, 4, 0) == 4, "6 GB free holds four 4K frames");
    CHECK(fitSlots(2 * G, 8 * G, 700 * M, 4, 0) == 1, "2 GB free less 512 MB headroom holds one");
    CHECK(fitSlots(2700 * M, 8 * G, 700 * M, 4, 0) == 2, "2.7 GB free holds two with headroom and slack");
    CHECK(fitSlots(300 * M, 8 * G, 700 * M, 4, 0) == 1, "at least one is attempted");
    CHECK(fitSlots(20 * G, 24 * G, 700 * M, 4, 1600 * M) == 2, "an explicit limit caps");
    CHECK(fitSlots(0, 0, 700 * M, 4, 1600 * M) == 2, "an explicit limit works without a budget");
    CHECK(fitSlots(1 * G, 24 * G, 700 * M, 4, 16 * G) == 1, "the free memory caps below the limit");
    CHECK(fitSlots(6 * G, 8 * G, 700 * M, 0, 0) == 1, "zero requested means one");
}

// The slot footprint estimate against what the driver reports, and the memory limit /
// soft out-of-memory path of the pipeline on every GPU of a backend.
static void testGpuMemory(bool cuda) {
    std::vector<std::unique_ptr<Device>> devs;
#ifdef GAIL_HAVE_CUDA
    if (cuda) devs = createCudaDevices({}, stderrLogger(false));
#endif
#ifdef GAIL_HAVE_VULKAN
    if (!cuda) devs = createVulkanDevices({}, stderrLogger(false));
#endif
    if (devs.empty()) {
        std::printf("  (skipped: no %s devices on this machine)\n", cuda ? "CUDA" : "Vulkan");
        return;
    }
    for (Profile prof : {Profile::Cinema2K, Profile::Cinema4K}) {
        int W, H;
        containerSize(prof, Container::Flat, W, H);
        FrameLayout L = makeLayout(W, H, prof, 1.0);
        EncodeSettings es;
        es.profile = prof;
        es.width = W;
        es.height = H;
        es.input = openPatternSource(W, H, 1, 3)->format();
        for (auto& d : devs) {
            const uint64_t est = d->slotBytes(L, es);
            CHECK(est > 0, "%s: slot size unknown", d->name().c_str());
            const DeviceMemory m0 = d->memory();
            CHECK(m0.total > 0, "%s: total memory unknown", d->name().c_str());
            if (!m0.budgeted) {
                std::printf("  %s: no memory budget from the driver\n", d->name().c_str());
                continue;
            }
            uint64_t used;
            {
                auto slot = d->createSlot(L, es);
                const DeviceMemory m1 = d->memory();
                used = m0.available > m1.available ? m0.available - m1.available : 0;
            }
            std::printf("  %s %s: estimate %.0f MiB, driver %.0f MiB per frame, %.0f of %.0f MiB free\n", d->name().c_str(),
                        prof == Profile::Cinema2K ? "2K" : "4K", est / 1048576.0, used / 1048576.0,
                        m0.available / 1048576.0, m0.total / 1048576.0);
            // the estimate (plus the tenth the planner adds) must not come out below the truth
            CHECK(used <= est + est / 10, "%s: a frame takes %.0f MiB, estimated %.0f", d->name().c_str(),
                  used / 1048576.0, est / 1048576.0);
            CHECK(used >= est / 2, "%s: estimate %.0f MiB far above the %.0f MiB used", d->name().c_str(),
                  est / 1048576.0, used / 1048576.0);
        }
        // a limit of 1.5 frames leaves one frame in flight; the run still completes
        auto src = openPatternSource(W, H, 4, 3);
        std::vector<std::unique_ptr<Device>> one;
        one.push_back(std::move(devs[0]));
        PipelineOptions po;
        po.slotsPerDevice = 3;
        po.deviceMemoryLimit = one[0]->slotBytes(L, es) * 3 / 2;
        po.check = false;
        po.fps = 24;
        po.limits = dciLimits(24, 230);
        int notes = 0;
        po.onLog = [&](const std::string& m) {
            ++notes;
            std::printf("  note: %s\n", m.c_str());
        };
        PipelineSummary sum = runPipeline(*src, one, L, es, po);
        CHECK(sum.frames == 4 && sum.slotsPerDevice.size() == 1 && sum.slotsPerDevice[0] == 1 && notes == 1,
              "memory limit: %d frames in flight, %lld frames, %d notes", sum.slotsPerDevice.empty() ? -1 : sum.slotsPerDevice[0],
              sum.frames, notes);
        devs[0] = std::move(one[0]);
    }
}

// Failures the pipeline must survive without hanging or leaking a thread: a device that
// dies in the middle of a frame, a slot whose memory the device refuses, no slot at all,
// a source that fails, a sink that fails, and a cancel from another thread. Every case
// must come back from runPipeline() (the error as an exception) with all workers joined.
namespace {
struct FaultyDevice : Device {
    std::unique_ptr<Device> inner;
    std::atomic<int> encodes{0};
    int failAt = -1;         // the encode() call that throws (-1: never)
    int slotsAllowed = 99;   // createSlot() throws DeviceOutOfMemory beyond this many
    int created = 0;
    struct Slot : EncoderSlot {
        FaultyDevice& d;
        std::unique_ptr<EncoderSlot> inner;
        Slot(FaultyDevice& dev, std::unique_ptr<EncoderSlot> s) : d(dev), inner(std::move(s)) {}
        void* inputBuffer() override { return inner->inputBuffer(); }
        void encode(const void* f, Tier1Frame& out, FrameTimings* t, FrameDebug* dbg) override {
            if (d.encodes++ == d.failAt) throw std::runtime_error("device lost (injected)");
            inner->encode(f, out, t, dbg);
        }
    };
    std::string name() const override { return "faulty " + inner->name(); }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override {
        if (created >= slotsAllowed) throw DeviceOutOfMemory("cannot allocate 1 MiB of test memory");
        ++created;
        return std::make_unique<Slot>(*this, inner->createSlot(L, s));
    }
};
struct FaultySource : FrameSource {
    std::unique_ptr<FrameSource> inner;
    int failAt = -1, reads = 0;
    const FrameFormat& format() const override { return inner->format(); }
    bool read(void* dst) override {
        if (reads++ == failAt) throw std::runtime_error("read failed (injected)");
        return inner->read(dst);
    }
};
}  // namespace

static void testFaults() {
    const int W = 1998, H = 1080;
    const FrameLayout L = makeLayout(W, H, Profile::Cinema2K, 1.0);
    EncodeSettings es;
    es.profile = Profile::Cinema2K;
    es.width = W;
    es.height = H;
    es.input = openPatternSource(W, H, 1, 3)->format();
    PipelineOptions po;
    po.check = false;
    po.fps = 24;
    po.limits = dciLimits(24, 230);
    po.slotsPerDevice = 2;
    std::vector<std::string> notes;
    po.onLog = [&](const std::string& m) { notes.push_back(m); };
    // onReady: once per run that got its slots, before any frame, never when set-up failed
    int ready = 0;
    long long readyAt = -1;
    auto faulty = [](int failAt, int slots) {
        auto d = std::make_unique<FaultyDevice>();
        d->inner = createCpuDevice(0);
        d->failAt = failAt;
        d->slotsAllowed = slots;
        return d;
    };
    auto noted = [&](const char* text) {
        return std::any_of(notes.begin(), notes.end(), [&](const std::string& m) { return m.find(text) != std::string::npos; });
    };
    // runPipeline() with its outcome as a string: "" when it completed, else the error
    auto attempt = [&](std::vector<std::unique_ptr<Device>>& devs, FrameSource& src, PipelineSummary* sum = nullptr) {
        notes.clear();
        ready = 0;
        try {
            PipelineSummary s = runPipeline(src, devs, L, es, po);
            if (sum) *sum = s;
            return std::string();
        } catch (const std::exception& e) {
            return std::string(e.what());
        }
    };
    long long delivered = 0;
    bool inOrder = true;
    po.onFrame = [&](const FrameStats& st, const std::vector<uint8_t>& cs) {
        inOrder = inOrder && st.index == delivered && !cs.empty();
        ++delivered;
    };
    po.onReady = [&] {
        ++ready;
        readyAt = delivered;
    };

    // one of two devices dies on its third frame: the run stops with that error, and what was
    // delivered before it is a complete, ordered prefix
    {
        std::vector<std::unique_ptr<Device>> devs;
        devs.push_back(faulty(2, 99));
        devs.push_back(faulty(-1, 99));
        auto src = openPatternSource(W, H, 40, 3);
        delivered = 0;
        const std::string err = attempt(devs, *src);
        CHECK(err.find("device lost") != std::string::npos, "device failure: error \"%s\"", err.c_str());
        CHECK(inOrder && delivered < 40, "device failure: %lld frames delivered, in order %d", delivered, (int)inOrder);
        std::printf("  device lost on its 3rd frame: %lld frames delivered first, error \"%s\"\n", delivered, err.c_str());
    }
    // the second frame in flight is refused: the device runs with one, and the run completes
    {
        std::vector<std::unique_ptr<Device>> devs;
        devs.push_back(faulty(-1, 1));
        auto src = openPatternSource(W, H, 6, 3);
        delivered = 0;
        PipelineSummary sum;
        const std::string err = attempt(devs, *src, &sum);
        CHECK(err.empty() && sum.frames == 6 && delivered == 6 && inOrder, "slot refused: error \"%s\", %lld frames",
              err.c_str(), sum.frames);
        CHECK(ready == 1 && readyAt == 0, "slot refused: onReady called %d times, after %lld frames", ready, readyAt);
        CHECK(sum.slotsPerDevice.size() == 1 && sum.slotsPerDevice[0] == 1 && noted("running with 1 frame(s) in flight"),
              "slot refused: %d frames in flight, %zu notes", sum.slotsPerDevice.empty() ? -1 : sum.slotsPerDevice[0],
              notes.size());
    }
    // no frame fits at all: an error, or the CPU when the caller allows that
    {
        std::vector<std::unique_ptr<Device>> devs;
        devs.push_back(faulty(-1, 0));
        auto src = openPatternSource(W, H, 4, 3);
        delivered = 0;
        po.cpuFallback = false;
        std::string err = attempt(devs, *src);
        CHECK(err.find("no device has memory") != std::string::npos && delivered == 0, "no slot: error \"%s\"", err.c_str());
        CHECK(ready == 0, "no slot: onReady called %d times although set-up failed", ready);
        po.cpuFallback = true;
        PipelineSummary sum;
        src = openPatternSource(W, H, 4, 3);
        err = attempt(devs, *src, &sum);
        CHECK(err.empty() && sum.frames == 4 && devs.size() == 2 && sum.framesPerDevice.size() == 2 &&
                  sum.framesPerDevice[1] == 4 && noted("encoding on the CPU"),
              "no slot, CPU fallback: error \"%s\", %lld frames on %zu devices", err.c_str(), sum.frames, devs.size());
        CHECK(ready == 1 && readyAt == 0, "CPU fallback: onReady called %d times, after %lld frames", ready, readyAt);
        po.cpuFallback = false;
    }
    // the source fails on its third frame
    {
        std::vector<std::unique_ptr<Device>> devs;
        devs.push_back(faulty(-1, 99));
        FaultySource src;
        src.inner = openPatternSource(W, H, 40, 3);
        src.failAt = 2;
        delivered = 0;
        const std::string err = attempt(devs, src);
        CHECK(err.find("read failed") != std::string::npos && inOrder && delivered <= 2, "source failure: error \"%s\", %lld frames",
              err.c_str(), delivered);
    }
    // the sink fails on the second frame: the error comes back and the workers stop
    {
        std::vector<std::unique_ptr<Device>> devs;
        devs.push_back(faulty(-1, 99));
        auto src = openPatternSource(W, H, 40, 3);
        delivered = 0;
        auto deliver = po.onFrame;
        po.onFrame = [&](const FrameStats& st, const std::vector<uint8_t>& cs) {
            deliver(st, cs);
            if (st.index == 1) throw std::runtime_error("sink failed (injected)");
        };
        const std::string err = attempt(devs, *src);
        po.onFrame = deliver;
        CHECK(err.find("sink failed") != std::string::npos && delivered == 2, "sink failure: error \"%s\", %lld frames",
              err.c_str(), delivered);
    }
    // cancelled from another thread after the first frame: returns without an error
    {
        std::vector<std::unique_ptr<Device>> devs;
        devs.push_back(faulty(-1, 99));
        auto src = openPatternSource(W, H, 400, 3);
        std::atomic<bool> cancel{false};
        po.cancel = &cancel;
        delivered = 0;
        auto deliver = po.onFrame;
        po.onFrame = [&](const FrameStats& st, const std::vector<uint8_t>& cs) {
            deliver(st, cs);
            if (st.index == 0) cancel = true;
        };
        const auto t0 = std::chrono::steady_clock::now();
        PipelineSummary sum;
        const std::string err = attempt(devs, *src, &sum);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        po.onFrame = deliver;
        po.cancel = nullptr;
        CHECK(err.empty() && sum.frames < 400 && delivered == sum.frames, "cancel: error \"%s\", %lld frames", err.c_str(),
              sum.frames);
        std::printf("  cancelled after the 1st frame: %lld frames, back in %.0f ms\n", sum.frames, ms);
    }
}

// The Y'CbCr matrix both input paths use: the stream's tag, else the HD/SD rule (the RGB path
// used to leave untagged sources to swscale, which takes BT.601 at any size).
static void testSourceMatrix() {
    SourceInfo s;
    s.width = 1920, s.height = 1080;
    CHECK(sourceMatrix(s) == "bt709", "untagged HD: %s", sourceMatrix(s).c_str());
    s.matrix = "unknown";
    CHECK(sourceMatrix(s) == "bt709", "HD tagged unknown: %s", sourceMatrix(s).c_str());
    s.height = 576;
    CHECK(sourceMatrix(s) == "bt601", "untagged SD: %s", sourceMatrix(s).c_str());
    s.primaries = "bt2020";
    CHECK(sourceMatrix(s) == "bt2020", "untagged with BT.2020 primaries: %s", sourceMatrix(s).c_str());
    s.matrix = "smpte170m";
    CHECK(sourceMatrix(s) == "bt601", "tagged smpte170m: %s", sourceMatrix(s).c_str());
    s.matrix = "bt470bg";
    CHECK(sourceMatrix(s) == "bt601", "tagged bt470bg: %s", sourceMatrix(s).c_str());
    s.matrix = "bt709";
    s.height = 480;
    CHECK(sourceMatrix(s) == "bt709", "tagged bt709 at SD: %s", sourceMatrix(s).c_str());
    s.matrix = "bt2020nc";
    CHECK(sourceMatrix(s) == "bt2020", "tagged bt2020nc: %s", sourceMatrix(s).c_str());
}

// Truncated and malformed codestreams must come back as errors from the checker without it
// reading outside the buffer (each case sits in an exactly sized heap block, so a
// sanitizer build catches any over-read) or looping.
static void testCheckerMalformed() {
    // SOC + SIZ (2K flat, three 12-bit components): the valid start every case shares
    std::vector<uint8_t> head = {0xFF, 0x4F, 0xFF, 0x51, 0, 47, 0, 3};
    for (uint32_t v : {1998u, 1080u, 0u, 0u, 1998u, 1080u, 0u, 0u})
        for (int s = 24; s >= 0; s -= 8) head.push_back((uint8_t)(v >> s));
    head.insert(head.end(), {0, 3, 11, 1, 1, 11, 1, 1, 11, 1, 1});
    struct Case {
        const char* name;
        std::vector<uint8_t> tail;
        const char* expect;  // part of an error message
    };
    const Case cases[] = {
        {"COM truncated after Lcom", {0xFF, 0x64, 0x00, 0xC8}, "runs past the end"},
        {"COM claiming 200 bytes", {0xFF, 0x64, 0x00, 0xC8, 0, 1, 'x', 'y'}, "runs past the end"},
        {"COD with Lcod 0", {0xFF, 0x52, 0, 0, 0, 0, 0, 0}, "runs past the end"},
        {"QCD with Lqcd 1", {0xFF, 0x5C, 0, 1, 0, 0}, "runs past the end"},
        {"TLM with Ltlm 0", {0xFF, 0x55, 0, 0, 0, 0}, "runs past the end"},
        {"POC past the end", {0xFF, 0x5F, 0, 16, 0, 0}, "runs past the end"},
        {"SOT past the end", {0xFF, 0x90, 0, 10, 0, 0, 0, 0, 1, 0, 0, 3, 0xFF, 0x93}, "invalid Psot"},
        {"nothing after SIZ", {}, "missing EOC"},
    };
    auto check = [&](const char* name, const std::vector<uint8_t>& v, const char* expect) {
        std::unique_ptr<uint8_t[]> buf(new uint8_t[v.size() + 1]);  // +1: a zero-length block is not portable
        std::copy(v.begin(), v.end(), buf.get());
        const CheckReport rep = checkDciCodestream(buf.get(), v.size(), 24);
        const bool found = std::any_of(rep.errors.begin(), rep.errors.end(),
                                       [&](const std::string& e) { return e.find(expect) != std::string::npos; });
        CHECK(!rep.ok && found, "%s: ok %d, errors: %s", name, (int)rep.ok, formatReport(rep, false).c_str());
    };
    for (const Case& c : cases) {
        std::vector<uint8_t> v = head;
        v.insert(v.end(), c.tail.begin(), c.tail.end());
        check(c.name, v, c.expect);
    }
    check("empty buffer", {}, "SOC");
    // every prefix of a header with a COM and a TLM
    std::vector<uint8_t> full = head;
    full.insert(full.end(), {0xFF, 0x64, 0, 8, 0, 1, 'G', 'A', 'I', 'L', 0xFF, 0x55, 0, 19, 0, 0x50});
    for (int i = 0; i < 3; ++i) full.insert(full.end(), {0, 0, 0, 0, 0});
    for (size_t n = 1; n < full.size(); ++n) {
        std::unique_ptr<uint8_t[]> buf(new uint8_t[n]);
        std::copy(full.begin(), full.begin() + n, buf.get());
        CHECK(!checkDciCodestream(buf.get(), n, 24).ok, "prefix of %zu bytes passed", n);
    }
    std::printf("  %zu malformed cases and %zu prefixes rejected\n", sizeof cases / sizeof cases[0] + 1, full.size() - 1);
}

// Frame budgets the codestream headers alone cannot meet: dciLimits() must not wrap around,
// allocateRate() must still return (with every pass dropped), and the pipeline must refuse
// the budget up front instead of producing frames that fail the DCI check.
static void testRateLimits() {
    const RateLimits dci = dciLimits(24, 230);
    CHECK(dci.maxTotal == 1197900 && dci.maxComp == 1041650, "DCI limits at 24 fps: %zu / %zu", dci.maxTotal, dci.maxComp);
    CHECK(dciLimits(24, 0.0001).maxTotal == 0 && dciLimits(24, 230, 0.0001).maxComp == 0, "a tiny budget wrapped around");
    CHECK(dciLimits(0, 230).maxTotal == 0 && dciLimits(24, 0).maxTotal == 0, "a zero rate or frame rate wrapped around");

    const int W = 1998, H = 1080;
    const FrameLayout L = makeLayout(W, H, Profile::Cinema2K, 1.0);
    const size_t headers = emptyCodestreamBytes(L);
    // 3 tile-parts and one byte per packet, a few hundred bytes in all
    CHECK(headers > 100 && headers < 2000, "empty 2K codestream is %zu bytes", headers);
    auto src = openPatternSource(W, H, 1, 3);
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    src->read(rgb.data());
    EncodeSettings es;
    es.width = W;
    es.height = H;
    es.input = src->format();
    Tier1Frame t1;
    createCpuDevice(0)->createSlot(L, es)->encode(rgb.data(), t1);
    // no budget at all: everything is dropped and the call comes back
    const RateResult rr = allocateRate(L, t1, RateLimits{0, 0});
    const bool none = std::all_of(rr.passes.begin(), rr.passes.end(), [](uint8_t p) { return p == 0; });
    CHECK(none && rr.sizes.total == headers, "zero budget: passes left %d, %zu bytes", !none, rr.sizes.total);
    // exactly the header size is met
    const RateResult rh = allocateRate(L, t1, RateLimits{headers, headers});
    CHECK(rh.sizes.total <= headers, "header-sized budget: %zu bytes", rh.sizes.total);
    std::printf("  2K headers %zu bytes; zero and header-sized budgets handled\n", headers);

    // the pipeline refuses a budget below the headers before encoding anything
    std::vector<std::unique_ptr<Device>> devs;
    devs.push_back(createCpuDevice(0));
    PipelineOptions po;
    po.check = false;
    po.fps = 24;
    po.limits = RateLimits{headers - 1, headers - 1};
    int frames = 0;
    po.onFrame = [&](const FrameStats&, const std::vector<uint8_t>&) { ++frames; };
    std::string err;
    try {
        runPipeline(*openPatternSource(W, H, 2, 3), devs, L, es, po);
    } catch (const std::exception& e) {
        err = e.what();
    }
    CHECK(err.find("headers") != std::string::npos && frames == 0, "budget below the headers: \"%s\", %d frames", err.c_str(),
          frames);
}

int main(int argc, char** argv) {
    std::string only = argc > 1 ? argv[1] : "";
    auto run = [&](const char* name, const std::function<void()>& fn) {
        if (!only.empty() && only != name) return;
        std::printf("[%s]\n", name);
        fn();
    };
    run("dwt", testDwtRoundTrip);
    run("norms", testNorms);
    run("t1", testT1Truncation);
    run("t1limit", testT1PassLimit);
    run("frame2k", [] { testFrameCpu(Profile::Cinema2K); });
    run("frame4k", [] { testFrameCpu(Profile::Cinema4K); });
    run("yuv4k", [] { testYuv(Profile::Cinema4K); });
    run("yuv2k", [] { testYuv(Profile::Cinema2K); });
    run("planar", testPlanar);
    run("color", testColor);
    run("fit", testFitSlots);
    run("cpus", testCpuCount);
    run("faults", testFaults);
    run("limits", testRateLimits);
    run("checker", testCheckerMalformed);
    run("matrix", testSourceMatrix);
#ifdef GAIL_HAVE_VULKAN
    run("gpumem", [] { testGpuMemory(false); });
    run("gpucolor", [] { testGpuColor(false); });
    run("gpu2k", [] { testGpu(Profile::Cinema2K, false); });
    run("gpu4k", [] { testGpu(Profile::Cinema4K, false); });
#endif
#ifdef GAIL_HAVE_CUDA
    run("cudamem", [] { testGpuMemory(true); });
    run("cudacolor", [] { testGpuColor(true); });
    run("cuda2k", [] { testGpu(Profile::Cinema2K, true); });
    run("cuda4k", [] { testGpu(Profile::Cinema4K, true); });
#endif
    std::printf(g_fail ? "%d FAILURES\n" : "all tests passed\n", g_fail);
    return g_fail ? 1 : 0;
}
