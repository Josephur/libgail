// SPDX-License-Identifier: MIT
// Unit / integration tests for the Gail encoder.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
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
#include "core/bitio.h"
#include "core/codestream_check.h"
#include "core/codestream_parse.h"
#include "core/decode_backend.h"
#include "core/dwt_cpu.h"
#include "core/frame_source.h"
#include "core/pipeline.h"
#include "core/progression.h"
#include "core/quant.h"
#include "core/rate_control.h"
#include "core/t1_dec.h"
#include "core/t1_ref.h"
#include "core/t2.h"
#include "core/t2_parse.h"
#include "core/tagtree.h"
#include "core/thread_pool.h"
#ifdef GAIL_HAVE_CUDA
#include "cuda/cuda_host.h"
#endif

#define MQ_OUT_T uint8_t*
#define MQ_PUT(o, i, v) ((o)[(i)] = (uint8_t)(v))
#define MQ_GET(o, i) ((uint32_t)(o)[(i)])
#include "t1_decoder.h"

// The GPU backends' block decoder, compiled for the host
#include "core/t1_dec_tables.h"
static uint8_t g_t1dZc[gail::kT1dZcSize], g_t1dSc[gail::kT1dScSize];
#define MQD_IN_T const uint8_t*
#define MQD_BYTE(in, i) ((uint32_t)(in)[(i)])
#define T1D_OUT_T uint32_t*
#define T1D_WR(out, i, v) ((out)[(i)] = (v))
#define T1D_RD(out, i) ((out)[(i)])
#define T1D_ZC(i) ((uint32_t)g_t1dZc[(i)])
#define T1D_SC(i) ((uint32_t)g_t1dSc[(i)])
#include "kernels/t1_decode.h"

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

// The GPUs the tests run on: all that a backend offers, unless the environment says
// otherwise.
//   GAIL_TEST_GPUS         a comma-separated list of GPUs, each by its UUID as nvidia-smi -L
//                          prints it or by its PCI bus ID: the tests leave every other GPU
//                          alone. Both name the same GPU under CUDA and under Vulkan.
//   GAIL_TEST_CUDA_PLUGIN  gail_cuda13 or gail_cuda12: the CUDA tests use that plugin for
//                          every GPU it can run, whatever the GPU's architecture. This is how
//                          a plugin is tested on a GPU that libgail would not give it.
[[maybe_unused]] static std::vector<std::unique_ptr<Device>> testGpus(bool cuda, const Logger& log) {
    auto normal = [](std::string id) {
        for (auto& c : id) c = (char)std::tolower((unsigned char)c);
        if (id.rfind("gpu-", 0) == 0) id = id.substr(4);
        if (std::count(id.begin(), id.end(), ':') == 1) id = "0000:" + id;
        return id;
    };
    std::vector<std::string> only;
    if (const char* env = std::getenv("GAIL_TEST_GPUS")) {
        std::string list = env;
        for (size_t at = 0; at <= list.size();) {
            const size_t end = std::min(list.find(',', at), list.size());
            if (end > at) only.push_back(normal(list.substr(at, end - at)));
            at = end + 1;
        }
    }
    // Indices of the wanted GPUs among a backend's devices; false when none is wanted.
    auto pick = [&](const std::vector<DeviceInfo>& devices, std::vector<int>& select) {
        if (only.empty()) return !devices.empty();
        for (const auto& d : devices)
            if (std::find(only.begin(), only.end(), normal(d.uuid)) != only.end() ||
                std::find(only.begin(), only.end(), normal(d.pciBusId)) != only.end())
                select.push_back(d.index);
        return !select.empty();
    };
    std::vector<int> select;
#ifdef GAIL_HAVE_CUDA
    if (cuda) {
        if (const char* plugin = std::getenv("GAIL_TEST_CUDA_PLUGIN")) {
            static CudaHost host(CudaHost::moduleDirectories(), {{plugin}});
            return pick(host.list(), select) ? host.create(select, log) : std::vector<std::unique_ptr<Device>>();
        }
        return pick(listCudaDevices(), select) ? createCudaDevices(select, log) : std::vector<std::unique_ptr<Device>>();
    }
#endif
#ifdef GAIL_HAVE_VULKAN
    if (!cuda && pick(listVulkanDevices(), select)) return createVulkanDevices(select, log);
#endif
    // a build without a GPU backend uses none of these
    (void)cuda;
    (void)log;
    (void)pick;
    return {};
}

#ifdef GAIL_TEST_PLUGINS
// The CUDA backend's plugin loader, on stand-ins for the plugins (tests/test_cuda_plugin.cpp)
// with five made-up GPUs: which plugins load, and which one each GPU is given to. The build
// puts the stand-ins in test-plugins/ next to this program.
static void testCudaPlugins() {
    namespace fs = std::filesystem;
    const auto specs = cudaPluginSpecs();
    CHECK(specs.size() == 2 && specs[0].name == "gail_cuda13" && specs[0].minSm == 75 && specs[1].name == "gail_cuda12" &&
              specs[1].maxSm == 74,
          "the rule is not Turing and newer on gail_cuda13, older on gail_cuda12");

    // The one place the library looks: the directory of the module the loader is in, here this program's.
    std::error_code ec;
    const fs::path own = CudaHost::moduleDirectory();
    CHECK(!own.empty() && own.is_absolute() && (fs::exists(own / "gail-tests", ec) || fs::exists(own / "gail-tests.exe", ec)),
          "module directory \"%s\" is not where gail-tests is", own.string().c_str());
    const auto dirs = CudaHost::moduleDirectories();
    CHECK(!dirs.empty() && dirs[0] == own, "the module's own directory is not searched first");

    const fs::path standIns = own / "test-plugins";
    if (!fs::is_directory(standIns / "cuda13", ec)) {
        std::printf("  (skipped: the stand-in plugins are not next to gail-tests)\n");
        return;
    }
    const fs::path cuda13 = standIns / "cuda13", cuda12 = standIns / "cuda12", otherAbi = standIns / "other_abi",
                   declines = standIns / "declines", noDriver = standIns / "no_driver", busy = standIns / "busy",
                   none = standIns;  // none: no plugin file in it
    auto listed = [](CudaHost& h) {
        std::string s;
        for (const auto& d : h.list()) s += (s.empty() ? "" : " | ") + std::to_string(d.index) + " " + d.name + " " + d.plugin;
        return s;
    };
    auto made = [](const std::vector<std::unique_ptr<Device>>& devices) {
        std::string s;
        for (const auto& d : devices) s += (s.empty() ? "" : " | ") + d->name();
        return s;
    };
    auto joined = [](const std::vector<std::string>& lines) {
        std::string s;
        for (const auto& l : lines) s += (s.empty() ? "" : " / ") + l;
        return s;
    };
    auto has = [](const std::vector<std::string>& lines, const std::string& text) {
        return std::any_of(lines.begin(), lines.end(), [&](const std::string& l) { return l.find(text) != std::string::npos; });
    };
    std::vector<std::string> said;
    int warnings = 0;
    const Logger log = [&](LogLevel level, const std::string& m) {
        said.push_back(m);
        warnings += level == LogLevel::Warning;
    };

    {  // no plugins: nothing to offer, nothing fails, and the log says why at info level
        CudaHost h({none}, specs);
        CHECK(h.loaded().empty() && h.list().empty(), "no plugins: loaded \"%s\", devices \"%s\"", h.loaded().c_str(), listed(h).c_str());
        said.clear();
        warnings = 0;
        CHECK(h.create({}, log).empty(), "no plugins: a device was made");
        CHECK(has(said, "gail_cuda13 not loaded: " + CudaHost::fileName("gail_cuda13") + " is not next to the gail library") &&
                  has(said, "gail_cuda12 not loaded: ") && warnings == 0,
              "no plugins: %zu lines logged, %d of them warnings", said.size(), warnings);
        CHECK(h.unusable(0).empty(), "no plugins: there is no device 0");
    }
    {  // nowhere to look: the same
        CudaHost h({}, specs);
        CHECK(h.loaded().empty() && h.list().empty() && has(h.notes(), "gail_cuda13 not loaded"), "no directory: \"%s\"", listed(h).c_str());
    }
    {  // both plugins: each GPU on the one its architecture belongs to, whatever number that plugin has it under
        CudaHost h({cuda13, cuda12}, specs);
        CHECK(h.loaded() == "gail_cuda13 gail_cuda12", "both: loaded \"%s\"; notes: %s", h.loaded().c_str(), joined(h.notes()).c_str());
        const std::string want = "0 Test Ampere gail_cuda13 | 1 Test Maxwell gail_cuda12 | 2 Test Turing gail_cuda13 | 3 Test Volta gail_cuda12";
        CHECK(listed(h) == want, "both: devices \"%s\"", listed(h).c_str());
        CHECK(h.unusable(0).empty() && h.unusable(3).empty() && h.unusable(7).empty(), "both: a usable or absent device has a reason");
        CHECK(h.unusable(4) == "Test Kepler (sm_35) skipped by gail_cuda12: not built for this architecture", "both: Kepler: \"%s\"",
              h.unusable(4).c_str());
        said.clear();
        warnings = 0;
        const std::string all = made(h.create({}, log));
        CHECK(all == "Test Ampere (CUDA) on gail_cuda13 | Test Maxwell (CUDA) on gail_cuda12 | Test Turing (CUDA) on gail_cuda13 | "
                     "Test Volta (CUDA) on gail_cuda12",
              "both: made \"%s\"", all.c_str());
        CHECK(has(said, "gail_cuda13 loaded: CUDA 0.0 (test), sm 75 86 89 120") && has(said, "Test Maxwell [gail_cuda12]") &&
                  has(said, "device 4 Test Kepler (sm_35) skipped by gail_cuda12") && warnings == 0,
              "both: %zu lines logged, %d of them warnings", said.size(), warnings);
        const std::string two = made(h.create({3, 1}, log));
        CHECK(two == "Test Maxwell (CUDA) on gail_cuda12 | Test Volta (CUDA) on gail_cuda12", "both: devices 1 and 3 made \"%s\"", two.c_str());
        CHECK(h.create({4}, log).empty(), "both: the Kepler GPU was made");
    }
    {  // gail_cuda12 missing: Turing and newer still work, the older GPUs are not offered
        CudaHost h({cuda13}, specs);
        CHECK(h.loaded() == "gail_cuda13", "only gail_cuda13: loaded \"%s\"; notes: %s", h.loaded().c_str(), joined(h.notes()).c_str());
        CHECK(listed(h) == "0 Test Ampere gail_cuda13 | 2 Test Turing gail_cuda13", "only gail_cuda13: devices \"%s\"", listed(h).c_str());
        CHECK(h.unusable(1) == "Test Maxwell (sm_52) skipped: it needs gail_cuda12, which is not loaded", "only gail_cuda13: Maxwell: \"%s\"",
              h.unusable(1).c_str());
        CHECK(has(h.notes(), "gail_cuda12 not loaded: ") && has(h.notes(), "device 3 Test Volta (sm_70) skipped: it needs gail_cuda12"),
              "only gail_cuda13: notes");
        CHECK(made(h.create({}, log)) == "Test Ampere (CUDA) on gail_cuda13 | Test Turing (CUDA) on gail_cuda13", "only gail_cuda13: made");
    }
    {  // gail_cuda13 missing: a Turing GPU is not handed to gail_cuda12, though its Volta code would run there
        CudaHost h({cuda12}, specs);
        CHECK(listed(h) == "0 Test Maxwell gail_cuda12 | 2 Test Volta gail_cuda12", "only gail_cuda12: devices \"%s\"", listed(h).c_str());
        CHECK(h.unusable(1) == "Test Turing (sm_75) skipped: it needs gail_cuda13, which is not loaded", "only gail_cuda12: Turing: \"%s\"",
              h.unusable(1).c_str());
        CHECK(!h.unusable(4).empty(), "only gail_cuda12: Ampere is usable");
    }
    {  // plugins of another interface version, one that says so itself and one that does not: neither is used
        CudaHost h({otherAbi, declines}, specs);
        CHECK(h.loaded().empty() && h.list().empty(), "other ABI: loaded \"%s\", devices \"%s\"", h.loaded().c_str(), listed(h).c_str());
        const auto notes = h.notes();
        CHECK(has(notes, "gail_cuda13 not loaded: it was built for another version of the gail library (plugin interface 2, this library has 1)") &&
                  has(notes, "gail_cuda12 not loaded: it was built for another version of the gail library (plugin interface unknown"),
              "other ABI: notes");
        CHECK(h.create({}, log).empty(), "other ABI: a device was made");
    }
    {  // a plugin whose CUDA runtime does not start (no driver, or too old a one): its GPUs are not offered, the other plugin's are
        CudaHost h({noDriver, cuda12}, specs);
        CHECK(h.loaded() == "gail_cuda13 gail_cuda12", "no driver: loaded \"%s\"; notes: %s", h.loaded().c_str(), joined(h.notes()).c_str());
        CHECK(listed(h) == "0 Test Maxwell gail_cuda12 | 2 Test Volta gail_cuda12", "no driver: devices \"%s\"", listed(h).c_str());
        CHECK(h.unusable(4) == "Test Ampere (sm_86) skipped: it needs gail_cuda13, whose CUDA runtime did not start: no NVIDIA driver",
              "no driver: Ampere: \"%s\"", h.unusable(4).c_str());
        CHECK(has(h.notes(), "gail_cuda13 cannot be used: no NVIDIA driver"), "no driver: notes");
    }
    {  // listing goes by architecture and sets nothing up; a GPU the kernels do not load on is found out when it is created
        CudaHost h({cuda13, busy}, specs);
        CHECK(listed(h) == "0 Test Ampere gail_cuda13 | 1 Test Maxwell gail_cuda12 | 2 Test Turing gail_cuda13 | 3 Test Volta gail_cuda12",
              "busy: devices \"%s\"", listed(h).c_str());
        CHECK(h.unusable(3).empty(), "busy: Volta ruled out by listing: \"%s\"", h.unusable(3).c_str());
        said.clear();
        warnings = 0;
        const std::string all = made(h.create({}, log));
        CHECK(all == "Test Ampere (CUDA) on gail_cuda13 | Test Maxwell (CUDA) on gail_cuda12 | Test Turing (CUDA) on gail_cuda13",
              "busy: made \"%s\"", all.c_str());
        CHECK(has(said, "cuda: device 3 Test Volta (sm_70) skipped by gail_cuda12: out of memory") && warnings == 1,
              "busy: %zu lines logged, %d of them warnings", said.size(), warnings);
        // from then on the device is reported as the ones listing rules out are
        CHECK(h.unusable(3) == "Test Volta (sm_70) skipped by gail_cuda12: out of memory", "busy: Volta: \"%s\"", h.unusable(3).c_str());
        CHECK(listed(h) == "0 Test Ampere gail_cuda13 | 1 Test Maxwell gail_cuda12 | 2 Test Turing gail_cuda13",
              "busy: devices after creation \"%s\"", listed(h).c_str());
        CHECK(has(h.notes(), "device 3 Test Volta (sm_70) skipped by gail_cuda12: out of memory"), "busy: notes");
        CHECK(h.create({3}, log).empty(), "busy: the Volta GPU was made");
    }
    {  // which architectures a plugin's code runs on
        struct Case {
            const char* archs;
            int sm;
            bool runs;
        };
        const Case cases[] = {
            {"75 86 89 120", 75, true},   {"75 86 89 120", 87, true},  {"75 86 89 120", 70, false},
            {"50 52 60 61 70", 86, true},  // through the PTX for 7.0
            {"50 52 60 61 70", 35, false}, {"86-real", 89, true},      {"86-real", 90, false},  {"86-real", 80, false},
            {"61-real", 60, false},        {"86-virtual", 120, true},  {"86-virtual", 75, false}, {"90a", 90, true},
            {"90a", 100, false},           {"100f", 103, true},        {"100f", 120, false},    {"", 86, false},
        };
        int bad = 0;
        for (const auto& c : cases)
            if (cudaArchsRun(c.archs, c.sm) != c.runs) {
                ++bad;
                std::printf("  archs \"%s\" on sm %d: %s\n", c.archs, c.sm, c.runs ? "not run" : "run");
            }
        CHECK(bad == 0, "%d architecture cases wrong", bad);
    }
    {  // the first directory that has a plugin's file is the one it comes from
        CudaHost h({otherAbi, cuda13}, specs);
        CHECK(h.loaded().empty(), "directory order: loaded \"%s\"", h.loaded().c_str());
    }
}
#endif

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
    CHECK(rep.comment == codestreamLabel(), "frame not labeled as GAIL's");
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

#if defined(GAIL_HAVE_VULKAN) || defined(GAIL_HAVE_CUDA)
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
    if (cuda) devs = testGpus(true, stderrLogger(true));
#endif
    if (!cuda) devs = testGpus(false, stderrLogger(true));
    if (devs.empty()) {
        std::printf("  (skipped: no %s devices on this machine)\n", cuda ? "CUDA" : "Vulkan");
        return;
    }
    // CPU reference for the color/DWT stages
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
            // Compare the GPU wavelet against the CPU transform of the GPU's own color output.
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
            coefDiff += std::llabs(a - b) > 64;  // more than one quantization step apart
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
        buildCodestream(L, tg, rr.passes.data(), &cs, nullptr, d->codestreamTag());
        CheckReport rep = checkDciCodestream(cs.data(), cs.size(), 24);
        CHECK(rep.ok, "GPU frame not compliant: %s", formatReport(rep, false).c_str());
        const std::string tag = d->codestreamTag();
        CHECK(cuda ? tag.rfind("cuda", 0) == 0 && tag.size() > 6 : tag == "vulk", "%s: device tag \"%s\"",
              d->name().c_str(), tag.c_str());
        CHECK(rep.comment == codestreamComment(tag), "GPU frame comment \"%s\"", rep.comment.c_str());
        std::printf("  comment: %s\n", rep.comment.c_str());
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
        CHECK(lim.numPasses == limit, "pass limit not honored");
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
    for (auto& d : testGpus(false, stderrLogger(false))) compare(d->name().c_str(), xyzOf(*d, yf, fy), 70.0);
#endif
#ifdef GAIL_HAVE_CUDA
    for (auto& d : testGpus(true, stderrLogger(false))) compare(d->name().c_str(), xyzOf(*d, yf, fy), 70.0);
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
    // linear ramps: chroma interpolation reproduces them, so only quantization remains
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
    for (auto& d : testGpus(false, stderrLogger(false))) gpus.push_back(std::move(d));
#endif
#ifdef GAIL_HAVE_CUDA
    for (auto& d : testGpus(true, stderrLogger(false))) gpus.push_back(std::move(d));
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
        // quantization (8-bit Y'CbCr: about 2.5 code values through the matrix) plus the
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

// Color handling on the CPU reference: exact X'Y'Z' pass-through, in-memory 3D LUTs,
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
    std::vector<int> grays[3];
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
            grays[k].push_back(out[1]);
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
    for (size_t i = 0; i < grays[0].size(); ++i)
        diff = std::max({diff, std::abs(grays[0][i] - grays[1][i]), std::abs(grays[0][i] - grays[2][i])});
    CHECK(diff <= 1, "tone-map variants disagree on neutrals by %d codes", diff);
}

// GPU color kernels against the CPU reference for every color mode (an HDR source
// exercises the tone map and gamut stages; the frame tests use SDR patterns).
[[maybe_unused]] static void testGpuColor(bool cuda) {
    const int W = 1998, H = 1080;
    FrameLayout L = makeLayout(W, H, Profile::Cinema2K, 1.0);
    auto src = openPatternSource(W, H, 1, 5);
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    src->read(rgb.data());
    std::vector<std::unique_ptr<Device>> devs;
#ifdef GAIL_HAVE_CUDA
    if (cuda) devs = testGpus(true, stderrLogger(false));
#endif
#ifdef GAIL_HAVE_VULKAN
    if (!cuda) devs = testGpus(false, stderrLogger(false));
#endif
    (void)cuda;  // a build without a GPU backend tests none
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
            CHECK(maxd <= 1 && differ * 50 < (size_t)W * H * 3, "%s color on %s deviates from the CPU", m.name,
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
[[maybe_unused]] static void testGpuMemory(bool cuda) {
    std::vector<std::unique_ptr<Device>> devs;
#ifdef GAIL_HAVE_CUDA
    if (cuda) devs = testGpus(true, stderrLogger(false));
#endif
#ifdef GAIL_HAVE_VULKAN
    if (!cuda) devs = testGpus(false, stderrLogger(false));
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
    // canceled from another thread after the first frame: returns without an error
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
        std::printf("  canceled after the 1st frame: %lld frames, back in %.0f ms\n", sum.frames, ms);
    }
}

// Frame sources must report a frame they cannot read as an error, not as the end of the
// stream, so a run never ends short and silently.
static void testSourceErrors() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "gail_test_xyz";
    fs::remove_all(dir);
    fs::create_directories(dir);
    const int W = 1998, H = 1080;
    const size_t frameBytes = (size_t)W * H * 6;
    const std::vector<uint8_t> good(frameBytes, 0), shortFrame(1000, 0);
    auto write = [&](const char* name, const std::vector<uint8_t>& data) {
        FILE* f = std::fopen((dir / name).string().c_str(), "wb");
        std::fwrite(data.data(), 1, data.size(), f);
        std::fclose(f);
    };
    write("a.xyz", good);
    write("b.xyz", shortFrame);
    write("c.xyz", good);
    auto src = openXyzDirSource(dir.string(), W, H, -1);
    std::vector<uint8_t> buf(frameBytes);
    CHECK(src->read(buf.data()), "first .xyz frame");
    std::string err;
    try {
        src->read(buf.data());
    } catch (const std::exception& e) {
        err = e.what();
    }
    CHECK(err.find("b.xyz") != std::string::npos && err.find("500 samples") != std::string::npos, "short .xyz frame: \"%s\"",
          err.c_str());  // 1000 bytes are 500 samples
    fs::remove_all(dir);

    // ffmpeg itself failing (here: an input that does not exist) must surface, not look like
    // a clip with no frames
    SourceOptions so;
    so.path = (fs::temp_directory_path() / "gail_test_missing_input.mkv").string();
    so.width = W;
    so.height = H;
    SourceInfo info;
    info.width = 1920, info.height = 1080, info.pixFmt = "yuv420p";
    err.clear();
    try {
        auto ff = openFfmpegSource(so, info);
        std::fprintf(stderr, "  (expect an ffmpeg error about a missing file here)\n");
        while (ff->read(buf.data())) {}
    } catch (const std::exception& e) {
        err = e.what();
    }
    if (err.find("failed to start") != std::string::npos) std::printf("  (ffmpeg not installed: child status check skipped)\n");
    else CHECK(err.find("ffmpeg failed") != std::string::npos, "failing ffmpeg child: \"%s\"", err.c_str());
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

// The codestream with the main header's COM segment taken out, and that segment's text.
static std::vector<uint8_t> withoutComment(const std::vector<uint8_t>& cs, std::string* text = nullptr) {
    size_t at = 2;  // past SOC
    while (at + 4 <= cs.size() && cs[at] == 0xFF && cs[at + 1] != 0x90) {  // up to the first SOT
        const size_t seg = 2 + ((size_t)cs[at + 2] << 8 | cs[at + 3]);
        if (cs[at + 1] == 0x64) {
            if (text && seg >= 6) text->assign((const char*)cs.data() + at + 6, seg - 6);
            std::vector<uint8_t> out(cs.begin(), cs.begin() + (ptrdiff_t)at);
            out.insert(out.end(), cs.begin() + (ptrdiff_t)(at + seg), cs.end());
            return out;
        }
        at += seg;
    }
    return cs;
}

// A device that encodes on the CPU but says it is another kind, for the device tag.
class TaggedCpuDevice : public Device {
public:
    explicit TaggedCpuDevice(std::string tag) : cpu_(createCpuDevice(0)), tag_(std::move(tag)) {}
    std::string name() const override { return cpu_->name() + " tagged " + tag_; }
    std::string codestreamTag() const override { return tag_; }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override {
        return cpu_->createSlot(L, s);
    }

private:
    std::unique_ptr<Device> cpu_;
    std::string tag_;
};

// The COM marker names the kind of device that encoded the frame, and the rate control
// budgets for the longest name there is: at the DCI cap no tag takes a frame over the
// limit, and the coded data is the same whichever tag is written.
static void testCodestreamComment() {
    CHECK(codestreamComment("") == codestreamLabel(), "the CPU's comment is not the label alone");
    CHECK(codestreamComment("vulk") == std::string(codestreamLabel()) + " vulk", "Vulkan comment: %s",
          codestreamComment("vulk").c_str());
    bool threw = false;
    try {
        codestreamComment("cuda100.10");
    } catch (const std::logic_error&) {
        threw = true;
    }
    CHECK(threw, "a tag longer than kMaxDeviceTagBytes was accepted");

    const char* tags[] = {"", "vulk", "cuda12.9", "cuda99.99"};
    for (Profile prof : {Profile::Cinema2K, Profile::Cinema4K}) {
        int W, H;
        containerSize(prof, Container::Flat, W, H);
        const FrameLayout L = makeLayout(W, H, prof, 1.0);
        // noise: more to code in every component than the cap allows
        std::vector<uint16_t> rgb((size_t)W * H * 3);
        std::mt19937 rng(7);
        for (auto& v : rgb) v = (uint16_t)rng();
        EncodeSettings es;
        es.profile = prof;
        es.width = W;
        es.height = H;
        es.input = openPatternSource(W, H, 1, 3)->format();
        Tier1Frame t1;
        createCpuDevice(0)->createSlot(L, es)->encode(rgb.data(), t1);
        const RateLimits lim = dciLimits(24, 250);  // the DCI maximum
        const RateResult rr = allocateRate(L, t1, lim);
        // the rate control fills the frame to within a few percent of the cap
        CHECK(rr.sizes.total <= lim.maxTotal && rr.sizes.total + lim.maxTotal / 25 >= lim.maxTotal,
              "%s frame not at the cap: %zu of %zu bytes", profileName(prof), rr.sizes.total, lim.maxTotal);
        std::vector<uint8_t> first;
        for (const char* tag : tags) {
            std::vector<uint8_t> cs;
            buildCodestream(L, t1, rr.passes.data(), &cs, nullptr, tag);
            // the space before the tag and the tag's unused bytes
            const size_t unused = 1 + kMaxDeviceTagBytes - (*tag ? 1 + std::strlen(tag) : 0);
            CHECK(cs.size() + unused == rr.sizes.total, "tag \"%s\": %zu bytes, budgeted %zu", tag, cs.size(),
                  rr.sizes.total);
            CHECK(cs.size() <= lim.maxTotal, "tag \"%s\": %zu bytes over the %zu limit", tag, cs.size(), lim.maxTotal);
            CheckReport rep = checkDciCodestream(cs.data(), cs.size(), 24);
            CHECK(rep.ok, "tag \"%s\": %s", tag, formatReport(rep, false).c_str());
            CHECK(rep.comment == codestreamComment(tag), "comment \"%s\", wanted \"%s\"", rep.comment.c_str(),
                  codestreamComment(tag).c_str());
            std::string text;
            const std::vector<uint8_t> coded = withoutComment(cs, &text);
            CHECK(text == rep.comment && coded.size() + 6 + text.size() == cs.size(), "COM segment not found");
            if (first.empty()) first = coded;
            CHECK(coded == first, "tag \"%s\" changed the coded data", tag);
            // and the decoder's parser takes it
            StreamHeader h;
            std::string err;
            CHECK(parseCodestream(cs.data(), cs.size(), false, h, err), "tag \"%s\": decoder parse: %s", tag, err.c_str());
        }
        std::printf("  %s: %zu of %zu bytes with the longest tag, the same coded data with every tag\n",
                    profileName(prof), rr.sizes.total, lim.maxTotal);
    }

    // the pipeline writes the tag of the device that encoded each frame
    const int W = 1998, H = 1080;
    const FrameLayout L = makeLayout(W, H, Profile::Cinema2K, 1.0);
    EncodeSettings es;
    es.width = W;
    es.height = H;
    auto src = openPatternSource(W, H, 6, 3);
    es.input = src->format();
    std::vector<std::unique_ptr<Device>> devs;
    devs.push_back(std::make_unique<TaggedCpuDevice>("cuda99.99"));
    devs.push_back(createCpuDevice(0));
    PipelineOptions po;
    po.slotsPerDevice = 1;
    po.check = true;
    po.fps = 24;
    po.limits = dciLimits(24, 250);
    int bad = 0, tagged = 0;
    po.onFrame = [&](const FrameStats& st, const std::vector<uint8_t>& cs) {
        const CheckReport rep = checkDciCodestream(cs.data(), cs.size(), 24);
        const std::string want = codestreamComment(st.device == 0 ? "cuda99.99" : "");
        tagged += st.device == 0;
        if (!rep.ok || !st.compliant || rep.comment != want) {
            ++bad;
            std::printf("  frame %lld from device %d: \"%s\", wanted \"%s\"\n", st.index, st.device, rep.comment.c_str(),
                        want.c_str());
        }
    };
    const PipelineSummary sum = runPipeline(*src, devs, L, es, po);
    CHECK(sum.frames == 6 && bad == 0, "pipeline: %lld frames, %d with the wrong comment", sum.frames, bad);
    std::printf("  pipeline: %d of 6 frames tagged by their device\n", tagged);
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

// Packet-header bits written by BitWriter come back from BitReader, including the 7-bit
// bytes after 0xFF and the way a header ends.
static void testBitIo() {
    std::mt19937 rng(11);
    size_t ffHeaders = 0, endOnFf = 0;
    for (int trial = 0; trial < 2000; ++trial) {
        std::vector<std::pair<uint32_t, int>> items;
        const int count = trial == 0 ? 0 : 1 + (int)(rng() % 60);
        const bool ones = trial % 3 == 0;  // long runs of ones produce 0xFF bytes
        for (int i = 0; i < count; ++i) {
            const int n = (int)(rng() % 33);
            uint32_t v = ones && rng() % 4 ? 0xFFFFFFFFu : (uint32_t)rng();
            if (n < 32) v &= (1u << n) - 1u;
            items.push_back({v, n});
        }
        if (trial == 1) items = {{0xFFu, 8}};  // ends exactly on 0xFF
        std::vector<uint8_t> bytes;
        BitWriter bw(&bytes);
        for (auto& it : items) {
            if (it.second == 1) bw.putBit(it.first);
            else bw.putBits(it.first, it.second);
        }
        const size_t written = bw.flush();
        CHECK(written == bytes.size(), "writer count %zu vs %zu bytes", written, bytes.size());
        for (size_t i = 0; i + 1 < bytes.size(); ++i)
            if (bytes[i] == 0xFF) {
                ++ffHeaders;
                CHECK(bytes[i + 1] < 0x80, "trial %d: marker code inside a header", trial);
            }
        endOnFf += bytes.size() >= 2 && bytes[bytes.size() - 2] == 0xFF;
        bytes.push_back(0xAA);  // the reader must not take what follows the header
        BitReader br(bytes.data(), bytes.size());
        bool same = true;
        for (auto& it : items) {
            const uint32_t v = it.second == 1 ? br.getBit() : br.getBits(it.second);
            same = same && v == it.first;
        }
        CHECK(same, "trial %d: bits read back differ", trial);
        const size_t used = br.finish();
        CHECK(used == written, "trial %d: reader took %zu bytes, writer wrote %zu", trial, used, written);
        CHECK(!br.overrun(), "trial %d: overrun on a complete header", trial);
    }
    CHECK(ffHeaders > 100 && endOnFf > 0, "test data never exercised bit stuffing (%zu, %zu)", ffHeaders, endOnFf);

    const uint8_t three[3] = {0xA5, 0x0F, 0x33};
    BitReader br(three, 3);
    CHECK(br.getBits(24) == 0xA50F33u && !br.overrun(), "24 bits of 3 bytes");
    CHECK(br.getBits(32) == 0 && br.overrun(), "bits past the end must read as zero and flag an overrun");
}

// Tag trees decode what they encode, for the two ways packet headers use them: the
// inclusion tree asked "below layer + 1?" layer after layer, and the zero-bit-plane tree
// read out in full for a block when it is first included.
static void testTagTree() {
    std::mt19937 rng(23);
    const int sizes[][2] = {{1, 1}, {1, 7}, {4, 4}, {5, 3}, {13, 9}};
    const int layers = 4;
    for (auto& sz : sizes) {
        const int w = sz[0], h = sz[1], n = w * h;
        for (int trial = 0; trial < 400; ++trial) {
            std::vector<int> first(n), zbp(n);
            for (int i = 0; i < n; ++i) {
                first[i] = (int)(rng() % (layers + 2));  // some blocks never included
                zbp[i] = (int)(rng() % 21);
            }
            std::vector<uint8_t> bytes;
            {
                TagTree incl, imsb;
                incl.init(w, h);
                imsb.init(w, h);
                for (int i = 0; i < n; ++i) {
                    incl.setLeaf(i, first[i]);
                    imsb.setLeaf(i, zbp[i]);
                }
                incl.finalize();
                imsb.finalize();
                BitWriter bw(&bytes);
                for (int layer = 0; layer < layers; ++layer)
                    for (int i = 0; i < n; ++i) {
                        if (first[i] < layer) continue;  // already included: no tag tree bits
                        incl.encode(bw, i, layer + 1);
                        if (first[i] == layer) imsb.encode(bw, i, 999);
                    }
                bw.flush();
            }
            TagTree incl, imsb;
            incl.init(w, h);
            imsb.init(w, h);
            BitReader br(bytes.data(), bytes.size());
            std::vector<int> gotFirst(n, -1), gotZbp(n, -1);
            for (int layer = 0; layer < layers; ++layer)
                for (int i = 0; i < n; ++i) {
                    if (gotFirst[i] >= 0) continue;
                    if (!incl.decode(br, i, layer + 1)) continue;
                    gotFirst[i] = layer;
                    const bool known = imsb.decode(br, i, 999);
                    CHECK(known, "%dx%d trial %d: zero bit-planes of leaf %d not resolved", w, h, trial, i);
                    gotZbp[i] = imsb.value(i);
                }
            int bad = 0;
            for (int i = 0; i < n; ++i) {
                const bool included = first[i] < layers;
                bad += included ? gotFirst[i] != first[i] || gotZbp[i] != zbp[i] : gotFirst[i] != -1;
            }
            CHECK(bad == 0, "%dx%d trial %d: %d leaves decoded wrongly", w, h, trial, bad);
            CHECK(br.finish() == bytes.size() && !br.overrun(), "%dx%d trial %d: reader and writer disagree on length",
                  w, h, trial);
        }
    }
}

// The packet order both the writer and a reader follow: every precinct of every
// resolution exactly once per component, in the order B.12.1.5 gives for
// component-position-resolution-layer, with the 4K frame's two volumes one after the other.
static void testProgression() {
    const struct {
        Profile p;
        int w, h;
    } cases[] = {{Profile::Cinema2K, 1998, 1080}, {Profile::Cinema2K, 2048, 858}, {Profile::Cinema4K, 4096, 2160},
                 {Profile::Cinema4K, 3996, 2160}, {Profile::Cinema4K, 4096, 1716}};
    for (auto& cs : cases) {
        const FrameLayout L = makeLayout(cs.w, cs.h, cs.p, 1.0);
        const int NL = L.levels;
        ProgressionVolume vols[2];
        const int nv = progressionVolumes(L, vols);
        CHECK(nv == (cs.p == Profile::Cinema4K ? 2 : 1), "%dx%d: %d volumes", cs.w, cs.h, nv);
        CHECK(vols[0].rs == 0 && vols[nv - 1].re == NL + 1 && (nv == 1 || vols[0].re == vols[1].rs),
              "%dx%d: volumes do not cover resolutions 0..%d once", cs.w, cs.h, NL);
        for (int c = 0; c < kNumComps; ++c) {
            std::vector<std::vector<int>> seen(NL + 1);
            for (int r = 0; r <= NL; ++r) seen[r].assign(L.resolution(c, r).precincts.size(), 0);
            for (int vi = 0; vi < nv; ++vi) {
                // B.12.1.5 written out directly: for each position on the reference grid, for each
                // resolution, the precinct whose top-left corner sits at that position
                std::vector<std::pair<int, int>> want;
                for (int y = 0; y < cs.h; ++y)
                    for (int x = 0; x < cs.w; ++x) {
                        if ((x | y) & 127) continue;  // no precinct corner is finer than 128 here
                        for (int r = vols[vi].rs; r < vols[vi].re; ++r) {
                            const Resolution& R = L.resolution(c, r);
                            const long long sx = 1LL << (R.ppx + NL - r), sy = 1LL << (R.ppy + NL - r);
                            if (x % sx || y % sy) continue;
                            const int px = (int)(x / sx), py = (int)(y / sy);
                            if (px < R.npx && py < R.npy) want.push_back({r, py * R.npx + px});
                        }
                    }
                std::vector<std::pair<int, int>> got;
                forEachPacket(L, vols[vi], c, [&](int r, int prec) {
                    got.push_back({r, prec});
                    if (r >= 0 && r <= NL && prec >= 0 && prec < (int)seen[r].size()) ++seen[r][prec];
                });
                CHECK(got == want, "%dx%d comp %d volume %d: packet order differs (%zu vs %zu packets)", cs.w, cs.h, c,
                      vi, got.size(), want.size());
            }
            int wrong = 0;
            for (auto& v : seen)
                for (int n : v) wrong += n != 1;
            CHECK(wrong == 0, "%dx%d comp %d: %d precincts not visited exactly once", cs.w, cs.h, c, wrong);
        }
    }
}

// A layout built from the frame's structure and the step sizes as signaled is the layout
// the encoder builds for itself, field for field.
static void testGeometry() {
    const struct {
        Profile p;
        int w, h;
    } cases[] = {{Profile::Cinema2K, 1998, 1080}, {Profile::Cinema2K, 2048, 858}, {Profile::Cinema2K, 2048, 1080},
                 {Profile::Cinema4K, 3996, 2160}, {Profile::Cinema4K, 4096, 1716}, {Profile::Cinema4K, 4096, 2160}};
    for (auto& cs : cases)
        for (double step : {1.0, 2.0, 0.75}) {
            const FrameLayout L = makeLayout(cs.w, cs.h, cs.p, step);
            FrameLayout G = makeGeometry(cs.w, cs.h, cs.p, L.levels, L.guardBits);
            std::vector<int> eps, mu;
            for (int i = 0; i < L.bandsPerComp(); ++i) {
                eps.push_back(L.bands[i].eps);
                mu.push_back(L.bands[i].mu);
            }
            setSignaledSteps(G, eps.data(), mu.data());
            bool same = G.profile == L.profile && G.width == L.width && G.height == L.height && G.levels == L.levels &&
                        G.guardBits == L.guardBits && G.bands.size() == L.bands.size() && G.cblks.size() == L.cblks.size() &&
                        G.resolutions.size() == L.resolutions.size();
            for (size_t i = 0; same && i < L.bands.size(); ++i) {
                const Subband &a = L.bands[i], &b = G.bands[i];
                same = a.comp == b.comp && a.res == b.res && a.level == b.level && a.orient == b.orient && a.px0 == b.px0 &&
                       a.py0 == b.py0 && a.w == b.w && a.h == b.h && a.ncbx == b.ncbx && a.ncby == b.ncby &&
                       a.cblkBase == b.cblkBase && a.eps == b.eps && a.mu == b.mu && a.Mb == b.Mb &&
                       std::memcmp(&a.delta, &b.delta, sizeof(float)) == 0;
            }
            for (size_t i = 0; same && i < L.cblks.size(); ++i) {
                const CodeBlock &a = L.cblks[i], &b = G.cblks[i];
                same = a.band == b.band && a.x0 == b.x0 && a.y0 == b.y0 && a.w == b.w && a.h == b.h;
            }
            for (size_t i = 0; same && i < L.resolutions.size(); ++i) {
                const Resolution &a = L.resolutions[i], &b = G.resolutions[i];
                same = a.comp == b.comp && a.res == b.res && a.w == b.w && a.h == b.h && a.ppx == b.ppx && a.ppy == b.ppy &&
                       a.npx == b.npx && a.npy == b.npy && a.precincts.size() == b.precincts.size();
                for (size_t k = 0; same && k < a.precincts.size(); ++k) {
                    same = a.precincts[k].nb == b.precincts[k].nb;
                    for (int j = 0; same && j < a.precincts[k].nb; ++j) {
                        const PrecinctBand &x = a.precincts[k].bands[j], &y = b.precincts[k].bands[j];
                        same = x.band == y.band && x.cbx0 == y.cbx0 && x.cby0 == y.cby0 && x.ncbx == y.ncbx && x.ncby == y.ncby;
                    }
                }
            }
            CHECK(same, "%dx%d step %.2f: geometry + signaled steps differs from makeLayout", cs.w, cs.h, step);
        }
    FrameLayout G = makeGeometry(2048, 1080, Profile::Cinema2K, 5, 1);
    std::vector<int> eps(G.bandsPerComp(), 23), mu(G.bandsPerComp(), 0);
    bool thrown = false;
    try {
        setSignaledSteps(G, eps.data(), mu.data());  // 1 + 23 - 1 = 23 bit-planes: one too many
    } catch (const std::runtime_error&) {
        thrown = true;
    }
    CHECK(thrown, "a step with more bit-planes than the coder carries was accepted");
}

// One frame through the CPU reference encoder, with everything a decoder test wants to
// compare against: the Tier-1 data, the passes rate control kept, the codestream, and the
// X'Y'Z' picture that went in.
struct EncodedFrame {
    FrameLayout L;
    Tier1Frame t1;
    RateResult rr;
    std::vector<uint8_t> cs;
    std::vector<uint16_t> xyz12;
};
static EncodedFrame encodeCpuFrame(Profile prof, Container cont = Container::Flat, int patternSeed = 3) {
    EncodedFrame f;
    int W, H;
    containerSize(prof, cont, W, H);
    f.L = makeLayout(W, H, prof, 1.0);
    auto src = openPatternSource(W, H, 1, patternSeed);
    std::vector<uint16_t> rgb((size_t)W * H * 3);
    src->read(rgb.data());
    EncodeSettings es;
    es.profile = prof;
    es.width = W;
    es.height = H;
    es.input = src->format();
    auto dev = createCpuDevice(0);
    auto slot = dev->createSlot(f.L, es);
    FrameTimings tm;
    FrameDebug dbg;
    slot->encode(rgb.data(), f.t1, &tm, &dbg);
    f.rr = allocateRate(f.L, f.t1, dciLimits(24));
    buildCodestream(f.L, f.t1, f.rr.passes.data(), &f.cs, nullptr);
    f.xyz12 = std::move(dbg.xyz12);
    return f;
}

// The headers of an encoded frame say what the encoder's layout says, and reading its
// packets gives back, for every code-block, exactly the passes and bytes that were written.
static void testParse(Profile prof, Container cont) {
    const EncodedFrame f = encodeCpuFrame(prof, cont);
    const FrameLayout& L = f.L;
    StreamHeader h;
    std::string err;
    CHECK(parseCodestream(f.cs.data(), f.cs.size(), false, h, err), "parse failed: %s", err.c_str());
    CHECK(h.supported, "own codestream reported unsupported: %s", h.unsupported.c_str());
    CHECK(h.width == L.width && h.height == L.height && h.components == 3 && h.bitDepth == 12, "SIZ: %dx%d, %d components, %d bits",
          h.width, h.height, h.components, h.bitDepth);
    CHECK(h.rsiz == (prof == Profile::Cinema4K ? 4 : 3) && h.levels == L.levels && h.layers == 1 && h.guardBits == L.guardBits,
          "Rsiz %d, %d levels, %d layers, %d guard bits", h.rsiz, h.levels, h.layers, h.guardBits);
    CHECK(!h.reversible && !h.highThroughput && !h.sop && !h.eph && h.tiles == 1, "coding style flags wrong");
    ProgressionVolume vols[2];
    const int nv = progressionVolumes(L, vols);
    bool volsSame = (int)h.volumes.size() == nv;
    for (int i = 0; volsSame && i < nv; ++i) volsSame = h.volumes[i].rs == vols[i].rs && h.volumes[i].re == vols[i].re;
    CHECK(volsSame, "progression volumes differ from the writer's");
    CHECK((int)h.tileParts.size() == nv * 3, "%zu tile-parts", h.tileParts.size());

    StreamHeader ho;
    CHECK(parseCodestream(f.cs.data(), f.cs.size(), true, ho, err) && ho.supported && ho.tileParts.empty() && ho.width == h.width,
          "header-only parse");

    const FrameLayout G = layoutFor(h);
    bool bandsSame = G.bands.size() == L.bands.size() && G.numCblks() == L.numCblks();
    for (size_t i = 0; bandsSame && i < L.bands.size(); ++i)
        bandsSame = G.bands[i].eps == L.bands[i].eps && G.bands[i].mu == L.bands[i].mu && G.bands[i].Mb == L.bands[i].Mb &&
                    G.bands[i].delta == L.bands[i].delta && G.bands[i].cblkBase == L.bands[i].cblkBase;
    CHECK(bandsSame, "layout from the headers differs from the encoder's");

    std::vector<CblkCode> blk;
    CHECK(parsePackets(f.cs.data(), h, G, L.levels, blk, err), "packet parse failed: %s", err.c_str());
    int wrong = 0, coded = 0;
    for (int i = 0; i < L.numCblks(); ++i) {
        const int np = f.rr.passes[i];
        const CblkCode& b = blk[i];
        if (b.passes != np) {
            ++wrong;
            continue;
        }
        if (!np) continue;
        ++coded;
        const uint32_t len = f.t1.len(i, np - 1);
        if (b.nbps != f.t1.info[i].nbps || b.len != len || (size_t)b.offset + len > f.cs.size() ||
            std::memcmp(f.cs.data() + b.offset, f.t1.data.data() + f.t1.info[i].dataOffset, len) != 0)
            ++wrong;
    }
    std::printf("  %s %dx%d: %d of %d code-blocks coded, %zu bytes\n", profileName(prof), L.width, L.height, coded,
                L.numCblks(), f.cs.size());
    CHECK(wrong == 0 && coded > 0, "%d code-blocks read back differently from what was written", wrong);

    // Stopping one resolution early: the same answer below it, nothing above
    std::vector<CblkCode> low;
    CHECK(parsePackets(f.cs.data(), h, G, L.levels - 1, low, err), "reduced packet parse failed: %s", err.c_str());
    wrong = 0;
    for (int i = 0; i < L.numCblks(); ++i) {
        const bool top = L.bands[L.cblks[i].band].res == L.levels;
        wrong += top ? low[i].passes != 0 : low[i].passes != blk[i].passes || low[i].offset != blk[i].offset || low[i].len != blk[i].len;
    }
    CHECK(wrong == 0, "%d code-blocks wrong when the top resolution is skipped", wrong);

    // Every way of cutting the codestream short is an error or a clean parse, never a crash
    int accepted = 0;
    for (size_t n = 0; n < f.cs.size(); n += n < 400 ? 1 : 997) {
        StreamHeader t;
        std::string e;
        if (!parseCodestream(f.cs.data(), n, false, t, e)) continue;
        ++accepted;
        std::vector<CblkCode> tb;
        if (t.supported) parsePackets(f.cs.data(), t, G, L.levels, tb, e);
    }
    CHECK(accepted == 0, "%d truncated codestreams parsed as complete", accepted);
}

// The production block decoder against the independent test decoder: the same signs and
// magnitudes for every block at every truncation point, with the mid-point of the
// undecoded bit-planes where the standard puts it, and nothing written outside the block.
static void testT1Decode() {
    std::mt19937 rng(42);
    int blocks = 0, checks = 0;
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
                if (mode == 1 && sp(rng) < 0.7) v *= 0.05;
                c[y * 32 + x] = quantizeCoeff((float)v, 1.0f, (1u << 26) - 1);
            }
        T1Result res;
        t1EncodeRef(c.data(), w, h, orient, res);
        if (!res.nbps) continue;
        ++blocks;
        for (int k = 1; k <= res.numPasses; ++k) {
            const uint32_t len = res.passLen[k - 1];
            const auto ref = test::t1Decode(res.data.data(), len, w, h, orient, res.nbps, k);
            std::vector<uint32_t> out(kCblkArea, 0xDEADBEEFu);
            t1DecodeBlock(res.data.data(), len, w, h, orient, res.nbps, k, out.data());
            int bad = 0, outside = 0;
            for (int y = 0; y < 32; ++y)
                for (int x = 0; x < 32; ++x) {
                    const uint32_t v = out[y * 32 + x];
                    if (x >= w || y >= h) {
                        outside += v != 0xDEADBEEFu;
                        continue;
                    }
                    const int i = y * w + x;
                    const uint32_t want = ref.sig[i] ? ((uint32_t)ref.sign[i] << 31) | (2 * ref.mag[i] + (1u << ref.low[i])) : 0u;
                    bad += v != want;
                }
            CHECK(bad == 0 && outside == 0, "trial %d, %d of %d passes: %d samples wrong, %d written outside the block", trial, k,
                  res.numPasses, bad, outside);
            ++checks;
        }
        // all passes: exactly the quantizer's integer part, mid-point half a step above it
        std::vector<uint32_t> out(kCblkArea, 0);
        t1DecodeBlock(res.data.data(), (uint32_t)res.data.size(), w, h, orient, res.nbps, res.numPasses, out.data());
        int bad = 0;
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                const uint32_t m = (c[y * 32 + x] & 0x7FFFFFFF) >> kFracBits;
                bad += out[y * 32 + x] != (m ? (c[y * 32 + x] & 0x80000000u) | (2 * m + 1) : 0u);
            }
        CHECK(bad == 0, "trial %d: full decode differs from the quantized block in %d samples", trial, bad);
    }
    std::printf("  %d blocks, %d truncation points\n", blocks, checks);

    // Damaged data decodes to something without touching memory outside the codeword
    for (int trial = 0; trial < 300; ++trial) {
        const uint32_t len = rng() % 200;
        std::vector<uint8_t> junk(len);
        for (auto& b : junk) b = trial % 3 ? (uint8_t)rng() : 0xFF;
        std::vector<uint32_t> out(kCblkArea, 0);
        const int nbps = 1 + (int)(rng() % 22);
        t1DecodeBlock(junk.data(), len, 32, 32, trial % 4, nbps, 3 * nbps - 2, out.data());
    }
}

static double psnr12(const uint16_t* a, const uint16_t* b, size_t n, int* maxDiff = nullptr) {
    double se = 0;
    int worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const int d = (int)a[i] - (int)b[i];
        se += (double)d * d;
        worst = std::max(worst, std::abs(d));
    }
    if (maxDiff) *maxDiff = worst;
    return se == 0 ? 99.0 : 10 * std::log10(4095.0 * 4095.0 * (double)n / se);
}

// Encode a frame, decode its codestream on the CPU, and get the picture back: as close as
// the encoder's own estimate says, in both word alignments, into a buffer with padded
// rows, and at half and quarter size.
static void testDecodeCpu(Profile prof) {
    const EncodedFrame f = encodeCpuFrame(prof);
    const int W = f.L.width, H = f.L.height;
    StreamHeader h;
    std::string err;
    CHECK(parseCodestream(f.cs.data(), f.cs.size(), false, h, err) && h.supported, "parse: %s %s", err.c_str(), h.unsupported.c_str());
    const FrameLayout L = layoutFor(h);
    std::vector<CblkCode> blk;
    CHECK(parsePackets(f.cs.data(), h, L, L.levels, blk, err), "packets: %s", err.c_str());
    auto pool = std::make_shared<ThreadPool>(usableCpuCount(), false);

    DecodeSettings ds;
    ds.highBits = false;
    auto slot = createCpuDecoderSlot(L, ds, pool);
    std::vector<uint16_t> out((size_t)W * H * 3);
    DecodeTimings tm;
    slot->decode(L, f.cs.data(), f.cs.size(), blk, out.data(), (ptrdiff_t)W * 6, &tm);
    int maxd = 0;
    const double psnr = psnr12(out.data(), f.xyz12.data(), out.size(), &maxd);
    const double est = 10 * std::log10(4095.0 * 4095 / (f.rr.distortion / ((double)W * H * 3)));
    std::printf("  %s decode: PSNR %.2f dB against the source (encoder's estimate %.2f), max diff %d; Tier-1 %.1f ms, wavelet %.1f, "
                "color %.1f\n", profileName(prof), psnr, est, maxd, tm.t1, tm.dwt, tm.color);
    CHECK(psnr > est - 1.0 && psnr > 40.0, "decoded picture is %.2f dB from the source, encoder estimated %.2f", psnr, est);

    // a second frame through the same slot: nothing left over from the first
    std::vector<uint16_t> again((size_t)W * H * 3);
    slot->decode(L, f.cs.data(), f.cs.size(), blk, again.data(), (ptrdiff_t)W * 6, nullptr);
    CHECK(again == out, "decoding the same frame twice gave different pictures");

    // codes in the high bits, rows 10 bytes longer than the picture
    ds.highBits = true;
    auto hi = createCpuDecoderSlot(L, ds, pool);
    const ptrdiff_t stride = (ptrdiff_t)W * 6 + 10;
    std::vector<uint8_t> padded((size_t)stride * H, 0xA5);
    hi->decode(L, f.cs.data(), f.cs.size(), blk, padded.data(), stride, nullptr);
    int wrong = 0, touched = 0;
    for (int y = 0; y < H; ++y) {
        const uint16_t* row = (const uint16_t*)(padded.data() + (size_t)y * stride);
        for (int i = 0; i < W * 3; ++i) wrong += row[i] != (uint16_t)(out[(size_t)y * W * 3 + i] << 4);
        for (int i = 0; i < 10; ++i) touched += padded[(size_t)y * stride + (size_t)W * 6 + i] != 0xA5;
    }
    CHECK(wrong == 0, "high-bit output differs from low-bit output shifted, %d words", wrong);
    CHECK(touched == 0, "%d padding bytes written", touched);

    // reduced size: the low-pass band of the full picture, so transforming the decoded
    // picture forward again must land on it
    for (int reduce = 1; reduce <= 2; ++reduce) {
        std::vector<float> ref[3];
        for (auto& p : ref) p.resize((size_t)W * H);
        for (size_t i = 0; i < (size_t)W * H; ++i)
            ictForward((float)(out[3 * i] - 2048), (float)(out[3 * i + 1] - 2048), (float)(out[3 * i + 2] - 2048), ref[0][i],
                       ref[1][i], ref[2][i]);
        for (auto& p : ref) fdwt97_2d(p.data(), W, H, reduce);
        ds.highBits = false;
        ds.reduce = reduce;
        std::vector<CblkCode> low;
        CHECK(parsePackets(f.cs.data(), h, L, L.levels - reduce, low, err), "packets: %s", err.c_str());
        auto narrowSlot = createCpuDecoderSlot(L, ds, pool);
        int w, hh;
        decodedSize(L, reduce, w, hh);
        CHECK(w == (W + (1 << reduce) - 1) >> reduce && hh == (H + (1 << reduce) - 1) >> reduce, "reduced size %dx%d", w, hh);
        std::vector<uint16_t> got((size_t)w * hh * 3), want((size_t)w * hh * 3);
        narrowSlot->decode(L, f.cs.data(), f.cs.size(), low, got.data(), (ptrdiff_t)w * 6, nullptr);
        for (int y = 0; y < hh; ++y)
            for (int x = 0; x < w; ++x) {
                // the low-pass band sits in the top-left corner of the full-width plane
                const size_t i = (size_t)y * W + x;
                const float y0 = ref[0][i], y1 = ref[1][i], y2 = ref[2][i];
                const float v[3] = {y0 + 1.402f * y2, y0 - 0.34413f * y1 - 0.71414f * y2, y0 + 1.772f * y1};
                for (int k = 0; k < 3; ++k)
                    want[((size_t)y * w + x) * 3 + k] = (uint16_t)std::clamp((int)std::floor(v[k] + 2048.5f), 0, 4095);
            }
        const double p = psnr12(got.data(), want.data(), got.size(), &maxd);
        std::printf("  reduce %d: %dx%d, %.2f dB against the low-pass band of the full decode, max diff %d\n", reduce, w, hh, p, maxd);
        CHECK(p > 55.0, "reduce %d: %.2f dB from the full picture's low-pass band", reduce, p);
    }
}

// Damaged codestreams: a frame with bits flipped, lengths edited or its end cut off is
// refused or decodes to some picture. It never reads or writes out of bounds, which the
// sanitizer build of this test checks, and the decoder works as before afterwards.
static void testDecodeMutations() {
    const EncodedFrame f = encodeCpuFrame(Profile::Cinema2K);
    StreamHeader h0;
    std::string err;
    CHECK(parseCodestream(f.cs.data(), f.cs.size(), false, h0, err) && h0.supported, "parse: %s", err.c_str());
    const FrameLayout L0 = layoutFor(h0);
    auto pool = std::make_shared<ThreadPool>(usableCpuCount(), false);
    auto slot = createCpuDecoderSlot(L0, DecodeSettings{}, pool);
    std::vector<uint16_t> good((size_t)L0.width * L0.height * 3), out(good.size());
    std::vector<CblkCode> blk;
    CHECK(parsePackets(f.cs.data(), h0, L0, L0.levels, blk, err), "packets: %s", err.c_str());
    slot->decode(L0, f.cs.data(), f.cs.size(), blk, good.data(), (ptrdiff_t)L0.width * 6, nullptr);

    const size_t headerEnd = h0.tileParts[0].begin;
    std::mt19937 rng(77);
    int malformed = 0, unsupported = 0, badPackets = 0, decoded = 0, otherShape = 0;
    for (int trial = 0; trial < 360; ++trial) {
        std::vector<uint8_t> cs = f.cs;
        switch (trial % 6) {
        case 0:  // a few bits in the main header
            for (int k = 1 + (int)(rng() % 3); k > 0; --k) cs[2 + rng() % (headerEnd - 2)] ^= (uint8_t)(1u << (rng() % 8));
            break;
        case 1:  // a byte of the main header replaced
            cs[2 + rng() % (headerEnd - 2)] = (uint8_t)rng();
            break;
        case 2:  // bits anywhere: mostly packet headers and code-block data
            for (int k = 1 + (int)(rng() % 8); k > 0; --k) cs[rng() % cs.size()] ^= (uint8_t)(1u << (rng() % 8));
            break;
        case 3:  // the start of a tile-part's packets, where the first packet headers are
        {
            const TilePartSpan& tp = h0.tileParts[rng() % h0.tileParts.size()];
            for (int k = 1 + (int)(rng() % 4); k > 0; --k) cs[tp.begin + rng() % 64] ^= (uint8_t)(1u << (rng() % 8));
            break;
        }
        case 4:  // cut short
            cs.resize(1 + rng() % (cs.size() - 1));
            break;
        default:  // a run of 0xFF, which packet headers and the MQ decoder both treat specially
        {
            const size_t at = rng() % (cs.size() - 40);
            std::fill(cs.begin() + (ptrdiff_t)at, cs.begin() + (ptrdiff_t)at + 1 + rng() % 32, (uint8_t)0xFF);
            break;
        }
        }
        // Exactly the size of the data, so that a read past the end is a read out of bounds
        std::vector<uint8_t> exact(cs.begin(), cs.end());
        StreamHeader h;
        if (!parseCodestream(exact.data(), exact.size(), false, h, err)) {
            ++malformed;
            continue;
        }
        if (!h.supported) {
            ++unsupported;
            continue;
        }
        FrameLayout L;
        try {
            L = layoutFor(h);
        } catch (const std::exception&) {
            ++unsupported;
            continue;
        }
        std::vector<CblkCode> b;
        if (!parsePackets(exact.data(), h, L, L.levels, b, err)) {
            ++badPackets;
            continue;
        }
        int bad = 0;
        for (const CblkCode& c : b) bad += c.passes && ((size_t)c.offset + c.len > exact.size() || c.nbps < 1 || c.passes > 3 * c.nbps - 2);
        CHECK(bad == 0, "trial %d: %d code-blocks located outside the data or with impossible passes", trial, bad);
        if (!sameGeometry(L, L0)) {  // the damage changed the picture's shape: its own slot
            ++otherShape;
            auto other = createCpuDecoderSlot(L, DecodeSettings{}, pool);
            std::vector<uint16_t> o((size_t)L.width * L.height * 3);
            other->decode(L, exact.data(), exact.size(), b, o.data(), (ptrdiff_t)L.width * 6, nullptr);
            continue;
        }
        slot->decode(L, exact.data(), exact.size(), b, out.data(), (ptrdiff_t)L0.width * 6, nullptr);
        ++decoded;
    }
    std::printf("  360 damaged frames: %d refused as malformed, %d as unsupported, %d with damaged packets, %d decoded to a picture "
                "(%d of another shape)\n", malformed, unsupported, badPackets, decoded + otherShape, otherShape);
    CHECK(malformed > 20 && badPackets > 5 && decoded > 20, "the damage did not reach every stage");
    slot->decode(L0, f.cs.data(), f.cs.size(), blk, out.data(), (ptrdiff_t)L0.width * 6, nullptr);
    CHECK(out == good, "the intact frame decodes differently after the damaged ones");
}

// The block decoder the GPU backends share, run on the host: the same words as the CPU
// backend's decoder for every block at every truncation point.
static void testT1DecodeKernel() {
    buildT1DecodeTables(g_t1dZc, g_t1dSc);
    std::mt19937 rng(42);
    int checks = 0;
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
                if (mode == 1 && sp(rng) < 0.7) v *= 0.05;
                c[y * 32 + x] = quantizeCoeff((float)v, 1.0f, (1u << 26) - 1);
            }
        T1Result res;
        t1EncodeRef(c.data(), w, h, orient, res);
        if (!res.nbps) continue;
        for (int k = 1; k <= res.numPasses; ++k) {
            const uint32_t len = res.passLen[k - 1];
            std::vector<uint32_t> want(kCblkArea, 0), got(kCblkArea, 0);
            t1DecodeBlock(res.data.data(), len, w, h, orient, res.nbps, k, want.data());
            T1dState st;
            MqDec q;
            mqd_init(q, res.data.data(), 0, len);
            t1d_decode_block(st, q, res.data.data(), (uint32_t)w, (uint32_t)h, (uint32_t)orient, (uint32_t)res.nbps, (uint32_t)k,
                             got.data());
            CHECK(got == want, "trial %d, %d of %d passes: the kernel decodes differently from the CPU decoder", trial, k,
                  res.numPasses);
            ++checks;
        }
    }
    std::printf("  %d truncation points\n", checks);
    // junk, and descriptors that make no sense, stay inside the block's 1024 words
    for (int trial = 0; trial < 300; ++trial) {
        const uint32_t len = rng() % 200;
        std::vector<uint8_t> junk(len);
        for (auto& b : junk) b = trial % 3 ? (uint8_t)rng() : 0xFF;
        std::vector<uint32_t> out(kCblkArea + 2, 0);
        out[0] = out[kCblkArea + 1] = 0xC0FFEEu;
        T1dState st;
        MqDec q;
        mqd_init(q, junk.data(), 0, len);
        t1d_decode_block(st, q, junk.data(), rng() % 40, rng() % 40, rng() % 8, rng() % 40, rng() % 200, out.data() + 1);
        CHECK(out[0] == 0xC0FFEEu && out[kCblkArea + 1] == 0xC0FFEEu, "trial %d: wrote outside the block", trial);
    }
}

#if defined(GAIL_HAVE_VULKAN) || defined(GAIL_HAVE_CUDA)
// A GPU decode slot against the CPU decoder: the dequantized coefficients (Tier-1 and
// step sizes) must be the same to the bit; the picture, which goes through the float
// wavelet on a different machine, within one code value.
static void testGpuDecode(Profile prof, bool cuda) {
    std::vector<std::unique_ptr<Device>> devs;
#ifdef GAIL_HAVE_CUDA
    if (cuda) devs = testGpus(true, stderrLogger(false));
#endif
#ifdef GAIL_HAVE_VULKAN
    if (!cuda) devs = testGpus(false, stderrLogger(false));
#endif
    devs.erase(std::remove_if(devs.begin(), devs.end(), [](const std::unique_ptr<Device>& d) { return !d->canDecode(); }), devs.end());
    if (devs.empty()) {
        std::printf("  (skipped: no %s device that decodes on this machine)\n", cuda ? "CUDA" : "Vulkan");
        return;
    }
    const EncodedFrame f = encodeCpuFrame(prof);
    StreamHeader h;
    std::string err;
    CHECK(parseCodestream(f.cs.data(), f.cs.size(), false, h, err) && h.supported, "parse: %s %s", err.c_str(), h.unsupported.c_str());
    const FrameLayout L = layoutFor(h);
    const int W = L.width;
    auto pool = std::make_shared<ThreadPool>(usableCpuCount(), false);
    for (int reduce = 0; reduce <= 1; ++reduce) {
        std::vector<CblkCode> blk;
        CHECK(parsePackets(f.cs.data(), h, L, L.levels - reduce, blk, err), "packets: %s", err.c_str());
        DecodeSettings ds;
        ds.reduce = reduce;
        ds.highBits = reduce == 0;
        int w, hh;
        decodedSize(L, reduce, w, hh);
        std::vector<uint16_t> want((size_t)w * hh * 3);
        std::vector<float> wantCoef[3];
        DecodeTimings tc;
        tc.wavelet = wantCoef;
        createCpuDecoderSlot(L, ds, pool)->decode(L, f.cs.data(), f.cs.size(), blk, want.data(), (ptrdiff_t)w * 6, &tc);
        for (auto& d : devs) {
            auto slot = d->createDecoderSlot(L, ds);
            const ptrdiff_t stride = (ptrdiff_t)w * 6 + 14;
            std::vector<uint8_t> got((size_t)stride * hh, 0x3C);
            std::vector<float> gotCoef[3];
            DecodeTimings tg;
            tg.wavelet = gotCoef;
            slot->decode(L, f.cs.data(), f.cs.size(), blk, got.data(), stride, &tg);
            // coefficients of the resolutions that were decoded
            size_t coefBad = 0;
            for (int i = 0; i < L.numCblks(); ++i) {
                const CodeBlock& cb = L.cblks[i];
                const Subband& b = L.bands[cb.band];
                if (b.res > L.levels - reduce) continue;
                for (int y = 0; y < cb.h; ++y) {
                    const size_t at = (size_t)(b.py0 + cb.y0 + y) * W + b.px0 + cb.x0;
                    coefBad += std::memcmp(&gotCoef[b.comp][at], &wantCoef[b.comp][at], (size_t)cb.w * sizeof(float)) != 0;
                }
            }
            CHECK(coefBad == 0, "%s reduce %d: %zu code-block rows of coefficients differ from the CPU decoder", d->name().c_str(),
                  reduce, coefBad);
            int maxd = 0, touched = 0;
            size_t differ = 0;
            for (int y = 0; y < hh; ++y) {
                const uint16_t* row = (const uint16_t*)(got.data() + (size_t)y * stride);
                for (int i = 0; i < w * 3; ++i) {
                    const int dd = std::abs((int)row[i] - (int)want[(size_t)y * w * 3 + i]) >> (ds.highBits ? 4 : 0);
                    maxd = std::max(maxd, dd);
                    differ += dd != 0;
                }
                for (int i = 0; i < 14; ++i) touched += got[(size_t)y * stride + (size_t)w * 6 + i] != 0x3C;
            }
            std::printf("  %s %s reduce %d: Tier-1 %.2f ms, wavelet %.2f, color + download %.2f, total %.2f; pixels: max diff %d, "
                        "%.4f%% differ\n", d->name().c_str(), profileName(prof), reduce, tg.t1, tg.dwt, tg.color, tg.total, maxd,
                        100.0 * (double)differ / ((double)w * hh * 3));
            CHECK(maxd <= 1 && differ * 50 < (size_t)w * hh * 3, "%s reduce %d: picture deviates from the CPU decoder (max %d, %zu "
                  "samples)", d->name().c_str(), reduce, maxd, differ);
            CHECK(touched == 0, "%s: %d padding bytes written", d->name().c_str(), touched);
            // the same slot again, timed without the test's extra download
            DecodeTimings t2;
            slot->decode(L, f.cs.data(), f.cs.size(), blk, got.data(), stride, &t2);
            slot->decode(L, f.cs.data(), f.cs.size(), blk, got.data(), stride, &t2);
            std::printf("    steady state: Tier-1 %.2f ms, wavelet %.2f, color + download %.2f, total %.2f\n", t2.t1, t2.dwt, t2.color,
                        t2.total);
        }
    }
}
#endif

// The shared thread pool: every index handed out once, concurrent run() calls take turns,
// and a job's exception reaches the caller without wedging the pool.
static void testThreadPool() {
    ThreadPool pool(4, false);
    CHECK(pool.threads() == 4, "pool has %d threads", pool.threads());
    const int n = 10000;
    std::atomic<long long> sum{0};
    std::vector<std::atomic<int>> hits(n);
    pool.run(n, [&](int i) {
        sum += i;
        ++hits[i];
    });
    CHECK(sum.load() == 49995000LL, "sum %lld", sum.load());
    int wrong = 0;
    for (auto& h : hits) wrong += h.load() != 1;
    CHECK(wrong == 0, "%d indices not run exactly once", wrong);

    int calls = 0;
    pool.run(0, [&](int) { ++calls; });
    CHECK(calls == 0, "run(0) called the job");

    bool thrown = false;
    try {
        pool.run(100, [&](int i) {
            if (i == 7) throw std::runtime_error("seven");
        });
    } catch (const std::runtime_error& e) {
        thrown = std::string(e.what()) == "seven";
    }
    CHECK(thrown, "exception from a job did not reach the caller");
    sum = 0;
    pool.run(n, [&](int i) { sum += i; });
    CHECK(sum.load() == 49995000LL, "pool unusable after an exception: sum %lld", sum.load());

    std::atomic<long long> a{0}, b{0};
    std::thread other([&] { pool.run(n, [&](int i) { a += i; }); });
    pool.run(n, [&](int i) { b += i; });
    other.join();
    CHECK(a.load() == 49995000LL && b.load() == 49995000LL, "concurrent runs: %lld, %lld", a.load(), b.load());

    ThreadPool one(1, false);
    sum = 0;
    one.run(100, [&](int i) { sum += i; });
    CHECK(sum.load() == 4950, "single-thread pool: sum %lld", sum.load());
}

int main(int argc, char** argv) {
    std::string only = argc > 1 ? argv[1] : "";
    auto run = [&](const char* name, const std::function<void()>& fn) {
        if (!only.empty() && only != name) return;
        std::printf("[%s]\n", name);
        fn();
    };
    run("pool", testThreadPool);
#ifdef GAIL_TEST_PLUGINS
    run("plugins", testCudaPlugins);
#endif
    run("bitio", testBitIo);
    run("tagtree", testTagTree);
    run("progression", testProgression);
    run("geometry", testGeometry);
    run("parse2k", [] { testParse(Profile::Cinema2K, Container::Flat); });
    run("parse4k", [] { testParse(Profile::Cinema4K, Container::Scope); });
    run("decode2k", [] { testDecodeCpu(Profile::Cinema2K); });
    run("decode4k", [] { testDecodeCpu(Profile::Cinema4K); });
    run("mutate", testDecodeMutations);
    run("dwt", testDwtRoundTrip);
    run("norms", testNorms);
    run("t1", testT1Truncation);
    run("t1limit", testT1PassLimit);
    run("t1dec", testT1Decode);
    run("t1kernel", testT1DecodeKernel);
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
    run("comment", testCodestreamComment);
    run("checker", testCheckerMalformed);
    run("matrix", testSourceMatrix);
    run("sources", testSourceErrors);
#ifdef GAIL_HAVE_VULKAN
    run("gpudec2k", [] { testGpuDecode(Profile::Cinema2K, false); });
    run("gpudec4k", [] { testGpuDecode(Profile::Cinema4K, false); });
    run("gpumem", [] { testGpuMemory(false); });
    run("gpucolor", [] { testGpuColor(false); });
    run("gpu2k", [] { testGpu(Profile::Cinema2K, false); });
    run("gpu4k", [] { testGpu(Profile::Cinema4K, false); });
#endif
#ifdef GAIL_HAVE_CUDA
    run("cudadec2k", [] { testGpuDecode(Profile::Cinema2K, true); });
    run("cudadec4k", [] { testGpuDecode(Profile::Cinema4K, true); });
    run("cudamem", [] { testGpuMemory(true); });
    run("cudacolor", [] { testGpuColor(true); });
    run("cuda2k", [] { testGpu(Profile::Cinema2K, true); });
    run("cuda4k", [] { testGpu(Profile::Cinema4K, true); });
#endif
    std::printf(g_fail ? "%d FAILURES\n" : "all tests passed\n", g_fail);
    return g_fail ? 1 : 0;
}
