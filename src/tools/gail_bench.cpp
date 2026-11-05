// SPDX-License-Identifier: MIT
// gail-bench: encoder throughput without decode cost (frames preloaded in RAM).
#include <cmath>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "core/backend.h"
#include "core/frame_source.h"
#include "core/pipeline.h"

using namespace gail;

int main(int argc, char** argv) {
    std::string input, backend = "vulkan", gpus = "all", profileStr = "4k";
    int frames = 24, loops = 4, slots = 3, threads = 0;
    bool slotsSet = false, lowPriority = false;
    double deviceMemoryMb = 0;
    double step = 1.0, fps = 24, mbps = 230, start = 0, rdSkip = 0;
    bool check = false, rgbInput = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {  // argv[argc] is NULL: std::string would throw and abort the process
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-i") input = next();
        else if (a == "--backend") backend = next();
        else if (a == "--gpus") gpus = next();
        else if (a == "--profile") profileStr = next();
        else if (a == "--frames") frames = std::atoi(next().c_str());
        else if (a == "--loops") loops = std::atoi(next().c_str());
        else if (a == "--slots") {
            slots = std::atoi(next().c_str());
            slotsSet = true;
        }
        else if (a == "--low-priority") lowPriority = true;
        else if (a == "--device-memory") deviceMemoryMb = std::atof(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--step") step = std::atof(next().c_str());
        else if (a == "--mbps") mbps = std::atof(next().c_str());
        else if (a == "--check") check = true;
        else if (a == "--start") start = std::atof(next().c_str());
        else if (a == "--rd-skip") rdSkip = std::atof(next().c_str());
        else if (a == "--rgb") rgbInput = true;
        else {
            std::fprintf(stderr, "usage: gail-bench [-i video] [--profile 2k|4k] [--backend vulkan|cuda|cpu] [--gpus 0,1]\n"
                                 "                  [--frames N] [--loops N] [--slots N] [--threads N] [--low-priority]\n"
                                 "                  [--device-memory MB] [--step X] [--check]\n");
            return 2;
        }
    }
    if (!(mbps > 0) || mbps > 250 || !(fps > 0)) {  // also catches NaN; a zero budget would switch the rate control off
        std::fprintf(stderr, "--mbps must be in (0, 250] and --fps positive\n");
        return 2;
    }
    try {
        const Profile prof = profileStr == "2k" ? Profile::Cinema2K : Profile::Cinema4K;
        int W, H;
        containerSize(prof, Container::Flat, W, H);
        EncodeSettings es;
        es.profile = prof;
        es.width = W;
        es.height = H;
        es.baseStep = step;
        es.slopeFactor = rdSkip;
        std::unique_ptr<FrameSource> base;
        if (!input.empty()) {
            SourceInfo info = probeSource(input);
            es.color.transfer = parseTransfer(info.transfer.empty() || info.transfer == "unknown" ? "bt1886" : info.transfer);
            es.color.primaries = parsePrimaries(info.primaries.empty() || info.primaries == "unknown" ? "bt709" : info.primaries);
            es.color.hdrPeak = info.masteringPeak > 0 ? info.masteringPeak : 1000;
            SourceOptions so;
            so.path = input;
            so.width = W;
            so.height = H;
            so.fps = fps;
            so.maxFrames = frames;
            so.start = start;
            so.yuv = !rgbInput;
            base = openFfmpegSource(so, info);
        } else {
            base = openPatternSource(W, H, frames, 1);
        }
        auto src = openMemorySource(*base, frames, loops);
        es.input = src->format();
        FrameLayout L = makeLayout(W, H, prof, step);
        std::vector<std::unique_ptr<Device>> devices;
        std::vector<int> sel;
        if (gpus != "all") {
            std::stringstream ss(gpus);
            std::string t;
            while (std::getline(ss, t, ',')) sel.push_back(std::atoi(t.c_str()));
        }
        if (backend == "cpu") devices.push_back(createCpuDevice(threads, lowPriority));
#ifdef GAIL_HAVE_VULKAN
        else if (backend == "vulkan") devices = createVulkanDevices(sel, stderrLogger(false));
#endif
#ifdef GAIL_HAVE_CUDA
        else if (backend == "cuda") devices = createCudaDevices(sel, stderrLogger(false));
#endif
        if (devices.empty()) throw std::runtime_error("no devices for backend " + backend);

        std::mutex m;
        double sum[12] = {};
        long long n = 0;
        double psnr = 0;
        PipelineOptions po;
        po.fps = fps;
        po.limits = dciLimits(fps, mbps);
        po.slotsPerDevice = slotsSet ? slots : backend == "cpu" ? kCpuSlots : slots;
        po.lowPriority = lowPriority;
        po.deviceMemoryLimit = (uint64_t)(deviceMemoryMb * 1048576.0);
        po.onLog = [](const std::string& msg) { std::fprintf(stderr, "gail-bench: %s\n", msg.c_str()); };
        po.check = check;
        size_t bytes = 0;
        po.onFrame = [&](const FrameStats& st, const std::vector<uint8_t>&) {
            std::lock_guard<std::mutex> l(m);
            const FrameTimings& t = st.tier1;
            const double v[] = {t.upload, t.color, t.dwt, t.quant, t.t1cm, t.t1mq, t.download, t.gpu, t.total, st.rateMs,
                                t.symbols, t.codedBytes};
            for (int i = 0; i < 12; ++i) sum[i] += v[i];
            psnr += st.mseEstimate > 0 ? 10 * std::log10(4095.0 * 4095.0 / st.mseEstimate) : 99;
            bytes += st.bytes;
            ++n;
        };
        PipelineSummary s = runPipeline(*src, devices, L, es, po);
        std::printf("%s %s %dx%d: %lld frames in %.2f s = %.2f fps (%zu devices x %d slots), avg %.0f bytes, "
                    "non-compliant %lld\n",
                    backend.c_str(), profileName(prof), W, H, s.frames, s.seconds, s.frames / s.seconds, devices.size(),
                    po.slotsPerDevice, (double)bytes / n, s.nonCompliant);
        const char* names[] = {"upload", "color", "dwt", "quant", "t1cm", "t1mq", "download", "gpu", "slot", "rc+t2"};
        std::printf("  avg ms:");
        for (int i = 0; i < 10; ++i) std::printf(" %s %.2f", names[i], sum[i] / n);
        std::printf("\n  est. PSNR %.3f dB, Tier-1 symbols %.2fM/frame, coded %.2f MB/frame, %.3f bits/symbol\n",
                    psnr / n, sum[10] / n / 1e6, sum[11] / n / 1e6, sum[10] > 0 ? sum[11] * 8 / sum[10] : 0.0);
        for (size_t d = 0; d < devices.size(); ++d)
            std::printf("  device %zu %s: %lld frames (%d in flight%s)\n", d, devices[d]->name().c_str(),
                        s.framesPerDevice[d], s.slotsPerDevice[d],
                        s.slotsRetired[d] ? (", " + std::to_string(s.slotsRetired[d]) + " given up").c_str() : "");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
