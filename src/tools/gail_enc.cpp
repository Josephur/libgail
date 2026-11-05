// SPDX-License-Identifier: MIT
// gail-enc: GPU accelerated DCI JPEG 2000 encoder.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "api/devices.h"
#include "core/backend.h"
#include "core/frame_source.h"
#include "core/pipeline.h"

using namespace gail;

namespace {

void usage() {
    std::puts(
        "usage: gail-enc -i INPUT -o OUTDIR [options]\n"
        "  -i, --input PATH        video file (anything FFmpeg can decode)\n"
        "      --pattern           synthetic test pattern instead of an input file\n"
        "      --xyz-dir DIR       raw 12-bit X'Y'Z' frames (*.xyz) instead of a video (passed through)\n"
        "  -o, --output DIR        directory for frame_NNNNNN.j2c codestreams\n"
        "      --profile 2k|4k     DCI profile (default: 4k for UHD+ sources, else 2k)\n"
        "      --container C       flat | scope | full (default flat)\n"
        "      --fps N             output frame rate (default 24)\n"
        "      --frames N          number of frames to encode\n"
        "      --start SEC         start time in the source\n"
        "      --mbps N            max frame bitrate in Mbit/s (default 230, DCI hard limit 250)\n"
        "      --step X            base quantisation step in 12-bit code values (default 1.0)\n"
        "      --backend B         auto | cuda | vulkan | cpu (default auto: CUDA when NVIDIA GPUs are\n"
        "                          present, else Vulkan, else CPU)\n"
        "      --gpus LIST         devices to use: 'all' (default), indices (0,1) or IDs as --list-devices\n"
        "                          shows them (UUIDs as nvidia-smi prints them, or PCI bus IDs)\n"
        "      --list-devices      list the devices of every backend and exit\n"
        "      --slots N           frames in flight per device (default 3; fewer when they do not fit\n"
        "                          the free device memory, and frames are given up while encoding\n"
        "                          when other programs leave too little)\n"
        "      --device-memory MB  device memory the frames in flight may take per GPU (default: what is\n"
        "                          free at the start, less 512 MB or 6%)\n"
        "      --threads N         CPU backend threads (default: the cores this process may use)\n"
        "      --low-priority      run the encoder's threads below normal priority (background encode)\n"
        "      --transfer T        source transfer: auto | bt1886 | srgb | pq | hlg | linear\n"
        "      --primaries P       source primaries: auto | bt709 | bt2020 | p3d65 | dcip3\n"
        "      --hdr-peak NITS     source mastering peak for PQ/HLG (default: metadata or 1000)\n"
        "\n  Colour handling (source -> DCI X'Y'Z'):\n"
        "      --tonemap M         for PQ/HLG sources: signal (default; HDR shown as gamma 2.4 video,\n"
        "                          as DCP-o-matic does) | shoulder | bt2390 | none\n"
        "      (the next four set the shoulder)\n"
        "      --ref-white-in N    source reference white in cd/m2 (default 203, BT.2408)\n"
        "      --ref-white-out N   where it lands in the cinema, cd/m2 (default 16 of 48)\n"
        "      --tone-norm N       channel (default: highlights drift toward white, like film) |\n"
        "                          maxrgb (keep hue and saturation) | luminance\n"
        "      --knee F            start of the highlight roll-off, fraction of peak (default 0.25)\n"
        "      --cinema-peak N     target peak white, cd/m2 (default 48)\n"
        "      --sdr-white N       cd/m2 for SDR code value 1.0 (default 48)\n"
        "      --gamut G           none (default, as DCP-o-matic) | p3d65 | dcip3: colours outside the\n"
        "                          projector gamut are pulled toward neutral at constant luminance\n"
        "      --gamut-threshold F compression starts at this fraction of the gamut edge (default 0.8)\n"
        "      --lut3d FILE.cube   app-supplied 3D LUT, source R'G'B' -> X'Y'Z' (0..1); replaces the above\n"
        "      --input-format F    auto | yuv | rgb: hand 4:2:0 sources to the GPU as Y'CbCr (default)\n"
        "                          or let FFmpeg convert to RGB48 on the CPU\n"
        "      --rd-skip F         Tier-1 early termination: stop coding a block once a pass's\n"
        "                          estimated RD slope < F * lambda of the previous frame (default 0.25, 0 = off)\n"
        "      --no-check          skip the per-frame DCI compliance check\n"
        "      --dump-xyz N        also write the 12-bit X'Y'Z' reference of every Nth frame (validation)\n"
        "  -v, --verbose");
}

std::vector<std::string> splitList(const std::string& s) {
    std::vector<std::string> v;
    if (s == "all" || s.empty()) return v;
    std::stringstream ss(s);
    std::string t;
    while (std::getline(ss, t, ',')) v.push_back(t);
    return v;
}

bool allDigits(const std::vector<std::string>& v) {
    return std::all_of(v.begin(), v.end(), [](const std::string& t) {
        return !t.empty() && std::all_of(t.begin(), t.end(), ::isdigit);
    });
}

// --gpus 0,1 selects by index; anything else (UUIDs, PCI bus IDs) by ID
std::vector<int> parseList(const std::string& s) {
    std::vector<int> v;
    const auto t = splitList(s);
    if (allDigits(t))
        for (const auto& x : t) v.push_back(std::atoi(x.c_str()));
    return v;
}

std::vector<std::string> parseIds(const std::string& s) {
    const auto t = splitList(s);
    return allDigits(t) ? std::vector<std::string>{} : t;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, outDir, xyzDir, profileStr, containerStr = "flat", backend = "auto", gpus = "all";
    std::string transfer = "auto", primaries = "auto", tonemap = "signal", toneNorm = "channel", inputFormat = "auto", gamut = "none",
                lut3d;
    double refWhiteIn = 203, refWhiteOut = 16, knee = 0.25, cinemaPeak = 48, sdrWhite = 48, gamutThreshold = 0.8;
    double fps = 24, start = 0, mbps = 230, step = 1.0, hdrPeak = 0, rdSkip = 0.25;
    long long frames = -1;
    int slots = 3, threads = 0;
    bool slotsSet = false, lowPriority = false;
    double deviceMemoryMb = 0;
    bool pattern = false, check = true, verbose = false;
    int dumpXyz = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                std::cerr << "missing value for " << a << "\n";
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-i" || a == "--input") input = next();
        else if (a == "-o" || a == "--output") outDir = next();
        else if (a == "--pattern") pattern = true;
        else if (a == "--xyz-dir") xyzDir = next();
        else if (a == "--profile") profileStr = next();
        else if (a == "--container") containerStr = next();
        else if (a == "--fps") fps = std::atof(next().c_str());
        else if (a == "--frames") frames = std::atoll(next().c_str());
        else if (a == "--start") start = std::atof(next().c_str());
        else if (a == "--mbps") mbps = std::atof(next().c_str());
        else if (a == "--step") step = std::atof(next().c_str());
        else if (a == "--backend") backend = next();
        else if (a == "--gpus") gpus = next();
        else if (a == "--list-devices") {
            for (const char* b : {"cuda", "vulkan", "cpu"}) {
                std::string chosen;
                for (const auto& d : listDevices(b, chosen))
                    std::printf("%-6s %d: %s%s%s%s\n", b, d.index, d.name.c_str(), d.uuid.empty() ? "" : "  ",
                                d.uuid.c_str(), d.pciBusId.empty() ? "" : ("  " + d.pciBusId).c_str());
            }
            return 0;
        }
        else if (a == "--slots") {
            slots = std::atoi(next().c_str());
            slotsSet = true;
        }
        else if (a == "--low-priority") lowPriority = true;
        else if (a == "--device-memory") deviceMemoryMb = std::atof(next().c_str());
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--transfer") transfer = next();
        else if (a == "--primaries") primaries = next();
        else if (a == "--hdr-peak") hdrPeak = std::atof(next().c_str());
        else if (a == "--tonemap") tonemap = next();
        else if (a == "--tone-norm") toneNorm = next();
        else if (a == "--ref-white-in") refWhiteIn = std::atof(next().c_str());
        else if (a == "--ref-white-out") refWhiteOut = std::atof(next().c_str());
        else if (a == "--knee") knee = std::atof(next().c_str());
        else if (a == "--cinema-peak") cinemaPeak = std::atof(next().c_str());
        else if (a == "--sdr-white") sdrWhite = std::atof(next().c_str());
        else if (a == "--gamut") gamut = next();
        else if (a == "--gamut-threshold") gamutThreshold = std::atof(next().c_str());
        else if (a == "--lut3d") lut3d = next();
        else if (a == "--input-format") inputFormat = next();
        else if (a == "--rd-skip") rdSkip = std::atof(next().c_str());
        else if (a == "--no-check") check = false;
        else if (a == "--dump-xyz") dumpXyz = std::atoi(next().c_str());
        else if (a == "-v" || a == "--verbose") verbose = true;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::cerr << "unknown option " << a << "\n";
            usage();
            return 2;
        }
    }
    if (!xyzDir.empty()) {
        pattern = false;
        if (transfer == "auto") transfer = "gamma26";
        if (primaries == "auto") primaries = "xyz";
    }
    if (input.empty() && !pattern && xyzDir.empty()) {
        usage();
        return 2;
    }
    if (mbps > 250) {
        std::cerr << "--mbps above 250 would violate the DCI bitrate limit\n";
        return 2;
    }
    if (!(mbps > 0) || !(fps > 0)) {  // also catches NaN; a zero budget would switch the rate control off
        std::cerr << "--mbps and --fps must be positive\n";
        return 2;
    }

    try {
        SourceInfo info;
        if (!pattern && xyzDir.empty()) info = probeSource(input);
        Profile profile = Profile::Cinema2K;
        if (profileStr == "4k") profile = Profile::Cinema4K;
        else if (profileStr.empty() && info.width >= 3840) profile = Profile::Cinema4K;
        int W, H;
        containerSize(profile, parseContainer(containerStr), W, H);

        EncodeSettings es;
        es.profile = profile;
        es.width = W;
        es.height = H;
        es.baseStep = step;
        es.slopeFactor = rdSkip;
        ColorParams& cp = es.color;
        if (pattern) {
            cp.transfer = Transfer::BT1886;
            cp.primaries = Primaries::BT709;
        } else if (!xyzDir.empty()) {
            cp.transfer = parseTransfer(transfer);
            cp.primaries = parsePrimaries(primaries);
        } else {
            cp.transfer = parseTransfer(transfer == "auto" ? (info.transfer.empty() || info.transfer == "unknown"
                                                                  ? std::string("bt1886")
                                                                  : info.transfer)
                                                           : transfer);
            cp.primaries = parsePrimaries(primaries == "auto" ? (info.primaries.empty() || info.primaries == "unknown"
                                                                     ? std::string("bt709")
                                                                     : info.primaries)
                                                              : primaries);
        }
        cp.toneMap = parseToneMap(tonemap);
        cp.toneNorm = parseToneNorm(toneNorm);
        cp.hdrRefWhite = refWhiteIn;
        cp.cinemaRefWhite = refWhiteOut;
        cp.knee = knee;
        cp.cinemaPeak = cinemaPeak;
        cp.sdrWhite = sdrWhite;
        cp.gamut = parseGamut(gamut);
        cp.gamutThreshold = gamutThreshold;
        cp.lut3d = lut3d;
        cp.hdrPeak = hdrPeak > 0 ? hdrPeak : (info.masteringPeak > 0 ? info.masteringPeak : 1000.0);

        FrameLayout L = makeLayout(W, H, profile, step);

        const bool autoBackend = backend == "auto";  // may fall back to the CPU when the GPUs are full
        auto devices = createDevices(backend, parseList(gpus), parseIds(gpus), threads, lowPriority, stderrLogger(verbose), backend);

        std::unique_ptr<FrameSource> src;
        if (pattern) {
            src = openPatternSource(W, H, frames < 0 ? 48 : frames, 1);
        } else if (!xyzDir.empty()) {
            src = openXyzDirSource(xyzDir, W, H, frames);
        } else {
            SourceOptions so;
            so.path = input;
            so.width = W;
            so.height = H;
            so.fps = fps;
            so.start = start;
            so.maxFrames = frames;
            so.yuv = inputFormat != "rgb";
            src = openFfmpegSource(so, info);
        }
        es.input = src->format();

        std::fprintf(stderr, "gail-enc: %s %dx%d, %d levels, %d code-blocks, %.3g fps, max %.0f Mbit/s, input %s\n",
                     profileName(profile), W, H, L.levels, L.numCblks(), fps, mbps,
                     es.input.kind != FrameFormat::Kind::Planar ? "RGB48"
                     : es.input.rgb                             ? "planar G'B'R' (GPU conversion)"
                     : es.input.sy                              ? "Y'CbCr 4:2:0 (GPU conversion)"
                     : es.input.sx                              ? "Y'CbCr 4:2:2 (GPU conversion)"
                                                                : "Y'CbCr 4:4:4 (GPU conversion)");
        if (!pattern && xyzDir.empty()) {
            const bool untagged = info.matrix.empty() || info.matrix == "unknown";
            std::fprintf(stderr, "  source %dx%d %.3f fps, transfer %s, primaries %s, matrix %s%s, peak %.0f cd/m2\n",
                         info.width, info.height, info.fps, transferName(cp.transfer), primariesName(cp.primaries),
                         sourceMatrix(info).c_str(), untagged ? " (assumed: the stream is not tagged)" : "", cp.hdrPeak);
        }
        std::fprintf(stderr, "  colour: %s\n", ColorConverter(cp).describe().c_str());
        for (size_t d = 0; d < devices.size(); ++d)
            std::fprintf(stderr, "  device %zu: %s\n", d, devices[d]->name().c_str());

        if (!outDir.empty()) std::filesystem::create_directories(outDir);
        PipelineOptions po;
        po.outDir = outDir;
        po.fps = fps;
        po.limits = dciLimits(fps, mbps);
        po.slotsPerDevice = slotsSet ? slots : backend == "cpu" ? kCpuSlots : slots;
        po.cpuThreads = threads;
        po.lowPriority = lowPriority;
        po.deviceMemoryLimit = (uint64_t)(deviceMemoryMb * 1048576.0);
        po.cpuFallback = autoBackend;
        po.onLog = [](const std::string& m) { std::fprintf(stderr, "gail-enc: %s\n", m.c_str()); };
        po.check = check;
        po.dumpReferenceEvery = dumpXyz;
        po.onFrame = [&](const FrameStats& st, const std::vector<uint8_t>&) {
            if (verbose || !st.compliant)
                std::fprintf(stderr,
                             "frame %6lld dev %d %8zu bytes  est.PSNR %6.2f dB  T1 %.1f ms (color %.1f dwt %.1f q %.1f "
                             "t1 %.1f) RC+T2 %.1f ms %s\n",
                             st.index, st.device, st.bytes,
                             st.mseEstimate > 0 ? 10 * std::log10(4095.0 * 4095.0 / st.mseEstimate) : 99.0,
                             st.tier1.total, st.tier1.color, st.tier1.dwt, st.tier1.quant, st.tier1.t1, st.rateMs,
                             st.compliant ? "" : st.checkSummary.c_str());
            else if (st.index % 24 == 0)
                std::fprintf(stderr, "frame %lld\r", st.index);
        };
        PipelineSummary sum = runPipeline(*src, devices, L, es, po);
        std::fprintf(stderr, "\nencoded %lld frames in %.2f s (%.2f fps), avg %.0f bytes/frame (%.1f Mbit/s at %g fps), "
                             "max %zu, non-compliant %lld\n",
                     sum.frames, sum.seconds, sum.frames / sum.seconds, sum.frames ? (double)sum.bytes / sum.frames : 0.0,
                     sum.frames ? sum.bytes * 8.0 * fps / sum.frames / 1e6 : 0.0, fps, sum.maxFrameBytes,
                     sum.nonCompliant);
        for (size_t d = 0; d < sum.framesPerDevice.size(); ++d)
            std::fprintf(stderr, "  device %zu: %lld frames (%d in flight%s)\n", d, sum.framesPerDevice[d],
                         sum.slotsPerDevice[d],
                         sum.slotsRetired[d] ? (", " + std::to_string(sum.slotsRetired[d]) + " given up").c_str() : "");
        return sum.nonCompliant ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
