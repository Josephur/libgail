// SPDX-License-Identifier: MIT
// gail-dec: decode DCI JPEG 2000 codestreams to raw X'Y'Z' frames, through the library's
// public API. Also the decoder's throughput test: without -o nothing is written.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "gail/gail.h"

namespace {

void usage() {
    std::fprintf(stderr,
        "usage: gail-dec -i INPUT [-o OUTDIR] [options]\n"
        "  -i, --input PATH        a JPEG 2000 codestream (.j2c / .j2k), or a directory of them\n"
        "                          (decoded in name order)\n"
        "  -o, --output DIR        directory for frame_NNNNNN.xyz: raw pictures, three 16-bit\n"
        "                          little-endian words per pixel (X', Y', Z'). Without it the\n"
        "                          frames are decoded and dropped (a speed test)\n"
        "      --low-bits          12-bit codes in the low bits of each word (default: the high\n"
        "                          bits, FFmpeg's xyz12le)\n"
        "      --reduce N          leave out N resolution levels: 1 = half size, 2 = quarter\n"
        "      --frames N          decode only the first N codestreams\n"
        "      --loops N           decode the set N times over (for timing; default 1)\n"
        "      --backend B         auto | cuda | vulkan | cpu (default auto)\n"
        "      --gpus LIST         devices to use: indices (0,1) or 'all' (default: the first)\n"
        "      --list-devices      list the devices of every backend and exit\n"
        "      --slots N           frames in flight per device\n"
        "      --threads N         CPU backend threads (default: the cores this process may use)\n"
        "      --low-priority      run the decoder's threads below normal priority\n"
        "      --info              print what each codestream is and exit\n"
        "  -v                      print the library's messages\n");
}

void onLog(void*, gail_log_level level, const char* message) {
    std::fprintf(stderr, "%s%s\n", level <= GAIL_LOG_WARNING ? "warning: " : "", message);
}

bool readFile(const std::string& path, std::vector<uint8_t>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    out.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    std::string input, outDir, backend = "auto", gpus;
    int reduce = 0, frames = -1, loops = 1, slots = 0, threads = 0;
    bool lowBits = false, lowPriority = false, verbose = false, info = false, listDevices = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "-i" || a == "--input") input = value();
        else if (a == "-o" || a == "--output") outDir = value();
        else if (a == "--low-bits") lowBits = true;
        else if (a == "--reduce") reduce = std::atoi(value());
        else if (a == "--frames") frames = std::atoi(value());
        else if (a == "--loops") loops = std::max(1, std::atoi(value()));
        else if (a == "--backend") backend = value();
        else if (a == "--gpus") gpus = value();
        else if (a == "--list-devices") listDevices = true;
        else if (a == "--slots") slots = std::atoi(value());
        else if (a == "--threads") threads = std::atoi(value());
        else if (a == "--low-priority") lowPriority = true;
        else if (a == "--info") info = true;
        else if (a == "-v") verbose = true;
        else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    static const char* const names[] = {"auto", "cuda", "vulkan", "cpu"};
    int be = -1;
    for (int i = 0; i < 4; ++i)
        if (backend == names[i]) be = i;
    if (be < 0) {
        std::fprintf(stderr, "unknown backend %s\n", backend.c_str());
        return 2;
    }
    if (listDevices) {
        for (int b = 1; b < 4; ++b) {
            char buf[4096] = "";
            if (gail_list_devices((gail_backend)b, buf, sizeof buf) > 0) std::fputs(buf, stdout);
        }
        return 0;
    }
    if (input.empty()) {
        usage();
        return 2;
    }

    namespace fs = std::filesystem;
    std::vector<std::string> files;
    std::error_code ec;
    if (fs::is_directory(input, ec)) {
        for (const auto& e : fs::directory_iterator(input, ec)) {
            std::string ext = e.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return (char)std::tolower(c); });
            if (ext == ".j2c" || ext == ".j2k") files.push_back(e.path().string());
        }
        std::sort(files.begin(), files.end());
    } else {
        files.push_back(input);
    }
    if (frames >= 0 && (size_t)frames < files.size()) files.resize((size_t)frames);
    if (files.empty()) {
        std::fprintf(stderr, "no .j2c / .j2k codestreams in %s\n", input.c_str());
        return 1;
    }
    // In memory first, so that reading them is not part of the timing
    std::vector<std::vector<uint8_t>> streams(files.size());
    size_t bytes = 0;
    for (size_t i = 0; i < files.size(); ++i) {
        if (!readFile(files[i], streams[i]) || streams[i].empty()) {
            std::fprintf(stderr, "%s: cannot read the file\n", files[i].c_str());
            return 1;
        }
        bytes += streams[i].size();
    }
    if (info) {
        for (size_t i = 0; i < files.size(); ++i) {
            gail_stream_info si;
            if (gail_probe(streams[i].data(), streams[i].size(), &si) != GAIL_OK) {
                std::printf("%s: %s\n", files[i].c_str(), gail_last_error());
                continue;
            }
            std::printf("%s: %dx%d, %d x %d-bit, Rsiz %d, %d levels, %s%s, %d tile%s, %d layer%s: %s\n", files[i].c_str(), si.width,
                        si.height, si.components, si.bit_depth, si.rsiz, si.levels, si.reversible ? "5/3" : "9/7",
                        si.high_throughput ? " HT" : "", si.tiles, si.tiles == 1 ? "" : "s", si.layers, si.layers == 1 ? "" : "s",
                        si.supported ? "supported" : gail_last_error());
        }
        return 0;
    }

    gail_decoder_config cfg;
    gail_decoder_config_defaults(&cfg);
    cfg.backend = (gail_backend)be;
    cfg.output = lowBits ? GAIL_PIX_XYZ12 : GAIL_PIX_XYZ12_MSB;
    cfg.reduce = reduce;
    cfg.frames_in_flight = slots;
    cfg.cpu_threads = threads;
    cfg.low_priority = lowPriority;
    if (verbose) cfg.log = onLog;
    std::vector<int> devices;
    if (gpus == "all") {
        cfg.num_devices = GAIL_ALL_DEVICES;
    } else if (!gpus.empty()) {
        for (size_t p = 0; p < gpus.size();) {
            const size_t q = gpus.find(',', p);
            devices.push_back(std::atoi(gpus.substr(p, q - p).c_str()));
            if (q == std::string::npos) break;
            p = q + 1;
        }
        cfg.devices = devices.data();
        cfg.num_devices = (int)devices.size();
    }
    gail_decoder* dec = nullptr;
    if (gail_decoder_create(&cfg, &dec) != GAIL_OK) {
        std::fprintf(stderr, "gail-dec: %s\n", gail_last_error());
        return 1;
    }
    if (!outDir.empty()) {
        fs::create_directories(outDir, ec);
        if (ec) {
            std::fprintf(stderr, "gail-dec: cannot create %s\n", outDir.c_str());
            return 1;
        }
    }

    long long decoded = 0, failed = 0;
    int width = 0, height = 0;
    bool writeError = false;
    auto take = [&](const gail_image& img) {
        if (img.status != GAIL_OK) {
            ++failed;
            std::fprintf(stderr, "%s: %s\n", files[(size_t)img.index % files.size()].c_str(), gail_last_error());
            return;
        }
        ++decoded;
        width = img.width;
        height = img.height;
        if (outDir.empty() || img.index >= (long long)files.size()) return;  // written on the first loop only
        char name[64];
        std::snprintf(name, sizeof name, "frame_%06lld.xyz", (long long)img.index);
        std::ofstream out(fs::path(outDir) / name, std::ios::binary);
        for (int y = 0; y < img.height && out; ++y)
            out.write((const char*)img.data[0] + (ptrdiff_t)y * img.stride[0], (std::streamsize)img.width * 6);
        if (!out) writeError = true;
    };

    // The first codestream sets the devices up (GPU programs, buffers): once, before the clock
    {
        gail_packet pkt{};
        pkt.data = streams[0].data();
        pkt.size = streams[0].size();
        gail_image warm{};
        if (gail_decoder_decode(dec, &pkt, &warm) == GAIL_OK) gail_image_free(&warm);
    }
    const auto t0 = std::chrono::steady_clock::now();
    int status = GAIL_OK;
    for (int loop = 0; loop < loops && status == GAIL_OK; ++loop)
        for (size_t i = 0; i < streams.size() && status == GAIL_OK; ++i) {
            gail_packet pkt{};
            pkt.data = streams[i].data();
            pkt.size = streams[i].size();
            gail_image img;
            // GAIL_EAGAIN: the decoder holds as many decoded pictures as it will; take one
            while ((status = gail_decoder_send(dec, &pkt, nullptr)) == GAIL_EAGAIN) {
                status = gail_decoder_receive(dec, &img, 1);
                if (status != GAIL_OK) break;
                take(img);
                gail_image_free(&img);
            }
            while (status == GAIL_OK && gail_decoder_receive(dec, &img, 0) == GAIL_OK) {
                take(img);
                gail_image_free(&img);
            }
        }
    if (status == GAIL_OK) status = gail_decoder_send(dec, nullptr, nullptr);
    gail_image img;
    while (status == GAIL_OK && gail_decoder_receive(dec, &img, 1) == GAIL_OK) {
        take(img);
        gail_image_free(&img);
    }
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (status != GAIL_OK) std::fprintf(stderr, "gail-dec: %s\n", gail_last_error());
    if (const char* err = gail_decoder_error(dec)) std::fprintf(stderr, "gail-dec: %s\n", err);
    const std::string devicesUsed = gail_decoder_devices(dec);
    gail_decoder_destroy(dec);
    if (writeError) std::fprintf(stderr, "gail-dec: could not write to %s\n", outDir.c_str());

    std::printf("%lld frames decoded", decoded);
    if (decoded) std::printf(" at %dx%d", width, height);
    if (failed) std::printf(", %lld failed", failed);
    std::printf(" in %.2f s: %.1f fps, %.0f Mbit/s of codestream at 24 fps, on %s\n", seconds, seconds > 0 ? decoded / seconds : 0.0,
                (double)bytes * 8 / 1e6 * 24 / (double)streams.size(), devicesUsed.c_str());
    return status != GAIL_OK || failed || writeError ? 1 : 0;
}
