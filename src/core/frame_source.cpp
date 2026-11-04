// SPDX-License-Identifier: MIT
#include "core/frame_source.h"

#if defined(_WIN32)
#include <cstdio>
#else
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

#include <algorithm>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>

namespace gail {

namespace {

// A child process (FFmpeg / ffprobe) whose stdout we read.
struct Child {
    FILE* f = nullptr;
#if !defined(_WIN32)
    pid_t pid = -1;
#endif
    // Closes the pipe and reaps the process; `stop` ends it first if it is still running.
    void close(bool stop) {
#if defined(_WIN32)
        (void)stop;  // the child gets a broken pipe and exits
        if (f) _pclose(f);
#else
        if (f) fclose(f);
        if (pid > 0) {
            if (stop) kill(pid, SIGTERM);
            int st;
            waitpid(pid, &st, 0);
        }
        pid = -1;
#endif
        f = nullptr;
    }
};

#if defined(_WIN32)
// cmd.exe / CommandLineToArgvW quoting
std::string quoteArg(const std::string& a) {
    if (!a.empty() && a.find_first_of(" \t\"&|<>^%") == std::string::npos) return a;
    std::string q = "\"";
    for (size_t i = 0; i < a.size(); ++i) {
        size_t bs = 0;
        while (i < a.size() && a[i] == '\\') ++bs, ++i;
        if (i == a.size()) {
            q.append(bs * 2, '\\');
            break;
        }
        if (a[i] == '"') q.append(bs * 2 + 1, '\\');
        else q.append(bs, '\\');
        q += a[i];
    }
    return q + "\"";
}
#endif

// Runs a command (argv) and returns its stdout.
Child spawnRead(const std::vector<std::string>& args) {
    Child c;
#if defined(_WIN32)
    std::string cmd = "\"";  // cmd /c strips one outer pair of quotes
    for (auto& a : args) cmd += quoteArg(a) + " ";
    cmd += "\"";
    c.f = _popen(cmd.c_str(), "rb");
    if (!c.f) throw std::runtime_error("failed to start " + args[0]);
#else
    int fds[2];
    if (pipe(fds) != 0) throw std::runtime_error("pipe failed");
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fds[1], 1);
    posix_spawn_file_actions_addclose(&fa, fds[0]);
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
    argv.push_back(nullptr);
    if (posix_spawnp(&c.pid, argv[0], &fa, nullptr, argv.data(), environ) != 0)
        throw std::runtime_error("failed to start " + args[0]);
    posix_spawn_file_actions_destroy(&fa);
    ::close(fds[1]);
    c.f = fdopen(fds[0], "rb");
#endif
    setvbuf(c.f, nullptr, _IOFBF, 1 << 22);
    return c;
}

double parseRational(const std::string& s) {
    auto p = s.find('/');
    if (p == std::string::npos) return std::atof(s.c_str());
    const double d = std::atof(s.c_str() + p + 1);
    return d != 0 ? std::atof(s.substr(0, p).c_str()) / d : 0.0;
}

class FfmpegSource : public FrameSource {
public:
    FfmpegSource(const SourceOptions& o, const SourceInfo& info) {
        const int W = o.width, H = o.height;
        int sw = W, sh = H;
        if (o.fitToContainer && info.width > 0 && info.height > 0) {
            const double s = std::min((double)W / info.width, (double)H / info.height);
            sw = std::min(W, (int)std::lround(info.width * s / 2.0) * 2);
            sh = std::min(H, (int)std::lround(info.height * s / 2.0) * 2);
        }
        fmt_.width = W;
        fmt_.height = H;
        // Native planar layouts: FFmpeg only decodes (and resizes if needed); colour
        // conversion, chroma upsampling and padding happen on the device.
        auto starts = [&](const char* p) { return info.pixFmt.rfind(p, 0) == 0; };
        const char* planar = starts("yuv420p") || info.pixFmt == "nv12" || info.pixFmt == "p010le" ? "yuv420p"
                             : starts("yuv422p") || starts("yuva422p")                              ? "yuv422p"
                             : starts("yuv444p") || starts("yuva444p")                              ? "yuv444p"
                             : starts("gbrp") || starts("gbrap")                                    ? "gbrp"
                                                                                                    : nullptr;
        std::string matrix = "auto";
        if (info.matrix == "bt2020nc" || info.matrix == "bt2020c") matrix = "bt2020";
        else if (info.matrix == "bt709") matrix = "bt709";
        else if (info.matrix == "smpte170m" || info.matrix == "bt470bg") matrix = "bt601";
        std::ostringstream vf;
        if (o.fps > 0) vf << "fps=" << o.fps << ",";
        if (o.yuv && planar) {
            const std::string layout = planar;
            fmt_.kind = FrameFormat::Kind::Planar;
            fmt_.rgb = layout == "gbrp";
            fmt_.sx = layout == "yuv420p" || layout == "yuv422p" ? 1 : 0;
            fmt_.sy = layout == "yuv420p" ? 1 : 0;
            fmt_.activeW = sw;
            fmt_.activeH = sh;
            fmt_.ox = (W - sw) / 2;  // same placement as the RGB path (FFmpeg pad)
            fmt_.oy = (H - sh) / 2;
            int depth = 8;
            for (int d : {16, 12, 10})
                if (info.pixFmt.find(std::to_string(d)) != std::string::npos || (d == 10 && info.pixFmt == "p010le")) {
                    depth = d;
                    break;
                }
            fmt_.bits = depth;
            fmt_.fullRange = info.range == "pc";
            if (matrix == "bt709") fmt_.kr = 0.2126, fmt_.kb = 0.0722;
            else if (matrix == "bt601") fmt_.kr = 0.299, fmt_.kb = 0.114;
            else if (matrix == "bt2020") fmt_.kr = 0.2627, fmt_.kb = 0.0593;
            else if (info.primaries == "bt2020") fmt_.kr = 0.2627, fmt_.kb = 0.0593;
            else if (info.height >= 720) fmt_.kr = 0.2126, fmt_.kb = 0.0722;
            else fmt_.kr = 0.299, fmt_.kb = 0.114;
            fmt_.chroma = info.chromaLocation == "center" ? FrameFormat::Chroma::Center
                          : info.chromaLocation == "topleft" ? FrameFormat::Chroma::TopLeft
                                                             : FrameFormat::Chroma::Left;
            if (sw != info.width || sh != info.height) vf << "scale=w=" << sw << ":h=" << sh << ":flags=lanczos,";
            vf << "format=" << (depth == 8 ? layout : layout + std::to_string(depth) + "le");
        } else {
            std::string range = info.range == "pc" ? "full" : (info.range == "tv" ? "limited" : "auto");
            vf << "scale=w=" << sw << ":h=" << sh << ":flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp"
               << ":in_color_matrix=" << matrix << ":in_range=" << range << ":out_range=full,format=rgb48le";
            if (sw != W || sh != H) vf << ",pad=" << W << ":" << H << ":(ow-iw)/2:(oh-ih)/2:color=black";
        }
        std::vector<std::string> args = {"ffmpeg", "-nostdin", "-v", "error"};
        if (o.start > 0) {
            args.push_back("-ss");
            args.push_back(std::to_string(o.start));
        }
        args.insert(args.end(), {"-i", o.path, "-an", "-sn", "-dn"});
        if (o.maxFrames > 0) {
            args.push_back("-frames:v");
            args.push_back(std::to_string(o.maxFrames));
        }
        args.insert(args.end(), {"-vf", vf.str(), "-f", "rawvideo", "-"});
        child_ = spawnRead(args);
    }
    ~FfmpegSource() override { child_.close(true); }
    const FrameFormat& format() const override { return fmt_; }
    bool read(void* dst) override { return fread(dst, 1, fmt_.bytes(), child_.f) == fmt_.bytes(); }

private:
    FrameFormat fmt_;
    Child child_;
};

class PatternSource : public FrameSource {
public:
    PatternSource(int w, int h, long long frames, unsigned seed) : w_(w), h_(h), frames_(frames), rng_(seed) {
        fmt_.width = w;
        fmt_.height = h;
    }
    const FrameFormat& format() const override { return fmt_; }
    bool read(void* dst) override {
        uint16_t* rgb = (uint16_t*)dst;
        if (frames_ >= 0 && n_ >= frames_) return false;
        std::uniform_int_distribution<int> noise(-600, 600);
        const double t = n_ * 0.05;
        for (int y = 0; y < h_; ++y)
            for (int x = 0; x < w_; ++x) {
                uint16_t* p = rgb + ((size_t)y * w_ + x) * 3;
                const double u = (double)x / w_, v = (double)y / h_;
                double r, g, b;
                if (v < 0.33) {  // colour bars, scrolling
                    const int bar = (int)((u + t * 0.1) * 8) & 7;
                    r = (bar & 1) ? 0.75 : 0.05;
                    g = (bar & 2) ? 0.75 : 0.05;
                    b = (bar & 4) ? 0.75 : 0.05;
                } else if (v < 0.66) {  // smooth gradients + zone plate
                    const double dx = u - 0.5, dy = v - 0.5;
                    const double z = 0.5 + 0.5 * std::cos(900.0 * (dx * dx + dy * dy) + t);
                    r = u;
                    g = 0.5 * z + 0.25;
                    b = 1.0 - u;
                } else {  // textured noise region
                    r = g = b = 0.3 + 0.2 * std::sin(40 * u + t) * std::cos(30 * v);
                }
                const int nz = noise(rng_);
                p[0] = (uint16_t)std::clamp((int)(r * 65535) + nz, 0, 65535);
                p[1] = (uint16_t)std::clamp((int)(g * 65535) + nz, 0, 65535);
                p[2] = (uint16_t)std::clamp((int)(b * 65535) + nz, 0, 65535);
            }
        ++n_;
        return true;
    }

private:
    FrameFormat fmt_;
    int w_, h_;
    long long frames_, n_ = 0;
    std::mt19937 rng_;
};

class MemorySource : public FrameSource {
public:
    MemorySource(FrameSource& from, int frames, int loops) : fmt_(from.format()), loops_(loops) {
        for (int i = 0; i < frames; ++i) {
            std::vector<uint8_t> f(fmt_.bytes());
            if (!from.read(f.data())) break;
            frames_.push_back(std::move(f));
        }
        if (frames_.empty()) throw std::runtime_error("memory source: no frames");
    }
    const FrameFormat& format() const override { return fmt_; }
    bool read(void* dst) override {
        if (pos_ >= (long long)frames_.size() * loops_) return false;
        const auto& f = frames_[pos_++ % frames_.size()];
        std::memcpy(dst, f.data(), f.size());
        return true;
    }

private:
    FrameFormat fmt_;
    int loops_;
    long long pos_ = 0;
    std::vector<std::vector<uint8_t>> frames_;
};

class XyzDirSource : public FrameSource {
public:
    XyzDirSource(const std::string& dir, int w, int h, long long maxFrames) {
        fmt_.width = w;
        fmt_.height = h;
        for (const auto& e : std::filesystem::directory_iterator(dir))
            if (e.path().extension() == ".xyz") files_.push_back(e.path().string());
        std::sort(files_.begin(), files_.end());
        if (maxFrames >= 0 && (long long)files_.size() > maxFrames) files_.resize(maxFrames);
        if (files_.empty()) throw std::runtime_error("no .xyz frames in " + dir);
    }
    const FrameFormat& format() const override { return fmt_; }
    bool read(void* dst) override {
        if (next_ >= files_.size()) return false;
        FILE* f = fopen(files_[next_++].c_str(), "rb");
        if (!f) return false;
        uint16_t* p = (uint16_t*)dst;
        const size_t n = (size_t)fmt_.width * fmt_.height * 3;
        const bool ok = fread(p, 2, n, f) == n;
        fclose(f);
        // 12-bit code -> 16-bit signal value; the gamma26/xyz colour path maps it back exactly
        for (size_t i = 0; i < n; ++i) p[i] = (uint16_t)((p[i] * 65535u + 2047u) / 4095u);
        return ok;
    }

private:
    FrameFormat fmt_;
    std::vector<std::string> files_;
    size_t next_ = 0;
};

}  // namespace

std::unique_ptr<FrameSource> openXyzDirSource(const std::string& dir, int width, int height, long long maxFrames) {
    return std::make_unique<XyzDirSource>(dir, width, height, maxFrames);
}

std::unique_ptr<FrameSource> openMemorySource(FrameSource& from, int frames, int loops) {
    return std::make_unique<MemorySource>(from, frames, loops);
}

SourceInfo probeSource(const std::string& path) {
    Child child = spawnRead({"ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=width,height,r_frame_rate,color_transfer,color_primaries,color_space,color_range,"
                         "pix_fmt,chroma_location,nb_frames:stream_side_data=max_luminance,max_content",
                         "-of", "default=noprint_wrappers=1", path});
    FILE* f = child.f;
    std::map<std::string, std::string> kv;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        std::string s(line);
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        auto eq = s.find('=');
        if (eq != std::string::npos && !kv.count(s.substr(0, eq))) kv[s.substr(0, eq)] = s.substr(eq + 1);
    }
    child.close(false);
    SourceInfo info;
    if (!kv.count("width")) throw std::runtime_error("ffprobe could not read " + path);
    info.width = std::atoi(kv["width"].c_str());
    info.height = std::atoi(kv["height"].c_str());
    info.fps = parseRational(kv["r_frame_rate"]);
    info.transfer = kv["color_transfer"];
    info.primaries = kv["color_primaries"];
    info.matrix = kv["color_space"];
    info.range = kv["color_range"];
    info.pixFmt = kv["pix_fmt"];
    info.chromaLocation = kv["chroma_location"];
    info.frames = std::atoll(kv["nb_frames"].c_str());
    if (kv.count("max_luminance")) info.masteringPeak = parseRational(kv["max_luminance"]);
    if (kv.count("max_content")) info.maxCLL = std::atof(kv["max_content"].c_str());
    return info;
}

std::unique_ptr<FrameSource> openFfmpegSource(const SourceOptions& o, const SourceInfo& info) {
    return std::make_unique<FfmpegSource>(o, info);
}

std::unique_ptr<FrameSource> openPatternSource(int width, int height, long long frames, unsigned seed) {
    return std::make_unique<PatternSource>(width, height, frames, seed);
}

}  // namespace gail
