// SPDX-License-Identifier: MIT
// Library API (include/gail/gail.h). Frames handed over by the app feed the same
// multi-device pipeline gail-enc uses, through a bounded queue; the pipeline's in-order
// writer hands each codestream to whoever is waiting for it: a gail_encoder_encode()
// call, the on_packet callback, or the queue gail_encoder_receive() reads.
#define GAIL_BUILDING
#include "gail/gail.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "api/devices.h"
#include "core/build_version.h"
#include "core/color.h"
#include "core/frame_input.h"
#include "core/layout.h"
#include "core/pipeline.h"

namespace {

using namespace gail;

thread_local std::string g_lastError = "no error";

int fail(int status, const std::string& msg) {
    g_lastError = msg;
    return status;
}

// gail_config of the 1.0 header, the smallest a program can pass: keep this at the 1.0 size
// when fields are added to gail_config.
constexpr size_t kConfigSizeV1 = sizeof(gail_config);

template <class T, size_t N>
bool allZero(const T (&words)[N]) {
    return std::all_of(words, words + N, [](T w) { return w == 0; });
}

bool allZero(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    return std::all_of(b, b + n, [](uint8_t v) { return v == 0; });
}

// Error carrying the status the API call should return.
struct ApiError : std::runtime_error {
    int status;
    ApiError(int s, const std::string& m) : std::runtime_error(m), status(s) {}
};

// A gail_encoder_encode() call waiting for its frame.
struct Waiter {
    std::mutex m;
    std::condition_variable cv;
    bool done = false;
    bool failed = false;
    gail_packet pkt{};
};

struct FrameMeta {
    void* opaque = nullptr;
    Waiter* waiter = nullptr;
};

// Frames handed over by the app; read() blocks until one arrives. Every queued frame gets
// the next sequence number, which is the index the pipeline gives it.
class PushSource : public FrameSource {
public:
    PushSource(const FrameFormat& f, size_t capacity) : fmt_(f), cap_(capacity) {}
    const FrameFormat& format() const override { return fmt_; }

    bool read(void* dst) override {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return !q_.empty() || closed_ || aborted_; });
        if (aborted_ || q_.empty()) return false;
        std::memcpy(dst, q_.front().data(), q_.front().size());
        free_.push_back(std::move(q_.front()));
        q_.pop_front();
        cv_.notify_all();
        return true;
    }

    // Fills a buffer with `fill` (outside the lock) and queues it with `meta`.
    template <class F>
    int push(F&& fill, const FrameMeta& meta, std::map<long long, FrameMeta>& metas, std::mutex& metaMutex) {
        std::vector<uint8_t> buf;
        {
            std::unique_lock<std::mutex> l(m_);
            cv_.wait(l, [&] { return q_.size() < cap_ || aborted_ || closed_; });
            if (aborted_) return GAIL_ERROR_FAILED;
            if (closed_) return GAIL_ERROR_STATE;
            if (!free_.empty()) {
                buf = std::move(free_.back());
                free_.pop_back();
            }
        }
        if (buf.size() != fmt_.bytes()) buf.assign(fmt_.bytes(), 0);  // borders stay black on reuse
        fill(buf.data());
        std::lock_guard<std::mutex> l(m_);
        if (aborted_) return GAIL_ERROR_FAILED;
        if (closed_) return GAIL_ERROR_STATE;
        {
            std::lock_guard<std::mutex> lm(metaMutex);
            metas[next_++] = meta;
        }
        q_.push_back(std::move(buf));
        cv_.notify_all();
        return GAIL_OK;
    }

    void close() {
        std::lock_guard<std::mutex> l(m_);
        closed_ = true;
        cv_.notify_all();
    }
    void abort() {
        std::lock_guard<std::mutex> l(m_);
        aborted_ = true;
        cv_.notify_all();
    }

private:
    FrameFormat fmt_;
    size_t cap_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::vector<uint8_t>> q_;
    std::vector<std::vector<uint8_t>> free_;
    long long next_ = 0;
    bool closed_ = false, aborted_ = false;
};

Transfer toTransfer(gail_transfer t) {
    switch (t) {
    case GAIL_TRANSFER_BT1886: return Transfer::BT1886;
    case GAIL_TRANSFER_SRGB: return Transfer::SRGB;
    case GAIL_TRANSFER_PQ: return Transfer::PQ;
    case GAIL_TRANSFER_HLG: return Transfer::HLG;
    case GAIL_TRANSFER_LINEAR: return Transfer::Linear;
    case GAIL_TRANSFER_GAMMA26: return Transfer::Gamma26;
    }
    throw ApiError(GAIL_ERROR_INVALID, "invalid transfer");
}

Primaries toPrimaries(gail_primaries p) {
    switch (p) {
    case GAIL_PRIMARIES_BT709: return Primaries::BT709;
    case GAIL_PRIMARIES_BT2020: return Primaries::BT2020;
    case GAIL_PRIMARIES_P3D65: return Primaries::P3D65;
    case GAIL_PRIMARIES_DCIP3: return Primaries::DCIP3;
    }
    throw ApiError(GAIL_ERROR_INVALID, "invalid primaries");
}

ColorParams toColorParams(const gail_color& c) {
    if (!allZero(c.reserved)) throw ApiError(GAIL_ERROR_INVALID, "gail_color.reserved must be zero (start from gail_color_preset())");
    ColorParams p;
    p.sdrWhite = c.sdr_white;
    p.hdrPeak = c.hdr_peak;
    p.hdrRefWhite = c.hdr_ref_white;
    p.cinemaRefWhite = c.cinema_ref_white;
    p.cinemaPeak = c.cinema_peak;
    p.knee = c.knee;
    p.gamutThreshold = c.gamut_threshold;
    if (!(p.sdrWhite > 0 && p.hdrPeak > 0 && p.hdrRefWhite > 0 && p.cinemaRefWhite > 0 && p.cinemaPeak > 0 &&
          p.cinemaPeak <= 52.37 && p.knee > 0 && p.knee < 1 && p.gamutThreshold >= 0 && p.gamutThreshold < 1))
        throw ApiError(GAIL_ERROR_INVALID, "colour parameter out of range");
    switch (c.tone_map) {
    case GAIL_TONE_MAP_SIGNAL: p.toneMap = ToneMap::Signal; break;
    case GAIL_TONE_MAP_SHOULDER: p.toneMap = ToneMap::Shoulder; break;
    case GAIL_TONE_MAP_BT2390: p.toneMap = ToneMap::BT2390; break;
    case GAIL_TONE_MAP_NONE: p.toneMap = ToneMap::None; break;
    default: throw ApiError(GAIL_ERROR_INVALID, "invalid tone map");
    }
    switch (c.tone_norm) {
    case GAIL_TONE_NORM_CHANNEL: p.toneNorm = ToneNorm::Channel; break;
    case GAIL_TONE_NORM_MAXRGB: p.toneNorm = ToneNorm::MaxRGB; break;
    case GAIL_TONE_NORM_LUMINANCE: p.toneNorm = ToneNorm::Luminance; break;
    default: throw ApiError(GAIL_ERROR_INVALID, "invalid tone norm");
    }
    switch (c.gamut) {
    case GAIL_GAMUT_NONE: p.gamut = TargetGamut::None; break;
    case GAIL_GAMUT_P3D65: p.gamut = TargetGamut::P3D65; break;
    case GAIL_GAMUT_DCIP3: p.gamut = TargetGamut::DCIP3; break;
    default: throw ApiError(GAIL_ERROR_INVALID, "invalid gamut");
    }
    p.transfer = toTransfer(c.transfer);
    p.primaries = toPrimaries(c.primaries);
    if (c.mode == GAIL_COLOR_LUT3D) {
        if (!c.lut3d || c.lut3d_size < 2 || c.lut3d_size > 256) throw ApiError(GAIL_ERROR_INVALID, "invalid 3D LUT");
        const size_t n = (size_t)c.lut3d_size * c.lut3d_size * c.lut3d_size * 3;
        p.lut3dData.assign(c.lut3d, c.lut3d + n);
        p.lut3dSize = c.lut3d_size;
    } else if (c.mode != GAIL_COLOR_MANAGED) {
        throw ApiError(GAIL_ERROR_INVALID, "invalid colour mode");
    }
    return p;
}

// An enum field as the integer the caller stored: a value outside the enum's range (a
// program with a wrong or stale header) cannot be loaded as the enum type itself.
template <class E> int enumValue(const E& e) {
    static_assert(sizeof(E) == sizeof(int));
    int v;
    std::memcpy(&v, &e, sizeof v);
    return v;
}

bool isXyz(gail_pixel_format f) { return f == GAIL_PIX_XYZ12 || f == GAIL_PIX_XYZ12_MSB; }
// Planar formats: bit depth, chroma subsampling (log2 across / down) and G'B'R' planes.
struct PlanarInfo {
    int bits = 0, sx = 0, sy = 0;
    bool rgb = false;
};

bool planarInfo(gail_pixel_format f, PlanarInfo& p) {
    switch (f) {
    case GAIL_PIX_YUV420P: p = {8, 1, 1, false}; return true;
    case GAIL_PIX_YUV420P10: p = {10, 1, 1, false}; return true;
    case GAIL_PIX_YUV420P12: p = {12, 1, 1, false}; return true;
    case GAIL_PIX_YUV422P: p = {8, 1, 0, false}; return true;
    case GAIL_PIX_YUV422P10: p = {10, 1, 0, false}; return true;
    case GAIL_PIX_YUV422P12: p = {12, 1, 0, false}; return true;
    case GAIL_PIX_YUV444P: p = {8, 0, 0, false}; return true;
    case GAIL_PIX_YUV444P10: p = {10, 0, 0, false}; return true;
    case GAIL_PIX_YUV444P12: p = {12, 0, 0, false}; return true;
    case GAIL_PIX_GBRP: p = {8, 0, 0, true}; return true;
    case GAIL_PIX_GBRP10: p = {10, 0, 0, true}; return true;
    case GAIL_PIX_GBRP12: p = {12, 0, 0, true}; return true;
    case GAIL_PIX_GBRP16: p = {16, 0, 0, true}; return true;
    default: return false;
    }
}

bool isPlanar(gail_pixel_format f) {
    PlanarInfo p;
    return planarInfo(f, p);
}

const char* backendsBuilt() {
    return ""
#ifdef GAIL_HAVE_CUDA
           "cuda "
#endif
#ifdef GAIL_HAVE_VULKAN
           "vulkan "
#endif
           "cpu";
}

}  // namespace

struct gail_encoder {
    gail_config cfg{};
    int width = 0, height = 0;        // container
    int picW = 0, picH = 0, ox = 0, oy = 0;
    std::string devicesDesc;
    std::unique_ptr<PushSource> src;
    std::vector<std::unique_ptr<Device>> devices;
    std::thread thread;
    std::atomic<bool> ended{false}, joined{false}, cancelled{false};
    std::mutex lifecycle;

    std::mutex metaMutex;
    std::map<long long, FrameMeta> metas;

    // receive() queue
    std::mutex outMutex;
    std::condition_variable outCv;
    std::deque<gail_packet> out;
    bool pipelineDone = false;

    std::mutex errMutex;
    std::atomic<bool> failed{false};
    std::string error;

    void setError(const std::string& msg) {
        {
            std::lock_guard<std::mutex> l(errMutex);
            if (failed.load()) return;
            error = msg;
            failed.store(true);
        }
        log(GAIL_LOG_ERROR, "%s", msg.c_str());
    }

    void log(gail_log_level level, const char* fmt, ...) {
        if (!cfg.log || level > cfg.log_level) return;
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        cfg.log(cfg.log_user, level, buf);
    }

    // Wakes every gail_encoder_encode() still waiting (after a failure).
    void failWaiters() {
        std::lock_guard<std::mutex> l(metaMutex);
        for (auto& [i, m] : metas)
            if (m.waiter) {
                std::lock_guard<std::mutex> lw(m.waiter->m);
                m.waiter->failed = true;
                m.waiter->done = true;
                m.waiter->cv.notify_all();
            }
        metas.clear();
    }

    void deliver(const FrameStats& st, const std::vector<uint8_t>& cs) {
        if (cancelled.load()) return;
        FrameMeta meta;
        {
            std::lock_guard<std::mutex> l(metaMutex);
            auto it = metas.find(st.index);
            if (it != metas.end()) {
                meta = it->second;
                metas.erase(it);
            }
        }
        if (!st.compliant) log(GAIL_LOG_WARNING, "frame %lld failed the DCI check: %s", st.index, st.checkSummary.c_str());
        gail_packet p{};
        p.size = cs.size();
        p.index = st.index;
        p.opaque = meta.opaque;
        p.device = st.device;
        p.compliant = st.compliant ? 1 : 0;
        p.psnr_estimate = st.mseEstimate > 0 ? 10 * std::log10(4095.0 * 4095.0 / st.mseEstimate) : 99.0;
        if (!meta.waiter && cfg.on_packet) {
            p.data = const_cast<uint8_t*>(cs.data());
            cfg.on_packet(cfg.user, &p);
            return;
        }
        p.data = (uint8_t*)std::malloc(cs.size());
        if (!p.data) throw std::bad_alloc();
        std::memcpy(p.data, cs.data(), cs.size());
        if (meta.waiter) {
            std::lock_guard<std::mutex> l(meta.waiter->m);
            meta.waiter->pkt = p;
            meta.waiter->done = true;
            meta.waiter->cv.notify_all();
            return;
        }
        std::lock_guard<std::mutex> l(outMutex);
        out.push_back(p);
        outCv.notify_all();
    }

    // Copies one app frame into the pipeline's input layout.
    void fill(const gail_frame& f, uint8_t* dst) const {
        const gail_pixel_format fmt = cfg.input.format;
        PlanarInfo pi;
        if (planarInfo(fmt, pi)) {
            const size_t bps = pi.bits > 8 ? 2 : 1;
            const int cw = (picW + pi.sx) >> pi.sx, ch = (picH + pi.sy) >> pi.sy;
            uint8_t* plane = dst;
            for (int c = 0; c < 3; ++c) {
                const int w = c ? cw : picW, h = c ? ch : picH;
                const size_t row = (size_t)w * bps;
                const ptrdiff_t stride = f.stride[c] ? f.stride[c] : (ptrdiff_t)row;
                const uint8_t* s = static_cast<const uint8_t*>(f.data[c]);
                for (int y = 0; y < h; ++y) std::memcpy(plane + y * row, s + y * stride, row);
                plane += row * h;
            }
            return;
        }
        const size_t row = (size_t)picW * 6;
        const ptrdiff_t stride = f.stride[0] ? f.stride[0] : (ptrdiff_t)row;
        for (int y = 0; y < picH; ++y) {
            const uint16_t* s = reinterpret_cast<const uint16_t*>(static_cast<const uint8_t*>(f.data[0]) + y * stride);
            uint16_t* o = reinterpret_cast<uint16_t*>(dst) + ((size_t)(y + oy) * width + ox) * 3;
            if (fmt == GAIL_PIX_RGB48) {
                std::memcpy(o, s, row);
                continue;
            }
            // 12-bit code -> 16-bit signal; the X'Y'Z' colour path maps it back exactly
            const bool msb = fmt == GAIL_PIX_XYZ12_MSB;
            for (int i = 0; i < picW * 3; ++i) {
                const uint32_t v = std::min<uint32_t>(msb ? s[i] >> 4 : s[i], 4095u);
                o[i] = (uint16_t)((v * 65535u + 2047u) / 4095u);
            }
        }
    }

    int send(const gail_frame* f, Waiter* w) {
        if (cancelled.load()) return fail(GAIL_ERROR_CANCELLED, "cancelled");
        if (failed.load()) return fail(GAIL_ERROR_FAILED, error);
        if (!f) {
            ended = true;
            src->close();
            return GAIL_OK;
        }
        if (ended) return fail(GAIL_ERROR_STATE, "the stream has already ended");
        if (!allZero(f->reserved)) return fail(GAIL_ERROR_INVALID, "gail_frame.reserved must be zero");
        const int planes = isPlanar(cfg.input.format) ? 3 : 1;
        for (int c = 0; c < planes; ++c)
            if (!f->data[c]) return fail(GAIL_ERROR_INVALID, "frame plane missing");
        FrameMeta meta{f->opaque, w};
        const int r = src->push([&](uint8_t* dst) { fill(*f, dst); }, meta, metas, metaMutex);
        if (r == GAIL_ERROR_FAILED && cancelled.load()) return fail(GAIL_ERROR_CANCELLED, "cancelled");
        if (r == GAIL_ERROR_FAILED) return fail(r, failed.load() ? error : "encoder stopped");
        if (r == GAIL_ERROR_STATE) return fail(r, "the stream has already ended");
        return r;
    }

    void cancel() {
        if (cancelled.exchange(true)) return;
        bool done;
        {
            std::lock_guard<std::mutex> l(outMutex);
            done = pipelineDone;
        }
        if (!done) log(GAIL_LOG_INFO, "cancelled");
        ended = true;
        if (src) src->abort();  // wakes the pipeline's readers and blocked sends
        failWaiters();
        std::lock_guard<std::mutex> l(outMutex);
        outCv.notify_all();
    }

    void join() {
        std::lock_guard<std::mutex> l(lifecycle);
        if (joined) return;
        ended = true;
        if (src) src->close();
        if (thread.joinable()) thread.join();
        joined = true;
    }
};

namespace {

std::vector<DeviceInfo> enumerate(gail_backend backend, gail_backend& chosen) {
    static const char* names[] = {"auto", "cuda", "vulkan", "cpu"};
    if ((unsigned)backend > 3) throw ApiError(GAIL_ERROR_INVALID, "invalid backend");
    std::string c;
    auto devices = listDevices(names[backend], c);
    chosen = c == "cuda" ? GAIL_BACKEND_CUDA : c == "vulkan" ? GAIL_BACKEND_VULKAN : GAIL_BACKEND_CPU;
    return devices;
}

}  // namespace

extern "C" {

const char* gail_last_error(void) { return g_lastError.c_str(); }

const char* gail_status_string(int s) {
    switch (s) {
    case GAIL_OK: return "ok";
    case GAIL_EAGAIN: return "no packet ready yet";
    case GAIL_EOF: return "end of stream";
    case GAIL_ERROR_INVALID: return "invalid argument";
    case GAIL_ERROR_UNSUPPORTED: return "not supported";
    case GAIL_ERROR_NOMEM: return "out of memory";
    case GAIL_ERROR_FAILED: return "encoding failed";
    case GAIL_ERROR_STATE: return "not allowed in this state";
    case GAIL_ERROR_CANCELLED: return "cancelled";
    }
    return "unknown status";
}

unsigned gail_version(void) { return GAIL_VERSION; }

const char* gail_version_string(void) {
    static const std::string s = std::string(gail::buildVersion()) + " (" + backendsBuilt() + ")";
    return s.c_str();
}

int gail_container_size(gail_profile profile, gail_container container, int* width, int* height) {
    if ((unsigned)profile > 1 || (unsigned)container > 2) return fail(GAIL_ERROR_INVALID, "invalid profile or container");
    int w, h;
    containerSize(profile == GAIL_PROFILE_4K ? Profile::Cinema4K : Profile::Cinema2K,
                  container == GAIL_CONTAINER_SCOPE  ? Container::Scope
                  : container == GAIL_CONTAINER_FULL ? Container::Full
                                                     : Container::Flat,
                  w, h);
    if (width) *width = w;
    if (height) *height = h;
    return GAIL_OK;
}

void gail_color_preset(gail_preset preset, gail_color* c) {
    *c = gail_color{};
    c->mode = GAIL_COLOR_MANAGED;
    c->transfer = GAIL_TRANSFER_BT1886;
    c->primaries = GAIL_PRIMARIES_BT709;
    c->tone_map = GAIL_TONE_MAP_SIGNAL;
    c->tone_norm = GAIL_TONE_NORM_CHANNEL;
    c->sdr_white = 48;
    c->hdr_peak = 1000;
    c->hdr_ref_white = 203;
    c->cinema_ref_white = 16;
    c->cinema_peak = 48;
    c->knee = 0.25;
    c->gamut = GAIL_GAMUT_NONE;
    c->gamut_threshold = 0.8;
    switch (preset) {
    case GAIL_PRESET_SDR_BT709: break;
    case GAIL_PRESET_SDR_BT2020: c->primaries = GAIL_PRIMARIES_BT2020; break;
    case GAIL_PRESET_HDR_PQ_BT2020:
        c->transfer = GAIL_TRANSFER_PQ;
        c->primaries = GAIL_PRIMARIES_BT2020;
        break;
    case GAIL_PRESET_HDR_HLG_BT2020:
        c->transfer = GAIL_TRANSFER_HLG;
        c->primaries = GAIL_PRIMARIES_BT2020;
        break;
    }
}

int gail_color_describe(const gail_color* c, char* buf, size_t size) {
    try {
        const std::string d = ColorConverter(toColorParams(*c)).describe();
        if (buf && size) std::snprintf(buf, size, "%s", d.c_str());
        return (int)d.size();
    } catch (const ApiError& e) {
        return fail(e.status, e.what());
    } catch (const std::exception& e) {
        return fail(GAIL_ERROR_INVALID, e.what());
    }
}

void gail_config_init(gail_config* out, size_t struct_size) {
    if (!out || struct_size < sizeof(size_t)) return;
    gail_config d{};
    gail_config* c = &d;
    c->struct_size = struct_size;
    c->profile = GAIL_PROFILE_2K;
    c->container = GAIL_CONTAINER_FLAT;
    c->fps = 24;
    c->max_mbps = 230;
    c->input.format = GAIL_PIX_RGB48;
    c->input.matrix = GAIL_MATRIX_BT709;
    c->input.chroma_location = GAIL_CHROMA_LEFT;
    gail_color_preset(GAIL_PRESET_SDR_BT709, &c->color);
    c->backend = GAIL_BACKEND_AUTO;
    c->check = 1;
    c->log_level = GAIL_LOG_INFO;
    // A program built against a newer header may have more fields: zero them
    std::memset(out, 0, struct_size);
    std::memcpy(out, &d, std::min(struct_size, sizeof d));
}

void gail_packet_free(gail_packet* pkt) {
    if (!pkt) return;
    std::free(pkt->data);
    pkt->data = nullptr;
    pkt->size = 0;
}

int gail_encoder_create(const gail_config* c, gail_encoder** encoder) {
    if (!encoder) return fail(GAIL_ERROR_INVALID, "encoder pointer is NULL");
    *encoder = nullptr;
    if (!c || c->struct_size < kConfigSizeV1)
        return fail(GAIL_ERROR_INVALID, "config not initialised with gail_config_defaults()");
    if (c->struct_size > sizeof(gail_config) &&
        !allZero(reinterpret_cast<const uint8_t*>(c) + sizeof(gail_config), c->struct_size - sizeof(gail_config)))
        return fail(GAIL_ERROR_UNSUPPORTED, std::string("the config uses settings this libgail (") + gail_version_string() +
                                                ") does not have: update libgail");
    std::unique_ptr<gail_encoder> e;
    try {
        e = std::make_unique<gail_encoder>();
        // A config from an older header is shorter: the fields it lacks keep their defaults.
        gail_config_init(&e->cfg, sizeof(gail_config));
        std::memcpy(&e->cfg, c, std::min(c->struct_size, sizeof(gail_config)));
        e->cfg.struct_size = sizeof(gail_config);
        gail_config& cfg = e->cfg;
        if (!(cfg.fps > 0)) throw ApiError(GAIL_ERROR_INVALID, "fps must be positive");
        if (!(cfg.max_mbps > 0) || cfg.max_mbps > 250)
            throw ApiError(GAIL_ERROR_INVALID, "max_mbps must be in (0, 250] (the DCI bitrate limit)");
        if (gail_container_size(cfg.profile, cfg.container, &e->width, &e->height) != GAIL_OK)
            throw ApiError(GAIL_ERROR_INVALID, g_lastError);
        const gail_input& in = cfg.input;
        if (enumValue(in.format) < 0 || enumValue(in.format) > GAIL_PIX_GBRP16 ||
            (!isPlanar(in.format) && in.format != GAIL_PIX_RGB48 && !isXyz(in.format)))
            throw ApiError(GAIL_ERROR_INVALID, "invalid pixel format");
        if (!allZero(in.reserved)) throw ApiError(GAIL_ERROR_INVALID, "gail_input.reserved must be zero");
        e->picW = in.width ? in.width : e->width;
        e->picH = in.height ? in.height : e->height;
        if (e->picW < 1 || e->picH < 1 || e->picW > e->width || e->picH > e->height)
            throw ApiError(GAIL_ERROR_INVALID, "picture " + std::to_string(e->picW) + "x" + std::to_string(e->picH) +
                                                   " does not fit the " + std::to_string(e->width) + "x" +
                                                   std::to_string(e->height) + " container (scale it first)");
        e->ox = (e->width - e->picW) / 2;
        e->oy = (e->height - e->picH) / 2;

        const Profile profile = cfg.profile == GAIL_PROFILE_4K ? Profile::Cinema4K : Profile::Cinema2K;
        EncodeSettings es;
        es.profile = profile;
        es.width = e->width;
        es.height = e->height;
        es.slopeFactor = 0.25;  // Tier-1 early termination, as gail-enc
        if (isXyz(in.format)) {
            es.color.transfer = Transfer::Gamma26;
            es.color.primaries = Primaries::XYZ;
        } else {
            es.color = toColorParams(cfg.color);
        }
        FrameFormat& ff = es.input;
        ff.width = e->width;
        ff.height = e->height;
        PlanarInfo pi;
        if (planarInfo(in.format, pi)) {
            ff.kind = FrameFormat::Kind::Planar;
            ff.activeW = e->picW;
            ff.activeH = e->picH;
            ff.ox = e->ox;
            ff.oy = e->oy;
            ff.bits = pi.bits;
            ff.sx = pi.sx;
            ff.sy = pi.sy;
            ff.rgb = pi.rgb;
            ff.fullRange = in.full_range != 0;
            switch (in.matrix) {
            case GAIL_MATRIX_BT709: ff.kr = 0.2126, ff.kb = 0.0722; break;
            case GAIL_MATRIX_BT2020: ff.kr = 0.2627, ff.kb = 0.0593; break;
            case GAIL_MATRIX_BT601: ff.kr = 0.299, ff.kb = 0.114; break;
            default: throw ApiError(GAIL_ERROR_INVALID, "invalid matrix");
            }
            switch (in.chroma_location) {
            case GAIL_CHROMA_LEFT: ff.chroma = FrameFormat::Chroma::Left; break;
            case GAIL_CHROMA_CENTER: ff.chroma = FrameFormat::Chroma::Center; break;
            case GAIL_CHROMA_TOPLEFT: ff.chroma = FrameFormat::Chroma::TopLeft; break;
            default: throw ApiError(GAIL_ERROR_INVALID, "invalid chroma location");
            }
        } else {
            ff.kind = FrameFormat::Kind::RGB48;
        }
        ColorConverter probe(es.color);  // validates the colour settings up front
        const FrameLayout L = makeLayout(e->width, e->height, profile, es.baseStep);

        static const char* names[] = {"auto", "cuda", "vulkan", "cpu"};
        if ((unsigned)cfg.backend > 3) throw ApiError(GAIL_ERROR_INVALID, "invalid backend");
        std::vector<int> select(cfg.devices, cfg.devices + (cfg.devices ? std::max(cfg.num_devices, 0) : 0));
        std::vector<std::string> ids;
        for (int i = 0; cfg.device_ids && i < cfg.num_device_ids; ++i)
            if (cfg.device_ids[i]) ids.emplace_back(cfg.device_ids[i]);
        cfg.devices = nullptr;  // not kept: only read here
        cfg.device_ids = nullptr;
        cfg.color.lut3d = nullptr;
        std::string chosen;
        try {
            gail_encoder* enc = e.get();
            Logger deviceLog = [enc](LogLevel level, const std::string& m) {
                enc->log(level == LogLevel::Info ? GAIL_LOG_INFO : GAIL_LOG_WARNING, "%s", m.c_str());
            };
            e->devices = createDevices(names[cfg.backend], select, ids, cfg.cpu_threads, cfg.low_priority != 0,
                                       deviceLog, chosen);
        } catch (const std::exception& ex) {
            throw ApiError(GAIL_ERROR_UNSUPPORTED, ex.what());
        }
        for (size_t d = 0; d < e->devices.size(); ++d) e->devicesDesc += (d ? ", " : "") + e->devices[d]->name();
        e->log(GAIL_LOG_INFO, "gail %s: %s %dx%d, %s", gail_version_string(), profileName(profile), e->width,
               e->height, e->devicesDesc.c_str());
        e->log(GAIL_LOG_INFO, "colour: %s", probe.describe().c_str());
        static const double dciRates[] = {24, 25, 30, 48, 50, 60};  // SMPTE ST 429-2
        if (std::none_of(std::begin(dciRates), std::end(dciRates), [&](double r) { return std::fabs(cfg.fps - r) < 1e-3; }))
            e->log(GAIL_LOG_WARNING,
                   "%.3f fps is not a DCI frame rate (24, 25, 30, 48, 50, 60); the bitrate is shared over %.3f "
                   "frames per second, so frames are smaller than at 24 fps",
                   cfg.fps, cfg.fps);

        const double fps = cfg.fps * (cfg.stereo ? 2 : 1);  // each eye gets half the frame budget
        PipelineOptions po;
        po.fps = fps;
        po.limits = dciLimits(fps, cfg.max_mbps);
        po.slotsPerDevice = cfg.frames_in_flight > 0 ? cfg.frames_in_flight : (chosen == "cpu" ? kCpuSlots : 3);
        po.deviceMemoryLimit = cfg.max_device_memory;
        po.cpuFallback = cfg.backend == GAIL_BACKEND_AUTO;
        po.cpuThreads = cfg.cpu_threads;
        po.lowPriority = cfg.low_priority != 0;
        po.check = cfg.check != 0;
        po.cancel = &e->cancelled;
        e->src = std::make_unique<PushSource>(es.input, (size_t)po.slotsPerDevice * e->devices.size() + 1);
        gail_encoder* raw = e.get();
        po.onError = [raw] { raw->src->abort(); };
        po.onLog = [raw](const std::string& m) { raw->log(GAIL_LOG_WARNING, "%s", m.c_str()); };
        po.onFrame = [raw](const FrameStats& st, const std::vector<uint8_t>& cs) { raw->deliver(st, cs); };
        e->thread = std::thread([raw, es, L, po] {
            try {
                runPipeline(*raw->src, raw->devices, L, es, po);
            } catch (const std::exception& ex) {
                raw->setError(ex.what());
            } catch (...) {
                raw->setError("unknown error");
            }
            raw->src->abort();  // a send waiting for room must not block forever
            raw->failWaiters();
            std::lock_guard<std::mutex> l(raw->outMutex);
            raw->pipelineDone = true;
            raw->outCv.notify_all();
        });
    } catch (const ApiError& ex) {
        return fail(ex.status, ex.what());
    } catch (const std::bad_alloc&) {
        return fail(GAIL_ERROR_NOMEM, "out of memory");
    } catch (const std::exception& ex) {
        return fail(GAIL_ERROR_FAILED, ex.what());
    }
    *encoder = e.release();
    return GAIL_OK;
}

void gail_encoder_frame_size(const gail_encoder* e, int* width, int* height) {
    if (width) *width = e ? e->width : 0;
    if (height) *height = e ? e->height : 0;
}

const char* gail_encoder_devices(const gail_encoder* e) { return e ? e->devicesDesc.c_str() : ""; }

int gail_encoder_send(gail_encoder* e, const gail_frame* frame) {
    if (!e) return fail(GAIL_ERROR_INVALID, "encoder is NULL");
    return e->send(frame, nullptr);
}

int gail_encoder_receive(gail_encoder* e, gail_packet* pkt, int wait) {
    if (!e || !pkt) return fail(GAIL_ERROR_INVALID, "NULL argument");
    if (e->cfg.on_packet) return fail(GAIL_ERROR_STATE, "packets go to on_packet");
    std::unique_lock<std::mutex> l(e->outMutex);
    if (wait) e->outCv.wait(l, [&] { return !e->out.empty() || e->pipelineDone || e->cancelled.load(); });
    if (e->cancelled.load()) return fail(GAIL_ERROR_CANCELLED, "cancelled");
    if (!e->out.empty()) {
        *pkt = e->out.front();
        e->out.pop_front();
        return GAIL_OK;
    }
    if (e->failed.load()) return fail(GAIL_ERROR_FAILED, e->error);
    if (e->pipelineDone) return GAIL_EOF;
    return GAIL_EAGAIN;
}

int gail_encoder_encode(gail_encoder* e, const gail_frame* frame, gail_packet* pkt) {
    if (!e || !frame || !pkt) return fail(GAIL_ERROR_INVALID, "NULL argument");
    Waiter w;
    const int r = e->send(frame, &w);
    if (r != GAIL_OK) return r;
    std::unique_lock<std::mutex> l(w.m);
    w.cv.wait(l, [&] { return w.done; });
    if (w.failed && e->cancelled.load()) {
        gail_packet_free(&w.pkt);
        return fail(GAIL_ERROR_CANCELLED, "cancelled");
    }
    if (w.failed) return fail(GAIL_ERROR_FAILED, e->failed.load() ? e->error : "encoder stopped");
    *pkt = w.pkt;
    return GAIL_OK;
}

int gail_encoder_finish(gail_encoder* e) {
    if (!e) return fail(GAIL_ERROR_INVALID, "encoder is NULL");
    e->join();
    if (e->cancelled.load()) return fail(GAIL_ERROR_CANCELLED, "cancelled");
    return e->failed.load() ? fail(GAIL_ERROR_FAILED, e->error) : GAIL_OK;
}

void gail_encoder_cancel(gail_encoder* e) {
    if (e) e->cancel();
}

const char* gail_encoder_error(const gail_encoder* e) { return e && e->failed.load() ? e->error.c_str() : nullptr; }

void gail_encoder_destroy(gail_encoder* e) {
    if (!e) return;
    e->join();
    for (auto& p : e->out) gail_packet_free(&p);
    delete e;
}

int gail_enumerate_devices(gail_backend backend, gail_device_info* out, int max_devices) {
    try {
        gail_backend chosen;
        const auto devices = enumerate(backend, chosen);
        for (int i = 0; out && i < max_devices && i < (int)devices.size(); ++i) {
            gail_device_info& d = out[i];
            d = gail_device_info{};
            d.backend = chosen;
            d.index = devices[i].index;
            std::snprintf(d.name, sizeof d.name, "%s", devices[i].name.c_str());
            std::snprintf(d.uuid, sizeof d.uuid, "%s", devices[i].uuid.c_str());
            std::snprintf(d.pci_bus_id, sizeof d.pci_bus_id, "%s", devices[i].pciBusId.c_str());
            d.memory_bytes = devices[i].memoryBytes;
        }
        return (int)devices.size();
    } catch (const ApiError& ex) {
        return fail(ex.status, ex.what());
    } catch (const std::exception& ex) {
        return fail(GAIL_ERROR_UNSUPPORTED, ex.what());
    }
}

int gail_list_devices(gail_backend backend, char* buf, size_t size) {
    static const char* names[] = {"auto", "cuda", "vulkan", "cpu"};
    try {
        gail_backend chosen;
        const auto devices = enumerate(backend, chosen);
        std::string s;
        for (const auto& d : devices) {
            char mem[32] = "";
            if (d.memoryBytes) std::snprintf(mem, sizeof mem, ", %.0f GB", d.memoryBytes / 1073741824.0);
            s += std::string(names[chosen]) + " " + std::to_string(d.index) + ": " + d.name +
                 (d.uuid.empty() ? "" : " " + d.uuid) + (d.pciBusId.empty() ? "" : " (" + d.pciBusId + mem + ")") + "\n";
        }
        if (buf && size) std::snprintf(buf, size, "%s", s.c_str());
        return (int)s.size();
    } catch (const ApiError& ex) {
        return fail(ex.status, ex.what());
    } catch (const std::exception& ex) {
        return fail(GAIL_ERROR_UNSUPPORTED, ex.what());
    }
}

}  // extern "C"
