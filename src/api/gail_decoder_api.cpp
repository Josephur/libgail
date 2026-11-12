// SPDX-License-Identifier: MIT
// The decoder half of the library API (include/gail/gail.h). Codestreams handed over by
// the app go into a queue; one worker thread per frame in flight takes them, reads the
// headers and packets on the host and decodes on its slot's device. Finished images go to
// whoever is waiting: a gail_decoder_decode() call, or in send order to the on_image
// callback or the queue gail_decoder_receive() reads.
#define GAIL_BUILDING
#include "gail/gail.h"

#include <atomic>
#include <condition_variable>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "api/api_internal.h"
#include "api/devices.h"
#include "core/backend.h"
#include "core/codestream_parse.h"
#include "core/decode_backend.h"
#include "core/t2_parse.h"

namespace {

using namespace gail;
using namespace gail::api;

// gail_decoder_config as first released, the smallest a program can pass: keep this at
// that size when fields are added.
constexpr size_t kDecoderConfigSizeV1 = sizeof(gail_decoder_config);

// In gail_image.reserved[3] of an image whose buffer the library allocated; reserved[2]
// then points at the pool the buffer goes back to.
constexpr uint64_t kOwnedImage = 0x6761696C6F776E64ull;

// The decoder's picture buffers. A 4K picture is 52 MB: taking that from the system for
// every frame, and handing it back, costs more than decoding the frame, so released
// buffers are kept and handed out again. Images may outlive the decoder: the pool is
// freed when the decoder and the last image are gone.
struct ImagePool {
    std::mutex m;
    std::vector<void*> spare;
    size_t bytes = 0;      // size of every buffer
    size_t keep = 0;       // how many spare ones to hold on to
    long users = 1;        // the decoder, plus one per image handed out
    bool closed = false;   // the decoder is gone: nothing is kept any more

    void* take() {
        std::unique_lock<std::mutex> l(m);
        ++users;
        if (!spare.empty()) {
            void* p = spare.back();
            spare.pop_back();
            return p;
        }
        l.unlock();
        void* p = std::malloc(bytes);
        if (!p) release();
        return p;
    }
    void give(void* p) {
        {
            std::lock_guard<std::mutex> l(m);
            if (!closed && spare.size() < keep) {
                spare.push_back(p);
                p = nullptr;
            }
        }
        std::free(p);
        release();
    }
    void close() {
        std::vector<void*> drop;
        {
            std::lock_guard<std::mutex> l(m);
            closed = true;
            drop.swap(spare);
        }
        for (void* p : drop) std::free(p);
        release();
    }

private:
    void release() {
        bool last;
        {
            std::lock_guard<std::mutex> l(m);
            last = --users == 0;
        }
        if (last) delete this;
    }
};

struct Job {
    std::vector<uint8_t> cs;
    int64_t index = 0;
    uint64_t epoch = 0;   // of the flush it was sent after
    bool ordered = true;  // sent with gail_decoder_send(), as opposed to a blocking decode
    void* opaque = nullptr;
    uint8_t* dst = nullptr;
    ptrdiff_t stride = 0;
    bool ownedDst = false;  // dst is the library's, from `pool`
    ImagePool* pool = nullptr;
    // result
    bool done = false;
    int status = GAIL_OK;
    std::string error;
    int device = 0;

    ~Job() {
        if (ownedDst) pool->give(dst);
    }
    void failWith(int s, const std::string& why) {
        status = s;
        error = why;
        if (ownedDst) {
            pool->give(dst);
            dst = nullptr;
            ownedDst = false;
        }
    }
};

}  // namespace

struct gail_decoder {
    gail_decoder_config cfg{};
    std::vector<std::unique_ptr<Device>> devices;
    std::string devicesDesc;
    std::string backend;  // the one in use: "cuda", "vulkan" or "cpu"

    // Set up from the first codestream, under setupMutex
    std::mutex setupMutex;
    std::atomic<bool> configured{false};
    StreamHeader first;
    std::shared_ptr<const FrameLayout> layout;
    DecodeSettings ds;
    int outW = 0, outH = 0;
    ImagePool* pool = nullptr;
    struct Worker {
        std::thread thread;
        std::unique_ptr<DecoderSlot> slot;
        int device = 0;
        std::vector<CblkCode> blocks;
    };
    std::vector<std::unique_ptr<Worker>> workers;

    std::mutex m;  // everything below
    std::condition_variable workCv;  // a job was queued, or the workers are to quit
    std::condition_variable doneCv;  // a job finished, was delivered or dropped; room in the queue
    std::deque<std::shared_ptr<Job>> queue;
    std::map<int64_t, std::shared_ptr<Job>> ready;  // finished, waiting for their turn in send order
    int64_t nextIndex = 0, nextOut = 0, nextBlocking = 0;
    uint64_t epoch = 0;
    int outstanding = 0;    // ordered jobs of this epoch queued or being decoded
    int activeOrdered = 0;  // ordered jobs of any epoch being decoded
    bool delivering = false;  // an on_image call is under way
    bool ended = false, quit = false;
    std::atomic<bool> canceled{false};
    std::atomic<bool> failed{false};
    std::string error;

    void log(gail_log_level level, const char* fmt, ...) {
        if (!cfg.log || level > cfg.log_level) return;
        char buf[1024];
        va_list ap;
        va_start(ap, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, ap);
        va_end(ap);
        cfg.log(cfg.log_user, level, buf);
    }

    void setError(const std::string& what) {
        std::lock_guard<std::mutex> l(m);
        if (failed.exchange(true)) return;
        error = what;
    }

    // First codestream: learn the picture's size and structure, make the frames in flight
    // and start their threads. Returns a status with the calling thread's last error set.
    int setUp(const uint8_t* data, size_t size) {
        std::lock_guard<std::mutex> l(setupMutex);
        if (configured.load()) return GAIL_OK;
        StreamHeader h;
        std::string err;
        if (!parseCodestream(data, size, true, h, err)) return fail(GAIL_ERROR_INVALID, err);
        if (!h.supported) return fail(GAIL_ERROR_UNSUPPORTED, "unsupported codestream: " + h.unsupported);
        if (cfg.reduce > h.levels)
            return fail(GAIL_ERROR_INVALID, "reduce " + std::to_string(cfg.reduce) + " is more than the stream's " +
                                                std::to_string(h.levels) + " decomposition levels");
        try {
            layout = std::make_shared<const FrameLayout>(layoutFor(h));
        } catch (const std::exception& e) {
            return fail(GAIL_ERROR_UNSUPPORTED, std::string("unsupported codestream: ") + e.what());
        }
        first = h;
        ds.reduce = cfg.reduce;
        ds.highBits = cfg.output == GAIL_PIX_XYZ12_MSB;
        decodedSize(*layout, ds.reduce, outW, outH);

        const int perDevice = cfg.frames_in_flight > 0 ? cfg.frames_in_flight : (backend == "cpu" ? kCpuSlots : 3);
        std::string refused;
        for (size_t dv = 0; dv < devices.size(); ++dv)
            for (int i = 0; i < perDevice; ++i) {
                try {
                    auto w = std::make_unique<Worker>();
                    w->slot = devices[dv]->createDecoderSlot(*layout, ds);
                    w->device = (int)dv;
                    workers.push_back(std::move(w));
                } catch (const DeviceOutOfMemory& e) {
                    refused = e.what();
                    log(GAIL_LOG_WARNING, "%s: %d frames in flight instead of %d (%s)", devices[dv]->name().c_str(), i, perDevice,
                        e.what());
                    break;
                } catch (const std::bad_alloc&) {
                    refused = "out of memory";
                    break;
                } catch (const std::exception& e) {
                    workers.clear();
                    setError(e.what());
                    return fail(GAIL_ERROR_FAILED, e.what());
                }
            }
        if (workers.empty()) {
            const std::string why = "no device has room for a " + std::to_string(layout->width) + "x" +
                                    std::to_string(layout->height) + " frame" + (refused.empty() ? "" : ": " + refused);
            setError(why);
            return fail(GAIL_ERROR_NOMEM, why);
        }
        pool = new ImagePool;
        pool->bytes = (size_t)outW * 6 * (size_t)outH;
        pool->keep = workers.size() * 2 + 2;
        for (auto& w : workers) w->thread = std::thread([this, wp = w.get()] { run(*wp); });
        log(GAIL_LOG_INFO, "gail %s: decoding %dx%d%s on %s, %zu frames in flight", gail_version_string(), layout->width,
            layout->height, ds.reduce ? (" at 1/" + std::to_string(1 << ds.reduce) + " size").c_str() : "", devicesDesc.c_str(),
            workers.size());
        configured.store(true);
        return GAIL_OK;
    }

    void run(Worker& w) {
        if (cfg.low_priority) lowerCurrentThreadPriority();
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> l(m);
                workCv.wait(l, [&] { return quit || !queue.empty(); });
                if (queue.empty()) return;  // quit, with nothing left to do
                job = std::move(queue.front());
                queue.pop_front();
                if (job->ordered) ++activeOrdered;
            }
            doneCv.notify_all();  // room in the queue
            decodeOne(*job, w);
            finished(job);
        }
    }

    void decodeOne(Job& job, Worker& w) {
        job.device = w.device;
        try {
            StreamHeader h;
            std::string err;
            const uint8_t* data = job.cs.data();
            const size_t size = job.cs.size();
            if (!parseCodestream(data, size, false, h, err)) return job.failWith(GAIL_ERROR_INVALID, "damaged codestream: " + err);
            if (!h.supported) return job.failWith(GAIL_ERROR_UNSUPPORTED, "unsupported codestream: " + h.unsupported);
            if (h.width != first.width || h.height != first.height || h.levels != first.levels)
                return job.failWith(GAIL_ERROR_UNSUPPORTED,
                                    "codestream is " + std::to_string(h.width) + "x" + std::to_string(h.height) + " with " +
                                        std::to_string(h.levels) + " levels, the decoder was set up by its first frame for " +
                                        std::to_string(first.width) + "x" + std::to_string(first.height) + " with " +
                                        std::to_string(first.levels) + ": use another decoder");
            // Step sizes usually stay the same for a whole track; a frame with its own gets its own layout
            const FrameLayout* L = layout.get();
            std::optional<FrameLayout> own;
            if (h.guardBits != first.guardBits || std::memcmp(h.eps, first.eps, sizeof h.eps) != 0 ||
                std::memcmp(h.mu, first.mu, sizeof h.mu) != 0) {
                own.emplace(layoutFor(h));
                L = &*own;
            }
            if (!parsePackets(data, h, *L, L->levels - ds.reduce, w.blocks, err))
                return job.failWith(GAIL_ERROR_INVALID, "damaged codestream: " + err);
            w.slot->decode(*L, data, size, w.blocks, job.dst, job.stride, nullptr);
        } catch (const std::bad_alloc&) {
            job.failWith(GAIL_ERROR_NOMEM, "out of memory");
        } catch (const std::exception& e) {
            job.failWith(GAIL_ERROR_FAILED, e.what());
        }
    }

    void fill(gail_image& img, const Job& job) const {
        img = gail_image{};
        img.data[0] = job.dst;
        img.stride[0] = job.dst ? job.stride : 0;
        img.width = job.status == GAIL_OK ? outW : 0;
        img.height = job.status == GAIL_OK ? outH : 0;
        img.format = cfg.output;
        img.index = job.index;
        img.opaque = job.opaque;
        img.device = job.device;
        img.status = job.status;
        if (job.ownedDst) {
            img.reserved[2] = (uint64_t)(uintptr_t)job.pool;
            img.reserved[3] = kOwnedImage;
        }
    }

    // A worker is done with a job (or it never needed one): hand it to its waiter, or put
    // it in line for delivery in send order.
    void finished(const std::shared_ptr<Job>& job, bool wasActive = true) {
        std::unique_lock<std::mutex> l(m);
        job->done = true;
        if (job->ordered) {
            if (wasActive) --activeOrdered;
            if (job->epoch == epoch && !canceled.load()) {
                --outstanding;
                ready[job->index] = job;
                if (cfg.on_image) deliver(l);
            }
        }
        l.unlock();
        doneCv.notify_all();
    }

    // Callback mode: pass on every image whose turn has come. One thread at a time does
    // this, without the lock held during the call, so callbacks arrive in order and may
    // call back into the decoder.
    void deliver(std::unique_lock<std::mutex>& l) {
        if (delivering) return;
        delivering = true;
        for (auto it = ready.find(nextOut); it != ready.end() && !canceled.load(); it = ready.find(nextOut)) {
            const std::shared_ptr<Job> job = it->second;
            ready.erase(it);
            ++nextOut;
            gail_image img;
            fill(img, *job);
            img.reserved[2] = img.reserved[3] = 0;  // valid during the call only: nothing for the app to release
            l.unlock();
            if (job->status != GAIL_OK) g_lastError = job->error;
            cfg.on_image(cfg.user, &img);
            l.lock();
        }
        delivering = false;
    }

    // Stops the workers and waits for them.
    void shutDown() {
        {
            std::lock_guard<std::mutex> l(m);
            quit = true;
            queue.clear();
        }
        workCv.notify_all();
        doneCv.notify_all();
        for (auto& w : workers)
            if (w->thread.joinable()) w->thread.join();
    }
};

namespace {

// Checks a caller's destination (or the lack of one) and gives the job its buffer.
int prepareDestination(gail_decoder* d, Job& job, const gail_image* dest) {
    const ptrdiff_t tight = (ptrdiff_t)d->outW * 6;
    if (dest && dest->data[0]) {
        if (!allZero(dest->reserved)) return fail(GAIL_ERROR_INVALID, "gail_image.reserved must be zero");
        const ptrdiff_t stride = dest->stride[0] ? dest->stride[0] : tight;
        if (stride < tight)
            return fail(GAIL_ERROR_INVALID, "stride " + std::to_string(stride) + " is less than the " + std::to_string(tight) +
                                                " bytes of a " + std::to_string(d->outW) + "-pixel row");
        job.dst = static_cast<uint8_t*>(dest->data[0]);
        job.stride = stride;
        return GAIL_OK;
    }
    job.dst = static_cast<uint8_t*>(d->pool->take());
    if (!job.dst) return fail(GAIL_ERROR_NOMEM, "out of memory");
    job.pool = d->pool;
    job.ownedDst = true;
    job.stride = tight;
    return GAIL_OK;
}

int checkPacket(const gail_packet* pkt) {
    if (!pkt->data || !pkt->size) return fail(GAIL_ERROR_INVALID, "packet has no data");
    if (!allZero(pkt->reserved)) return fail(GAIL_ERROR_INVALID, "gail_packet.reserved must be zero");
    return GAIL_OK;
}

}  // namespace

extern "C" {

int gail_probe(const void* data, size_t size, gail_stream_info* info) {
    if (!data || !info) return fail(GAIL_ERROR_INVALID, "data or info is NULL");
    *info = gail_stream_info{};
    StreamHeader h;
    std::string err;
    if (!parseCodestream(static_cast<const uint8_t*>(data), size, true, h, err)) return fail(GAIL_ERROR_INVALID, err);
    info->width = h.width;
    info->height = h.height;
    info->components = h.components;
    info->bit_depth = h.bitDepth;
    info->rsiz = h.rsiz;
    info->levels = h.levels;
    info->reversible = h.reversible;
    info->high_throughput = h.highThroughput;
    info->tiles = h.tiles;
    info->layers = h.layers;
    info->supported = h.supported;
    if (!h.supported) g_lastError = "unsupported codestream: " + h.unsupported;
    return GAIL_OK;
}

void gail_image_free(gail_image* img) {
    if (!img) return;
    if (img->reserved[3] == kOwnedImage) {
        reinterpret_cast<ImagePool*>((uintptr_t)img->reserved[2])->give(img->data[0]);
        img->data[0] = nullptr;
        img->reserved[2] = img->reserved[3] = 0;
    }
}

void gail_decoder_config_init(gail_decoder_config* out, size_t struct_size) {
    if (!out || struct_size < sizeof(size_t)) return;
    gail_decoder_config d{};
    d.struct_size = struct_size;
    d.backend = GAIL_BACKEND_AUTO;
    d.output = GAIL_PIX_XYZ12_MSB;
    d.log_level = GAIL_LOG_INFO;
    // A program built against a newer header may have more fields: zero them
    std::memset(out, 0, struct_size);
    std::memcpy(out, &d, std::min(struct_size, sizeof d));
}

int gail_decoder_create(const gail_decoder_config* c, gail_decoder** decoder) {
    if (!decoder) return fail(GAIL_ERROR_INVALID, "decoder pointer is NULL");
    *decoder = nullptr;
    if (!c || c->struct_size < kDecoderConfigSizeV1)
        return fail(GAIL_ERROR_INVALID, "config not initialized with gail_decoder_config_defaults()");
    if (c->struct_size > sizeof(gail_decoder_config) &&
        !allZero(reinterpret_cast<const uint8_t*>(c) + sizeof(gail_decoder_config), c->struct_size - sizeof(gail_decoder_config)))
        return fail(GAIL_ERROR_UNSUPPORTED, std::string("the config uses settings this libgail (") + gail_version_string() +
                                                ") does not have: update libgail");
    try {
        auto d = std::make_unique<gail_decoder>();
        gail_decoder_config_init(&d->cfg, sizeof(gail_decoder_config));
        std::memcpy(&d->cfg, c, std::min(c->struct_size, sizeof(gail_decoder_config)));
        d->cfg.struct_size = sizeof(gail_decoder_config);
        gail_decoder_config& cfg = d->cfg;
        if ((unsigned)cfg.backend > 3) throw ApiError(GAIL_ERROR_INVALID, "invalid backend");
        if (cfg.output != GAIL_PIX_XYZ12_MSB && cfg.output != GAIL_PIX_XYZ12)
            throw ApiError(GAIL_ERROR_INVALID, "output must be GAIL_PIX_XYZ12_MSB or GAIL_PIX_XYZ12");
        if (cfg.reduce < 0 || cfg.reduce > 32) throw ApiError(GAIL_ERROR_INVALID, "invalid reduce");
        if (cfg.frames_in_flight < 0) throw ApiError(GAIL_ERROR_INVALID, "invalid frames_in_flight");

        std::vector<int> select(cfg.devices, cfg.devices + (cfg.devices ? std::max(cfg.num_devices, 0) : 0));
        std::vector<std::string> ids;
        for (int i = 0; cfg.device_ids && i < cfg.num_device_ids; ++i)
            if (cfg.device_ids[i]) ids.emplace_back(cfg.device_ids[i]);
        cfg.devices = nullptr;  // not kept: only read here
        cfg.device_ids = nullptr;

        gail_decoder* raw = d.get();
        const Logger deviceLog = [raw](LogLevel level, const std::string& m) {
            raw->log(level == LogLevel::Info ? GAIL_LOG_INFO : GAIL_LOG_WARNING, "%s", m.c_str());
        };
        // The backend asked for, or with AUTO the first of CUDA, Vulkan and CPU that decodes
        // and has a device
        std::vector<std::string> candidates;
        if (cfg.backend == GAIL_BACKEND_AUTO) candidates = {"cuda", "vulkan", "cpu"};
        else candidates = {kBackendNames[cfg.backend]};
        std::string why = "no backend";
        for (const std::string& b : candidates) {
            if (!backendDecodes(b)) {
                why = "the " + b + " backend of this libgail does not decode";
                continue;
            }
            try {
                std::string chosen;
                d->devices = createDevices(b, select, ids, cfg.cpu_threads, cfg.low_priority != 0, deviceLog, chosen);
                d->backend = chosen;
                break;
            } catch (const std::exception& e) {
                why = e.what();
            }
        }
        if (d->devices.empty()) throw ApiError(GAIL_ERROR_UNSUPPORTED, why);
        for (size_t i = 0; i < d->devices.size(); ++i) d->devicesDesc += (i ? ", " : "") + d->devices[i]->name();
        *decoder = d.release();
        return GAIL_OK;
    } catch (const ApiError& e) {
        return fail(e.status, e.what());
    } catch (const std::bad_alloc&) {
        return fail(GAIL_ERROR_NOMEM, "out of memory");
    } catch (const std::exception& e) {
        return fail(GAIL_ERROR_FAILED, e.what());
    }
}

void gail_decoder_destroy(gail_decoder* d) {
    if (!d) return;
    gail_decoder_cancel(d);
    d->shutDown();
    ImagePool* pool = d->pool;
    delete d;  // with the jobs still holding buffers
    if (pool) pool->close();
}

const char* gail_decoder_devices(const gail_decoder* d) { return d ? d->devicesDesc.c_str() : ""; }

int gail_decoder_send(gail_decoder* d, const gail_packet* pkt, const gail_image* dest) {
    if (!d) return fail(GAIL_ERROR_INVALID, "decoder is NULL");
    if (d->canceled.load()) return fail(GAIL_ERROR_CANCELLED, "canceled");
    if (!pkt) {  // end of stream
        {
            std::lock_guard<std::mutex> l(d->m);
            d->ended = true;
        }
        d->doneCv.notify_all();
        return GAIL_OK;
    }
    if (int r = checkPacket(pkt)) return r;
    try {
        auto job = std::make_shared<Job>();
        job->opaque = pkt->opaque;
        // A frame the decoder cannot even be set up from still takes its place in line
        const int setUp = d->setUp(pkt->data, pkt->size);
        if (setUp != GAIL_OK && d->failed.load()) return setUp;  // a device failure, not this frame's fault
        if (setUp == GAIL_OK) {
            if (int r = prepareDestination(d, *job, dest)) return r;
            job->cs.assign(pkt->data, pkt->data + pkt->size);
        } else {
            job->status = setUp;
            job->error = g_lastError;
        }
        std::unique_lock<std::mutex> l(d->m);
        if (d->ended) return fail(GAIL_ERROR_STATE, "send after the end of the stream");
        d->doneCv.wait(l, [&] { return d->canceled.load() || d->queue.size() < std::max<size_t>(d->workers.size(), 1); });
        if (d->canceled.load()) return fail(GAIL_ERROR_CANCELLED, "canceled");
        job->index = d->nextIndex++;
        job->epoch = d->epoch;
        ++d->outstanding;
        if (setUp == GAIL_OK) {
            d->queue.push_back(job);
            l.unlock();
            d->workCv.notify_one();
        } else {
            l.unlock();
            d->finished(job, false);
        }
        return GAIL_OK;
    } catch (const std::bad_alloc&) {
        return fail(GAIL_ERROR_NOMEM, "out of memory");
    }
}

int gail_decoder_receive(gail_decoder* d, gail_image* img, int wait) {
    if (!d || !img) return fail(GAIL_ERROR_INVALID, "decoder or image is NULL");
    if (d->cfg.on_image) return fail(GAIL_ERROR_STATE, "images go to the on_image callback");
    std::unique_lock<std::mutex> l(d->m);
    auto have = [&] { return d->ready.count(d->nextOut) != 0; };
    auto drained = [&] { return d->ended && d->outstanding == 0 && d->ready.empty(); };
    if (wait) d->doneCv.wait(l, [&] { return have() || drained() || d->canceled.load(); });
    if (d->canceled.load()) return fail(GAIL_ERROR_CANCELLED, "canceled");
    if (!have()) return drained() ? GAIL_EOF : GAIL_EAGAIN;
    const std::shared_ptr<Job> job = d->ready[d->nextOut];
    d->ready.erase(d->nextOut++);
    d->fill(*img, *job);
    job->ownedDst = false;  // the app's now, through the image
    if (job->status != GAIL_OK) g_lastError = job->error;
    return GAIL_OK;
}

int gail_decoder_decode(gail_decoder* d, const gail_packet* pkt, gail_image* img) {
    if (!d || !pkt || !img) return fail(GAIL_ERROR_INVALID, "decoder, packet or image is NULL");
    if (img->reserved[3] == kOwnedImage)
        return fail(GAIL_ERROR_INVALID, "the image still holds a picture: release it with gail_image_free() first");
    if (d->canceled.load()) return fail(GAIL_ERROR_CANCELLED, "canceled");
    if (int r = checkPacket(pkt)) return r;
    try {
        if (int r = d->setUp(pkt->data, pkt->size)) return r;
        auto job = std::make_shared<Job>();
        job->ordered = false;
        job->opaque = pkt->opaque;
        if (int r = prepareDestination(d, *job, img)) return r;
        job->cs.assign(pkt->data, pkt->data + pkt->size);
        std::unique_lock<std::mutex> l(d->m);
        d->doneCv.wait(l, [&] { return d->canceled.load() || d->queue.size() < d->workers.size(); });
        if (d->canceled.load()) return fail(GAIL_ERROR_CANCELLED, "canceled");
        job->index = d->nextBlocking++;
        d->queue.push_back(job);
        d->workCv.notify_one();
        // Once queued the job may be writing to the caller's buffer, so this waits for it
        // even when the decoder is canceled meanwhile (a canceled decoder drops what is
        // still queued, and marks it done)
        d->doneCv.wait(l, [&] { return job->done; });
        l.unlock();
        d->fill(*img, *job);
        job->ownedDst = false;
        if (job->status != GAIL_OK) return fail(job->status, job->error);
        return GAIL_OK;
    } catch (const std::bad_alloc&) {
        return fail(GAIL_ERROR_NOMEM, "out of memory");
    }
}

int gail_decoder_flush(gail_decoder* d) {
    if (!d) return fail(GAIL_ERROR_INVALID, "decoder is NULL");
    std::unique_lock<std::mutex> l(d->m);
    ++d->epoch;
    for (auto it = d->queue.begin(); it != d->queue.end();) it = (*it)->ordered ? d->queue.erase(it) : it + 1;
    d->ready.clear();
    d->outstanding = 0;
    d->nextOut = d->nextIndex;
    d->ended = false;
    // Frames the devices are working on may be writing to buffers the app handed over:
    // they are let finish (and are then dropped) before this returns
    d->doneCv.wait(l, [&] { return d->activeOrdered == 0; });
    l.unlock();
    d->doneCv.notify_all();
    return d->canceled.load() ? fail(GAIL_ERROR_CANCELLED, "canceled") : GAIL_OK;
}

int gail_decoder_finish(gail_decoder* d) {
    if (!d) return fail(GAIL_ERROR_INVALID, "decoder is NULL");
    std::unique_lock<std::mutex> l(d->m);
    d->ended = true;
    d->doneCv.notify_all();
    const bool callback = d->cfg.on_image != nullptr;
    d->doneCv.wait(l, [&] {
        return d->canceled.load() || (d->outstanding == 0 && (!callback || (d->ready.empty() && !d->delivering)));
    });
    if (d->canceled.load()) return fail(GAIL_ERROR_CANCELLED, "canceled");
    return GAIL_OK;
}

void gail_decoder_cancel(gail_decoder* d) {
    if (!d) return;
    {
        std::lock_guard<std::mutex> l(d->m);
        if (d->canceled.exchange(true)) return;
        for (auto& job : d->queue) {  // never started: done, as far as their waiters go
            job->failWith(GAIL_ERROR_CANCELLED, "canceled");
            job->done = true;
        }
        d->queue.clear();
        d->ready.clear();
        d->outstanding = 0;
    }
    d->doneCv.notify_all();
}

const char* gail_decoder_error(const gail_decoder* d) { return d && d->failed.load() ? d->error.c_str() : nullptr; }

}  // extern "C"
