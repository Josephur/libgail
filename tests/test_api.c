// SPDX-License-Identifier: MIT
/* Library API test, compiled as C against include/gail/gail.h: every way of getting
   codestreams (pull, callback, blocking from several threads), every input format,
   letterboxed pictures, cancelling, errors and the binary-compatibility rules. Uses
   whatever devices are present (CPU fallback). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "gail/gail.h"

#ifdef _WIN32
#include <windows.h>
typedef HANDLE thread_t;
typedef DWORD thread_ret;
#define THREAD_FN WINAPI
static void thread_start(thread_t* t, LPTHREAD_START_ROUTINE fn, void* arg) { *t = CreateThread(NULL, 0, fn, arg, 0, NULL); }
static void thread_join(thread_t t) {
    WaitForSingleObject(t, INFINITE);
    CloseHandle(t);
}
static double now_ms(void) { return (double)GetTickCount64(); }
static void sleep_ms(int ms) { Sleep(ms); }
#else
#include <pthread.h>
#include <time.h>
typedef pthread_t thread_t;
typedef void* thread_ret;
#define THREAD_FN
static void thread_start(thread_t* t, void* (*fn)(void*), void* arg) { pthread_create(t, NULL, fn, arg); }
static void thread_join(thread_t t) { pthread_join(t, NULL); }
static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}
static void sleep_ms(int ms) {
    struct timespec ts = {ms / 1000, (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}
#endif

static int failures = 0;
#define CHECK(c, ...)                                     \
    do {                                                  \
        if (!(c)) {                                       \
            printf("  FAIL %s:%d: ", __FILE__, __LINE__); \
            printf(__VA_ARGS__);                          \
            printf("\n");                                 \
            ++failures;                                   \
        }                                                 \
    } while (0)

static int is_codestream(const uint8_t* d, size_t n) {
    return n > 4 && d[0] == 0xFF && d[1] == 0x4F && d[n - 2] == 0xFF && d[n - 1] == 0xD9;
}

/* A moving gradient, as interleaved 16-bit samples (12-bit values for X'Y'Z'). */
static void fill_interleaved(uint16_t* px, int w, int h, int f, gail_pixel_format fmt) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            for (int c = 0; c < 3; ++c) {
                const unsigned v = (unsigned)((x + 7 * f) * (c + 1) + y * 3) % 4096u;
                px[((size_t)y * w + x) * 3 + c] =
                    (uint16_t)(fmt == GAIL_PIX_XYZ12 ? v : fmt == GAIL_PIX_XYZ12_MSB ? v << 4 : v * 16u);
            }
}

/* Planar layouts: bit depth and chroma subsampling (log2 across / down); 0 bits = interleaved. */
static void planar_layout(gail_pixel_format fmt, int* bits, int* sx, int* sy) {
    static const int tab[][3] = {
        {0, 0, 0},  {0, 0, 0},  {0, 0, 0},  {8, 1, 1}, {10, 1, 1}, {12, 1, 1}, {8, 1, 0},  {10, 1, 0},
        {12, 1, 0}, {8, 0, 0},  {10, 0, 0}, {12, 0, 0}, {8, 0, 0},  {10, 0, 0}, {12, 0, 0}, {16, 0, 0}};
    *bits = tab[fmt][0];
    *sx = tab[fmt][1];
    *sy = tab[fmt][2];
}

static gail_encoder* make(const char* name, gail_config* cfg) {
    gail_encoder* e = NULL;
    const int r = gail_encoder_create(cfg, &e);
    CHECK(r == GAIL_OK && e, "%s: create: %s (%s)", name, gail_status_string(r), gail_last_error());
    return r == GAIL_OK ? e : NULL;
}

/* ---- pull: send / receive ------------------------------------------------------------- */
static void test_pull(const char* name, gail_config* cfg, int frames) {
    gail_encoder* e = make(name, cfg);
    if (!e) return;
    int w = cfg->input.width, h = cfg->input.height;
    if (!w || !h) gail_encoder_frame_size(e, &w, &h); /* 0: the picture fills the container */
    const gail_pixel_format fmt = cfg->input.format;
    int bits, sx, sy;
    planar_layout(fmt, &bits, &sx, &sy);
    const int yuv = bits > 0; /* any planar layout */
    const size_t bps = bits > 8 ? 2 : 1;
    const int cw = (w + sx) >> sx, ch = (h + sy) >> sy;
    uint8_t* buf = (uint8_t*)malloc(yuv ? ((size_t)w * h + 2 * (size_t)cw * ch) * bps : (size_t)w * h * 6);
    int64_t got = 0;
    size_t bytes = 0;
    for (int f = 0; f <= frames; ++f) {
        if (f < frames) {
            gail_frame fr;
            memset(&fr, 0, sizeof fr);
            if (yuv) {
                uint8_t* p = buf;
                for (int c = 0; c < 3; ++c) {
                    const int pw = c ? cw : w, ph = c ? ch : h;
                    const unsigned maxv = (1u << bits) - 1;
                    for (int y = 0; y < ph; ++y)
                        for (int x = 0; x < pw; ++x) {
                            const unsigned v = c ? maxv / 2 + ((x + f) % 64) * (maxv / 255)
                                                 : (unsigned)(x * 3 + y + f * 5) * (maxv / 255) % (maxv + 1);
                            if (bps == 1) p[(size_t)y * pw + x] = (uint8_t)v;
                            else ((uint16_t*)p)[(size_t)y * pw + x] = (uint16_t)v;
                        }
                    fr.data[c] = p;
                    p += (size_t)pw * ph * bps;
                }
            } else {
                fill_interleaved((uint16_t*)buf, w, h, f, fmt);
                fr.data[0] = buf;
            }
            fr.opaque = (void*)(intptr_t)(1000 + f);
            CHECK(gail_encoder_send(e, &fr) == GAIL_OK, "%s: send %d: %s", name, f, gail_last_error());
        } else {
            CHECK(gail_encoder_send(e, NULL) == GAIL_OK, "%s: end of stream", name);
        }
        /* drain whatever is ready (blocking at the end) */
        for (;;) {
            gail_packet pkt;
            const int r = gail_encoder_receive(e, &pkt, f == frames);
            if (r == GAIL_EAGAIN || r == GAIL_EOF) break;
            CHECK(r == GAIL_OK, "%s: receive: %s", name, gail_last_error());
            if (r != GAIL_OK) break;
            CHECK(pkt.index == got && pkt.opaque == (void*)(intptr_t)(1000 + got), "%s: packet %lld out of order", name,
                  (long long)pkt.index);
            CHECK(pkt.compliant && is_codestream(pkt.data, pkt.size), "%s: packet %lld not a compliant codestream", name,
                  (long long)pkt.index);
            bytes += pkt.size;
            ++got;
            gail_packet_free(&pkt);
        }
    }
    gail_packet pkt;
    CHECK(gail_encoder_receive(e, &pkt, 1) == GAIL_EOF, "%s: no EOF", name);
    CHECK(got == frames, "%s: %lld of %d packets", name, (long long)got, frames);
    gail_frame fr = {{buf}, {0}, NULL};
    CHECK(gail_encoder_send(e, &fr) == GAIL_ERROR_STATE, "%s: send after the end accepted", name);
    int cwd, chh;
    gail_encoder_frame_size(e, &cwd, &chh);
    printf("  %-14s %dx%d in %dx%d on %s: %lld packets, %zu bytes\n", name, w, h, cwd, chh, gail_encoder_devices(e),
           (long long)got, bytes);
    free(buf);
    gail_encoder_destroy(e);
}

/* ---- callback ------------------------------------------------------------------------- */
typedef struct {
    int64_t next;
    int bad;
} sink;

static void on_packet(void* user, const gail_packet* p) {
    sink* s = (sink*)user;
    if (p->index != s->next || !p->compliant || !is_codestream(p->data, p->size)) s->bad++;
    s->next++;
}

static void test_callback(const char* name, gail_config* cfg, int frames) {
    sink s = {0, 0};
    cfg->on_packet = on_packet;
    cfg->user = &s;
    gail_encoder* e = make(name, cfg);
    cfg->on_packet = NULL;
    if (!e) return;
    int w = cfg->input.width, h = cfg->input.height;
    if (!w || !h) gail_encoder_frame_size(e, &w, &h);
    uint16_t* px = (uint16_t*)malloc((size_t)w * h * 6);
    for (int f = 0; f < frames; ++f) {
        fill_interleaved(px, w, h, f, cfg->input.format);
        gail_frame fr = {{px}, {0}, NULL};
        CHECK(gail_encoder_send(e, &fr) == GAIL_OK, "%s: send: %s", name, gail_last_error());
    }
    CHECK(gail_encoder_finish(e) == GAIL_OK, "%s: finish: %s", name, gail_last_error());
    CHECK(s.next == frames && s.bad == 0, "%s: %lld callbacks, %d bad", name, (long long)s.next, s.bad);
    gail_packet pkt;
    CHECK(gail_encoder_receive(e, &pkt, 0) == GAIL_ERROR_STATE, "%s: receive allowed in callback mode", name);
    printf("  %-14s %d callbacks in order\n", name, (int)s.next);
    free(px);
    gail_encoder_destroy(e);
}

/* ---- blocking encode from several threads ---------------------------------------------- */
#define THREADS 4
#define PER_THREAD 3
typedef struct {
    gail_encoder* e;
    int id, w, h, ok;
} job;

static thread_ret THREAD_FN worker(void* arg) {
    job* j = (job*)arg;
    uint16_t* px = (uint16_t*)malloc((size_t)j->w * j->h * 6);
    for (int k = 0; k < PER_THREAD; ++k) {
        fill_interleaved(px, j->w, j->h, j->id * 10 + k, GAIL_PIX_XYZ12);
        gail_frame fr = {{px}, {0}, (void*)(intptr_t)(j->id * 100 + k)};
        gail_packet pkt;
        if (gail_encoder_encode(j->e, &fr, &pkt) == GAIL_OK && pkt.opaque == fr.opaque && pkt.compliant &&
            is_codestream(pkt.data, pkt.size))
            j->ok++;
        gail_packet_free(&pkt);
    }
    free(px);
    return 0;
}

static void test_threads(gail_config* cfg) {
    gail_encoder* e = make("threads", cfg);
    if (!e) return;
    job jobs[THREADS];
    thread_t th[THREADS];
    for (int i = 0; i < THREADS; ++i) {
        jobs[i].e = e;
        jobs[i].id = i;
        jobs[i].w = cfg->input.width;
        jobs[i].h = cfg->input.height;
        jobs[i].ok = 0;
        thread_start(&th[i], worker, &jobs[i]);
    }
    int ok = 0;
    for (int i = 0; i < THREADS; ++i) {
        thread_join(th[i]);
        ok += jobs[i].ok;
    }
    CHECK(ok == THREADS * PER_THREAD, "threads: %d of %d frames came back right", ok, THREADS * PER_THREAD);
    printf("  %-14s %d threads x %d frames, each got its own codestream: %d ok\n", "threads", THREADS, PER_THREAD, ok);
    gail_encoder_destroy(e);
}

/* ---- cancelling ------------------------------------------------------------------------ */
typedef struct {
    gail_encoder* e;
    int w, h, sent, last, mode; /* mode 0: send forever; 1: encode forever */
} feeder;

static thread_ret THREAD_FN feed(void* arg) {
    feeder* f = (feeder*)arg;
    uint16_t* px = (uint16_t*)malloc((size_t)f->w * f->h * 6);
    fill_interleaved(px, f->w, f->h, 0, GAIL_PIX_XYZ12);
    for (;;) {
        gail_frame fr = {{px}, {0}, NULL};
        gail_packet pkt;
        f->last = f->mode ? gail_encoder_encode(f->e, &fr, &pkt) : gail_encoder_send(f->e, &fr);
        if (f->last != GAIL_OK) break;
        if (f->mode) gail_packet_free(&pkt);
        f->sent++;
    }
    free(px);
    return 0;
}

typedef struct {
    gail_encoder* e;
    int calls;
} canceller;

static void cancel_in_callback(void* user, const gail_packet* p) {
    canceller* c = (canceller*)user;
    if (c->calls++ == 1) gail_encoder_cancel(c->e);
    (void)p;
}

static void test_cancel(gail_config* cfg) {
    /* senders blocked on a full queue and receivers waiting are woken; destroy is quick */
    for (int mode = 0; mode < 2; ++mode) {
        const char* name = mode ? "cancel encode" : "cancel send";
        gail_encoder* e = make(name, cfg);
        if (!e) return;
        feeder f[2];
        thread_t th[2];
        for (int i = 0; i < 2; ++i) {
            f[i] = (feeder){e, cfg->input.width, cfg->input.height, 0, 0, mode};
            thread_start(&th[i], feed, &f[i]);
        }
        sleep_ms(300);
        const double t0 = now_ms();
        gail_encoder_cancel(e);
        for (int i = 0; i < 2; ++i) thread_join(th[i]);
        const double woken = now_ms() - t0;
        for (int i = 0; i < 2; ++i)
            CHECK(f[i].last == GAIL_ERROR_CANCELLED, "%s: feeder %d ended with %s", name, i, gail_status_string(f[i].last));
        gail_packet pkt;
        gail_frame fr = {{cfg}, {0}, NULL};
        CHECK(gail_encoder_receive(e, &pkt, 1) == GAIL_ERROR_CANCELLED,
              "%s: receive after cancel", name);
        CHECK(gail_encoder_send(e, &fr) == GAIL_ERROR_CANCELLED, "%s: send after cancel", name);
        CHECK(gail_encoder_finish(e) == GAIL_ERROR_CANCELLED, "%s: finish after cancel", name);
        CHECK(gail_encoder_error(e) == NULL, "%s: cancel reported as an error", name);
        gail_encoder_cancel(e); /* twice is fine */
        gail_encoder_destroy(e);
        const double total = now_ms() - t0;
        printf("  %-14s %d+%d frames in, callers woken in %.0f ms, destroyed after %.0f ms\n", name, f[0].sent, f[1].sent,
               woken, total);
        CHECK(woken < 2000, "%s: callers took %.0f ms to wake", name, woken);
    }

    /* from inside on_packet */
    canceller c = {NULL, 0};
    cfg->on_packet = cancel_in_callback;
    cfg->user = &c;
    const int r = gail_encoder_create(cfg, &c.e);
    cfg->on_packet = NULL;
    CHECK(r == GAIL_OK, "cancel cb: create: %s", gail_last_error());
    if (r != GAIL_OK) return;
    gail_encoder* e = c.e;
    uint16_t* px = (uint16_t*)malloc((size_t)cfg->input.width * cfg->input.height * 6);
    fill_interleaved(px, cfg->input.width, cfg->input.height, 0, GAIL_PIX_XYZ12);
    int rs = GAIL_OK, sent = 0;
    for (; sent < 40 && rs == GAIL_OK; ++sent) {
        gail_frame fr = {{px}, {0}, NULL};
        rs = gail_encoder_send(e, &fr);
    }
    CHECK(gail_encoder_finish(e) == GAIL_ERROR_CANCELLED, "cancel cb: finish: %s", gail_last_error());
    CHECK(c.calls >= 2 && c.calls < 40, "cancel cb: %d callbacks", c.calls);
    printf("  %-14s cancelled in the 2nd callback: %d callbacks, %d sends\n", "cancel cb", c.calls, sent);
    free(px);
    gail_encoder_destroy(e);
}

/* ---- worst-case content: full-scale noise overflows the GPUs' Tier-1 buffers and must be
   redone on the CPU, not crash the device ---------------------------------------------- */
static void test_noise(void) {
    gail_config cfg;
    gail_config_defaults(&cfg);
    gail_encoder* e = make("noise", &cfg);
    if (!e) return;
    int w, h;
    gail_encoder_frame_size(e, &w, &h);
    uint16_t* px = (uint16_t*)malloc((size_t)w * h * 6);
    uint32_t x = 1;
    for (size_t i = 0; i < (size_t)w * h * 3; ++i) {
        x = x * 1664525u + 1013904223u;
        px[i] = (uint16_t)(x >> 16);
    }
    for (int k = 0; k < 2; ++k) { /* the second frame shows the device is still usable */
        gail_frame fr = {{px}, {0}, NULL};
        gail_packet pkt;
        const int r = gail_encoder_encode(e, &fr, &pkt);
        CHECK(r == GAIL_OK, "noise frame %d: %s", k, gail_last_error());
        if (r != GAIL_OK) break;
        CHECK(pkt.compliant && is_codestream(pkt.data, pkt.size), "noise frame %d not compliant", k);
        printf("  %-14s frame %d: %zu bytes on %s\n", "noise", k, pkt.size, gail_encoder_devices(e));
        gail_packet_free(&pkt);
    }
    free(px);
    gail_encoder_destroy(e);
}

/* ---- binary compatibility ---------------------------------------------------------------- */
static void test_abi(void) {
    gail_config cfg;
    gail_encoder* e = NULL;
    printf("  sizes: config %zu, input %zu, color %zu, frame %zu, packet %zu, device_info %zu\n", sizeof(gail_config),
           sizeof(gail_input), sizeof(gail_color), sizeof(gail_frame), sizeof(gail_packet), sizeof(gail_device_info));

    /* a program built against a newer header: its extra fields are zero unless it uses them */
    struct {
        gail_config c;
        uint64_t later[4];
    } newer;
    memset(&newer, 0xAB, sizeof newer);
    gail_config_init(&newer.c, sizeof newer);
    CHECK(newer.c.struct_size == sizeof newer && newer.later[0] == 0 && newer.later[3] == 0 && newer.c.fps == 24,
          "gail_config_init with a larger size");
    newer.c.input.format = GAIL_PIX_XYZ12;
    CHECK(gail_encoder_create(&newer.c, &e) == GAIL_OK, "newer header, unused fields: %s", gail_last_error());
    gail_encoder_destroy(e);
    e = NULL;
    newer.later[1] = 7;
    CHECK(gail_encoder_create(&newer.c, &e) == GAIL_ERROR_UNSUPPORTED && !e, "newer header, a setting we lack accepted");
    printf("  rejected as expected: %s\n", gail_last_error());

    /* gail_config_init never writes past the size it is given */
    struct {
        gail_config c;
        uint64_t guard;
    } g;
    g.guard = 0x1234;
    gail_config_init(&g.c, sizeof g.c);
    CHECK(g.guard == 0x1234, "gail_config_init wrote past the struct");

    gail_config_defaults(&cfg);
    cfg.struct_size = offsetof(gail_config, log);
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID, "config smaller than 1.0 accepted");

    /* reserved words must be zero */
    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_XYZ12;
    cfg.input.reserved[0] = 1;
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID && strstr(gail_last_error(), "reserved"),
          "non-zero gail_input.reserved accepted");
    gail_config_defaults(&cfg);
    cfg.color.reserved[7] = 1;
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID, "non-zero gail_color.reserved accepted");
    char d[256];
    CHECK(gail_color_describe(&cfg.color, d, sizeof d) == GAIL_ERROR_INVALID, "describe: non-zero reserved accepted");
    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_XYZ12;
    e = make("abi frame", &cfg);
    if (e) {
        uint16_t* px = (uint16_t*)calloc((size_t)2048 * 1080, 6);
        gail_frame fr = {{px}, {0}, NULL};
        fr.reserved[2] = 1;
        CHECK(gail_encoder_send(e, &fr) == GAIL_ERROR_INVALID, "non-zero gail_frame.reserved accepted");
        fr.reserved[2] = 0;
        gail_packet pkt;
        memset(&pkt, 0xAB, sizeof pkt);
        const int r = gail_encoder_encode(e, &fr, &pkt);
        CHECK(r == GAIL_OK, "abi frame: encode: %s", gail_last_error());
        if (r == GAIL_OK) {
            CHECK(pkt.reserved[0] == 0 && pkt.reserved[3] == 0, "packet reserved words not zeroed");
            gail_packet_free(&pkt);
        }
        free(px);
        gail_encoder_destroy(e);
    }
    gail_device_info di;
    memset(&di, 0xAB, sizeof di);
    CHECK(gail_enumerate_devices(GAIL_BACKEND_CPU, &di, 1) == 1 && di.reserved[0] == 0 && di.reserved[7] == 0,
          "device_info reserved words not zeroed");
}

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    char d[512] = "";
    gail_config cfg;
    printf("gail %s (header %d.%d.%d)\n", gail_version_string(), GAIL_VERSION_MAJOR, GAIL_VERSION_MINOR,
           GAIL_VERSION_PATCH);
    CHECK(gail_version() == GAIL_VERSION, "library version %x != header %x", gail_version(), GAIL_VERSION);
    CHECK(gail_list_devices(GAIL_BACKEND_AUTO, d, sizeof d) > 0, "no devices listed: %s", gail_last_error());
    printf("  devices:\n%s", d);

    /* device listing: friendly names and nvidia-smi style UUIDs, selection by UUID / PCI bus ID */
    gail_device_info devs[16];
    const int ndev = gail_enumerate_devices(GAIL_BACKEND_AUTO, devs, 16);
    CHECK(ndev > 0, "no devices enumerated: %s", gail_last_error());
    for (int i = 0; i < ndev && i < 16; ++i) {
        printf("  device %d: %s  uuid %s  pci %s  %.0f GB\n", devs[i].index, devs[i].name,
               devs[i].uuid[0] ? devs[i].uuid : "-", devs[i].pci_bus_id[0] ? devs[i].pci_bus_id : "-",
               devs[i].memory_bytes / 1073741824.0);
        CHECK(devs[i].name[0], "device %d has no name", i);
        CHECK(devs[i].backend == GAIL_BACKEND_CPU || strncmp(devs[i].uuid, "GPU-", 4) == 0, "GPU %d has no UUID", i);
    }
    if (ndev > 0 && devs[0].uuid[0]) {
        const gail_device_info* last = &devs[(ndev < 16 ? ndev : 16) - 1];
        const char* by_uuid[] = {last->uuid};
        gail_encoder* e = NULL;
        gail_config_defaults(&cfg);
        cfg.input.format = GAIL_PIX_XYZ12;
        cfg.device_ids = by_uuid;
        cfg.num_device_ids = 1;
        CHECK(gail_encoder_create(&cfg, &e) == GAIL_OK && strstr(gail_encoder_devices(e), last->name) &&
                  !strchr(gail_encoder_devices(e), ','),
              "select by UUID %s: %s", last->uuid, e ? gail_encoder_devices(e) : gail_last_error());
        if (e) printf("  by UUID %s -> %s\n", last->uuid, gail_encoder_devices(e));
        gail_encoder_destroy(e);
        if (last->pci_bus_id[0]) {
            const char* by_pci[] = {last->pci_bus_id + 5}; /* "01:00.0": the domain is optional */
            cfg.device_ids = by_pci;
            e = NULL;
            CHECK(gail_encoder_create(&cfg, &e) == GAIL_OK && strstr(gail_encoder_devices(e), last->name),
                  "select by PCI bus ID %s: %s", by_pci[0], gail_last_error());
            gail_encoder_destroy(e);
        }
        const char* bogus[] = {"GPU-00000000-0000-0000-0000-000000000000"};
        cfg.device_ids = bogus;
        e = NULL;
        CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_UNSUPPORTED && !e && strstr(gail_last_error(), last->uuid),
              "unknown UUID accepted or not explained: %s", gail_last_error());
    }

    int w, h;
    CHECK(gail_container_size(GAIL_PROFILE_4K, GAIL_CONTAINER_SCOPE, &w, &h) == GAIL_OK && w == 4096 && h == 1716,
          "4K scope is %dx%d", w, h);

    /* CPU backend: thread count and priority are the caller's to set */
    {
        gail_encoder* e = NULL;
        gail_config_defaults(&cfg);
        cfg.backend = GAIL_BACKEND_CPU;
        cfg.cpu_threads = 2;
        cfg.low_priority = 1;
        CHECK(gail_encoder_create(&cfg, &e) == GAIL_OK && strstr(gail_encoder_devices(e), "CPU (2 threads)"),
              "cpu_threads 2: %s", e ? gail_encoder_devices(e) : gail_last_error());
        gail_encoder_destroy(e);
        test_pull("cpu low prio", &cfg, 2);
    }

    gail_config_defaults(&cfg);
    CHECK(gail_color_describe(&cfg.color, d, sizeof d) > 0, "describe SDR");
    printf("  SDR preset: %s\n", d);
    gail_color_preset(GAIL_PRESET_HDR_PQ_BT2020, &cfg.color);
    CHECK(gail_color_describe(&cfg.color, d, sizeof d) > 0 && strstr(d, "DCP-o-matic"), "describe PQ: %s", d);
    printf("  PQ preset:  %s\n", d);
    cfg.color.knee = 1.5;
    CHECK(gail_color_describe(&cfg.color, d, sizeof d) == GAIL_ERROR_INVALID, "knee 1.5 accepted");

    /* X'Y'Z' (libdcp layout), full container, pull */
    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_XYZ12;
    cfg.input.width = 1998;
    cfg.input.height = 1080;
    test_pull("xyz12", &cfg, 4);

    /* X'Y'Z' (FFmpeg layout), 4K scope, pull */
    gail_config_defaults(&cfg);
    cfg.profile = GAIL_PROFILE_4K;
    cfg.container = GAIL_CONTAINER_SCOPE;
    cfg.input.format = GAIL_PIX_XYZ12_MSB;
    cfg.input.width = 4096;
    cfg.input.height = 1716;
    test_pull("xyz12-msb 4k", &cfg, 2);

    /* HDR10 Y'CbCr 10-bit, 1920x1080 pillarboxed into flat 2K (what FFmpeg hands over) */
    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_YUV420P10;
    cfg.input.width = 1920;
    cfg.input.height = 1080;
    cfg.input.matrix = GAIL_MATRIX_BT2020;
    gail_color_preset(GAIL_PRESET_HDR_PQ_BT2020, &cfg.color);
    test_pull("yuv420p10", &cfg, 3);

    /* 8-bit Y'CbCr, letterboxed and odd-sized */
    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_YUV420P;
    cfg.input.width = 1997;
    cfg.input.height = 833;
    test_pull("yuv420p odd", &cfg, 2);

    /* the other planar layouts, odd-sized: 4:2:2 and 4:4:4 Y'CbCr (ProRes, DNxHR), G'B'R' (DPX) */
    {
        static const struct {
            const char* name;
            gail_pixel_format fmt;
        } planar[] = {{"yuv422p", GAIL_PIX_YUV422P},     {"yuv422p10", GAIL_PIX_YUV422P10}, {"yuv422p12", GAIL_PIX_YUV422P12},
                      {"yuv444p", GAIL_PIX_YUV444P},     {"yuv444p10", GAIL_PIX_YUV444P10}, {"yuv444p12", GAIL_PIX_YUV444P12},
                      {"gbrp", GAIL_PIX_GBRP},           {"gbrp10", GAIL_PIX_GBRP10},       {"gbrp12", GAIL_PIX_GBRP12},
                      {"gbrp16", GAIL_PIX_GBRP16}};
        for (size_t i = 0; i < sizeof planar / sizeof planar[0]; ++i) {
            gail_config_defaults(&cfg);
            cfg.input.format = planar[i].fmt;
            cfg.input.width = 1919;
            cfg.input.height = 1037;
            test_pull(planar[i].name, &cfg, 2);
        }
        gail_encoder* e = NULL;
        gail_config_defaults(&cfg);
        cfg.input.format = (gail_pixel_format)(GAIL_PIX_GBRP16 + 1);
        CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID && !e, "unknown pixel format accepted");
    }

    /* R'G'B' 16-bit, smaller than the container, callback */
    gail_config_defaults(&cfg);
    cfg.input.width = 1920;
    cfg.input.height = 800;
    test_callback("rgb48 cb", &cfg, 3);

    /* app LUT: 2^3 nodes, X'Y'Z' = R'G'B' scaled into 0.1..0.9 */
    static float lut[2 * 2 * 2 * 3];
    for (int b = 0; b < 2; ++b)
        for (int g = 0; g < 2; ++g)
            for (int r = 0; r < 2; ++r) {
                float* n = lut + ((b * 2 + g) * 2 + r) * 3;
                n[0] = 0.1f + 0.8f * r;
                n[1] = 0.1f + 0.8f * g;
                n[2] = 0.1f + 0.8f * b;
            }
    gail_config_defaults(&cfg);
    cfg.color.mode = GAIL_COLOR_LUT3D;
    cfg.color.lut3d = lut;
    cfg.color.lut3d_size = 2;
    cfg.input.width = 1998;
    cfg.input.height = 1080;
    CHECK(gail_color_describe(&cfg.color, d, sizeof d) > 0 && strstr(d, "3D LUT"), "describe LUT: %s", d);
    test_callback("lut3d cb", &cfg, 2);

    /* worker-thread style: blocking encode from several threads, stereo budget */
    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_XYZ12;
    cfg.input.width = 1998;
    cfg.input.height = 1080;
    cfg.stereo = 1;
    test_threads(&cfg);

    gail_config_defaults(&cfg);
    cfg.input.format = GAIL_PIX_XYZ12;
    cfg.input.width = 1998;
    cfg.input.height = 1080;
    test_cancel(&cfg);

    test_noise();
    test_abi();

    /* errors are reported, not thrown */
    gail_encoder* e = NULL;
    gail_config_defaults(&cfg);
    cfg.max_mbps = 300;
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID && !e && strstr(gail_last_error(), "250"),
          "max_mbps 300 accepted");
    printf("  rejected as expected: %s\n", gail_last_error());
    gail_config_defaults(&cfg);
    cfg.max_mbps = 0.05; /* 260 bytes per frame: less than the codestream headers; used to hang the first encode */
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID && !e && strstr(gail_last_error(), "headers"),
          "bitrate below the header size accepted: %s", gail_last_error());
    printf("  rejected as expected: %s\n", gail_last_error());
    gail_config_defaults(&cfg);
    cfg.max_mbps = 0.0001; /* a budget under the 16-byte margin used to wrap around and switch the limit off */
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID && !e, "bitrate below the safety margin accepted");
    gail_config_defaults(&cfg);
    cfg.input.width = 2048; /* wider than flat 2K */
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID && !e, "oversized picture accepted");
    printf("  rejected as expected: %s\n", gail_last_error());
    memset(&cfg, 0, sizeof cfg);
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID, "uninitialised config accepted");

    printf(failures ? "%d FAILURES\n" : "all API tests passed\n", failures);
    return failures ? 1 : 0;
}
