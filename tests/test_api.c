// SPDX-License-Identifier: MIT
/* Library API test, compiled as C against include/gail/gail.h: every way of getting
   codestreams (pull, callback, blocking from several threads), every input format,
   letterboxed pictures, canceling, errors and the binary-compatibility rules. Uses
   whatever devices are present (CPU fallback). */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
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
typedef DWORD thread_id;
static thread_id thread_self(void) { return GetCurrentThreadId(); }
static int thread_same(thread_id a, thread_id b) { return a == b; }
typedef CRITICAL_SECTION lock_t;
static void lock_init(lock_t* l) { InitializeCriticalSection(l); }
static void lock_free(lock_t* l) { DeleteCriticalSection(l); }
static void lock_take(lock_t* l) { EnterCriticalSection(l); }
static void lock_give(lock_t* l) { LeaveCriticalSection(l); }
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
typedef pthread_t thread_id;
static thread_id thread_self(void) { return pthread_self(); }
static int thread_same(thread_id a, thread_id b) { return pthread_equal(a, b); }
typedef pthread_mutex_t lock_t;
static void lock_init(lock_t* l) { pthread_mutex_init(l, NULL); }
static void lock_free(lock_t* l) { pthread_mutex_destroy(l); }
static void lock_take(lock_t* l) { pthread_mutex_lock(l); }
static void lock_give(lock_t* l) { pthread_mutex_unlock(l); }
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

/* ---- canceling ------------------------------------------------------------------------ */
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
} canceler;

static void cancel_in_callback(void* user, const gail_packet* p) {
    canceler* c = (canceler*)user;
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
    canceler c = {NULL, 0};
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
    printf("  %-14s canceled in the 2nd callback: %d callbacks, %d sends\n", "cancel cb", c.calls, sent);
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

/* ---- decoder ----------------------------------------------------------------------------- */

#define DEC_FRAMES 6
typedef struct dec_set {
    int w, h;
    gail_packet pkt[DEC_FRAMES];
    uint16_t* src[DEC_FRAMES]; /* 12-bit codes in the low bits */
} dec_set;

/* PSNR of a decoded picture (codes shifted left by `shift`) against 12-bit source codes. */
static double psnr_xyz(const uint8_t* img, ptrdiff_t stride, const uint16_t* src, int w, int h, int shift) {
    double se = 0;
    for (int y = 0; y < h; ++y) {
        const uint16_t* row = (const uint16_t*)(img + (ptrdiff_t)y * stride);
        for (int i = 0; i < w * 3; ++i) {
            const double d = (double)(row[i] >> shift) - (double)src[(size_t)y * w * 3 + i];
            se += d * d;
        }
    }
    return se == 0 ? 99.0 : 10.0 * log10(4095.0 * 4095.0 * (double)w * h * 3 / se);
}

/* One image of the pull test: in order, the damaged frame 2 failed, frame 4 in our buffer. */
static void dec_pull_check(const dec_set* s, gail_image* img, int got, const uint8_t* mine, ptrdiff_t stride) {
    const int w = s->w, h = s->h;
    CHECK(img->index == got && img->opaque == s->pkt[got].opaque, "pull: image %d arrived as %lld", got, (long long)img->index);
    if (got == 2) {
        CHECK(img->status == GAIL_ERROR_INVALID && !img->data[0] && strstr(gail_last_error(), "damaged"),
              "pull: damaged frame came back with status %d (%s)", img->status, gail_last_error());
    } else {
        CHECK(img->status == GAIL_OK && img->data[0] && img->width == w && img->height == h, "pull: image %d: status %d", got,
              img->status);
        if (img->status == GAIL_OK)
            CHECK(psnr_xyz((const uint8_t*)img->data[0], img->stride[0], s->src[got], w, h, 4) > 40, "pull: image %d is wrong", got);
    }
    if (got == 4) {
        int touched = 0;
        for (int y = 0; y < h; ++y)
            for (int i = 0; i < 26; ++i) touched += mine[(size_t)y * stride + (size_t)w * 6 + i] != 0x5A;
        CHECK(img->data[0] == mine && img->stride[0] == stride && touched == 0, "pull: caller's buffer: %d padding bytes written",
              touched);
    }
    gail_image_free(img);
}

static int dec_cb_calls, dec_cb_bad;
static void dec_on_image(void* user, const gail_image* img) {
    const dec_set* s = (const dec_set*)user;
    const int rw = (s->w + 1) / 2, rh = (s->h + 1) / 2;
    if (img->index != dec_cb_calls || img->status != GAIL_OK || img->width != rw || img->height != rh || !img->data[0] ||
        img->format != GAIL_PIX_XYZ12 || img->opaque != (void*)(intptr_t)(500 + img->index))
        ++dec_cb_bad;
    else {
        /* half size, codes in the low bits: every word is a 12-bit code */
        const uint16_t* px = (const uint16_t*)img->data[0];
        for (int i = 0; i < rw * 3; ++i) dec_cb_bad += px[i] > 4095;
    }
    ++dec_cb_calls;
}

static volatile int dec_slow_calls;
static void dec_on_image_slow(void* user, const gail_image* img) {
    (void)user;
    (void)img;
    sleep_ms(20); /* a consumer slower than the decoder */
    dec_slow_calls = dec_slow_calls + 1;
}

static int dec_log_fit;
static void dec_on_log(void* user, gail_log_level level, const char* message) {
    (void)user;
    if (level <= GAIL_LOG_WARNING && strstr(message, "frames in flight fit")) ++dec_log_fit;
}

typedef struct dec_thread {
    gail_decoder* d;
    const dec_set* s;
    const uint8_t* want; /* frame 0 as decoded before, tightly packed */
    int bad;
} dec_thread;
static thread_ret THREAD_FN dec_worker(void* arg) {
    dec_thread* t = (dec_thread*)arg;
    for (int i = 0; i < 3; ++i) {
        gail_image img;
        memset(&img, 0, sizeof img);
        if (gail_decoder_decode(t->d, &t->s->pkt[0], &img) != GAIL_OK ||
            memcmp(img.data[0], t->want, (size_t)t->s->w * t->s->h * 6) != 0)
            ++t->bad;
        gail_image_free(&img);
    }
    return 0;
}

/* ---- decoder: the threading contracts of gail.h, on the CPU backend ---------------------- */

/* A flag one thread raises and another waits for, with a time limit. */
typedef struct dec_flag {
    lock_t lock;
    int value;
} dec_flag;
static void flag_raise(dec_flag* f, int v) {
    lock_take(&f->lock);
    f->value = v;
    lock_give(&f->lock);
}
static int flag_get(dec_flag* f) {
    lock_take(&f->lock);
    const int v = f->value;
    lock_give(&f->lock);
    return v;
}
/* Waits up to ms for the flag to be raised; returns its value (0: it was not). */
static int flag_wait(dec_flag* f, int ms) {
    int v = 0;
    for (double end = now_ms() + ms; !(v = flag_get(f)) && now_ms() < end;) sleep_ms(2);
    return v;
}

static gail_decoder* dec_cpu(gail_image_cb on_image, void* user) {
    gail_decoder_config dc;
    gail_decoder_config_defaults(&dc);
    dc.backend = GAIL_BACKEND_CPU;
    dc.cpu_threads = 1; /* slow enough to be caught in the middle of a frame */
    dc.frames_in_flight = 1;
    dc.on_image = on_image;
    dc.user = user;
    gail_decoder* d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK && d, "CPU decoder create: %s", gail_last_error());
    return d;
}

/* A blocking decode on another thread, for gail_decoder_cancel() to catch under way. */
typedef struct dec_blocked {
    gail_decoder* d;
    const gail_packet* pkt;
    gail_image img;
    int result;
    double returned; /* now_ms() when the call came back */
} dec_blocked;
static thread_ret THREAD_FN dec_blocked_run(void* arg) {
    dec_blocked* b = (dec_blocked*)arg;
    memset(&b->img, 0, sizeof b->img);
    b->result = gail_decoder_decode(b->d, b->pkt, &b->img);
    b->returned = now_ms();
    return 0;
}

/* on_image of the contract tests: on which thread and in what order images arrive, and
   what a send or a decode from inside the callback does */
typedef struct dec_contract {
    const dec_set* s;
    gail_decoder* d;
    thread_id app; /* the thread that sends */
    int calls, on_app, out_of_order;
    int status[8];
    int nest;        /* on image 0: 1 = send a codestream, 2 = decode one */
    int nested;      /* what that call returned */
    dec_flag queued; /* the app has queued its second codestream */
    dec_flag back;   /* the nested call has come back */
} dec_contract;
static void dec_contract_image(void* user, const gail_image* img) {
    dec_contract* c = (dec_contract*)user;
    if (thread_same(thread_self(), c->app)) ++c->on_app;
    if (img->index != c->calls) ++c->out_of_order;
    if (c->calls < 8) c->status[c->calls] = img->status;
    ++c->calls;
    if (img->index == 0 && c->nest) {
        /* the only frame in flight is this thread's, and the app's next codestream waits
           in the queue: nothing else can make room for a call that waits for room */
        flag_wait(&c->queued, 10000);
        if (c->nest == 1) {
            c->nested = gail_decoder_send(c->d, &c->s->pkt[2], NULL);
        } else {
            gail_image out;
            memset(&out, 0, sizeof out);
            c->nested = gail_decoder_decode(c->d, &c->s->pkt[2], &out);
            if (c->nested == GAIL_OK &&
                psnr_xyz((const uint8_t*)out.data[0], out.stride[0], c->s->src[2], out.width, out.height, 4) < 40)
                c->nested = GAIL_ERROR_FAILED;
            gail_image_free(&out);
        }
        flag_raise(&c->back, 1);
    }
}

static void test_decoder_contracts(const dec_set* s) {
    /* A cancel while a blocking decode is under way: the call waits for the frame (it may
       be writing into the caller's buffer), then returns GAIL_ERROR_CANCELLED, not the
       picture */
    {
        gail_decoder* d = dec_cpu(NULL, NULL);
        if (d) {
            gail_image img;
            memset(&img, 0, sizeof img);
            const double t0 = now_ms();
            CHECK(gail_decoder_decode(d, &s->pkt[0], &img) == GAIL_OK, "CPU decode: %s", gail_last_error());
            const double took = now_ms() - t0;
            gail_image_free(&img);
            dec_blocked b;
            b.d = d;
            b.pkt = &s->pkt[1];
            thread_t t;
            thread_start(&t, dec_blocked_run, &b);
            sleep_ms((int)(took / 3) + 2);
            const double canceled = now_ms();
            gail_decoder_cancel(d);
            thread_join(t);
            if (b.returned < canceled) {
                printf("  note: the decode was over before the cancel (%.0f ms a frame)\n", took);
            } else {
                CHECK(b.result == GAIL_ERROR_CANCELLED && !b.img.data[0] && b.img.status == GAIL_ERROR_CANCELLED,
                      "cancel during a blocking decode: it returned %s with %s picture", gail_status_string(b.result),
                      b.img.data[0] ? "a" : "no");
            }
            gail_image_free(&b.img);
            gail_decoder_destroy(d);
        }
    }

    static uint8_t soc_only[2] = {0xFF, 0x4F}; /* a codestream cut off after SOC */
    gail_packet cut;
    memset(&cut, 0, sizeof cut);
    cut.data = soc_only;
    cut.size = sizeof soc_only;

    /* A first codestream the decoder cannot be set up from: its image still comes from a
       decoder thread, never from inside gail_decoder_send() */
    {
        dec_contract c;
        memset(&c, 0, sizeof c);
        c.s = s;
        c.app = thread_self();
        gail_decoder* d = dec_cpu(dec_contract_image, &c);
        if (d) {
            CHECK(gail_decoder_send(d, &cut, NULL) == GAIL_OK, "send of a cut-off first codestream: %s", gail_last_error());
            CHECK(gail_decoder_finish(d) == GAIL_OK, "finish after a cut-off codestream: %s", gail_last_error());
            CHECK(c.calls == 1 && c.status[0] == GAIL_ERROR_INVALID, "cut-off codestream alone: %d images, status %d", c.calls,
                  c.status[0]);
            CHECK(c.on_app == 0, "on_image ran on the sending thread for a codestream that cannot set up the decoder");
            gail_decoder_destroy(d);
        }
        memset(&c, 0, sizeof c);
        c.s = s;
        c.app = thread_self();
        d = dec_cpu(dec_contract_image, &c);
        if (d) {
            CHECK(gail_decoder_send(d, &cut, NULL) == GAIL_OK, "send of a cut-off first codestream: %s", gail_last_error());
            CHECK(gail_decoder_send(d, &cut, NULL) == GAIL_OK, "send of a second cut-off codestream: %s", gail_last_error());
            CHECK(gail_decoder_send(d, &s->pkt[0], NULL) == GAIL_OK, "send after cut-off codestreams: %s", gail_last_error());
            CHECK(gail_decoder_send(d, &cut, NULL) == GAIL_OK, "send of a cut-off codestream: %s", gail_last_error());
            CHECK(gail_decoder_finish(d) == GAIL_OK, "finish: %s", gail_last_error());
            CHECK(c.calls == 4 && c.out_of_order == 0 && c.status[0] == GAIL_ERROR_INVALID && c.status[1] == GAIL_ERROR_INVALID &&
                      c.status[2] == GAIL_OK && c.status[3] == GAIL_ERROR_INVALID,
                  "cut-off codestreams around a good one: %d images, %d out of order, statuses %d %d %d %d", c.calls,
                  c.out_of_order, c.status[0], c.status[1], c.status[2], c.status[3]);
            CHECK(c.on_app == 0, "on_image ran on the sending thread %d times", c.on_app);
            gail_decoder_destroy(d);
        }
    }

    /* A send, or a blocking decode, from inside on_image with one frame in flight and the
       app's next codestream queued: neither waits for the queue that only this thread
       could work off */
    for (int nest = 1; nest <= 2; ++nest) {
        const char* what = nest == 1 ? "send" : "decode";
        dec_contract c;
        memset(&c, 0, sizeof c);
        lock_init(&c.queued.lock);
        lock_init(&c.back.lock);
        c.s = s;
        c.app = thread_self();
        c.nest = nest;
        c.d = dec_cpu(dec_contract_image, &c);
        if (c.d) {
            CHECK(gail_decoder_send(c.d, &s->pkt[0], NULL) == GAIL_OK, "nested %s: send 0: %s", what, gail_last_error());
            CHECK(gail_decoder_send(c.d, &s->pkt[1], NULL) == GAIL_OK, "nested %s: send 1: %s", what, gail_last_error());
            flag_raise(&c.queued, 1);
            const int back = flag_wait(&c.back, 10000);
            CHECK(back, "a %s from on_image was still waiting after 10 s", what);
            if (!back) {
                gail_decoder_cancel(c.d); /* lets it go */
            } else {
                CHECK(gail_decoder_finish(c.d) == GAIL_OK, "nested %s: finish: %s", what, gail_last_error());
                CHECK(c.nested == GAIL_OK, "%s from on_image: %s", what, gail_status_string(c.nested));
                const int want = nest == 1 ? 3 : 2; /* a frame decoded with gail_decoder_decode() skips on_image */
                CHECK(c.calls == want && c.out_of_order == 0 && c.status[0] == GAIL_OK && c.status[want - 1] == GAIL_OK,
                      "nested %s: %d images (%d out of order)", what, c.calls, c.out_of_order);
            }
            gail_decoder_destroy(c.d);
        }
        lock_free(&c.queued.lock);
        lock_free(&c.back.lock);
    }
}

static void test_decoder(void) {
    dec_set s;
    memset(&s, 0, sizeof s);
    gail_config ec;
    gail_config_defaults(&ec);
    ec.input.format = GAIL_PIX_XYZ12;
    gail_encoder* e = make("decoder source", &ec);
    if (!e) return;
    gail_encoder_frame_size(e, &s.w, &s.h);
    const int w = s.w, h = s.h;
    for (int f = 0; f < DEC_FRAMES; ++f) {
        s.src[f] = (uint16_t*)malloc((size_t)w * h * 6);
        /* a smooth picture that moves: something a codec can represent well */
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x)
                for (int c = 0; c < 3; ++c)
                    s.src[f][((size_t)y * w + x) * 3 + c] = (uint16_t)(600 + ((x + 9 * f) * (c + 2) + y * (3 - c)) % 2800);
        gail_frame fr = {{s.src[f]}, {0}, NULL};
        CHECK(gail_encoder_encode(e, &fr, &s.pkt[f]) == GAIL_OK, "decoder source: encode %d: %s", f, gail_last_error());
        s.pkt[f].opaque = (void*)(intptr_t)(500 + f);
    }
    gail_encoder_destroy(e);

    /* probe */
    gail_stream_info info;
    CHECK(gail_probe(s.pkt[0].data, s.pkt[0].size, &info) == GAIL_OK && info.supported == 1 && info.width == w &&
              info.height == h && info.components == 3 && info.bit_depth == 12 && info.rsiz == 3 && info.levels == 5 &&
              !info.reversible && !info.high_throughput && info.tiles == 1 && info.layers == 1,
          "probe: %dx%d, supported %d (%s)", info.width, info.height, info.supported, gail_last_error());
    CHECK(gail_probe(s.pkt[0].data, 40, &info) == GAIL_ERROR_INVALID, "probe of a cut-off header accepted");
    CHECK(gail_probe("not a codestream at all", 23, &info) == GAIL_ERROR_INVALID, "probe of text accepted");
    {
        /* a lossless stream is JPEG 2000 but not one for this decoder: say so, do not fail */
        uint8_t* copy = (uint8_t*)malloc(s.pkt[0].size);
        memcpy(copy, s.pkt[0].data, s.pkt[0].size);
        size_t cod = 0;
        for (size_t i = 2; i + 1 < 200; ++i)
            if (copy[i] == 0xFF && copy[i + 1] == 0x52) cod = i;
        CHECK(cod, "no COD marker found");
        copy[cod + 13] = 1; /* SPcod transformation: 5/3 reversible */
        CHECK(gail_probe(copy, s.pkt[0].size, &info) == GAIL_OK && info.supported == 0 && info.reversible == 1 &&
                  strstr(gail_last_error(), "5/3"),
              "probe of a 5/3 stream: supported %d, %s", info.supported, gail_last_error());
        free(copy);
    }

    gail_decoder_config dc;
    gail_decoder* d = NULL;

    /* blocking, library buffer, codes in the high bits (the default) */
    gail_decoder_config_defaults(&dc);
    CHECK(dc.output == GAIL_PIX_XYZ12_MSB && dc.reduce == 0 && dc.backend == GAIL_BACKEND_AUTO, "decoder defaults");
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK && d, "decoder create: %s", gail_last_error());
    if (!d) return;
    printf("  decoder on %s\n", gail_decoder_devices(d));
    uint8_t* frame0 = (uint8_t*)malloc((size_t)w * h * 6);
    for (int f = 0; f < DEC_FRAMES; ++f) {
        gail_image img;
        memset(&img, 0, sizeof img);
        const int r = gail_decoder_decode(d, &s.pkt[f], &img);
        CHECK(r == GAIL_OK && img.status == GAIL_OK && img.width == w && img.height == h && img.data[0] &&
                  img.stride[0] == (ptrdiff_t)w * 6 && img.format == GAIL_PIX_XYZ12_MSB && img.opaque == s.pkt[f].opaque,
              "decode %d: %s", f, gail_last_error());
        if (r != GAIL_OK) continue;
        const double p = psnr_xyz((const uint8_t*)img.data[0], img.stride[0], s.src[f], w, h, 4);
        if (f == 0) {
            printf("  decoded frame 0: %.2f dB against the source (encoder's estimate %.2f)\n", p, s.pkt[0].psnr_estimate);
            memcpy(frame0, img.data[0], (size_t)w * h * 6);
        }
        /* as close as the encoder said it would be (an estimate above 60 dB means nothing was cut) */
        const double promised = s.pkt[f].psnr_estimate < 60 ? s.pkt[f].psnr_estimate : 60;
        CHECK(p > promised - 1.5, "decode %d: %.2f dB, encoder estimated %.2f", f, p, s.pkt[f].psnr_estimate);
        /* an image that still holds a picture is not silently overwritten or leaked */
        CHECK(gail_decoder_decode(d, &s.pkt[f], &img) == GAIL_ERROR_INVALID, "decode into an unreleased image accepted");
        gail_image_free(&img);
        CHECK(img.data[0] == NULL, "gail_image_free left the pointer");
        gail_image_free(&img); /* twice is harmless */
    }

    /* several threads at once: the same picture every time */
    {
        dec_thread th[4];
        thread_t tid[4];
        for (int i = 0; i < 4; ++i) {
            th[i].d = d;
            th[i].s = &s;
            th[i].want = frame0;
            th[i].bad = 0;
            thread_start(&tid[i], dec_worker, &th[i]);
        }
        int bad = 0;
        for (int i = 0; i < 4; ++i) {
            thread_join(tid[i]);
            bad += th[i].bad;
        }
        CHECK(bad == 0, "decode from 4 threads: %d wrong pictures", bad);
    }

    /* a frame that cannot be decoded fails alone, and says why */
    {
        gail_image img;
        gail_packet junk;
        memset(&junk, 0, sizeof junk);
        static uint8_t noise[5000];
        for (size_t i = 0; i < sizeof noise; ++i) noise[i] = (uint8_t)(i * 31 + 7);
        junk.data = noise;
        junk.size = sizeof noise;
        memset(&img, 0, sizeof img);
        CHECK(gail_decoder_decode(d, &junk, &img) == GAIL_ERROR_INVALID && !img.data[0], "noise decoded: %s", gail_last_error());
        junk.data = s.pkt[1].data;
        junk.size = s.pkt[1].size / 2;
        CHECK(gail_decoder_decode(d, &junk, &img) == GAIL_ERROR_INVALID && !img.data[0] && strstr(gail_last_error(), "damaged"),
              "half a codestream decoded: %s", gail_last_error());
        printf("  rejected as expected: %s\n", gail_last_error());
        CHECK(gail_decoder_error(d) == NULL, "a bad frame stopped the decoder: %s", gail_decoder_error(d));
        CHECK(gail_decoder_decode(d, &s.pkt[2], &img) == GAIL_OK, "decode after a bad frame: %s", gail_last_error());
        gail_image_free(&img);

        /* a caller's buffer that is too narrow, reserved words, an empty packet */
        uint8_t* ownBuf = (uint8_t*)malloc((size_t)w * h * 6);
        memset(&img, 0, sizeof img);
        img.data[0] = ownBuf;
        img.stride[0] = (ptrdiff_t)w * 6 - 2;
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_ERROR_INVALID && strstr(gail_last_error(), "stride"),
              "short stride accepted");
        img.stride[0] = 0;
        img.reserved[0] = 1;
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_ERROR_INVALID, "non-zero gail_image.reserved accepted");
        img.reserved[0] = 0;
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_OK && img.data[0] == ownBuf && img.stride[0] == (ptrdiff_t)w * 6 &&
                  memcmp(ownBuf, frame0, (size_t)w * h * 6) == 0,
              "decode into the caller's buffer: %s", gail_last_error());
        gail_image_free(&img); /* not the library's: must leave it alone */
        CHECK(img.data[0] == ownBuf, "gail_image_free touched the caller's buffer");
        free(ownBuf);
        gail_packet bad = s.pkt[0];
        bad.reserved[1] = 1;
        memset(&img, 0, sizeof img);
        CHECK(gail_decoder_decode(d, &bad, &img) == GAIL_ERROR_INVALID, "non-zero gail_packet.reserved accepted");
        bad.reserved[1] = 0;
        bad.size = 0;
        CHECK(gail_decoder_decode(d, &bad, &img) == GAIL_ERROR_INVALID, "empty packet accepted");
    }

    /* a frame of another size needs another decoder */
    {
        gail_config_defaults(&ec);
        ec.input.format = GAIL_PIX_XYZ12;
        ec.container = GAIL_CONTAINER_SCOPE;
        gail_encoder* e2 = make("scope source", &ec);
        if (e2) {
            uint16_t* px = (uint16_t*)calloc((size_t)2048 * 858, 6);
            gail_frame fr = {{px}, {0}, NULL};
            gail_packet scope;
            gail_image img;
            memset(&img, 0, sizeof img);
            CHECK(gail_encoder_encode(e2, &fr, &scope) == GAIL_OK, "scope source: %s", gail_last_error());
            CHECK(gail_decoder_decode(d, &scope, &img) == GAIL_ERROR_UNSUPPORTED && strstr(gail_last_error(), "another decoder"),
                  "a 2048x858 frame in a 1998x1080 decoder: %s", gail_last_error());
            printf("  rejected as expected: %s\n", gail_last_error());
            gail_packet_free(&scope);
            free(px);
            gail_encoder_destroy(e2);
        }
    }
    gail_decoder_destroy(d);

    /* a buffer that says what size it is: only a picture of that size goes into it */
    gail_decoder_config_defaults(&dc);
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create (sized buffer): %s", gail_last_error());
    if (d) {
        uint8_t* mine = (uint8_t*)malloc((size_t)w * h * 6);
        gail_image img;
        memset(&img, 0, sizeof img);
        img.data[0] = mine;
        img.width = w / 2;
        img.height = h / 2;
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_ERROR_INVALID && strstr(gail_last_error(), "buffer is for"),
              "a %dx%d picture into a buffer for %dx%d: %s", w, h, w / 2, h / 2, gail_last_error());
        printf("  rejected as expected: %s\n", gail_last_error());
        CHECK(gail_decoder_send(d, &s.pkt[0], &img) == GAIL_ERROR_INVALID, "send into a buffer of another size accepted");
        img.width = w;
        img.height = h;
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_OK && img.data[0] == mine &&
                  memcmp(mine, frame0, (size_t)w * h * 6) == 0,
              "decode into a buffer of the right size: %s", gail_last_error());
        gail_decoder_destroy(d);
        free(mine);
    }

    /* pull: order kept, a damaged frame in the middle, one frame into a padded buffer of ours */
    gail_decoder_config_defaults(&dc);
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create: %s", gail_last_error());
    if (d) {
        const ptrdiff_t stride = (ptrdiff_t)w * 6 + 26;
        uint8_t* mine = (uint8_t*)malloc((size_t)stride * h);
        memset(mine, 0x5A, (size_t)stride * h);
        gail_image img;
        CHECK(gail_decoder_receive(d, &img, 0) == GAIL_EAGAIN, "receive before anything was sent");
        int got = 0, r, refused = 0;
        for (int f = 0; f < DEC_FRAMES; ++f) {
            gail_packet p = s.pkt[f];
            gail_image dest;
            memset(&dest, 0, sizeof dest);
            if (f == 2) p.size = p.size * 2 / 3; /* damaged in transit */
            if (f == 4) {
                dest.data[0] = mine;
                dest.stride[0] = stride;
            }
            /* GAIL_EAGAIN: as many decoded images wait as the decoder holds; take one first */
            while ((r = gail_decoder_send(d, &p, f == 4 ? &dest : NULL)) == GAIL_EAGAIN) {
                ++refused;
                const int rr = gail_decoder_receive(d, &img, 0);
                CHECK(rr == GAIL_OK, "pull: send said receive first, but receive says %s", gail_status_string(rr));
                if (rr != GAIL_OK) break;
                dec_pull_check(&s, &img, got++, mine, stride);
            }
            CHECK(r == GAIL_OK, "send %d: %s", f, gail_last_error());
        }
        CHECK(gail_decoder_send(d, NULL, NULL) == GAIL_OK, "end of stream");
        CHECK(gail_decoder_send(d, &s.pkt[0], NULL) == GAIL_ERROR_STATE, "send after the end of the stream accepted");
        while ((r = gail_decoder_receive(d, &img, 1)) == GAIL_OK) dec_pull_check(&s, &img, got++, mine, stride);
        printf("  pull: %d sends refused until an image was received\n", refused);
        CHECK(r == GAIL_EOF && got == DEC_FRAMES, "pull: %d images, then %s", got, gail_status_string(r));
        CHECK(gail_decoder_finish(d) == GAIL_OK, "finish after the end");
        gail_decoder_destroy(d);
        free(mine);
    }

    /* flush (a seek): what was sent before it never arrives, what is sent after it does */
    gail_decoder_config_defaults(&dc);
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create: %s", gail_last_error());
    if (d) {
        gail_image img;
        for (int f = 0; f < 4; ++f) CHECK(gail_decoder_send(d, &s.pkt[f], NULL) == GAIL_OK, "send %d", f);
        CHECK(gail_decoder_flush(d) == GAIL_OK, "flush: %s", gail_last_error());
        CHECK(gail_decoder_receive(d, &img, 0) == GAIL_EAGAIN, "an image survived the flush");
        CHECK(gail_decoder_send(d, &s.pkt[5], NULL) == GAIL_OK, "send after flush");
        CHECK(gail_decoder_receive(d, &img, 1) == GAIL_OK && img.status == GAIL_OK && img.opaque == s.pkt[5].opaque && img.index == 4,
              "after flush: index %lld", (long long)img.index);
        gail_image_free(&img);
        /* cancel: everything stops, nothing hangs */
        for (int f = 0; f < 3; ++f) gail_decoder_send(d, &s.pkt[f], NULL);
        gail_decoder_cancel(d);
        CHECK(gail_decoder_receive(d, &img, 1) == GAIL_ERROR_CANCELLED, "receive after cancel");
        CHECK(gail_decoder_send(d, &s.pkt[0], NULL) == GAIL_ERROR_CANCELLED, "send after cancel");
        memset(&img, 0, sizeof img);
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_ERROR_CANCELLED, "decode after cancel");
        CHECK(gail_decoder_finish(d) == GAIL_ERROR_CANCELLED, "finish after cancel");
        gail_decoder_destroy(d);
    }

    /* callback, half size, codes in the low bits */
    gail_decoder_config_defaults(&dc);
    dc.reduce = 1;
    dc.output = GAIL_PIX_XYZ12;
    dc.on_image = dec_on_image;
    dc.user = &s;
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create (callback): %s", gail_last_error());
    if (d) {
        gail_image img;
        dec_cb_calls = dec_cb_bad = 0;
        for (int f = 0; f < DEC_FRAMES; ++f) CHECK(gail_decoder_send(d, &s.pkt[f], NULL) == GAIL_OK, "send %d (callback)", f);
        CHECK(gail_decoder_receive(d, &img, 0) == GAIL_ERROR_STATE, "receive allowed in callback mode");
        CHECK(gail_decoder_finish(d) == GAIL_OK, "finish (callback): %s", gail_last_error());
        CHECK(dec_cb_calls == DEC_FRAMES && dec_cb_bad == 0, "callback: %d calls, %d wrong", dec_cb_calls, dec_cb_bad);
        gail_decoder_destroy(d);
    }

    /* back-pressure, callback mode: a consumer slower than the decoder holds up send, so
       the decoded pictures waiting for it stay few (with 2 frames in flight on one device:
       2 queued, 2 decoding, about 4 decoded) instead of growing with the stream */
    gail_decoder_config_defaults(&dc);
    dc.frames_in_flight = 2;
    dc.on_image = dec_on_image_slow;
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create (slow callback): %s", gail_last_error());
    if (d) {
        dec_slow_calls = 0;
        int worst = 0;
        for (int f = 0; f < 24; ++f) {
            gail_packet p = s.pkt[f % DEC_FRAMES];
            CHECK(gail_decoder_send(d, &p, NULL) == GAIL_OK, "send %d (slow callback): %s", f, gail_last_error());
            const int waiting = f + 1 - dec_slow_calls;
            if (waiting > worst) worst = waiting;
        }
        CHECK(gail_decoder_finish(d) == GAIL_OK, "finish (slow callback): %s", gail_last_error());
        printf("  slow callback: at most %d images sent and not yet delivered\n", worst);
        CHECK(dec_slow_calls == 24 && worst <= 9, "slow callback: %d calls, up to %d images waiting", dec_slow_calls, worst);
        gail_decoder_destroy(d);
    }

    /* back-pressure, pull mode: sending without receiving soon gets GAIL_EAGAIN, and the
       codestream refused is not taken: it goes in again once an image has been received */
    gail_decoder_config_defaults(&dc);
    dc.frames_in_flight = 2;
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create (backlog): %s", gail_last_error());
    if (d) {
        int accepted = 0, r = GAIL_OK;
        while (accepted < 40 && (r = gail_decoder_send(d, &s.pkt[accepted % DEC_FRAMES], NULL)) == GAIL_OK) ++accepted;
        CHECK(r == GAIL_EAGAIN && accepted <= 9, "pull backlog: %d codestreams taken, then %s", accepted, gail_status_string(r));
        gail_image img;
        CHECK(gail_decoder_receive(d, &img, 0) == GAIL_OK && img.index == 0, "pull backlog: no image ready after GAIL_EAGAIN");
        gail_image_free(&img);
        CHECK(gail_decoder_send(d, &s.pkt[accepted % DEC_FRAMES], NULL) == GAIL_OK, "pull backlog: send after a receive: %s",
              gail_last_error());
        ++accepted;
        CHECK(gail_decoder_send(d, NULL, NULL) == GAIL_OK, "end of stream (backlog)");
        int got = 1;
        while ((r = gail_decoder_receive(d, &img, 1)) == GAIL_OK) {
            CHECK(img.index == got && img.status == GAIL_OK && img.opaque == s.pkt[got % DEC_FRAMES].opaque,
                  "pull backlog: image %d arrived as %lld, status %d", got, (long long)img.index, img.status);
            gail_image_free(&img);
            ++got;
        }
        CHECK(r == GAIL_EOF && got == accepted, "pull backlog: %d of %d images, then %s", got, accepted, gail_status_string(r));
        gail_decoder_destroy(d);
    }

    /* a memory limit that holds one frame: the decoder runs with one frame in flight per
       GPU and says so (the CPU backend has no device memory to limit) */
    gail_decoder_config_defaults(&dc);
    dc.max_device_memory = (uint64_t)100 << 20;
    dc.log = dec_on_log;
    dc.log_level = GAIL_LOG_WARNING;
    d = NULL;
    dec_log_fit = 0;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create (memory limit): %s", gail_last_error());
    if (d) {
        gail_image img;
        memset(&img, 0, sizeof img);
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_OK && memcmp(img.data[0], frame0, (size_t)w * h * 6) == 0,
              "decode under a memory limit: %s", gail_last_error());
        gail_image_free(&img);
        const int gpu = strstr(gail_decoder_devices(d), "CUDA") || strstr(gail_decoder_devices(d), "Vulkan");
        CHECK(!gpu || dec_log_fit > 0, "frames in flight not reduced to fit max_device_memory");
        gail_decoder_destroy(d);
    }

    /* leaving out more levels than the stream has: that frame fails, with the reason */
    gail_decoder_config_defaults(&dc);
    dc.reduce = 6;
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_OK, "decoder create (reduce 6): %s", gail_last_error());
    if (d) {
        gail_image img;
        memset(&img, 0, sizeof img);
        CHECK(gail_decoder_decode(d, &s.pkt[0], &img) == GAIL_ERROR_INVALID && strstr(gail_last_error(), "levels"),
              "reduce 6 on a 5-level stream: %s", gail_last_error());
        gail_decoder_destroy(d);
    }

    /* settings and the binary-compatibility rules */
    gail_decoder_config_defaults(&dc);
    dc.output = GAIL_PIX_RGB48;
    d = NULL;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_ERROR_INVALID && !d, "RGB output accepted");
    gail_decoder_config_defaults(&dc);
    dc.reduce = -1;
    CHECK(gail_decoder_create(&dc, &d) == GAIL_ERROR_INVALID && !d, "negative reduce accepted");
    memset(&dc, 0, sizeof dc);
    CHECK(gail_decoder_create(&dc, &d) == GAIL_ERROR_INVALID && !d, "uninitialized decoder config accepted");
    struct {
        gail_decoder_config c;
        uint64_t later[4];
    } newer;
    memset(&newer, 0xAB, sizeof newer);
    gail_decoder_config_init(&newer.c, sizeof newer);
    CHECK(newer.c.struct_size == sizeof newer && newer.later[0] == 0 && newer.later[3] == 0 && newer.c.output == GAIL_PIX_XYZ12_MSB,
          "gail_decoder_config_init with a larger size");
    CHECK(gail_decoder_create(&newer.c, &d) == GAIL_OK, "newer header, unused fields: %s", gail_last_error());
    gail_decoder_destroy(d);
    d = NULL;
    newer.later[2] = 1;
    CHECK(gail_decoder_create(&newer.c, &d) == GAIL_ERROR_UNSUPPORTED && !d, "newer header, a setting we lack accepted");
    printf("  sizes: decoder_config %zu, image %zu, stream_info %zu\n", sizeof(gail_decoder_config), sizeof(gail_image),
           sizeof(gail_stream_info));

    test_decoder_contracts(&s);

    free(frame0);
    for (int f = 0; f < DEC_FRAMES; ++f) {
        gail_packet_free(&s.pkt[f]);
        free(s.src[f]);
    }
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
    test_decoder();

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
    CHECK(gail_encoder_create(&cfg, &e) == GAIL_ERROR_INVALID, "uninitialized config accepted");

    printf(failures ? "%d FAILURES\n" : "all API tests passed\n", failures);
    return failures ? 1 : 0;
}
