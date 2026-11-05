/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Joseph Stackhouse
 *
 * GAIL, GPU Accelerated Intraframe Library: a JPEG 2000 encoder for DCI (SMPTE ST 428-1 /
 * ST 429-4) picture tracks.
 *
 * Library API, version 1. Plain C (usable from C++ and any FFI); libgail.so / libgail.dll /
 * libgail.dylib, or the static libgail.a. pkg-config name: gail.
 *
 * An app describes its frames once (gail_config), then hands frames over and gets one
 * DCI-compliant JPEG 2000 codestream back per frame. Three ways to get codestreams:
 *
 *   1. Pull (FFmpeg-style):  gail_encoder_send() frames, gail_encoder_receive() packets
 *      in send order; gail_encoder_send(enc, NULL) ends the stream, then receive until
 *      GAIL_EOF.
 *   2. Callback:             set config.on_packet; it is called in send order from an
 *      encoder thread; gail_encoder_finish() waits for the last one.
 *   3. Blocking, per frame (apps with worker threads): gail_encoder_encode() from any number
 *      of threads at once; each call returns its own frame's codestream. Frames from
 *      all threads share the GPUs.
 *
 * Packets returned by receive / encode belong to the caller: release them with
 * gail_packet_free() (never free() them directly: on Windows the DLL has its own heap).
 *
 *   gail_config cfg;
 *   gail_config_defaults(&cfg);
 *   cfg.profile = GAIL_PROFILE_2K;
 *   cfg.input.format = GAIL_PIX_XYZ12;                 // already DCI X'Y'Z': encoded as is
 *   gail_encoder* enc;
 *   if (gail_encoder_create(&cfg, &enc) != GAIL_OK) fprintf(stderr, "%s\n", gail_last_error());
 *   gail_frame f = {{pixels}, {0}, NULL};
 *   gail_packet pkt;
 *   if (gail_encoder_encode(enc, &f, &pkt) == GAIL_OK) { write(pkt.data, pkt.size); gail_packet_free(&pkt); }
 *   gail_encoder_destroy(enc);
 */
#ifndef GAIL_GAIL_H
#define GAIL_GAIL_H

#include <stddef.h>
#include <stdint.h>

/* Only the DLL build itself needs a declaration: callers link functions through the import
   library or the static library alike, so the same header serves both. */
#if defined(_MSC_VER) && !defined(__cplusplus)
#define GAIL_INLINE static __inline
#else
#define GAIL_INLINE static inline
#endif

#if defined(_WIN32) && defined(GAIL_BUILDING) && !defined(GAIL_STATIC)
#define GAIL_API __declspec(dllexport)
#elif defined(_WIN32)
#define GAIL_API
#elif defined(__GNUC__) || defined(__clang__)
#define GAIL_API __attribute__((visibility("default")))
#else
#define GAIL_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define GAIL_VERSION_MAJOR 1
#define GAIL_VERSION_MINOR 0
#define GAIL_VERSION_PATCH 1
#define GAIL_VERSION ((GAIL_VERSION_MAJOR << 16) | (GAIL_VERSION_MINOR << 8) | GAIL_VERSION_PATCH)

/* ---- binary compatibility -----------------------------------------------------------
 * A program built against any 1.x header runs with any later 1.x library:
 *   - gail_config only grows at its end; gail_config_defaults() records the size the
 *     program was built with in struct_size, and fields it lacks keep their defaults.
 *   - Every other struct keeps its size: fields added later take the place of `reserved`
 *     words, and a 0 there keeps the 1.0 behaviour. Structs the program fills in
 *     (gail_frame, and gail_input / gail_color through the config) must have them zeroed:
 *     start from gail_config_defaults() / gail_color_preset() and zero-initialise frames
 *     (gail_frame f = {0}). The encoder rejects non-zero reserved words, so a mistake
 *     shows up now rather than with a later version.
 *   - Enum values and functions may be added; existing ones keep their meaning. */

/* ---- status ------------------------------------------------------------------------ */

typedef enum gail_status {
    GAIL_OK = 0,
    GAIL_EAGAIN = 1,              /* receive: no packet ready yet */
    GAIL_EOF = 2,                 /* receive: the stream has ended and every packet was returned */
    GAIL_ERROR_INVALID = -1,      /* invalid argument or setting */
    GAIL_ERROR_UNSUPPORTED = -2,  /* backend, device or format not available */
    GAIL_ERROR_NOMEM = -3,
    GAIL_ERROR_FAILED = -4,       /* encoding failed: see gail_encoder_error() */
    GAIL_ERROR_STATE = -5,        /* not allowed now (e.g. send after the end of the stream) */
    GAIL_ERROR_CANCELLED = -6     /* gail_encoder_cancel() was called */
} gail_status;

/* Message for the most recent failure in the calling thread (never NULL). */
GAIL_API const char* gail_last_error(void);
GAIL_API const char* gail_status_string(int status);

/* GAIL_VERSION of the library actually loaded, and its full version and backends, e.g.
 * "1.1.0 (cuda vulkan cpu)" for a release or "1.1.0+3.gabc1234 (cuda vulkan cpu)" between releases. */
GAIL_API unsigned gail_version(void);
GAIL_API const char* gail_version_string(void);

/* ---- picture format ---------------------------------------------------------------- */

typedef enum gail_profile { GAIL_PROFILE_2K = 0, GAIL_PROFILE_4K = 1 } gail_profile;

/* DCI containers: flat 1998x1080 / 3996x2160, scope 2048x858 / 4096x1716,
   full 2048x1080 / 4096x2160 */
typedef enum gail_container { GAIL_CONTAINER_FLAT = 0, GAIL_CONTAINER_SCOPE = 1, GAIL_CONTAINER_FULL = 2 } gail_container;

GAIL_API int gail_container_size(gail_profile profile, gail_container container, int* width, int* height);

/* Pixel layouts. Multi-byte samples are native-endian (little-endian on x86 / ARM).
   FFmpeg names in brackets. Planar formats take data[0..2] in FFmpeg's plane order;
   deeper than 8 bits, each sample is a 16-bit word holding the value in its low bits.
   Planar formats are converted to R'G'B' on the GPU, so handing over the decoder's own
   layout is both faster and more exact than converting it first. */
typedef enum gail_pixel_format {
    GAIL_PIX_RGB48 = 0,      /* interleaved 16-bit R'G'B' [rgb48le] */
    GAIL_PIX_XYZ12 = 1,      /* interleaved 16-bit words, 12-bit DCI X'Y'Z' codes in the low bits
                                [libdcp rgb_to_xyz output]; encoded exactly, colour settings unused */
    GAIL_PIX_XYZ12_MSB = 2,  /* the same with the codes in the high bits [xyz12le] */
    GAIL_PIX_YUV420P = 3,    /* planar Y'CbCr 4:2:0, 8-bit [yuv420p] */
    GAIL_PIX_YUV420P10 = 4,  /* [yuv420p10le]: HDR10 H.264 / HEVC */
    GAIL_PIX_YUV420P12 = 5,  /* [yuv420p12le] */
    GAIL_PIX_YUV422P = 6,    /* planar Y'CbCr 4:2:2, 8-bit [yuv422p] */
    GAIL_PIX_YUV422P10 = 7,  /* [yuv422p10le]: ProRes 422, DNxHR HQX */
    GAIL_PIX_YUV422P12 = 8,  /* [yuv422p12le] */
    GAIL_PIX_YUV444P = 9,    /* planar Y'CbCr 4:4:4, 8-bit [yuv444p] */
    GAIL_PIX_YUV444P10 = 10, /* [yuv444p10le]: ProRes 4444, DNxHR 444 (alpha dropped) */
    GAIL_PIX_YUV444P12 = 11, /* [yuv444p12le] */
    GAIL_PIX_GBRP = 12,      /* planar full-range R'G'B', 8-bit, planes G', B', R' [gbrp] */
    GAIL_PIX_GBRP10 = 13,    /* [gbrp10le]: DPX, TIFF */
    GAIL_PIX_GBRP12 = 14,    /* [gbrp12le] */
    GAIL_PIX_GBRP16 = 15     /* [gbrp16le] */
} gail_pixel_format;

typedef enum gail_matrix { GAIL_MATRIX_BT709 = 0, GAIL_MATRIX_BT2020 = 1, GAIL_MATRIX_BT601 = 2 } gail_matrix;

typedef enum gail_chroma_location {
    GAIL_CHROMA_LEFT = 0,    /* MPEG-2 / H.264 / HEVC default */
    GAIL_CHROMA_CENTER = 1,  /* JPEG / MPEG-1 */
    GAIL_CHROMA_TOPLEFT = 2  /* BT.2020 / UHD masters */
} gail_chroma_location;

typedef struct gail_input {
    gail_pixel_format format;
    int width, height;   /* picture size; 0 = the container size. A smaller picture is
                            centred in the container with black borders (no scaling). */
    /* Y'CbCr formats only */
    gail_matrix matrix;
    int full_range;      /* 0 = limited (video) range */
    gail_chroma_location chroma_location;  /* 4:2:0 and 4:2:2 only */
    uint64_t reserved[4];  /* zero */
} gail_input;

/* ---- colour (R'G'B' and Y'CbCr input; X'Y'Z' input is never altered) ---------------- */

typedef enum gail_color_mode {
    GAIL_COLOR_MANAGED = 0,  /* transfer + primaries -> X'Y'Z' (below) */
    GAIL_COLOR_LUT3D = 1     /* an app-supplied 3D LUT from R'G'B' to X'Y'Z' */
} gail_color_mode;

typedef enum gail_transfer {
    GAIL_TRANSFER_BT1886 = 0, /* SDR video (gamma 2.4) */
    GAIL_TRANSFER_SRGB = 1,
    GAIL_TRANSFER_PQ = 2,     /* SMPTE ST 2084 */
    GAIL_TRANSFER_HLG = 3,    /* ARIB STD-B67 / BT.2100 */
    GAIL_TRANSFER_LINEAR = 4,
    GAIL_TRANSFER_GAMMA26 = 5 /* DCI gamma 2.6, code 1.0 = 52.37 cd/m^2 */
} gail_transfer;

typedef enum gail_primaries {
    GAIL_PRIMARIES_BT709 = 0,
    GAIL_PRIMARIES_BT2020 = 1,
    GAIL_PRIMARIES_P3D65 = 2,
    GAIL_PRIMARIES_DCIP3 = 3
} gail_primaries;

typedef enum gail_tone_map {
    GAIL_TONE_MAP_SIGNAL = 0,   /* default: the HDR signal is shown as if it were gamma 2.4 video
                                   with code 1.0 at sdr_white, as DCP-o-matic does: dim and flat,
                                   no crushed highlights */
    GAIL_TONE_MAP_SHOULDER = 1, /* scene-relative: hdr_ref_white -> cinema_ref_white, then a smooth
                                   roll-off reaching cinema_peak at hdr_peak */
    GAIL_TONE_MAP_BT2390 = 2,   /* ITU-R BT.2390 EETF on absolute PQ luminance (bright) */
    GAIL_TONE_MAP_NONE = 3      /* absolute luminance; above cinema_peak clips */
} gail_tone_map;

typedef enum gail_tone_norm {   /* what SHOULDER / BT2390 are applied to */
    GAIL_TONE_NORM_CHANNEL = 0,  /* each channel: bright saturated colours drift toward white */
    GAIL_TONE_NORM_MAXRGB = 1,   /* brightest channel: keeps hue and saturation */
    GAIL_TONE_NORM_LUMINANCE = 2 /* Y: keeps luminance; saturated highlights bleach */
} gail_tone_norm;

typedef enum gail_gamut {
    GAIL_GAMUT_NONE = 0,  /* default, as DCP-o-matic: colours beyond the projector are left to it */
    GAIL_GAMUT_P3D65 = 1, /* out-of-gamut colours are pulled toward neutral, not clipped */
    GAIL_GAMUT_DCIP3 = 2
} gail_gamut;

typedef struct gail_color {
    gail_color_mode mode;
    gail_transfer transfer;
    gail_primaries primaries;
    gail_tone_map tone_map;   /* PQ / HLG only */
    gail_tone_norm tone_norm;
    double sdr_white;         /* cd/m^2 for code 1.0 of BT.1886 / sRGB / linear (and SIGNAL): 48 */
    double hdr_peak;          /* source mastering peak, cd/m^2 */
    double hdr_ref_white;     /* source reference white, cd/m^2 (203, BT.2408) */
    double cinema_ref_white;  /* where it lands in the cinema, cd/m^2 (SHOULDER) */
    double cinema_peak;       /* projector peak white, cd/m^2 (48) */
    double knee;              /* SHOULDER: roll-off start, fraction of cinema_peak */
    gail_gamut gamut;
    double gamut_threshold;   /* compression starts at this fraction of the gamut edge */
    /* GAIL_COLOR_LUT3D: lut3d_size^3 nodes of 3 floats (X', Y', Z' in 0..1 = code / 4095),
       red index fastest as in a .cube file; copied by gail_encoder_create() */
    const float* lut3d;
    int lut3d_size;
    uint64_t reserved[8];     /* zero */
} gail_color;

typedef enum gail_preset {
    GAIL_PRESET_SDR_BT709 = 0,
    GAIL_PRESET_SDR_BT2020 = 1,
    GAIL_PRESET_HDR_PQ_BT2020 = 2,  /* HDR10 */
    GAIL_PRESET_HDR_HLG_BT2020 = 3
} gail_preset;

/* Fills *c with the defaults for that kind of source; change any field afterwards. */
GAIL_API void gail_color_preset(gail_preset preset, gail_color* c);

/* One-line description of the colour handling. Returns the full length like snprintf,
   or a negative status for invalid settings. */
GAIL_API int gail_color_describe(const gail_color* c, char* buf, size_t size);

/* ---- encoder ----------------------------------------------------------------------- */

typedef enum gail_backend {
    GAIL_BACKEND_AUTO = 0, /* CUDA if NVIDIA GPUs are present, else Vulkan, else CPU */
    GAIL_BACKEND_CUDA = 1,
    GAIL_BACKEND_VULKAN = 2,
    GAIL_BACKEND_CPU = 3
} gail_backend;

typedef enum gail_log_level { GAIL_LOG_ERROR = 0, GAIL_LOG_WARNING = 1, GAIL_LOG_INFO = 2, GAIL_LOG_DEBUG = 3 } gail_log_level;

typedef struct gail_packet {
    uint8_t* data;          /* JPEG 2000 codestream (SOC ... EOC) */
    size_t size;
    int64_t index;          /* 0-based order in which the frame was handed over */
    void* opaque;           /* the frame's opaque pointer */
    int device;             /* which GPU / device encoded it */
    int compliant;          /* passed the built-in DCI codestream check (1 when checks are off) */
    double psnr_estimate;   /* dB on 12-bit X'Y'Z', from the rate control */
    uint64_t reserved[4];   /* set to zero */
} gail_packet;

GAIL_API void gail_packet_free(gail_packet* pkt);

typedef struct gail_frame {
    const void* data[3];    /* R'G'B' / X'Y'Z': data[0]. Y'CbCr: the Y, Cb and Cr planes */
    ptrdiff_t stride[3];    /* bytes per row; 0 = tightly packed */
    void* opaque;           /* returned in this frame's packet */
    uint64_t reserved[4];   /* zero */
} gail_frame;

typedef void (*gail_packet_cb)(void* user, const gail_packet* pkt); /* pkt->data valid during the call only */
typedef void (*gail_log_cb)(void* user, gail_log_level level, const char* message);

typedef struct gail_config {
    size_t struct_size;       /* sizeof(gail_config) in the program's header, set by gail_config_defaults() */
    gail_profile profile;
    gail_container container;
    double fps;               /* frame rate of the track (per eye for stereo) */
    double max_mbps;          /* max bitrate, Mbit/s (<= 250, the DCI limit; default 230) */
    int stereo;               /* 3D: frames alternate eyes, each gets half the frame budget */
    gail_input input;
    gail_color color;
    gail_backend backend;
    const int* devices;       /* device indices to use; NULL = all */
    int num_devices;
    int frames_in_flight;     /* per device (0 = default) */
    int check;                /* run the DCI codestream check on every frame (default 1) */
    gail_packet_cb on_packet; /* NULL: packets are collected for gail_encoder_receive() */
    void* user;
    gail_log_cb log;          /* NULL: no messages (the library never writes to stderr) */
    void* log_user;
    gail_log_level log_level;
    /* Devices to use by ID, instead of `devices`: each a UUID as nvidia-smi / gail_enumerate_devices
       show it ("GPU-6c7b...", the "GPU-" prefix and case optional) or a PCI bus ID ("0000:01:00.0").
       IDs stay the same when GPUs are added or drivers change; indices may not. With
       GAIL_BACKEND_AUTO, CUDA is used if it has every listed device, else Vulkan. */
    const char* const* device_ids;
    int num_device_ids;
    /* Device memory the frames in flight on one device may take, in bytes. 0 (default): as
       much as the driver reports free when the encoder starts, less a headroom for other
       users of the GPU (512 MiB or 6 %). Fewer frames than frames_in_flight are used when
       they would not fit (a 4K frame takes ~0.7 GB, a 2K frame ~0.2 GB), a frame the driver
       still refuses memory for is dropped rather than failing, and when another process
       later leaves less than half the headroom free, frames in flight are given up one at
       a time (one always stays). Reductions are reported at GAIL_LOG_WARNING. With
       GAIL_BACKEND_AUTO an encoder whose GPUs have no room for a single frame falls back to
       the CPU; with an explicit backend gail_encoder_create() fails with GAIL_ERROR_NOMEM. */
    uint64_t max_device_memory;
    /* CPU backend worker threads (chosen, auto fallback, or GAIL_BACKEND_CPU). 0 (default):
       every core this process may run on, i.e. its affinity mask rather than the machine's
       core count. */
    int cpu_threads;
    /* Non-zero: the encoder's threads run below normal priority (nice +10 on Linux, below
       normal on Windows): the CPU backend's workers, and the per-frame threads that do rate
       control and Tier-2 for the GPU backends. For encodes in the background of a desktop;
       throughput on an otherwise idle machine is unchanged. */
    int low_priority;
} gail_config;

/* Defaults: 2K flat, 24 fps, 230 Mbit/s, R'G'B' 16-bit BT.709 SDR input, all GPUs.
   gail_config_init() is what it calls: bindings from other languages call it with their
   size of the struct. */
GAIL_API void gail_config_init(gail_config* c, size_t struct_size);
GAIL_INLINE void gail_config_defaults(gail_config* c) { gail_config_init(c, sizeof(gail_config)); }

typedef struct gail_encoder gail_encoder;

/* Sets up the devices (this can take a moment: GPU programs are prepared here). */
GAIL_API int gail_encoder_create(const gail_config* c, gail_encoder** encoder);
GAIL_API void gail_encoder_destroy(gail_encoder* e);  /* waits for frames still in flight */

GAIL_API void gail_encoder_frame_size(const gail_encoder* e, int* width, int* height); /* container */
GAIL_API const char* gail_encoder_devices(const gail_encoder* e);  /* "NVIDIA ... (CUDA), ..." */

/* Hands over one frame (copied before returning; blocks while every device is busy).
   NULL ends the stream. */
GAIL_API int gail_encoder_send(gail_encoder* e, const gail_frame* frame);

/* Next codestream in send order. wait = 0 returns GAIL_EAGAIN when none is ready yet;
   wait = 1 blocks until one is. GAIL_EOF after the end of the stream once all are out.
   Only when config.on_packet is NULL. */
GAIL_API int gail_encoder_receive(gail_encoder* e, gail_packet* pkt, int wait);

/* Encodes one frame and returns its codestream. Thread-safe; calls from several threads
   are encoded concurrently. Frames encoded this way never appear in receive / on_packet. */
GAIL_API int gail_encoder_encode(gail_encoder* e, const gail_frame* frame, gail_packet* pkt);

/* Ends the stream (if not already) and waits until every frame has been delivered. */
GAIL_API int gail_encoder_finish(gail_encoder* e);

/* Stops encoding as soon as possible, e.g. for a Cancel button. Returns at once and may be
   called from any thread, including the on_packet and log callbacks. Frames not yet
   delivered are dropped: blocked and later send / receive / encode / finish calls return
   GAIL_ERROR_CANCELLED, on_packet may still see at most the calls already under way, and
   gail_encoder_destroy() then only waits for the frames the devices are working on. */
GAIL_API void gail_encoder_cancel(gail_encoder* e);

/* NULL while all is well, else what stopped the encoder. */
GAIL_API const char* gail_encoder_error(const gail_encoder* e);

typedef struct gail_device_info {
    gail_backend backend;   /* CUDA, VULKAN or CPU */
    int index;              /* index within that backend, as in gail_config.devices */
    char name[256];         /* "NVIDIA GeForce RTX 3090" */
    char uuid[48];          /* "GPU-6c7b3f1a-0d3e-6b5c-1f4e-9a8b7c6d5e4f" as nvidia-smi -L prints it
                               (the same GPU has the same UUID under CUDA and Vulkan); "" for the CPU */
    char pci_bus_id[32];    /* "0000:01:00.0", or "" when the driver does not say */
    uint64_t memory_bytes;  /* device memory */
    uint64_t reserved[8];   /* set to zero */
} gail_device_info;

/* Lists the devices a backend offers, without setting them up (cheap enough for a
   preferences dialog). GAIL_BACKEND_AUTO lists the backend an encoder with AUTO would use.
   Fills up to max_devices entries and returns how many there are (which may be more), or
   a negative status. */
GAIL_API int gail_enumerate_devices(gail_backend backend, gail_device_info* devices, int max_devices);

/* The same as text, one "backend index: name UUID (PCI bus ID)" per line. Returns the full
   length like snprintf, or a negative status. */
GAIL_API int gail_list_devices(gail_backend backend, char* buf, size_t size);

#ifdef __cplusplus
}
#endif

#endif
