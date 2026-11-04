/*
 * SPDX-License-Identifier: MIT OR LGPL-2.1-or-later
 * DCI JPEG 2000 encoding through GAIL (GPU, CUDA / Vulkan)
 * Copyright (c) 2026 Joseph Stackhouse
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * JPEG 2000 encoder for DCI picture tracks (SMPTE ST 428-1 / ST 429-4) using libgail.
 * Every packet is one complete, DCI-compliant codestream (2K / 4K, 12-bit X'Y'Z',
 * CPRL, 9/7, under the 250 Mbit/s frame limit). The picture is centred in the DCI
 * container chosen from its size (scale it first); R'G'B' / Y'CbCr input is converted
 * to X'Y'Z' on the GPU from the stream's colour tags, xyz12le is encoded as is.
 */

#include <gail/gail.h>

#include "libavutil/avstring.h"
#include "libavutil/fifo.h"
#include "libavutil/frame.h"
#include "libavutil/mastering_display_metadata.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "encode.h"

typedef struct LibGailContext {
    const AVClass *class;
    gail_encoder *enc;
    AVFrame *frame;
    AVFifo *timestamps;   /* pts / duration of frames in flight, in send order */
    int eof_sent;

    int profile;          /* -1 auto, else gail_profile */
    int container;        /* -1 auto, else gail_container */
    double mbps;
    int backend;
    char *devices;
    int frames_in_flight;
    int tone_map;
    int gamut;
    int check;
} LibGailContext;

typedef struct Timestamp {
    int64_t pts, duration;
} Timestamp;

static void gail_log(void *user, gail_log_level level, const char *msg)
{
    static const int levels[] = { AV_LOG_ERROR, AV_LOG_WARNING, AV_LOG_VERBOSE, AV_LOG_DEBUG };
    av_log(user, levels[level], "%s\n", msg);
}

static int gail_error(AVCodecContext *avctx, int status, const char *what)
{
    av_log(avctx, AV_LOG_ERROR, "%s: %s\n", what, gail_last_error());
    switch (status) {
    case GAIL_ERROR_INVALID:     return AVERROR(EINVAL);
    case GAIL_ERROR_UNSUPPORTED: return AVERROR(ENOSYS);
    case GAIL_ERROR_NOMEM:       return AVERROR(ENOMEM);
    default:                     return AVERROR_EXTERNAL;
    }
}

/* Smallest DCI container that holds the picture; scope for wide pictures. */
static void pick_container(AVCodecContext *avctx, LibGailContext *s, gail_profile *profile, gail_container *container)
{
    *profile = s->profile >= 0 ? s->profile
                               : (avctx->width > 2048 || avctx->height > 1080 ? GAIL_PROFILE_4K : GAIL_PROFILE_2K);
    if (s->container >= 0) {
        *container = s->container;
        return;
    }
    static const gail_container order_normal[] = { GAIL_CONTAINER_FLAT, GAIL_CONTAINER_SCOPE, GAIL_CONTAINER_FULL };
    static const gail_container order_wide[]   = { GAIL_CONTAINER_SCOPE, GAIL_CONTAINER_FLAT, GAIL_CONTAINER_FULL };
    const gail_container *order = avctx->width > 2.0 * avctx->height ? order_wide : order_normal;
    *container = GAIL_CONTAINER_FULL;
    for (int i = 0; i < 3; i++) {
        int w, h;
        gail_container_size(*profile, order[i], &w, &h);
        if (avctx->width <= w && avctx->height <= h) {
            *container = order[i];
            return;
        }
    }
}

static void map_colour(AVCodecContext *avctx, LibGailContext *s, gail_config *cfg)
{
    gail_color *c = &cfg->color;
    switch (avctx->color_trc) {
    case AVCOL_TRC_SMPTE2084:    gail_color_preset(GAIL_PRESET_HDR_PQ_BT2020, c); break;
    case AVCOL_TRC_ARIB_STD_B67: gail_color_preset(GAIL_PRESET_HDR_HLG_BT2020, c); break;
    default:                     gail_color_preset(GAIL_PRESET_SDR_BT709, c); break;
    }
    if (avctx->color_trc == AVCOL_TRC_IEC61966_2_1)
        c->transfer = GAIL_TRANSFER_SRGB;
    else if (avctx->color_trc == AVCOL_TRC_LINEAR)
        c->transfer = GAIL_TRANSFER_LINEAR;
    switch (avctx->color_primaries) {
    case AVCOL_PRI_BT2020:   c->primaries = GAIL_PRIMARIES_BT2020; break;
    case AVCOL_PRI_SMPTE432: c->primaries = GAIL_PRIMARIES_P3D65;  break;
    case AVCOL_PRI_SMPTE431: c->primaries = GAIL_PRIMARIES_DCIP3;  break;
    case AVCOL_PRI_BT709:    c->primaries = GAIL_PRIMARIES_BT709;  break;
    default: break; /* keep the preset's */
    }
    c->tone_map = s->tone_map;
    c->gamut = s->gamut;

    /* mastering peak, used by the tone-curve modes */
    const AVFrameSideData *sd = av_frame_side_data_get(avctx->decoded_side_data, avctx->nb_decoded_side_data,
                                                       AV_FRAME_DATA_MASTERING_DISPLAY_METADATA);
    if (sd) {
        const AVMasteringDisplayMetadata *m = (const AVMasteringDisplayMetadata *)sd->data;
        if (m->has_luminance && av_q2d(m->max_luminance) > 0)
            c->hdr_peak = av_q2d(m->max_luminance);
    }

    gail_input *in = &cfg->input;
    switch (avctx->colorspace) {
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:  in->matrix = GAIL_MATRIX_BT2020; break;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:  in->matrix = GAIL_MATRIX_BT601;  break;
    case AVCOL_SPC_BT709:      in->matrix = GAIL_MATRIX_BT709;  break;
    default:
        in->matrix = c->primaries == GAIL_PRIMARIES_BT2020 ? GAIL_MATRIX_BT2020 : GAIL_MATRIX_BT709;
        break;
    }
    in->full_range = avctx->color_range == AVCOL_RANGE_JPEG;
    switch (avctx->chroma_sample_location) {
    case AVCHROMA_LOC_CENTER:  in->chroma_location = GAIL_CHROMA_CENTER;  break;
    case AVCHROMA_LOC_TOPLEFT: in->chroma_location = GAIL_CHROMA_TOPLEFT; break;
    default:                   in->chroma_location = GAIL_CHROMA_LEFT;    break;
    }
}

static av_cold int libgail_init(AVCodecContext *avctx)
{
    LibGailContext *s = avctx->priv_data;
    gail_config cfg;
    char desc[256];
    int devices[16], num_devices = 0, ret;
    const char *ids[16];
    int num_ids = 0, numeric = 1;
    char *ids_buf = NULL;

    gail_config_defaults(&cfg);
    gail_profile profile;
    gail_container container;
    pick_container(avctx, s, &profile, &container);
    cfg.profile = profile;
    cfg.container = container;

    if (avctx->framerate.num > 0 && avctx->framerate.den > 0)
        cfg.fps = av_q2d(avctx->framerate);
    else
        cfg.fps = 1.0 / av_q2d(avctx->time_base);
    cfg.max_mbps = avctx->bit_rate > 0 ? avctx->bit_rate / 1e6 : s->mbps;

    switch (avctx->pix_fmt) {
    case AV_PIX_FMT_XYZ12:      cfg.input.format = GAIL_PIX_XYZ12_MSB; break;
    case AV_PIX_FMT_RGB48:      cfg.input.format = GAIL_PIX_RGB48;     break;
    case AV_PIX_FMT_YUV420P:    cfg.input.format = GAIL_PIX_YUV420P;   break;
    case AV_PIX_FMT_YUV420P10:  cfg.input.format = GAIL_PIX_YUV420P10; break;
    case AV_PIX_FMT_YUV420P12:  cfg.input.format = GAIL_PIX_YUV420P12; break;
    case AV_PIX_FMT_YUV422P:    cfg.input.format = GAIL_PIX_YUV422P;   break;
    case AV_PIX_FMT_YUV422P10:  cfg.input.format = GAIL_PIX_YUV422P10; break;
    case AV_PIX_FMT_YUV422P12:  cfg.input.format = GAIL_PIX_YUV422P12; break;
    case AV_PIX_FMT_YUV444P:    cfg.input.format = GAIL_PIX_YUV444P;   break;
    case AV_PIX_FMT_YUV444P10:  cfg.input.format = GAIL_PIX_YUV444P10; break;
    case AV_PIX_FMT_YUV444P12:  cfg.input.format = GAIL_PIX_YUV444P12; break;
    case AV_PIX_FMT_GBRP:       cfg.input.format = GAIL_PIX_GBRP;      break;
    case AV_PIX_FMT_GBRP10:     cfg.input.format = GAIL_PIX_GBRP10;    break;
    case AV_PIX_FMT_GBRP12:     cfg.input.format = GAIL_PIX_GBRP12;    break;
    case AV_PIX_FMT_GBRP16:     cfg.input.format = GAIL_PIX_GBRP16;    break;
    default: return AVERROR(EINVAL);
    }
    cfg.input.width = avctx->width;
    cfg.input.height = avctx->height;
    map_colour(avctx, s, &cfg);

    cfg.backend = s->backend;
    if (s->devices && !strcmp(s->devices, "list")) {
        static const gail_backend backends[] = { GAIL_BACKEND_CUDA, GAIL_BACKEND_VULKAN, GAIL_BACKEND_CPU };
        char list[4096];
        for (int i = 0; i < FF_ARRAY_ELEMS(backends); i++)
            if (gail_list_devices(backends[i], list, sizeof(list)) > 0)
                av_log(avctx, AV_LOG_INFO, "%s", list);
        return AVERROR_EXIT;
    }
    if (s->devices && *s->devices) {
        /* indices ("0,1") or IDs: UUIDs as nvidia-smi prints them, or PCI bus IDs */
        char *save = NULL, *tok;
        if (!(ids_buf = av_strdup(s->devices)))
            return AVERROR(ENOMEM);
        for (tok = av_strtok(ids_buf, ",", &save); tok && num_ids < FF_ARRAY_ELEMS(ids);
             tok = av_strtok(NULL, ",", &save)) {
            ids[num_ids++] = tok;
            numeric &= tok[strspn(tok, "0123456789")] == '\0';
        }
        if (numeric) {
            for (int i = 0; i < num_ids; i++)
                devices[num_devices++] = atoi(ids[i]);
            cfg.devices = devices;
            cfg.num_devices = num_devices;
        } else {
            cfg.device_ids = ids;
            cfg.num_device_ids = num_ids;
        }
    }
    cfg.frames_in_flight = s->frames_in_flight;
    cfg.check = s->check;
    cfg.log = gail_log;
    cfg.log_user = avctx;
    cfg.log_level = GAIL_LOG_DEBUG;

    ret = gail_encoder_create(&cfg, &s->enc);
    av_freep(&ids_buf);
    if (ret != GAIL_OK)
        return gail_error(avctx, ret, "cannot create the Gail encoder");

    int cw, ch;
    gail_encoder_frame_size(s->enc, &cw, &ch);
    if (cfg.input.format != GAIL_PIX_XYZ12_MSB && gail_color_describe(&cfg.color, desc, sizeof(desc)) > 0)
        av_log(avctx, AV_LOG_INFO, "colour: %s\n", desc);
    av_log(avctx, AV_LOG_INFO, "%s %dx%d in a %dx%d container, max %.0f Mbit/s, on %s\n",
           profile == GAIL_PROFILE_4K ? "4K" : "2K", avctx->width, avctx->height, cw, ch, cfg.max_mbps,
           gail_encoder_devices(s->enc));

    s->frame = av_frame_alloc();
    s->timestamps = av_fifo_alloc2(64, sizeof(Timestamp), AV_FIFO_FLAG_AUTO_GROW);
    if (!s->frame || !s->timestamps)
        return AVERROR(ENOMEM);
    avctx->bit_rate = cfg.max_mbps * 1e6;
    return 0;
}

static int output_packet(AVCodecContext *avctx, AVPacket *pkt, gail_packet *gp)
{
    LibGailContext *s = avctx->priv_data;
    Timestamp ts = { AV_NOPTS_VALUE, 0 };
    int ret = av_new_packet(pkt, gp->size);

    if (ret >= 0) {
        memcpy(pkt->data, gp->data, gp->size);
        av_fifo_read(s->timestamps, &ts, 1);
        pkt->pts = pkt->dts = ts.pts;
        pkt->duration = ts.duration;
        pkt->flags |= AV_PKT_FLAG_KEY;
        if (!gp->compliant)
            av_log(avctx, AV_LOG_WARNING, "frame %"PRId64" failed the DCI codestream check\n", gp->index);
    }
    gail_packet_free(gp);
    return ret;
}

static int libgail_receive_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    LibGailContext *s = avctx->priv_data;
    gail_packet gp;
    int ret;

    for (;;) {
        ret = gail_encoder_receive(s->enc, &gp, s->eof_sent);
        if (ret == GAIL_OK)
            return output_packet(avctx, pkt, &gp);
        if (ret == GAIL_EOF)
            return AVERROR_EOF;
        if (ret != GAIL_EAGAIN)
            return gail_error(avctx, ret, "encoding failed");

        ret = ff_encode_get_frame(avctx, s->frame);
        if (ret == AVERROR_EOF) {
            if ((ret = gail_encoder_send(s->enc, NULL)) != GAIL_OK)
                return gail_error(avctx, ret, "cannot end the stream");
            s->eof_sent = 1;
            continue;
        }
        if (ret < 0)
            return ret; /* EAGAIN: need more input */

        gail_frame f = { { s->frame->data[0], s->frame->data[1], s->frame->data[2] },
                         { s->frame->linesize[0], s->frame->linesize[1], s->frame->linesize[2] },
                         NULL };
        Timestamp ts = { s->frame->pts, s->frame->duration };
        av_fifo_write(s->timestamps, &ts, 1);
        ret = gail_encoder_send(s->enc, &f); /* copies the frame */
        av_frame_unref(s->frame);
        if (ret != GAIL_OK)
            return gail_error(avctx, ret, "cannot send a frame");
    }
}

static av_cold int libgail_close(AVCodecContext *avctx)
{
    LibGailContext *s = avctx->priv_data;
    gail_encoder_cancel(s->enc); /* frames never received (an aborted run) are not waited for */
    gail_encoder_destroy(s->enc);
    s->enc = NULL;
    av_frame_free(&s->frame);
    av_fifo_freep2(&s->timestamps);
    return 0;
}

#define OFFSET(x) offsetof(LibGailContext, x)
#define VE AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_ENCODING_PARAM
static const AVOption options[] = {
    { "dci_profile", "DCI profile", OFFSET(profile), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 1, VE, .unit = "profile" },
        { "auto", "from the picture size", 0, AV_OPT_TYPE_CONST, { .i64 = -1 },               0, 0, VE, .unit = "profile" },
        { "2k",   NULL,                    0, AV_OPT_TYPE_CONST, { .i64 = GAIL_PROFILE_2K }, 0, 0, VE, .unit = "profile" },
        { "4k",   NULL,                    0, AV_OPT_TYPE_CONST, { .i64 = GAIL_PROFILE_4K }, 0, 0, VE, .unit = "profile" },
    { "container", "DCI container the picture is centred in", OFFSET(container), AV_OPT_TYPE_INT, { .i64 = -1 }, -1, 2, VE, .unit = "container" },
        { "auto",  "smallest that fits", 0, AV_OPT_TYPE_CONST, { .i64 = -1 },                  0, 0, VE, .unit = "container" },
        { "flat",  "1.85:1",             0, AV_OPT_TYPE_CONST, { .i64 = GAIL_CONTAINER_FLAT },  0, 0, VE, .unit = "container" },
        { "scope", "2.39:1",             0, AV_OPT_TYPE_CONST, { .i64 = GAIL_CONTAINER_SCOPE }, 0, 0, VE, .unit = "container" },
        { "full",  "1.90:1",             0, AV_OPT_TYPE_CONST, { .i64 = GAIL_CONTAINER_FULL },  0, 0, VE, .unit = "container" },
    { "mbps", "max bitrate in Mbit/s when -b:v is not set (DCI limit 250)", OFFSET(mbps), AV_OPT_TYPE_DOUBLE, { .dbl = 230 }, 1, 250, VE },
    { "backend", "compute backend", OFFSET(backend), AV_OPT_TYPE_INT, { .i64 = GAIL_BACKEND_AUTO }, 0, 3, VE, .unit = "backend" },
        { "auto",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_AUTO },   0, 0, VE, .unit = "backend" },
        { "cuda",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_CUDA },   0, 0, VE, .unit = "backend" },
        { "vulkan", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_VULKAN }, 0, 0, VE, .unit = "backend" },
        { "cpu",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_CPU },    0, 0, VE, .unit = "backend" },
    { "gpus", "devices: indices (0,1), UUIDs as nvidia-smi prints them, PCI bus IDs, or \"list\" (default: all)", OFFSET(devices), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VE },
    { "inflight", "frames in flight per device (0 = default)", OFFSET(frames_in_flight), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 16, VE },
    { "tonemap", "HDR (PQ / HLG) handling", OFFSET(tone_map), AV_OPT_TYPE_INT, { .i64 = GAIL_TONE_MAP_SIGNAL }, 0, 3, VE, .unit = "tonemap" },
        { "signal",   "as DCP-o-matic: the HDR signal read as gamma 2.4", 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_TONE_MAP_SIGNAL },   0, 0, VE, .unit = "tonemap" },
        { "shoulder", "scene-relative tone curve",                         0, AV_OPT_TYPE_CONST, { .i64 = GAIL_TONE_MAP_SHOULDER }, 0, 0, VE, .unit = "tonemap" },
        { "bt2390",   "ITU-R BT.2390 EETF",                                0, AV_OPT_TYPE_CONST, { .i64 = GAIL_TONE_MAP_BT2390 },   0, 0, VE, .unit = "tonemap" },
        { "none",     "absolute luminance, clipped",                       0, AV_OPT_TYPE_CONST, { .i64 = GAIL_TONE_MAP_NONE },     0, 0, VE, .unit = "tonemap" },
    { "gamut", "compress colours into the projector gamut", OFFSET(gamut), AV_OPT_TYPE_INT, { .i64 = GAIL_GAMUT_NONE }, 0, 2, VE, .unit = "gamut" },
        { "none",  NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_GAMUT_NONE },  0, 0, VE, .unit = "gamut" },
        { "p3d65", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_GAMUT_P3D65 }, 0, 0, VE, .unit = "gamut" },
        { "dcip3", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_GAMUT_DCIP3 }, 0, 0, VE, .unit = "gamut" },
    { "dci_check", "verify every codestream against the DCI rules", OFFSET(check), AV_OPT_TYPE_BOOL, { .i64 = 1 }, 0, 1, VE },
    { NULL },
};

static const AVClass libgail_class = {
    .class_name = "libgail",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

static const FFCodecDefault libgail_defaults[] = {
    { "b", "0" },  /* 0: use -mbps (230); FFmpeg's generic default would starve the frames */
    { NULL },
};

const FFCodec ff_libgail_encoder = {
    .p.name         = "libgail",
    CODEC_LONG_NAME("Gail GPU JPEG 2000 (DCI)"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_JPEG2000,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_OTHER_THREADS | AV_CODEC_CAP_HARDWARE,
    .p.priv_class   = &libgail_class,
    .p.wrapper_name = "libgail",
    .priv_data_size = sizeof(LibGailContext),
    .init           = libgail_init,
    FF_CODEC_RECEIVE_PACKET_CB(libgail_receive_packet),
    .close          = libgail_close,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP,
    /* FFmpeg keeps a source in its own layout when it is listed here (ProRes 422 stays
       4:2:2, DPX stays planar R'G'B'), and otherwise converts to the closest one */
    CODEC_PIXFMTS(AV_PIX_FMT_YUV420P10, AV_PIX_FMT_YUV420P12, AV_PIX_FMT_YUV420P,
                  AV_PIX_FMT_RGB48, AV_PIX_FMT_XYZ12,
                  AV_PIX_FMT_YUV422P10, AV_PIX_FMT_YUV422P12, AV_PIX_FMT_YUV422P,
                  AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P12, AV_PIX_FMT_YUV444P,
                  AV_PIX_FMT_GBRP10, AV_PIX_FMT_GBRP12, AV_PIX_FMT_GBRP16, AV_PIX_FMT_GBRP),
    .color_ranges   = AVCOL_RANGE_MPEG | AVCOL_RANGE_JPEG,
    .defaults       = libgail_defaults,
};
