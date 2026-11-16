/*
 * SPDX-License-Identifier: MIT OR LGPL-2.1-or-later
 * DCI JPEG 2000 decoding through GAIL (GPU, CUDA / Vulkan)
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
 * JPEG 2000 decoder for DCI picture tracks (SMPTE ST 428-1 / ST 429-4) using libgail.
 * Every packet is one codestream (2K / 4K, 12-bit X'Y'Z'); the output is xyz12, as from
 * FFmpeg's own JPEG 2000 decoder, decoded straight into FFmpeg's frame buffers. Several
 * frames are in flight on the GPUs at once, so frames come out a few packets after they
 * go in. -lowres 1 / 2 decodes at half / quarter size. JPEG 2000 that is not a DCI
 * stream (lossless, other bit depths, HTJ2K) is refused: use the jpeg2000 decoder for it.
 */

#include <gail/gail.h>

#include "libavutil/avstring.h"
#include "libavutil/common.h"
#include "libavutil/fifo.h"
#include "libavutil/frame.h"
#include "libavutil/mem.h"
#include "libavutil/opt.h"
#include "avcodec.h"
#include "codec_internal.h"
#include "decode.h"

typedef struct LibGailDecContext {
    const AVClass *class;
    gail_decoder *dec;
    AVPacket *pkt;
    AVFifo *frames;       /* AVFrame* of the codestreams in flight, in send order */
    int capacity;         /* how many to keep in flight before waiting for one */
    int eof_sent;
    int pending;          /* s->pkt was refused for now (GAIL_EAGAIN): send it again */
    int full;             /* ... after receiving a picture, which is then waited for */
    int configured;       /* dimensions known from the first codestream */

    int backend;
    char *devices;
    int frames_in_flight;
    int queue;
} LibGailDecContext;

static void gail_log(void *user, gail_log_level level, const char *msg)
{
    static const int levels[] = { AV_LOG_ERROR, AV_LOG_WARNING, AV_LOG_VERBOSE, AV_LOG_DEBUG };
    av_log(user, levels[level], "%s\n", msg);
}

static int gail_error(AVCodecContext *avctx, int status, const char *what)
{
    av_log(avctx, AV_LOG_ERROR, "%s: %s\n", what, gail_last_error());
    switch (status) {
    case GAIL_ERROR_INVALID:     return AVERROR_INVALIDDATA;
    case GAIL_ERROR_UNSUPPORTED: return AVERROR_PATCHWELCOME;
    case GAIL_ERROR_NOMEM:       return AVERROR(ENOMEM);
    default:                     return AVERROR_EXTERNAL;
    }
}

static av_cold int libgail_dec_init(AVCodecContext *avctx)
{
    LibGailDecContext *s = avctx->priv_data;
    gail_decoder_config cfg;
    int devices[16], num_devices = 0, ret;
    const char *ids[16];
    int num_ids = 0, numeric = 1;
    char *ids_buf = NULL;

    gail_decoder_config_defaults(&cfg);
    cfg.backend = s->backend;
    cfg.output = GAIL_PIX_XYZ12_MSB;
    cfg.reduce = avctx->lowres;
    if (s->devices && !strcmp(s->devices, "list")) {
        static const gail_backend backends[] = { GAIL_BACKEND_CUDA, GAIL_BACKEND_VULKAN, GAIL_BACKEND_CPU };
        char list[4096];
        for (int i = 0; i < FF_ARRAY_ELEMS(backends); i++)
            if (gail_list_devices(backends[i], list, sizeof(list)) > 0)
                av_log(avctx, AV_LOG_INFO, "%s", list);
        return AVERROR_EXIT;
    }
    if (s->devices && !strcmp(s->devices, "all")) {
        cfg.num_devices = GAIL_ALL_DEVICES;
    } else if (s->devices && *s->devices) {
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
    cfg.log = gail_log;
    cfg.log_user = avctx;
    cfg.log_level = GAIL_LOG_DEBUG;

    ret = gail_decoder_create(&cfg, &s->dec);
    av_freep(&ids_buf);
    if (ret != GAIL_OK)
        return gail_error(avctx, ret, "cannot create the Gail decoder");
    av_log(avctx, AV_LOG_INFO, "decoding on %s\n", gail_decoder_devices(s->dec));

    /* codestreams handed over before waiting for a picture: enough to keep every device
       busy, and more to ride out a slow frame; the library queues beyond what is decoding */
    s->capacity = s->queue;
    s->pkt = av_packet_alloc();
    s->frames = av_fifo_alloc2(32, sizeof(AVFrame *), AV_FIFO_FLAG_AUTO_GROW);
    if (!s->pkt || !s->frames)
        return AVERROR(ENOMEM);
    avctx->pix_fmt = AV_PIX_FMT_XYZ12;
    avctx->bits_per_raw_sample = 12;
    return 0;
}

/* What the stream is, from a codestream's header: the picture size and profile. */
static int configure(AVCodecContext *avctx, const AVPacket *pkt)
{
    LibGailDecContext *s = avctx->priv_data;
    gail_stream_info info;
    int ret = gail_probe(pkt->data, pkt->size, &info);

    if (ret != GAIL_OK)
        return gail_error(avctx, ret, "not a JPEG 2000 codestream");
    if (!info.supported) {
        av_log(avctx, AV_LOG_ERROR, "%s (the jpeg2000 decoder handles it)\n", gail_last_error());
        return AVERROR_PATCHWELCOME;
    }
    if (avctx->lowres > info.levels) {
        av_log(avctx, AV_LOG_ERROR, "lowres %d is more than the stream's %d resolution levels\n", avctx->lowres, info.levels);
        return AVERROR(EINVAL);
    }
    /* the full size: ff_set_dimensions() applies lowres itself, rounding up as the library does */
    ret = ff_set_dimensions(avctx, info.width, info.height);
    if (ret < 0)
        return ret;
    avctx->profile = info.rsiz;  /* AV_PROFILE_JPEG2000_DCINEMA_2K / _4K */
    s->configured = 1;
    return 0;
}

static void drop_frames(LibGailDecContext *s)
{
    AVFrame *frame;
    while (s->frames && av_fifo_read(s->frames, &frame, 1) >= 0)
        av_frame_free(&frame);
}

/* Hands one packet to the decoder, with the FFmpeg frame it is to be decoded into. */
static int send_packet(AVCodecContext *avctx, AVPacket *pkt)
{
    LibGailDecContext *s = avctx->priv_data;
    AVFrame *frame;
    gail_packet gp = { 0 };
    gail_image dest = { 0 };
    int ret;

    if (!s->configured && (ret = configure(avctx, pkt)) < 0)
        return ret;
    if (avctx->skip_frame >= AVDISCARD_ALL)
        return 0;
    if (!(frame = av_frame_alloc()))
        return AVERROR(ENOMEM);
    if ((ret = ff_decode_frame_props_from_pkt(avctx, frame, pkt)) < 0 ||
        (ret = ff_get_buffer(avctx, frame, 0)) < 0) {
        av_frame_free(&frame);
        return ret;
    }
    frame->pict_type = AV_PICTURE_TYPE_I;
    frame->flags |= AV_FRAME_FLAG_KEY;

    gp.data = pkt->data;
    gp.size = pkt->size;
    dest.data[0] = frame->data[0];
    dest.stride[0] = frame->linesize[0];
    ret = gail_decoder_send(s->dec, &gp, &dest); /* copies the codestream */
    if (ret == GAIL_EAGAIN) {
        /* as many pictures decoded as the library holds: receive one, then send this again */
        av_frame_free(&frame);
        return AVERROR(EAGAIN);
    }
    if (ret != GAIL_OK) {
        av_frame_free(&frame);
        return gail_error(avctx, ret, "cannot send a codestream");
    }
    if ((ret = av_fifo_write(s->frames, &frame, 1)) < 0) {
        /* out of memory for the list of frames in flight: drop them all, so that the
           decoder writes to none of their buffers any more */
        gail_decoder_flush(s->dec);
        av_frame_free(&frame);
        drop_frames(s);
        return ret;
    }
    return 0;
}

static int libgail_dec_receive_frame(AVCodecContext *avctx, AVFrame *out)
{
    LibGailDecContext *s = avctx->priv_data;
    int ret;

    for (;;) {
        gail_image img;
        /* wait for a picture once enough are in flight, or when no more input will come */
        const int wait = s->eof_sent || s->full || av_fifo_can_read(s->frames) >= s->capacity;

        ret = gail_decoder_receive(s->dec, &img, wait);
        if (ret == GAIL_OK) {
            AVFrame *frame = NULL;
            s->full = 0;
            av_fifo_read(s->frames, &frame, 1); /* pictures arrive in send order */
            if (img.status != GAIL_OK) {
                ret = gail_error(avctx, img.status, "cannot decode a codestream");
                av_frame_free(&frame);
                return ret;
            }
            av_frame_move_ref(out, frame);
            av_frame_free(&frame);
            return 0;
        }
        if (ret == GAIL_EOF)
            return AVERROR_EOF;
        if (ret != GAIL_EAGAIN)
            return gail_error(avctx, ret, "decoding failed");

        if (!s->pending) {
            ret = ff_decode_get_packet(avctx, s->pkt);
            if (ret == AVERROR_EOF) {
                gail_decoder_send(s->dec, NULL, NULL);
                s->eof_sent = 1;
                continue;
            }
            if (ret < 0)
                return ret; /* EAGAIN: need more input */
        }
        ret = send_packet(avctx, s->pkt);
        if (ret == AVERROR(EAGAIN)) {
            s->pending = s->full = 1; /* kept in s->pkt; a picture is ready to be received */
            continue;
        }
        s->pending = 0;
        av_packet_unref(s->pkt);
        if (ret < 0)
            return ret;
    }
}

/* A seek: what is in flight is dropped, and its frame buffers are no longer written to
   once gail_decoder_flush() has returned. */
static void libgail_dec_flush(AVCodecContext *avctx)
{
    LibGailDecContext *s = avctx->priv_data;
    gail_decoder_flush(s->dec);
    drop_frames(s);
    av_packet_unref(s->pkt);
    s->pending = s->full = 0;
    s->eof_sent = 0;
}

static av_cold int libgail_dec_close(AVCodecContext *avctx)
{
    LibGailDecContext *s = avctx->priv_data;
    gail_decoder_cancel(s->dec);
    gail_decoder_destroy(s->dec); /* waits for the frames the devices are working on */
    s->dec = NULL;
    drop_frames(s);
    av_fifo_freep2(&s->frames);
    av_packet_free(&s->pkt);
    return 0;
}

#define OFFSET(x) offsetof(LibGailDecContext, x)
#define VD AV_OPT_FLAG_VIDEO_PARAM | AV_OPT_FLAG_DECODING_PARAM
static const AVOption options[] = {
    { "backend", "compute backend", OFFSET(backend), AV_OPT_TYPE_INT, { .i64 = GAIL_BACKEND_AUTO }, 0, 3, VD, .unit = "backend" },
        { "auto",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_AUTO },   0, 0, VD, .unit = "backend" },
        { "cuda",   NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_CUDA },   0, 0, VD, .unit = "backend" },
        { "vulkan", NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_VULKAN }, 0, 0, VD, .unit = "backend" },
        { "cpu",    NULL, 0, AV_OPT_TYPE_CONST, { .i64 = GAIL_BACKEND_CPU },    0, 0, VD, .unit = "backend" },
    { "gpus", "devices: indices (0,1), UUIDs as nvidia-smi prints them, PCI bus IDs, \"all\" or \"list\" (default: the first)", OFFSET(devices), AV_OPT_TYPE_STRING, { .str = NULL }, 0, 0, VD },
    { "inflight", "frames in flight per device (0 = default)", OFFSET(frames_in_flight), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 16, VD },
    { "queue", "codestreams handed to the decoder ahead of the picture returned", OFFSET(queue), AV_OPT_TYPE_INT, { .i64 = 16 }, 1, 64, VD },
    { NULL },
};

static const AVClass libgail_dec_class = {
    .class_name = "libgail decoder",
    .item_name  = av_default_item_name,
    .option     = options,
    .version    = LIBAVUTIL_VERSION_INT,
};

const FFCodec ff_libgail_decoder = {
    .p.name         = "libgail",
    CODEC_LONG_NAME("Gail GPU JPEG 2000 (DCI)"),
    .p.type         = AVMEDIA_TYPE_VIDEO,
    .p.id           = AV_CODEC_ID_JPEG2000,
    .p.capabilities = AV_CODEC_CAP_DELAY | AV_CODEC_CAP_OTHER_THREADS | AV_CODEC_CAP_DR1 | AV_CODEC_CAP_HARDWARE,
    .p.max_lowres   = 5,
    .p.priv_class   = &libgail_dec_class,
    .p.wrapper_name = "libgail",
    .priv_data_size = sizeof(LibGailDecContext),
    .init           = libgail_dec_init,
    FF_CODEC_RECEIVE_FRAME_CB(libgail_dec_receive_frame),
    .flush          = libgail_dec_flush,
    .close          = libgail_dec_close,
    .caps_internal  = FF_CODEC_CAP_NOT_INIT_THREADSAFE | FF_CODEC_CAP_INIT_CLEANUP | FF_CODEC_CAP_SETS_FRAME_PROPS,
};
