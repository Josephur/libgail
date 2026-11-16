// SPDX-License-Identifier: MIT
// The CUDA backend behind the plugin interface (cuda/cuda_plugin.h). This file is the
// whole of what a plugin adds to the backend: it turns the plain data of the interface
// back into the library's structures, calls the backend, and turns exceptions into
// status codes. It is built into each plugin, and into a static libgail next to the one
// backend that links in.
#include "cuda/cuda_plugin.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda/cuda_backend.h"

#ifndef GAIL_CUDA_PLUGIN_NAME
#error "GAIL_CUDA_PLUGIN_NAME, GAIL_CUDA_PLUGIN_TOOLKIT and GAIL_CUDA_PLUGIN_ARCHS come from the build"
#endif

using namespace gail;

// The objects behind the interface's handles.
struct GailCudaDevice {
    std::unique_ptr<Device> device;
    std::string name;
};

struct GailCudaEncoder {
    FrameLayout layout;  // the slot keeps a reference to it
    std::unique_ptr<EncoderSlot> slot;
    CudaEncoderSlot* cuda = nullptr;
    FrameDebug debug;
};

struct GailCudaDecoder {
    FrameLayout layout, frame;
    std::unique_ptr<DecoderSlot> slot;
    std::vector<CblkCode> blocks;
    std::vector<float> wavelet[3];
};

namespace {

thread_local std::string g_error = "no error";

// Runs fn; an exception becomes the status the library turns back into that exception.
template <class F>
int guarded(F&& fn) noexcept {
    try {
        fn();
        return GAIL_CUDA_OK;
    } catch (const DeviceOutOfMemory& e) {
        g_error = e.what();
        return GAIL_CUDA_DEVICE_MEMORY;
    } catch (const std::bad_alloc&) {
        g_error = "out of memory";
        return GAIL_CUDA_HOST_MEMORY;
    } catch (const std::logic_error& e) {
        g_error = e.what();
        return GAIL_CUDA_MISUSE;
    } catch (const std::exception& e) {
        g_error = e.what();
        return GAIL_CUDA_ERROR;
    } catch (...) {
        g_error = "unknown error";
        return GAIL_CUDA_ERROR;
    }
}

const char* lastError() { return g_error.c_str(); }

void copyText(char* dst, size_t size, const std::string& s) { std::snprintf(dst, size, "%s", s.c_str()); }

// The device list of the first successful enumeration: the runtime's numbering does not
// change while the process lives.
const std::vector<CudaDeviceDesc>& devices() {
    static const std::vector<CudaDeviceDesc> list = cudaEnumerate();
    return list;
}

int deviceCount() {
    int n = -1;
    guarded([&] { n = (int)devices().size(); });
    return n;
}

int deviceDesc(int index, GailCudaDeviceDesc* out) {
    return guarded([&] {
        const CudaDeviceDesc& d = devices().at((size_t)index);
        *out = {};
        out->index = d.info.index;
        out->smMajor = d.smMajor;
        out->smMinor = d.smMinor;
        out->memoryBytes = d.info.memoryBytes;
        copyText(out->name, sizeof out->name, d.info.name);
        copyText(out->uuid, sizeof out->uuid, d.info.uuid);
        copyText(out->pciBusId, sizeof out->pciBusId, d.info.pciBusId);
    });
}

int deviceUsable(int index) {
    return guarded([&] {
        const std::string why = cudaKernelsUnusable(devices().at((size_t)index).info.index);
        if (!why.empty()) throw std::runtime_error(why);
    });
}

int deviceCreate(int index, GailCudaLog log, void* user, GailCudaDevice** out) {
    return guarded([&] {
        Logger logger;
        if (log) logger = [log, user](LogLevel level, const std::string& m) { log(user, (int)level, m.c_str()); };
        auto d = std::make_unique<GailCudaDevice>();
        d->device = createCudaDevice(devices().at((size_t)index).info.index, logger);
        d->name = d->device->name();
        *out = d.release();
    });
}

void deviceDestroy(GailCudaDevice* d) { delete d; }

const char* deviceName(GailCudaDevice* d) { return d->name.c_str(); }

int deviceMemory(GailCudaDevice* d, GailCudaMemory* out) {
    return guarded([&] {
        const DeviceMemory m = d->device->memory();
        const auto& [total, available, budgeted] = m;
        *out = {total, available, budgeted ? 1 : 0};
    });
}

uint64_t encoderBytes(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaEncodeSettings* settings) {
    uint64_t bytes = 0;
    guarded([&] {
        FrameLayout L;
        fromPlugin(*layout, L);
        bytes = d->device->slotBytes(L, fromPlugin(*settings));
    });
    return bytes;
}

int encoderCreate(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaEncodeSettings* settings, GailCudaEncoder** out) {
    return guarded([&] {
        auto e = std::make_unique<GailCudaEncoder>();
        fromPlugin(*layout, e->layout);
        e->slot = d->device->createSlot(e->layout, fromPlugin(*settings));
        e->cuda = dynamic_cast<CudaEncoderSlot*>(e->slot.get());
        if (!e->cuda) throw std::logic_error("not a CUDA slot");
        *out = e.release();
    });
}

void encoderDestroy(GailCudaEncoder* e) { delete e; }

void* encoderInput(GailCudaEncoder* e) { return e->slot->inputBuffer(); }

void encoderSetSlope(GailCudaEncoder* e, double threshold) { e->slot->setSlopeThreshold(threshold); }

int encoderEncode(GailCudaEncoder* e, const void* frame, GailCudaTier1* out, FrameTimings* timings, GailCudaDebug* debug) {
    return guarded([&] {
        auto& [full, xyz12, coeffs, wavelet] = e->debug;
        if (debug) full = debug->full != 0;
        const Tier1View v = e->cuda->encodeView(frame, timings, debug ? &e->debug : nullptr);
        *out = {v.info, v.passLen, v.passDist, v.data, v.numInfo, v.numPasses, v.numData};
        if (debug) {
            // only what this frame filled: the vectors may hold an earlier frame's taps
            const bool all = debug->full != 0;
            *debug = {debug->full, xyz12.data(), all ? coeffs.data() : nullptr, all ? wavelet.data() : nullptr,
                      xyz12.size(), all ? coeffs.size() : 0, all ? wavelet.size() : 0};
        }
    });
}

uint64_t decoderBytes(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaDecodeSettings* settings) {
    uint64_t bytes = 0;
    guarded([&] {
        FrameLayout L;
        fromPlugin(*layout, L);
        bytes = d->device->decoderSlotBytes(L, fromPlugin(*settings));
    });
    return bytes;
}

int decoderCreate(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaDecodeSettings* settings, GailCudaDecoder** out) {
    return guarded([&] {
        auto dec = std::make_unique<GailCudaDecoder>();
        fromPlugin(*layout, dec->layout);
        dec->frame = dec->layout;
        dec->slot = d->device->createDecoderSlot(dec->layout, fromPlugin(*settings));
        *out = dec.release();
    });
}

void decoderDestroy(GailCudaDecoder* d) { delete d; }

int decoderDecode(GailCudaDecoder* d, const GailCudaLayout* frame, const uint8_t* data, size_t size, const CblkCode* blocks,
                  size_t numBlocks, void* dst, ptrdiff_t strideBytes, GailCudaDecodeTimings* timings) {
    return guarded([&] {
        fromPlugin(*frame, d->frame);
        d->blocks.assign(blocks, blocks + numBlocks);
        DecodeTimings tm;
        auto& [t1, dwt, color, total, wavelet] = tm;
        if (timings && timings->wantWavelet) wavelet = d->wavelet;
        d->slot->decode(d->frame, data, size, d->blocks, dst, strideBytes, timings ? &tm : nullptr);
        if (timings) {
            timings->t1 = t1;
            timings->dwt = dwt;
            timings->color = color;
            timings->total = total;
            timings->numWavelet = wavelet ? d->wavelet[0].size() : 0;
            for (int c = 0; c < 3; ++c) timings->wavelet[c] = wavelet ? d->wavelet[c].data() : nullptr;
        }
    });
}

const GailCudaPlugin kPlugin = {
    GAIL_CUDA_PLUGIN_ABI,
    (uint32_t)sizeof(GailCudaPlugin),
    cudaPluginLayout(),
    GAIL_CUDA_PLUGIN_NAME,
    GAIL_CUDA_PLUGIN_TOOLKIT,
    GAIL_CUDA_PLUGIN_ARCHS,
    lastError,
    deviceCount,
    deviceDesc,
    deviceUsable,
    deviceCreate,
    deviceDestroy,
    deviceName,
    deviceMemory,
    encoderBytes,
    encoderCreate,
    encoderDestroy,
    encoderInput,
    encoderSetSlope,
    encoderEncode,
    decoderBytes,
    decoderCreate,
    decoderDestroy,
    decoderDecode,
};

}  // namespace

extern "C" const GailCudaPlugin* gail_cuda_plugin_get(uint32_t abi) { return abi == GAIL_CUDA_PLUGIN_ABI ? &kPlugin : nullptr; }
