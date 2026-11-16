// SPDX-License-Identifier: MIT
// The interface between libgail and its CUDA backend plugins.
//
// The CUDA backend is built once per CUDA toolkit, each time into a shared library of
// its own with that toolkit's runtime linked in statically:
//
//   gail_cuda13   CUDA 13   Turing and newer (compute capability 7.5 and up)
//   gail_cuda12   CUDA 12   Maxwell, Pascal and Volta (5.0 to 7.2), which CUDA 13 dropped
//
// libgail links no CUDA itself. It loads the plugins it finds in its own directory
// (cuda/cuda_host.cpp) and reaches each through the table below. A static libgail, which
// cannot carry plugins, links one backend in and reaches it through the same table.
//
// Only plain data crosses this boundary: no standard library types and no exceptions,
// so a plugin may be built by another compiler than the library (CUDA 12 does not accept
// host compilers as new as CUDA 13 does). A call that can fail returns a GailCudaStatus
// and leaves its message in last_error() for the calling thread. Memory a plugin hands
// out (names, output views) is the plugin's and stays valid until the next call on the
// same object, or until the object is destroyed.
//
// The interface is internal: it is not part of include/gail/gail.h, and a plugin is only
// ever used by the libgail it was built with. GAIL_CUDA_PLUGIN_ABI changes whenever
// anything in this file does, or any structure that passes through it.
#pragma once
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "core/backend.h"
#include "core/color.h"
#include "core/decode_backend.h"
#include "core/layout.h"
#include "core/t2_parse.h"
#include "core/tier1_types.h"

#define GAIL_CUDA_PLUGIN_ABI 1u
#define GAIL_CUDA_PLUGIN_ENTRY "gail_cuda_plugin_get"

// GAIL_CUDA_PLUGIN_BUILD is set where a plugin is built, which exports the entry point.
// The library only calls it: through the pointer it gets from the plugin it loaded, or
// directly in a static library, where the backend is linked in.
#if !defined(GAIL_CUDA_PLUGIN_BUILD)
#define GAIL_CUDA_PLUGIN_EXPORT
#elif defined(_WIN32)
#define GAIL_CUDA_PLUGIN_EXPORT __declspec(dllexport)
#else
#define GAIL_CUDA_PLUGIN_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

enum GailCudaStatus {
    GAIL_CUDA_OK = 0,
    GAIL_CUDA_ERROR = 1,          // std::runtime_error on the library's side
    GAIL_CUDA_DEVICE_MEMORY = 2,  // gail::DeviceOutOfMemory: the device stays usable
    GAIL_CUDA_HOST_MEMORY = 3,    // std::bad_alloc
    GAIL_CUDA_MISUSE = 4,         // std::logic_error
    // device_usable() only: the plugin has no code the device can run, and never will
    // (as opposed to GAIL_CUDA_ERROR: it cannot run it now, e.g. out of memory or busy).
    // A library that predates it takes it as GAIL_CUDA_ERROR, so it needs no new
    // GAIL_CUDA_PLUGIN_ABI.
    GAIL_CUDA_NO_CODE = 5
};

// level: gail::LogLevel as an int.
typedef void (*GailCudaLog)(void* user, int level, const char* message);

typedef struct GailCudaDevice GailCudaDevice;
typedef struct GailCudaEncoder GailCudaEncoder;
typedef struct GailCudaDecoder GailCudaDecoder;

struct GailCudaDeviceDesc {
    int32_t index;              // as this plugin's CUDA runtime numbers the device
    int32_t smMajor, smMinor;   // compute capability
    uint64_t memoryBytes;
    char name[256];
    char uuid[48];              // "GPU-xxxxxxxx-..." as nvidia-smi prints it
    char pciBusId[32];          // "0000:01:00.0"
};

// gail::FrameLayout. The plugin rebuilds the code-blocks and precincts from the geometry
// (makeGeometry) and takes the subbands, which carry the step sizes, as they are.
struct GailCudaLayout {
    int32_t profile, width, height, levels, guardBits;
    int32_t numBands, numCblks;
    double baseStep;
    const gail::Subband* bands;
};

// gail::ColorParams
struct GailCudaColor {
    int32_t transfer, primaries, toneMap, toneNorm, gamut, lut3dSize;
    double sdrWhite, hdrPeak, cinemaPeak, hdrRefWhite, cinemaRefWhite, knee, gamutThreshold;
    const char* lut3d;        // path of a .cube file, or ""
    const float* lut3dData;
    uint64_t lut3dCount;
};

// gail::EncodeSettings
struct GailCudaEncodeSettings {
    int32_t profile, width, height;
    double baseStep, slopeFactor, bitsPerSymbol;
    gail::FrameFormat input;
    GailCudaColor color;
};

// gail::DecodeSettings
struct GailCudaDecodeSettings {
    int32_t reduce, highBits;
};

// gail::DeviceMemory
struct GailCudaMemory {
    uint64_t total, available;
    int32_t budgeted;
};

// gail::Tier1Frame as the slot left it
struct GailCudaTier1 {
    const gail::CblkInfo* info;
    const uint32_t* passLen;
    const int32_t* passDist;
    const uint8_t* data;
    uint64_t numInfo, numPasses, numData;
};

// gail::FrameDebug. full is set by the caller; the views are filled in.
struct GailCudaDebug {
    int32_t full;
    const uint16_t* xyz12;
    const uint32_t* coeffs;
    const float* wavelet;
    uint64_t numXyz12, numCoeffs, numWavelet;
};

// gail::DecodeTimings. wantWavelet is set by the caller.
struct GailCudaDecodeTimings {
    double t1, dwt, color, total;
    int32_t wantWavelet;
    const float* wavelet[3];
    uint64_t numWavelet;  // samples in each plane
};

struct GailCudaPlugin {
    uint32_t abi;      // GAIL_CUDA_PLUGIN_ABI the plugin was built with
    uint32_t size;     // sizeof(GailCudaPlugin)
    uint64_t layout;   // gail::cudaPluginLayout() of the plugin's build
    const char* name;      // "gail_cuda13"
    const char* toolkit;   // CUDA toolkit it was built with, "13.4.92"
    const char* archs;     // architectures it holds code for, "75 86 89 120"

    // Message of the calling thread's most recent failed call (never NULL).
    const char* (*last_error)(void);

    // How many devices this plugin's CUDA runtime sees, or a negative number when the
    // runtime cannot start (no NVIDIA driver, or one too old for this toolkit).
    int (*device_count)(void);
    int (*device_desc)(int index, GailCudaDeviceDesc* out);
    // Whether the plugin's kernels can run on the device. This loads them with the
    // device current, as a launch would, so it also covers code the driver has to
    // compile from PTX. GAIL_CUDA_NO_CODE and the reason when the plugin has no code for
    // the device, GAIL_CUDA_ERROR and the reason when they cannot be loaded now (out of
    // memory, the device busy or in exclusive use by another process). It sets the
    // device up (a CUDA context), so the library calls it when it creates a device,
    // never when it only lists them.
    int (*device_usable)(int index);

    int (*device_create)(int index, GailCudaLog log, void* user, GailCudaDevice** out);
    void (*device_destroy)(GailCudaDevice* d);
    const char* (*device_name)(GailCudaDevice* d);
    int (*device_memory)(GailCudaDevice* d, GailCudaMemory* out);

    uint64_t (*encoder_bytes)(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaEncodeSettings* settings);
    int (*encoder_create)(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaEncodeSettings* settings,
                          GailCudaEncoder** out);
    void (*encoder_destroy)(GailCudaEncoder* e);
    void* (*encoder_input)(GailCudaEncoder* e);
    void (*encoder_set_slope)(GailCudaEncoder* e, double threshold);
    // timings and debug may be NULL.
    int (*encoder_encode)(GailCudaEncoder* e, const void* frame, GailCudaTier1* out, gail::FrameTimings* timings,
                          GailCudaDebug* debug);

    uint64_t (*decoder_bytes)(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaDecodeSettings* settings);
    int (*decoder_create)(GailCudaDevice* d, const GailCudaLayout* layout, const GailCudaDecodeSettings* settings,
                          GailCudaDecoder** out);
    void (*decoder_destroy)(GailCudaDecoder* d);
    // timings may be NULL.
    int (*decoder_decode)(GailCudaDecoder* d, const GailCudaLayout* frame, const uint8_t* data, size_t size,
                          const gail::CblkCode* blocks, size_t numBlocks, void* dst, ptrdiff_t strideBytes,
                          GailCudaDecodeTimings* timings);
};

// The one symbol a plugin exports. Returns its table, or NULL when the plugin was built
// for another GAIL_CUDA_PLUGIN_ABI than the caller's.
GAIL_CUDA_PLUGIN_EXPORT const GailCudaPlugin* gail_cuda_plugin_get(uint32_t abi);

}  // extern "C"

namespace gail {

// The library's structures that pass through the interface as they are.
static_assert(std::is_trivially_copyable_v<Subband> && std::is_trivially_copyable_v<CblkInfo> &&
              std::is_trivially_copyable_v<CblkCode> && std::is_trivially_copyable_v<FrameFormat> &&
              std::is_trivially_copyable_v<FrameTimings>);

// Their sizes folded into one number, which both sides compare: a plugin from a build
// whose structures differ is turned away even if GAIL_CUDA_PLUGIN_ABI was not raised.
constexpr uint64_t cudaPluginLayout() {
    uint64_t h = 1469598103934665603ull;
    for (uint64_t v : {(uint64_t)sizeof(Subband), (uint64_t)sizeof(CblkInfo), (uint64_t)sizeof(CblkCode),
                       (uint64_t)sizeof(FrameFormat), (uint64_t)sizeof(FrameTimings), (uint64_t)sizeof(GailCudaDeviceDesc),
                       (uint64_t)sizeof(GailCudaLayout), (uint64_t)sizeof(GailCudaEncodeSettings),
                       (uint64_t)sizeof(GailCudaTier1), (uint64_t)sizeof(GailCudaDebug),
                       (uint64_t)sizeof(GailCudaDecodeTimings), (uint64_t)sizeof(GailCudaPlugin)})
        h = (h ^ v) * 1099511628211ull;
    return h;
}

// ---- the library's structures to and from their plain forms ---------------------------
// Each conversion names every member of its structure in a structured binding, so adding
// a member stops the build here until the member is carried across (and the ABI raised).
// The plain forms point into the structure they were made from.

inline GailCudaLayout toPlugin(const FrameLayout& L) {
    [[maybe_unused]] const auto& [profile, width, height, levels, guardBits, baseStep, bands, cblks, resolutions] = L;
    return {(int32_t)profile, width, height, levels, guardBits, (int32_t)bands.size(), (int32_t)cblks.size(), baseStep, bands.data()};
}

// Fills `L` from the plain form. The geometry is only rebuilt when it differs from what
// L holds, so a slot that sees one geometry frame after frame just takes the subbands.
inline void fromPlugin(const GailCudaLayout& p, FrameLayout& L) {
    if (p.numBands < 0 || p.numCblks < 0 || !p.bands) throw std::logic_error("frame layout is empty");
    if ((int32_t)L.profile != p.profile || L.width != p.width || L.height != p.height || L.levels != p.levels ||
        L.guardBits != p.guardBits || (int32_t)L.bands.size() != p.numBands)
        L = makeGeometry(p.width, p.height, (Profile)p.profile, p.levels, p.guardBits);
    if ((int32_t)L.bands.size() != p.numBands || L.numCblks() != p.numCblks)
        throw std::logic_error("frame layout does not have the geometry of its size and levels");
    L.baseStep = p.baseStep;
    L.bands.assign(p.bands, p.bands + p.numBands);
}

inline GailCudaColor toPlugin(const ColorParams& c) {
    const auto& [transfer, primaries, toneMap, toneNorm, sdrWhite, hdrPeak, cinemaPeak, hdrRefWhite, cinemaRefWhite, knee, gamut,
                 gamutThreshold, lut3d, lut3dData, lut3dSize] = c;
    return {(int32_t)transfer, (int32_t)primaries, (int32_t)toneMap, (int32_t)toneNorm, (int32_t)gamut, lut3dSize,
            sdrWhite, hdrPeak, cinemaPeak, hdrRefWhite, cinemaRefWhite, knee, gamutThreshold,
            lut3d.c_str(), lut3dData.data(), (uint64_t)lut3dData.size()};
}

inline ColorParams fromPlugin(const GailCudaColor& p) {
    ColorParams c;
    auto& [transfer, primaries, toneMap, toneNorm, sdrWhite, hdrPeak, cinemaPeak, hdrRefWhite, cinemaRefWhite, knee, gamut,
           gamutThreshold, lut3d, lut3dData, lut3dSize] = c;
    transfer = (Transfer)p.transfer;
    primaries = (Primaries)p.primaries;
    toneMap = (ToneMap)p.toneMap;
    toneNorm = (ToneNorm)p.toneNorm;
    sdrWhite = p.sdrWhite;
    hdrPeak = p.hdrPeak;
    cinemaPeak = p.cinemaPeak;
    hdrRefWhite = p.hdrRefWhite;
    cinemaRefWhite = p.cinemaRefWhite;
    knee = p.knee;
    gamut = (TargetGamut)p.gamut;
    gamutThreshold = p.gamutThreshold;
    lut3d = p.lut3d ? p.lut3d : "";
    if (p.lut3dData) lut3dData.assign(p.lut3dData, p.lut3dData + p.lut3dCount);
    lut3dSize = p.lut3dSize;
    return c;
}

inline GailCudaEncodeSettings toPlugin(const EncodeSettings& s) {
    const auto& [profile, width, height, baseStep, color, input, slopeFactor, bitsPerSymbol] = s;
    return {(int32_t)profile, width, height, baseStep, slopeFactor, bitsPerSymbol, input, toPlugin(color)};
}

inline EncodeSettings fromPlugin(const GailCudaEncodeSettings& p) {
    EncodeSettings s;
    auto& [profile, width, height, baseStep, color, input, slopeFactor, bitsPerSymbol] = s;
    profile = (Profile)p.profile;
    width = p.width;
    height = p.height;
    baseStep = p.baseStep;
    color = fromPlugin(p.color);
    input = p.input;
    slopeFactor = p.slopeFactor;
    bitsPerSymbol = p.bitsPerSymbol;
    return s;
}

inline GailCudaDecodeSettings toPlugin(const DecodeSettings& s) {
    const auto& [reduce, highBits] = s;
    return {reduce, highBits ? 1 : 0};
}

inline DecodeSettings fromPlugin(const GailCudaDecodeSettings& p) {
    DecodeSettings s;
    auto& [reduce, highBits] = s;
    reduce = p.reduce;
    highBits = p.highBits != 0;
    return s;
}

}  // namespace gail
