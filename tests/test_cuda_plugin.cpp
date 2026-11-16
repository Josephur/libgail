// SPDX-License-Identifier: MIT
// Stand-in for a CUDA backend plugin, for the loader's tests (test "plugins" in
// test_main.cpp): it has the entry point and the table of a real plugin and a made-up
// list of GPUs, and needs neither a GPU nor a CUDA runtime. The build compiles it several
// times, each under the name of a real plugin in a directory of its own:
//
//   TEST_PLUGIN_NAME   "gail_cuda13" or "gail_cuda12": which plugin it stands in for
//   TEST_PLUGIN_MODE   0  a working plugin
//                      1  a plugin built for another interface version, which hands out
//                         its table all the same
//                      2  a plugin that turns the library's interface version down
//                      3  a plugin whose CUDA runtime does not start (no driver)
#define GAIL_CUDA_PLUGIN_BUILD 1  // exports the entry point, as a plugin's build does
#include "cuda/cuda_plugin.h"

#include <cstdio>
#include <cstring>
#include <string>

namespace {

struct Gpu {
    const char* name;
    int major, minor;
};

// Both stand-ins see the same five GPUs, as two CUDA runtimes on one machine would, but
// number them differently, so the tests also cover matching them up by UUID.
const bool kIs13 = std::strcmp(TEST_PLUGIN_NAME, "gail_cuda13") == 0;
const Gpu kGpus13[] = {{"Test Ampere", 8, 6}, {"Test Maxwell", 5, 2}, {"Test Turing", 7, 5}, {"Test Volta", 7, 0}, {"Test Kepler", 3, 5}};
const Gpu kGpus12[] = {{"Test Maxwell", 5, 2}, {"Test Turing", 7, 5}, {"Test Volta", 7, 0}, {"Test Kepler", 3, 5}, {"Test Ampere", 8, 6}};
const Gpu* const kGpus = kIs13 ? kGpus13 : kGpus12;
const int kNumGpus = 5;

thread_local std::string g_error = "no error";
std::string g_names[kNumGpus];

const char* lastError() { return g_error.c_str(); }

int deviceCount() {
    if (TEST_PLUGIN_MODE == 3) {
        g_error = "no NVIDIA driver";
        return -1;
    }
    return kNumGpus;
}

int deviceDesc(int index, GailCudaDeviceDesc* out) {
    if (index < 0 || index >= kNumGpus) return GAIL_CUDA_ERROR;
    const Gpu& g = kGpus[index];
    *out = {};
    out->index = index;
    out->smMajor = g.major;
    out->smMinor = g.minor;
    out->memoryBytes = 8ull << 30;
    std::snprintf(out->name, sizeof out->name, "%s", g.name);
    std::snprintf(out->uuid, sizeof out->uuid, "GPU-00000000-0000-0000-0000-0000000000%d%d", g.major, g.minor);
    std::snprintf(out->pciBusId, sizeof out->pciBusId, "0000:%d%d:00.0", g.major, g.minor);
    return GAIL_CUDA_OK;
}

// What the real plugins hold code for: the CUDA 13 one for Turing and newer; the CUDA 12
// one for Maxwell to Volta, whose Volta code a Turing GPU could run as well.
int deviceUsable(int index) {
    const int sm = kGpus[index].major * 10 + kGpus[index].minor;
    if (kIs13 ? sm >= 75 : (sm >= 50 && sm <= 75)) return GAIL_CUDA_OK;
    g_error = "not built for this architecture";
    return GAIL_CUDA_ERROR;
}

// A device is nothing but its name here, which says which stand-in made it.
int deviceCreate(int index, GailCudaLog log, void* user, GailCudaDevice** out) {
    g_names[index] = std::string(kGpus[index].name) + " (CUDA) on " TEST_PLUGIN_NAME;
    if (log) log(user, (int)gail::LogLevel::Info, ("cuda: device " + std::to_string(index) + " " + kGpus[index].name).c_str());
    *out = reinterpret_cast<GailCudaDevice*>(&g_names[index]);
    return GAIL_CUDA_OK;
}

void deviceDestroy(GailCudaDevice*) {}

const char* deviceName(GailCudaDevice* d) { return reinterpret_cast<std::string*>(d)->c_str(); }

int deviceMemory(GailCudaDevice*, GailCudaMemory* out) {
    *out = {};
    return GAIL_CUDA_OK;
}

// Nothing is encoded or decoded by a stand-in.
int unsupported() {
    g_error = "a test plugin has no kernels";
    return GAIL_CUDA_ERROR;
}
uint64_t encoderBytes(GailCudaDevice*, const GailCudaLayout*, const GailCudaEncodeSettings*) { return 0; }
int encoderCreate(GailCudaDevice*, const GailCudaLayout*, const GailCudaEncodeSettings*, GailCudaEncoder**) { return unsupported(); }
void encoderDestroy(GailCudaEncoder*) {}
void* encoderInput(GailCudaEncoder*) { return nullptr; }
void encoderSetSlope(GailCudaEncoder*, double) {}
int encoderEncode(GailCudaEncoder*, const void*, GailCudaTier1*, gail::FrameTimings*, GailCudaDebug*) { return unsupported(); }
uint64_t decoderBytes(GailCudaDevice*, const GailCudaLayout*, const GailCudaDecodeSettings*) { return 0; }
int decoderCreate(GailCudaDevice*, const GailCudaLayout*, const GailCudaDecodeSettings*, GailCudaDecoder**) { return unsupported(); }
void decoderDestroy(GailCudaDecoder*) {}
int decoderDecode(GailCudaDecoder*, const GailCudaLayout*, const uint8_t*, size_t, const gail::CblkCode*, size_t, void*, ptrdiff_t,
                  GailCudaDecodeTimings*) {
    return unsupported();
}

const GailCudaPlugin kPlugin = {
    TEST_PLUGIN_MODE == 1 ? GAIL_CUDA_PLUGIN_ABI + 1 : GAIL_CUDA_PLUGIN_ABI,
    (uint32_t)sizeof(GailCudaPlugin),
    gail::cudaPluginLayout(),
    TEST_PLUGIN_NAME,
    "0.0 (test)",
    kIs13 ? "75 86 89 120" : "50 52 60 61 70",
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

extern "C" const GailCudaPlugin* gail_cuda_plugin_get(uint32_t abi) {
    if (TEST_PLUGIN_MODE == 2) return nullptr;
    if (TEST_PLUGIN_MODE == 1) return &kPlugin;  // whatever the caller asked for
    return abi == GAIL_CUDA_PLUGIN_ABI ? &kPlugin : nullptr;
}
