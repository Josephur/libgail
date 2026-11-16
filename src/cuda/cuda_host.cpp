// SPDX-License-Identifier: MIT
// The library's side of the CUDA backend (see cuda/cuda_host.h and cuda/cuda_plugin.h).
//
// Plugins are looked for in one place only: the directory of the module this code is
// linked into, which is the gail library, or a tool that has the library's internals
// linked in. Each is opened by its full path; the PATH, the working directory and the
// loader's own search order play no part. A plugin that is missing, was built for
// another interface version or cannot start its CUDA runtime is left out, with a note
// saying why, and the Vulkan and CPU backends carry on.
#include "cuda/cuda_host.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <sstream>
#include <map>
#include <new>
#include <stdexcept>
#include <system_error>

#include "core/t2.h"
#include "cuda/cuda_plugin.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#ifdef __linux__
#include <link.h>
#endif
#endif

namespace gail {

namespace fs = std::filesystem;

namespace {

int smOf(const GailCudaDeviceDesc& d) { return d.smMajor * 10 + d.smMinor; }

// "cuda13.4" for a plugin built with CUDA 13.4.92: the toolkit's major.minor.
std::string cudaCodestreamTag(const char* toolkit) {
    std::string v;
    int dots = 0;
    for (const char* c = toolkit ? toolkit : ""; *c && (std::isdigit((unsigned char)*c) || *c == '.'); ++c) {
        if (*c == '.' && ++dots == 2) break;
        v += *c;
    }
    return ("cuda" + v).substr(0, kMaxDeviceTagBytes);
}

std::string smName(const GailCudaDeviceDesc& d) { return "sm_" + std::to_string(d.smMajor) + std::to_string(d.smMinor); }

#ifndef GAIL_CUDA_BUILTIN
// Opens a plugin by its full path and returns its entry point, or null and the reason.
using EntryPoint = const GailCudaPlugin* (*)(uint32_t);

#ifdef _WIN32
std::string systemError(DWORD code) {
    char buf[256] = "";
    FormatMessageA(FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr, code, 0, buf, sizeof buf, nullptr);
    std::string s = buf;
    while (!s.empty() && (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' || s.back() == '.')) s.pop_back();
    return s.empty() ? "error " + std::to_string(code) : s;
}

EntryPoint openPlugin(const fs::path& file, std::string& why) {
    // What the plugin itself needs is looked for next to it and in the system directory.
    const HMODULE h = LoadLibraryExW(file.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!h) {
        why = systemError(GetLastError());
        return nullptr;
    }
    const auto entry = reinterpret_cast<EntryPoint>(reinterpret_cast<void*>(GetProcAddress(h, GAIL_CUDA_PLUGIN_ENTRY)));
    if (!entry) why = "it has no " GAIL_CUDA_PLUGIN_ENTRY;
    return entry;  // the module stays loaded: devices keep pointers into it
}
#else
EntryPoint openPlugin(const fs::path& file, std::string& why) {
    void* h = dlopen(file.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) {
        const char* e = dlerror();
        why = e ? e : "cannot be opened";
        return nullptr;
    }
    const auto entry = reinterpret_cast<EntryPoint>(dlsym(h, GAIL_CUDA_PLUGIN_ENTRY));
    if (!entry) why = "it has no " GAIL_CUDA_PLUGIN_ENTRY;
    return entry;  // the module stays loaded: devices keep pointers into it
}
#endif
#endif  // !GAIL_CUDA_BUILTIN

#if !defined(_WIN32) && defined(__linux__)
// The loaded object that holds an address: its name as the dynamic linker has it, and
// whether it is the program itself (whose name the linker leaves empty).
struct ObjectOf {
    uintptr_t address;
    std::string name;
    bool found = false;
};

int findObject(dl_phdr_info* info, size_t, void* data) {
    auto* f = static_cast<ObjectOf*>(data);
    for (int i = 0; i < info->dlpi_phnum; ++i) {
        const auto& ph = info->dlpi_phdr[i];
        const uintptr_t begin = info->dlpi_addr + ph.p_vaddr;
        if (ph.p_type == PT_LOAD && f->address >= begin && f->address < begin + ph.p_memsz) {
            f->name = info->dlpi_name ? info->dlpi_name : "";
            f->found = true;
            return 1;
        }
    }
    return 0;
}
#endif

void anchor() {}  // an address inside this module

// Turns the status of a plugin call back into the exception the backend threw.
[[noreturn]] void rethrow(const GailCudaPlugin* api, int status) {
    const std::string message = api->last_error();
    switch (status) {
    case GAIL_CUDA_DEVICE_MEMORY: throw DeviceOutOfMemory(message);
    case GAIL_CUDA_HOST_MEMORY: throw std::bad_alloc();
    case GAIL_CUDA_MISUSE: throw std::logic_error(message);
    default: throw std::runtime_error(message);
    }
}

void check(const GailCudaPlugin* api, int status) {
    if (status != GAIL_CUDA_OK) rethrow(api, status);
}

// What a device and its slots share: the plugin, and the log the plugin writes to.
struct Shared {
    const GailCudaPlugin* api = nullptr;
    Logger log;
    std::string tag;  // " [gail_cuda13]"
    GailCudaDevice* device = nullptr;
    ~Shared() {
        if (device) api->device_destroy(device);
    }
    static void logThunk(void* user, int level, const char* message) {
        const auto* s = static_cast<const Shared*>(user);
        if (s->log) s->log((LogLevel)level, message + s->tag);
    }
};

class PluginEncoderSlot : public EncoderSlot {
public:
    PluginEncoderSlot(std::shared_ptr<Shared> shared, GailCudaEncoder* e) : shared_(std::move(shared)), api_(shared_->api), e_(e) {}
    ~PluginEncoderSlot() override { api_->encoder_destroy(e_); }
    void* inputBuffer() override { return api_->encoder_input(e_); }
    void setSlopeThreshold(double t) override { api_->encoder_set_slope(e_, t); }
    void encode(const void* frame, Tier1Frame& out, FrameTimings* timings, FrameDebug* debug) override {
        GailCudaTier1 t{};
        GailCudaDebug d{};
        if (debug) d.full = debug->full ? 1 : 0;
        check(api_, api_->encoder_encode(e_, frame, &t, timings, debug ? &d : nullptr));
        auto& [info, passLen, passDist, data] = out;
        info.assign(t.info, t.info + t.numInfo);
        passLen.assign(t.passLen, t.passLen + t.numPasses);
        passDist.assign(t.passDist, t.passDist + t.numPasses);
        data.assign(t.data, t.data + t.numData);
        if (debug) {
            auto& [full, xyz12, coeffs, wavelet] = *debug;
            (void)full;
            xyz12.assign(d.xyz12, d.xyz12 + d.numXyz12);
            if (d.coeffs) coeffs.assign(d.coeffs, d.coeffs + d.numCoeffs);
            if (d.wavelet) wavelet.assign(d.wavelet, d.wavelet + d.numWavelet);
        }
    }

private:
    std::shared_ptr<Shared> shared_;  // keeps the device and its log alive
    const GailCudaPlugin* api_;
    GailCudaEncoder* e_;
};

class PluginDecoderSlot : public DecoderSlot {
public:
    PluginDecoderSlot(std::shared_ptr<Shared> shared, GailCudaDecoder* d) : shared_(std::move(shared)), api_(shared_->api), d_(d) {}
    ~PluginDecoderSlot() override { api_->decoder_destroy(d_); }
    void decode(const FrameLayout& frame, const uint8_t* data, size_t size, const std::vector<CblkCode>& blocks, void* dst,
                ptrdiff_t strideBytes, DecodeTimings* tm) override {
        const GailCudaLayout layout = toPlugin(frame);
        GailCudaDecodeTimings t{};
        if (tm) t.wantWavelet = tm->wavelet ? 1 : 0;
        check(api_, api_->decoder_decode(d_, &layout, data, size, blocks.data(), blocks.size(), dst, strideBytes, tm ? &t : nullptr));
        if (tm) {
            auto& [t1, dwt, color, total, wavelet] = *tm;
            t1 = t.t1;
            dwt = t.dwt;
            color = t.color;
            total = t.total;
            for (int c = 0; wavelet && c < 3; ++c) wavelet[c].assign(t.wavelet[c], t.wavelet[c] + t.numWavelet);
        }
    }

private:
    std::shared_ptr<Shared> shared_;
    const GailCudaPlugin* api_;
    GailCudaDecoder* d_;
};

class PluginDevice : public Device {
public:
    explicit PluginDevice(std::shared_ptr<Shared> shared) : shared_(std::move(shared)), api_(shared_->api), d_(shared_->device) {}
    std::string name() const override { return api_->device_name(d_); }
    std::string codestreamTag() const override { return cudaCodestreamTag(api_->toolkit); }
    std::unique_ptr<EncoderSlot> createSlot(const FrameLayout& L, const EncodeSettings& s) override {
        const GailCudaLayout layout = toPlugin(L);
        const GailCudaEncodeSettings settings = toPlugin(s);
        GailCudaEncoder* e = nullptr;
        check(api_, api_->encoder_create(d_, &layout, &settings, &e));
        return std::make_unique<PluginEncoderSlot>(shared_, e);
    }
    uint64_t slotBytes(const FrameLayout& L, const EncodeSettings& s) const override {
        const GailCudaLayout layout = toPlugin(L);
        const GailCudaEncodeSettings settings = toPlugin(s);
        return api_->encoder_bytes(d_, &layout, &settings);
    }
    bool canDecode() const override { return true; }
    std::unique_ptr<DecoderSlot> createDecoderSlot(const FrameLayout& L, const DecodeSettings& s) override {
        const GailCudaLayout layout = toPlugin(L);
        const GailCudaDecodeSettings settings = toPlugin(s);
        GailCudaDecoder* d = nullptr;
        check(api_, api_->decoder_create(d_, &layout, &settings, &d));
        return std::make_unique<PluginDecoderSlot>(shared_, d);
    }
    uint64_t decoderSlotBytes(const FrameLayout& L, const DecodeSettings& s) const override {
        const GailCudaLayout layout = toPlugin(L);
        const GailCudaDecodeSettings settings = toPlugin(s);
        return api_->decoder_bytes(d_, &layout, &settings);
    }
    DeviceMemory memory() const override {
        DeviceMemory m;
        GailCudaMemory p{};
        if (api_->device_memory(d_, &p) == GAIL_CUDA_OK) {
            auto& [total, available, budgeted] = m;
            total = p.total;
            available = p.available;
            budgeted = p.budgeted != 0;
        }
        return m;
    }

private:
    std::shared_ptr<Shared> shared_;
    const GailCudaPlugin* api_;
    GailCudaDevice* d_;
};

}  // namespace

struct CudaHost::Plugin {
    CudaPluginSpec spec;
    const GailCudaPlugin* api = nullptr;  // null: not loaded
    std::string note;                     // what it is, or why it is not loaded
    std::string runtimeError;             // set when its CUDA runtime did not start
    std::vector<GailCudaDeviceDesc> devices;
};

// One GPU, whichever plugins see it.
struct CudaHost::Entry {
    DeviceInfo info;     // index: its place in the list, which is the CUDA runtime's own number when the plugins agree
    std::string sm;      // "sm_52"
    Plugin* plugin = nullptr;  // the one that serves it
    int local = -1;      // its index there
    std::string reason;  // why it cannot be used; "" when it can
};

bool cudaArchsRun(const std::string& archs, int sm) {
    std::istringstream in(archs);
    std::string a;
    while (in >> a) {
        size_t n = 0;
        while (n < a.size() && std::isdigit((unsigned char)a[n])) ++n;
        if (n == 0) continue;
        const int built = std::stoi(a.substr(0, n));
        std::string rest = a.substr(n);
        char variant = 0;  // 'a': this architecture only, 'f': its family
        if (!rest.empty() && (rest[0] == 'a' || rest[0] == 'f')) {
            variant = rest[0];
            rest = rest.substr(1);
        }
        const bool real = rest.empty() || rest == "-real", virt = rest.empty() || rest == "-virtual";
        const bool sameMajor = sm / 10 == built / 10 && sm % 10 >= built % 10;
        if (variant == 'a') {
            if ((real || virt) && sm == built) return true;
        } else if (variant == 'f') {
            if ((real || virt) && sameMajor) return true;
        } else if ((real && sameMajor) || (virt && sm >= built)) {
            return true;
        }
    }
    return false;
}

std::vector<CudaPluginSpec> cudaPluginSpecs() { return {{"gail_cuda13", 75, 1 << 30}, {"gail_cuda12", 0, 74}}; }

std::string CudaHost::fileName(const std::string& plugin) {
#ifdef _WIN32
    return plugin + ".dll";
#else
    return "lib" + plugin + ".so";
#endif
}

fs::path CudaHost::moduleDirectory() {
#ifdef _WIN32
    HMODULE h = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                            reinterpret_cast<LPCWSTR>(&anchor), &h))
        return {};
    std::wstring path(512, L'\0');
    for (;;) {
        const DWORD n = GetModuleFileNameW(h, path.data(), (DWORD)path.size());
        if (n == 0) return {};
        if (n < path.size()) {
            path.resize(n);
            break;
        }
        path.resize(path.size() * 2);
    }
    return fs::path(path).parent_path();
#else
    fs::path file;
#ifdef __linux__
    ObjectOf f{reinterpret_cast<uintptr_t>(&anchor)};
    dl_iterate_phdr(findObject, &f);
    if (!f.found) return {};
    // The program's own name is not a path that can be trusted (it is argv[0]).
    file = f.name.empty() ? fs::path("/proc/self/exe") : fs::path(f.name);
#else
    Dl_info info;
    if (!dladdr(reinterpret_cast<void*>(&anchor), &info) || !info.dli_fname) return {};
    file = info.dli_fname;
#endif
    std::error_code ec;
    file = fs::canonical(file, ec);  // through symbolic links, to where the file really is
    return ec ? fs::path() : file.parent_path();
#endif
}

CudaHost& CudaHost::instance() {
    // Never destroyed: devices may outlive static destruction order.
    static CudaHost* host = new CudaHost(moduleDirectories(), cudaPluginSpecs());
    return *host;
}

std::vector<fs::path> CudaHost::moduleDirectories() {
    std::vector<fs::path> dirs;
    const fs::path own = moduleDirectory();
    if (own.empty()) return dirs;
    dirs.push_back(own);
#ifdef GAIL_CUDA_TOOL_PLUGIN_DIR
    // An installed tool is in bin/, the library and its plugins in lib/.
    std::error_code ec;
    const fs::path lib = fs::canonical(own / GAIL_CUDA_TOOL_PLUGIN_DIR, ec);
    if (!ec && lib != own) dirs.push_back(lib);
#endif
    return dirs;
}

CudaHost::CudaHost(const std::vector<fs::path>& dirs, const std::vector<CudaPluginSpec>& specs) {
#ifdef GAIL_CUDA_BUILTIN
    // One backend is linked in and takes every device its kernels can run.
    (void)dirs;
    (void)specs;
    auto p = std::make_unique<Plugin>();
    p->api = gail_cuda_plugin_get(GAIL_CUDA_PLUGIN_ABI);
    p->spec.name = p->api->name;
    p->note = std::string("cuda: ") + p->api->name + ", CUDA " + p->api->toolkit + ", sm " + p->api->archs;
    plugins_.push_back(std::move(p));
#else
    for (const auto& spec : specs) load(dirs, spec);
#endif
}

CudaHost::~CudaHost() = default;

void CudaHost::load(const std::vector<fs::path>& dirs, const CudaPluginSpec& spec) {
    auto p = std::make_unique<Plugin>();
    p->spec = spec;
#ifndef GAIL_CUDA_BUILTIN
    const std::string name = fileName(spec.name);
    fs::path file;
    std::error_code ec;
    for (const auto& dir : dirs)
        if (file.empty() && fs::is_regular_file(dir / name, ec)) file = dir / name;
    std::string why;
    if (dirs.empty()) {
        why = "the directory of the gail library cannot be told";
    } else if (file.empty()) {
        why = name + " is not next to the gail library";
    } else if (const EntryPoint entry = openPlugin(file, why)) {
        const GailCudaPlugin* api = entry(GAIL_CUDA_PLUGIN_ABI);
        if (!api || api->abi != GAIL_CUDA_PLUGIN_ABI)
            why = "it was built for another version of the gail library (plugin interface " +
                  (api ? std::to_string(api->abi) : std::string("unknown")) + ", this library has " +
                  std::to_string(GAIL_CUDA_PLUGIN_ABI) + ")";
        else if (api->size != sizeof(GailCudaPlugin) || api->layout != cudaPluginLayout())
            why = "it was built for another version of the gail library (its interface tables are " +
                  std::to_string(api->size) + " bytes, layout " + std::to_string(api->layout) + "; this library's are " +
                  std::to_string(sizeof(GailCudaPlugin)) + " bytes, layout " + std::to_string(cudaPluginLayout()) + ")";
        else
            p->api = api;
    }
    if (p->api)
        p->note = "cuda: " + spec.name + " loaded: CUDA " + p->api->toolkit + ", sm " + p->api->archs;
    else
        p->note = "cuda: " + spec.name + " not loaded: " + why;
#else
    (void)dirs;
#endif
    plugins_.push_back(std::move(p));
}

std::string CudaHost::loaded() const {
    std::string s;
#ifndef GAIL_CUDA_BUILTIN
    for (const auto& p : plugins_)
        if (p->api) s += (s.empty() ? "" : " ") + p->spec.name;
#endif
    return s;
}

void CudaHost::probe() {
    if (probed_) return;
    probed_ = true;
    for (auto& p : plugins_) {
        if (!p->api) continue;
        const int n = p->api->device_count();
        if (n < 0) {
            p->runtimeError = p->api->last_error();
            continue;
        }
        for (int i = 0; i < n; ++i) {
            GailCudaDeviceDesc d{};
            if (p->api->device_desc(i, &d) == GAIL_CUDA_OK) p->devices.push_back(d);
            else p->devices.push_back({});  // keeps the indices; an empty UUID matches nothing
        }
    }
    // Every GPU once, in the order the first plugin that sees it has it.
    std::vector<const GailCudaDeviceDesc*> descs;
    for (auto& p : plugins_)
        for (const auto& d : p->devices) {
            if (!d.uuid[0]) continue;
            if (std::any_of(entries_.begin(), entries_.end(), [&](const Entry& e) { return e.info.uuid == d.uuid; })) continue;
            Entry e;
            e.info.index = (int)entries_.size();
            e.info.name = d.name;
            e.info.uuid = d.uuid;
            e.info.pciBusId = d.pciBusId;
            e.info.memoryBytes = d.memoryBytes;
            e.sm = smName(d);
            entries_.push_back(e);
            descs.push_back(&d);
        }
    for (size_t k = 0; k < entries_.size(); ++k) {
        Entry& e = entries_[k];
        const int sm = smOf(*descs[k]);
        // The plugin this architecture belongs to, and no other.
        Plugin* owner = nullptr;
        for (auto& p : plugins_)
            if (!owner && sm >= p->spec.minSm && sm <= p->spec.maxSm) owner = p.get();
        if (!owner) {
            e.reason = "no plugin is for this architecture";
            continue;
        }
        const std::string& name = owner->spec.name;
        if (!owner->api) {
            e.reason = "it needs " + name + ", which is not loaded";
            continue;
        }
        if (!owner->runtimeError.empty()) {
            e.reason = "it needs " + name + ", whose CUDA runtime did not start: " + owner->runtimeError;
            continue;
        }
        for (size_t i = 0; i < owner->devices.size(); ++i)
            if (e.info.uuid == owner->devices[i].uuid) e.local = (int)i;
        if (e.local < 0) {
            e.reason = "it needs " + name + ", whose CUDA runtime does not list it";
            continue;
        }
        // By architecture alone: listing sets nothing up on the GPU. Whether the kernels
        // really load there is found out when the device is created.
        if (!cudaArchsRun(owner->api->archs, sm)) {
            e.reason = "skipped by " + name + ": not built for this architecture";
            continue;
        }
        e.plugin = owner;
        e.info.plugin = name;
    }
}

std::vector<DeviceInfo> CudaHost::list() {
    std::lock_guard<std::mutex> lock(mutex_);
    probe();
    std::vector<DeviceInfo> out;
    for (const auto& e : entries_)
        if (e.plugin) out.push_back(e.info);
    return out;
}

std::string CudaHost::unusable(int index) {
    std::lock_guard<std::mutex> lock(mutex_);
    probe();
    for (const auto& e : entries_)
        if (e.info.index == index && !e.plugin) return e.info.name + " (" + e.sm + ") " + describe(e.reason);
    return "";
}

std::string CudaHost::describe(const std::string& reason) {
    return reason.rfind("skipped by ", 0) == 0 ? reason : "skipped: " + reason;
}

std::vector<std::string> CudaHost::notes() {
    std::lock_guard<std::mutex> lock(mutex_);
    probe();
    std::vector<std::string> out;
    for (const auto& p : plugins_) {
        out.push_back(p->note);
        if (!p->runtimeError.empty()) out.push_back("cuda: " + p->spec.name + " cannot be used: " + p->runtimeError);
    }
    for (const auto& e : entries_)
        if (!e.plugin) out.push_back("cuda: device " + std::to_string(e.info.index) + " " + e.info.name + " (" + e.sm + ") " + describe(e.reason));
    return out;
}

std::vector<std::unique_ptr<Device>> CudaHost::create(const std::vector<int>& select, const Logger& log) {
    std::vector<Entry> entries;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        probe();
        entries = entries_;
    }
    if (log)
        for (const auto& note : notes()) log(LogLevel::Info, note);
    std::vector<std::unique_ptr<Device>> out;
    for (const auto& e : entries) {
        if (!e.plugin) continue;
        if (!select.empty() && std::find(select.begin(), select.end(), e.info.index) == select.end()) continue;
        // The kernels must load on the device, as a launch would load them (or have the
        // driver compile them from PTX). A device where they do not is set aside, with
        // the reason, as listing sets aside one the plugin has no code for.
        if (e.plugin->api->device_usable(e.local) != GAIL_CUDA_OK) {
            const std::string reason = "skipped by " + e.plugin->spec.name + ": " + e.plugin->api->last_error();
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (auto& f : entries_)
                    if (f.info.index == e.info.index) {
                        f.plugin = nullptr;
                        f.info.plugin.clear();
                        f.reason = reason;
                    }
            }
            if (log) log(LogLevel::Warning, "cuda: device " + std::to_string(e.info.index) + " " + e.info.name + " (" + e.sm + ") " + reason);
            continue;
        }
        auto shared = std::make_shared<Shared>();
        shared->api = e.plugin->api;
        shared->log = log;
#ifndef GAIL_CUDA_BUILTIN
        shared->tag = " [" + e.plugin->spec.name + "]";
#endif
        if (shared->api->device_create(e.local, Shared::logThunk, shared.get(), &shared->device) != GAIL_CUDA_OK) {
            if (log) log(LogLevel::Warning, "cuda: skipping device " + std::to_string(e.info.index) + ": " + shared->api->last_error());
            continue;
        }
        out.push_back(std::make_unique<PluginDevice>(std::move(shared)));
    }
    return out;
}

// ---- the CUDA backend as core/backend.h declares it -----------------------------------

std::vector<std::unique_ptr<Device>> createCudaDevices(const std::vector<int>& select, const Logger& log) {
    return CudaHost::instance().create(select, log);
}

std::vector<DeviceInfo> listCudaDevices() { return CudaHost::instance().list(); }

std::string cudaDeviceUnusable(int index) { return CudaHost::instance().unusable(index); }

std::vector<std::string> cudaBackendNotes() { return CudaHost::instance().notes(); }

std::string cudaPluginsLoaded() {
#ifdef GAIL_CUDA_BUILTIN
    return "";
#else
    const std::string names = CudaHost::instance().loaded();
    return names.empty() ? "no CUDA plugins" : names;
#endif
}

}  // namespace gail
