// SPDX-License-Identifier: MIT
// The library's side of the CUDA backend: finds the backend's plugins, decides which one
// serves each GPU, and wraps what they hand out in the Device interface. See
// cuda/cuda_plugin.h for the plugins and the interface to them.
#pragma once
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/backend.h"

namespace gail {

// A plugin and the compute capabilities it is given (major * 10 + minor, both inclusive).
struct CudaPluginSpec {
    std::string name;  // "gail_cuda13": the file is gail_cuda13.dll / libgail_cuda13.so
    int minSm = 0, maxSm = 1 << 30;
};

// Which plugin serves which GPUs. The rule is fixed by architecture: Turing and newer
// run on the CUDA 13 plugin, and the architectures CUDA 13 dropped (Maxwell, Pascal,
// Volta) on the CUDA 12 plugin. No device is ever given to the other one.
std::vector<CudaPluginSpec> cudaPluginSpecs();

// Whether a build for these architectures has code a GPU of compute capability sm
// (major * 10 + minor) can run. archs is a plugin's list, CMake's CUDA_ARCHITECTURES
// joined by spaces: "NN" is machine code and PTX for NN, "NN-real" the machine code alone
// and "NN-virtual" the PTX alone. Machine code runs on the same major version at the same
// or a higher minor, PTX on NN and everything after it; code for an "NNa" architecture
// runs on NN only, and "NNf" on its family (the same major version).
bool cudaArchsRun(const std::string& archs, int sm);

class CudaHost {
public:
    // The host the library uses: the plugins next to the module this code is part of,
    // or the one backend a static library has linked in.
    static CudaHost& instance();
    // The directory of that module (empty if it cannot be told), and the directories
    // instance() looks in: that one, and for an installed tool the library's.
    static std::filesystem::path moduleDirectory();
    static std::vector<std::filesystem::path> moduleDirectories();
    // File name of a plugin on this platform.
    static std::string fileName(const std::string& plugin);

    // Loads the plugins of `specs` from `dirs` (the first that has the file), and from
    // nowhere else. Tests use this with directories and rules of their own.
    CudaHost(const std::vector<std::filesystem::path>& dirs, const std::vector<CudaPluginSpec>& specs);
    ~CudaHost();

    // The devices that can be used, each with the plugin that serves it. The first call
    // starts the CUDA runtime of every plugin and asks it about the GPUs, which sets
    // nothing up on them (no context, no device memory): a GPU is offered when its
    // plugin has code for its architecture (cudaArchsRun). Later calls answer from that.
    std::vector<DeviceInfo> list();
    // Sets the devices up. One whose kernels do not load (device_usable) is left out
    // with a warning, and from then on is not listed and unusable() says why.
    std::vector<std::unique_ptr<Device>> create(const std::vector<int>& select, const Logger& log);
    // Why the device with this index is not in list(): "" when it is, or there is none.
    std::string unusable(int index);
    // A line for each plugin (what it is, or why it is not loaded) and for each device
    // that cannot be used.
    std::vector<std::string> notes();
    // Names of the plugins loaded, separated by spaces. Does not start a CUDA runtime.
    std::string loaded() const;

private:
    struct Plugin;
    struct Entry;
    void load(const std::vector<std::filesystem::path>& dirs, const CudaPluginSpec& spec);
    void probe();  // with mutex_ held
    static std::string describe(const std::string& reason);

    std::vector<std::unique_ptr<Plugin>> plugins_;
    std::vector<Entry> entries_;
    bool probed_ = false;
    std::mutex mutex_;
};

}  // namespace gail
