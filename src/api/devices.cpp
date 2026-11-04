// SPDX-License-Identifier: MIT
#include "api/devices.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>

namespace gail {

namespace {

std::vector<DeviceInfo> listBackend(const std::string& backend) {
#ifdef GAIL_HAVE_CUDA
    if (backend == "cuda") return listCudaDevices();
#endif
#ifdef GAIL_HAVE_VULKAN
    if (backend == "vulkan") return listVulkanDevices();
#endif
    if (backend == "cpu") return {cpuDeviceInfo(0)};
    return {};
}

std::string lower(std::string s) {
    for (auto& c : s) c = (char)std::tolower((unsigned char)c);
    return s;
}

// UUID without the "GPU-" prefix, PCI bus ID with its domain, lower case.
std::string normaliseId(const std::string& id) {
    std::string s = lower(id);
    if (s.rfind("gpu-", 0) == 0) s = s.substr(4);
    if (std::count(s.begin(), s.end(), ':') == 1) s = "0000:" + s;
    return s;
}

// Indices into `devices` for `ids`; empty `missing` when all were found.
std::vector<int> resolve(const std::vector<DeviceInfo>& devices, const std::vector<std::string>& ids,
                         std::string& missing) {
    std::vector<int> out;
    missing.clear();
    for (const auto& id : ids) {
        const std::string want = normaliseId(id);
        int found = -1;
        if (!want.empty() && std::all_of(want.begin(), want.end(), ::isdigit)) {
            const int i = std::stoi(want);
            for (const auto& d : devices)
                if (d.index == i) found = i;
        } else {
            for (const auto& d : devices)
                if ((!d.uuid.empty() && normaliseId(d.uuid) == want) || (!d.pciBusId.empty() && normaliseId(d.pciBusId) == want))
                    found = d.index;
        }
        if (found < 0) {
            missing = id;
            return {};
        }
        if (std::find(out.begin(), out.end(), found) == out.end()) out.push_back(found);
    }
    return out;
}

std::string describe(const std::vector<DeviceInfo>& devices, const std::string& backend) {
    std::string s;
    for (const auto& d : devices)
        s += "\n  " + backend + " " + std::to_string(d.index) + ": " + d.name + (d.uuid.empty() ? "" : " " + d.uuid) +
             (d.pciBusId.empty() ? "" : " (" + d.pciBusId + ")");
    return s;
}

}  // namespace

std::vector<DeviceInfo> listDevices(std::string backend, std::string& chosen) {
    if (backend != "auto") {
        chosen = backend;
        return listBackend(backend);
    }
    for (const char* b : {"cuda", "vulkan"}) {
        auto d = listBackend(b);
        if (!d.empty()) {
            chosen = b;
            return d;
        }
    }
    chosen = "cpu";
    return listBackend("cpu");
}

std::vector<std::unique_ptr<Device>> createDevices(std::string backend, const std::vector<int>& select,
                                                   const std::vector<std::string>& ids, int cpuThreads,
                                                   bool cpuLowPriority, const Logger& log, std::string& chosen) {
    std::vector<int> picked = select;
    if (!ids.empty()) {
        // Resolve the IDs against the backend(s) they could belong to.
        std::vector<std::string> candidates = backend == "auto" ? std::vector<std::string>{"cuda", "vulkan"}
                                                                : std::vector<std::string>{backend};
        std::string missing, available;
        bool ok = false;
        for (const auto& b : candidates) {
            const auto devices = listBackend(b);
            picked = resolve(devices, ids, missing);
            available += describe(devices, b);
            if (missing.empty()) {
                backend = b;
                ok = true;
                break;
            }
        }
        if (!ok)
            throw std::runtime_error("no " + (backend == "auto" ? std::string("GPU") : backend + " device") +
                                     " matches \"" + missing + "\"; available:" +
                                     (available.empty() ? std::string(" none") : available));
    }

    std::vector<std::unique_ptr<Device>> devices;
    chosen = backend;
    if (backend == "auto") {
#ifdef GAIL_HAVE_CUDA
        devices = createCudaDevices(picked, log);
        if (!devices.empty()) chosen = "cuda";
#endif
#ifdef GAIL_HAVE_VULKAN
        if (devices.empty()) {
            devices = createVulkanDevices(picked, log);
            if (!devices.empty()) chosen = "vulkan";
        }
#endif
        if (devices.empty()) {
            devices.push_back(createCpuDevice(cpuThreads, cpuLowPriority));
            chosen = "cpu";
        }
    } else if (backend == "cpu") {
        devices.push_back(createCpuDevice(cpuThreads, cpuLowPriority));
    } else if (backend == "vulkan") {
#ifdef GAIL_HAVE_VULKAN
        devices = createVulkanDevices(picked, log);
#else
        throw std::runtime_error("built without Vulkan support");
#endif
    } else if (backend == "cuda") {
#ifdef GAIL_HAVE_CUDA
        devices = createCudaDevices(picked, log);
#else
        throw std::runtime_error("built without CUDA support");
#endif
    } else {
        throw std::runtime_error("unknown backend " + backend);
    }
    if (devices.empty()) throw std::runtime_error("no usable " + chosen + " devices");
    return devices;
}

}  // namespace gail
