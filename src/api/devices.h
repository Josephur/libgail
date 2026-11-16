// SPDX-License-Identifier: MIT
// Device listing and selection shared by gail-enc and the library API.
#pragma once
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "core/backend.h"

namespace gail {

// Devices a backend offers, without setting them up. backend "auto" lists the backend
// that "auto" would encode with (CUDA, else Vulkan, else CPU); `chosen` is set to it.
// CUDA offers the GPUs one of its plugins can run, and no others.
std::vector<DeviceInfo> listDevices(std::string backend, std::string& chosen);

// backend: "auto" (CUDA, else Vulkan, else CPU), "cuda", "vulkan" or "cpu".
// Devices are picked by `ids` when given (each a UUID as nvidia-smi prints it, with or
// without "GPU-", a PCI bus ID such as "0000:01:00.0" / "01:00.0", or an index), else by
// `select` (indices; empty = all). With IDs, "auto" uses CUDA if it has every device,
// else Vulkan. cpuThreads / cpuLowPriority configure the CPU backend (see createCpuDevice).
// The backends' notes (devices skipped, what each device is) go to `log`.
// Sets `chosen` to the backend used. Throws when nothing usable is found, an ID matches
// no device, or "cuda" is asked for a GPU that no CUDA plugin can run ("auto" uses
// Vulkan for it instead).
std::vector<std::unique_ptr<Device>> createDevices(std::string backend, const std::vector<int>& select,
                                                   const std::vector<std::string>& ids, int cpuThreads,
                                                   bool cpuLowPriority, const Logger& log, std::string& chosen);

// Whether a backend ("cuda", "vulkan", "cpu") of this build can decode.
bool backendDecodes(const std::string& backend);

// For the tests, which link the library's sources in: when set, gail_decoder_create()
// hands the devices it has made to this before using them, to wrap them in devices that
// fail on purpose. Not part of the library's interface.
using DecoderDeviceHook = std::function<void(std::vector<std::unique_ptr<Device>>&)>;
DecoderDeviceHook& decoderDeviceHook();

}  // namespace gail
