// SPDX-License-Identifier: MIT
// Error handling shared by the CUDA translation units.
#pragma once
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "core/backend.h"

namespace gail {

#define CUCHECK(x)                                                                                        \
    do {                                                                                                  \
        cudaError_t e_ = (x);                                                                             \
        if (e_ != cudaSuccess) throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e_) + " at " #x); \
    } while (0)

// cudaMalloc / cudaHostAlloc failing for lack of memory becomes DeviceOutOfMemory (and the
// error is cleared, so the device stays usable); anything else is a plain CUDA error.
inline void cuAlloc(cudaError_t e, const char* what, size_t bytes) {
    if (e == cudaSuccess) return;
    if (e == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        throw DeviceOutOfMemory(std::string("cannot allocate ") + std::to_string(bytes >> 20) + " MiB of " + what);
    }
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(e) + " allocating " + what);
}

}  // namespace gail
