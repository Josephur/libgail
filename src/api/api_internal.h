// SPDX-License-Identifier: MIT
// What the encoder and decoder halves of the library API share: the calling thread's
// last error, and the checks behind the binary-compatibility rules of gail.h.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace gail::api {

// The calling thread's last error (gail_last_error()). A function defined in one place
// rather than an inline thread_local variable: MinGW-w64's GCC emits the initialization
// of such a variable in every translation unit that uses it, and the link fails.
std::string& lastError();

inline int fail(int status, const std::string& msg) {
    lastError() = msg;
    return status;
}

template <class T, size_t N>
bool allZero(const T (&words)[N]) {
    return std::all_of(words, words + N, [](T w) { return w == 0; });
}

inline bool allZero(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    return std::all_of(b, b + n, [](uint8_t v) { return v == 0; });
}

// Error carrying the status the API call should return.
struct ApiError : std::runtime_error {
    int status;
    ApiError(int s, const std::string& m) : std::runtime_error(m), status(s) {}
};

// "auto", "cuda", "vulkan", "cpu" by gail_backend value.
inline const char* const kBackendNames[] = {"auto", "cuda", "vulkan", "cpu"};

}  // namespace gail::api
