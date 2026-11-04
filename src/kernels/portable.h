// SPDX-License-Identifier: MIT
// Portability layer so the same kernel source compiles as C++, CUDA and GLSL.
//
// Shared kernel code uses: U32/I32/U64 scalar types, GAIL_FN for helper
// functions, INOUT(T) for by-reference parameters, GAIL_CONST for constant
// tables and ARRAY_BEGIN/ARRAY_END for table initialisers.
#ifndef GAIL_PORTABLE_H
#define GAIL_PORTABLE_H

#if defined(GAIL_GLSL)
#define U32 uint
#define I32 int
#define U64 uint64_t
#define GAIL_FN
#define INOUT(T) inout T
#define GAIL_CONST const
#define ARRAY_BEGIN(T, N) T[N](
#define ARRAY_END )
#define TO_U32(x) uint(x)
#define TO_I32(x) int(x)
#define TO_U64(x) uint64_t(x)
#define FIND_MSB(x) findMSB(x)
#define POPCOUNT(x) bitCount(x)
#define TO_F32(x) float(x)
#define FMIN(a, b) min(a, b)
#define FMAX(a, b) max(a, b)
#define FPOW(a, b) pow(a, b)
#define FFLOOR(a) floor(a)
#define IMIN(a, b) min(a, b)
#elif defined(__CUDACC__)
#include <cstdint>
#define U32 uint32_t
#define I32 int32_t
#define U64 uint64_t
#define GAIL_FN __device__ __forceinline__
#define INOUT(T) T&
#define GAIL_CONST __device__ const
#define ARRAY_BEGIN(T, N) {
#define ARRAY_END }
#define TO_U32(x) ((uint32_t)(x))
#define TO_I32(x) ((int32_t)(x))
#define TO_U64(x) ((uint64_t)(x))
#define FIND_MSB(x) (31 - __clz((int)(x)))
#define POPCOUNT(x) __popc(x)
#define TO_F32(x) ((float)(x))
#define FMIN(a, b) fminf(a, b)
#define FMAX(a, b) fmaxf(a, b)
#define FPOW(a, b) powf(a, b)
#define FFLOOR(a) floorf(a)
#define IMIN(a, b) min(a, b)
#else
#include <cstdint>
#define U32 uint32_t
#define I32 int32_t
#define U64 uint64_t
#define GAIL_FN static inline
#define INOUT(T) T&
#define GAIL_CONST static const
#define ARRAY_BEGIN(T, N) {
#define ARRAY_END }
#define TO_U32(x) ((uint32_t)(x))
#define TO_I32(x) ((int32_t)(x))
#define TO_U64(x) ((uint64_t)(x))
#if defined(_MSC_VER)
#include <intrin.h>
static inline int gail_find_msb(uint32_t x) {
    unsigned long i;
    return _BitScanReverse(&i, x) ? (int)i : -1;
}
#define FIND_MSB(x) gail_find_msb((uint32_t)(x))
#define POPCOUNT(x) ((int)__popcnt((uint32_t)(x)))
#else
#define FIND_MSB(x) ((x) == 0 ? -1 : 31 - __builtin_clz((uint32_t)(x)))
#define POPCOUNT(x) __builtin_popcount((uint32_t)(x))
#endif
#include <cmath>
#define TO_F32(x) ((float)(x))
#define FMIN(a, b) std::fmin((float)(a), (float)(b))
#define FMAX(a, b) std::fmax((float)(a), (float)(b))
#define FPOW(a, b) std::pow((float)(a), (float)(b))
#define FFLOOR(a) std::floor((float)(a))
#define IMIN(a, b) ((a) < (b) ? (a) : (b))
#endif

#endif
