// SPDX-License-Identifier: MIT
// CPU reference implementation of the ISO/IEC 15444-1 Annex F irreversible
// 9/7 wavelet (lifting form, whole-sample symmetric extension).
#pragma once
#include <functional>

namespace gail {

// Lifting constants (Annex F.4.8.2 / F.3.8.2).
constexpr double kAlpha = -1.586134342059924;
constexpr double kBeta = -0.052980118572961;
constexpr double kGamma = 0.882911075530934;
constexpr double kDelta = 0.443506852043971;
constexpr double kK = 1.230174104914001;

// Forward 1-D transform of n samples starting at an even coordinate.
// Output is de-interleaved: ceil(n/2) low-pass samples followed by floor(n/2) high-pass.
template <typename T> void fdwt97_1d(const T* in, T* out, int n);
template <typename T> void idwt97_1d(const T* in, T* out, int n);

// Runs f(i) for every i in [0, n), possibly on several threads.
using ParallelFor = std::function<void(int n, const std::function<void(int)>& f)>;

// In-place multi-level forward transform of a width x height plane (Mallat layout).
void fdwt97_2d(float* plane, int width, int height, int levels);
// The same for several planes of equal size at once; the rows and column strips of all
// planes are distributed with `par` (nullptr: this thread). Bit-identical to the above.
void fdwt97_2d(float* const* planes, int nplanes, int width, int height, int levels, const ParallelFor* par);
// Inverse of fdwt97_2d (double precision helper used for tests / norms).
void idwt97_2d(double* plane, int width, int height, int levels);

}  // namespace gail
