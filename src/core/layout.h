// SPDX-License-Identifier: MIT
// Frame geometry for DCI JPEG 2000 codestreams: subbands, code-blocks,
// precincts, and quantization parameters for one single-tile frame.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gail {

constexpr int kNumComps = 3;
constexpr int kCblkLog2 = 5;           // 32x32 code-blocks (DCI requires xcb=ycb=5)
constexpr int kCblkSize = 1 << kCblkLog2;
constexpr int kCblkArea = kCblkSize * kCblkSize;
constexpr int kFracBits = 6;           // extra magnitude bits kept for distortion estimates
constexpr int kMaxPasses = 64;         // per code-block pass table capacity
constexpr int kMaxBitplanes = 24;      // magnitude bitplanes we can carry (with kFracBits)
constexpr int kBitDepth = 12;          // DCI X'Y'Z' precision

enum class Profile { Cinema2K, Cinema4K };

const char* profileName(Profile p);

// DCI containers: flat 1998x1080 / 3996x2160, scope 2048x858 / 4096x1716, full 2048x1080 / 4096x2160.
enum class Container { Flat, Scope, Full };
Container parseContainer(const std::string& s);
void containerSize(Profile p, Container c, int& w, int& h);

struct Subband {
    int comp = 0, res = 0, level = 0, orient = 0;  // orient: 0 LL, 1 HL, 2 LH, 3 HH
    int px0 = 0, py0 = 0;   // top-left inside the Mallat-layout component plane
    int w = 0, h = 0;
    int ncbx = 0, ncby = 0; // code-block grid
    int cblkBase = 0;       // global index of the first code-block
    int eps = 0, mu = 0;    // signalled step size (exponent, mantissa)
    int Mb = 0;             // number of magnitude bitplanes (G + eps - 1)
    float delta = 1.f;      // quantization step derived from (eps, mu)
    double weight = 1.0;    // squared-error weight: image-domain MSE per unit coefficient error
};

struct CodeBlock {
    int band = 0;
    int x0 = 0, y0 = 0, w = 0, h = 0;  // inside the subband
};

struct PrecinctBand {
    int band = -1;          // subband index
    int cbx0 = 0, cby0 = 0; // first code-block (band grid coordinates)
    int ncbx = 0, ncby = 0; // code-blocks in this precinct (0 when empty)
};

struct Precinct {
    int nb = 0;             // 1 band at r=0, 3 bands otherwise
    PrecinctBand bands[3];
};

struct Resolution {
    int comp = 0, res = 0;
    int w = 0, h = 0;       // resolution extent (origin is always 0)
    int ppx = 0, ppy = 0;   // log2 precinct size on the resolution grid
    int npx = 0, npy = 0;
    std::vector<Precinct> precincts;
};

struct FrameLayout {
    Profile profile = Profile::Cinema2K;
    int width = 0, height = 0;
    int levels = 5;         // decomposition levels NL
    int guardBits = 1;
    double baseStep = 1.0;  // image-domain quantization step (12-bit code values)

    std::vector<Subband> bands;          // comp-major, then resolution order (QCD order)
    std::vector<CodeBlock> cblks;
    std::vector<Resolution> resolutions; // comp * (levels+1) + res

    int numRes() const { return levels + 1; }
    int bandsPerComp() const { return 3 * levels + 1; }
    int bandIndex(int comp, int res, int orient) const {
        return comp * bandsPerComp() + (res == 0 ? 0 : 1 + 3 * (res - 1) + (orient - 1));
    }
    const Resolution& resolution(int comp, int res) const { return resolutions[comp * numRes() + res]; }
    // Largest legal per-band step signalling (shared by all components via one QCD).
    int numCblks() const { return (int)cblks.size(); }
};

// Build the layout. width/height must fit the profile container (2048x1080 / 4096x2160).
FrameLayout makeLayout(int width, int height, Profile profile, double baseStep);

// L2 norm of the synthesis basis function of a subband (standard 9/7 normalisation).
double synthesisNorm(int level, int orient);
// Synthesis gain of the inverse ICT for component c (0=Y, 1=Cb, 2=Cr).
double ictSynthesisNorm(int comp);

}  // namespace gail
