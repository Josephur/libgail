// SPDX-License-Identifier: MIT
// Source RGB -> DCI X'Y'Z' (SMPTE ST 428-1) conversion.
//
// Pipeline per pixel: code value -> linear light (cd/m^2) via a 64K-entry LUT,
// 3x3 primaries matrix to CIE XYZ, optional HDR luminance tone mapping
// (BT.2390 EETF towards the 48 cd/m^2 cinema peak), normalisation by 52.37,
// 1/2.6 power, 12-bit quantisation, DC level shift and the JPEG 2000 ICT.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace gail {

enum class Transfer { BT1886, SRGB, PQ, HLG, Linear, Gamma26 };
enum class Primaries { BT709, BT2020, P3D65, DCIP3, XYZ };
// How PQ / HLG sources are brought into cinema range.
// Signal (default): the HDR signal is shown as if it were gamma 2.4 video with code 1.0 at
// sdrWhite, which is what DCP-o-matic does (libdcp's RGB -> XYZ with a gamma 2.4 input
// transfer); dim and flat compared with a true HDR rendering, but no highlight is ever
// crushed. Shoulder: scene-relative (source reference white -> cinemaRefWhite, smooth
// roll-off to the cinema peak); BT2390: the ITU-R BT.2390 EETF in absolute PQ terms
// (bright); None: absolute luminance, no tone mapping (values above the cinema peak clip).
enum class ToneMap { Signal, Shoulder, BT2390, None };
enum class TargetGamut { P3D65, DCIP3, None };
// What the tone curve is applied to. Channel: each target-gamut channel on its own (bright
// saturated colours drift toward white the way film and SDR grades render them; keeps
// texture in highlights); MaxRGB: the brightest channel (keeps hue and saturation
// exactly, so bright warm light stays deep orange); Luminance: Y (keeps luminance, so
// saturated highlights are bleached to white by the gamut stage).
enum class ToneNorm { Channel, MaxRGB, Luminance };

Transfer parseTransfer(const std::string& s);
Primaries parsePrimaries(const std::string& s);
const char* transferName(Transfer t);
const char* primariesName(Primaries p);

// Layout of the frames handed to the encoder.
//  RGB48:  interleaved 16-bit R'G'B', already fitted into the container (width x height)
//  Planar: three planes of the active picture (activeW x activeH), placed at (ox, oy)
//          inside the container: Y'CbCr with the chroma planes subsampled by 2^sx across
//          and 2^sy down (4:2:0 = 1/1, 4:2:2 = 1/0, 4:4:4 = 0/0), or full-range G'B'R'
//          (rgb; FFmpeg's gbrp plane order, never subsampled). The encoder converts to
//          R'G'B' on the device and fills the rest of the container with black.
struct FrameFormat {
    enum class Kind { RGB48, Planar } kind = Kind::RGB48;
    enum class Chroma { Left, Center, TopLeft } chroma = Chroma::Left;
    int width = 0, height = 0;                       // container size
    int activeW = 0, activeH = 0, ox = 0, oy = 0;    // planar picture area
    int sx = 1, sy = 1;                              // chroma subsampling (log2)
    bool rgb = false;                                // planes are G', B', R'
    int bits = 8;                                    // bit depth (8: bytes, >8: LSB-aligned words)
    bool fullRange = false;                          // Y'CbCr only (G'B'R' is always full range)
    double kr = 0.2627, kb = 0.0593;                 // Y'CbCr matrix (BT.2020 default)

    int chromaW() const { return (activeW + sx) >> sx; }
    int chromaH() const { return (activeH + sy) >> sy; }
    size_t bytes() const {
        if (kind == Kind::RGB48) return (size_t)width * height * 6;
        const size_t bps = bits > 8 ? 2 : 1;
        return ((size_t)activeW * activeH + 2 * (size_t)chromaW() * chromaH()) * bps;
    }
    // R' = Y' + crR*Cr; G' = Y' - cbG*Cb - crG*Cr; B' = Y' + cbB*Cb
    void matrix(float& crR, float& cbG, float& crG, float& cbB) const {
        const double kg = 1.0 - kr - kb;
        crR = (float)(2 * (1 - kr));
        cbB = (float)(2 * (1 - kb));
        cbG = (float)(2 * kb * (1 - kb) / kg);
        crG = (float)(2 * kr * (1 - kr) / kg);
    }
};

// Reference conversion of a Planar frame to container-size RGB48 (CPU backend / tests).
void planarToRgb48(const FrameFormat& f, const void* src, uint16_t* rgb);
// One container row y of the same conversion into rgbRow (f.width * 3 samples; the part
// outside the active picture is black). Rows are independent, so callers may thread them.
void planarToRgb48Row(const FrameFormat& f, const void* src, uint16_t* rgbRow, int y);

// How source pictures become DCI X'Y'Z'. Defaults give a cinema-safe rendering of HDR
// sources; apps can tune every step, pass X'Y'Z' through untouched (primaries XYZ +
// transfer gamma26), or supply their own 3D LUT.
struct ColorParams {
    Transfer transfer = Transfer::BT1886;
    Primaries primaries = Primaries::BT709;
    ToneMap toneMap = ToneMap::Signal;  // applied to PQ / HLG sources only
    ToneNorm toneNorm = ToneNorm::Channel;
    double sdrWhite = 48.0;        // cd/m^2 that SDR code value 1.0 maps to
    double hdrPeak = 1000.0;       // source mastering peak (PQ/HLG), cd/m^2
    double cinemaPeak = 48.0;      // target peak white, cd/m^2
    double hdrRefWhite = 203.0;    // source reference (diffuse) white, cd/m^2 (BT.2408)
    double cinemaRefWhite = 16.0;  // where that white lands, cd/m^2 (shoulder; leaves ~1.5 stops of highlight room)
    double knee = 0.25;            // shoulder start, fraction of cinemaPeak
    TargetGamut gamut = TargetGamut::None;  // projector gamut colours are compressed into (None: as DCP-o-matic)
    double gamutThreshold = 0.8;   // compression starts at this fraction of the gamut edge
    std::string lut3d;             // optional .cube: source R'G'B' -> X'Y'Z' (0..1); overrides the above
    std::vector<float> lut3dData;  // or the same LUT in memory: lut3dSize^3 * 3 floats, red fastest
    int lut3dSize = 0;
};

class ColorConverter {
public:
    explicit ColorConverter(const ColorParams& p);
    const std::vector<float>& lut() const { return lut_; }        // 65536 entries, EOTF in cd/m^2
    const std::vector<float>& params() const { return params_; }  // CP_COUNT floats (kernels/color_math.h)
    const std::vector<float>& lut3d() const { return lut3_; }     // N^3 * 3 floats, empty without a LUT
    int lut3dSize() const { return lut3n_; }
    // One-line description of the active colour handling.
    std::string describe() const;

    // Convert one RGB48 pixel to integer 12-bit DCI X'Y'Z' codes.
    void toXYZ12(uint16_t r, uint16_t g, uint16_t b, int out[3]) const;

    // Full frame: interleaved RGB48 -> three float planes holding DC-shifted ICT
    // components (Y, Cb, Cr) ready for the wavelet transform. Also optionally
    // returns the integer X'Y'Z' image (for PSNR checks).
    void convertFrame(const uint16_t* rgb, int width, int height, float* planes[3],
                      uint16_t* xyz12 = nullptr) const;

private:
    ColorParams p_;
    std::vector<float> lut_, params_, lut3_;
    int lut3n_ = 0;
};

double pqEotf(double e);         // PQ code (0..1) -> cd/m^2
double pqInverseEotf(double nits);
ToneMap parseToneMap(const std::string& s);
ToneNorm parseToneNorm(const std::string& s);
TargetGamut parseGamut(const std::string& s);
// Reads a .cube 3D LUT (LUT_3D_SIZE N, red fastest). Throws on errors.
std::vector<float> readCubeLut(const std::string& path, int& size);

// Forward ICT used on DC-shifted X'Y'Z' (Annex G.3).
inline void ictForward(float x, float y, float z, float& c0, float& c1, float& c2) {
    c0 = 0.299f * x + 0.587f * y + 0.114f * z;
    c1 = -0.16875f * x - 0.33126f * y + 0.5f * z;
    c2 = 0.5f * x - 0.41869f * y - 0.08131f * z;
}

}  // namespace gail
