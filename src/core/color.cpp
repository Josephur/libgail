// SPDX-License-Identifier: MIT
#include "core/color.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace gail {

namespace {

constexpr double kPqM1 = 2610.0 / 16384.0;
constexpr double kPqM2 = 2523.0 / 4096.0 * 128.0;
constexpr double kPqC1 = 3424.0 / 4096.0;
constexpr double kPqC2 = 2413.0 / 4096.0 * 32.0;
constexpr double kPqC3 = 2392.0 / 4096.0 * 32.0;
constexpr double kDciNorm = 52.37;  // SMPTE ST 428-1 normalising constant

struct XY {
    double x, y;
};

void primariesXY(Primaries p, XY out[4]) {
    switch (p) {
    case Primaries::BT709:
        out[0] = {0.640, 0.330}; out[1] = {0.300, 0.600}; out[2] = {0.150, 0.060}; out[3] = {0.3127, 0.3290};
        break;
    case Primaries::BT2020:
        out[0] = {0.708, 0.292}; out[1] = {0.170, 0.797}; out[2] = {0.131, 0.046}; out[3] = {0.3127, 0.3290};
        break;
    case Primaries::P3D65:
        out[0] = {0.680, 0.320}; out[1] = {0.265, 0.690}; out[2] = {0.150, 0.060}; out[3] = {0.3127, 0.3290};
        break;
    case Primaries::DCIP3:
        out[0] = {0.680, 0.320}; out[1] = {0.265, 0.690}; out[2] = {0.150, 0.060}; out[3] = {0.314, 0.351};
        break;
    case Primaries::XYZ:
        break;
    }
}

void invert3(const double m[9], double r[9]) {
    const double det = m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) +
                       m[2] * (m[3] * m[7] - m[4] * m[6]);
    r[0] = (m[4] * m[8] - m[5] * m[7]) / det;
    r[1] = (m[2] * m[7] - m[1] * m[8]) / det;
    r[2] = (m[1] * m[5] - m[2] * m[4]) / det;
    r[3] = (m[5] * m[6] - m[3] * m[8]) / det;
    r[4] = (m[0] * m[8] - m[2] * m[6]) / det;
    r[5] = (m[2] * m[3] - m[0] * m[5]) / det;
    r[6] = (m[3] * m[7] - m[4] * m[6]) / det;
    r[7] = (m[1] * m[6] - m[0] * m[7]) / det;
    r[8] = (m[0] * m[4] - m[1] * m[3]) / det;
}

// RGB (linear, white = Y 1.0) -> XYZ.
void rgbToXyzMatrix(Primaries p, double M[9]) {
    if (p == Primaries::XYZ) {
        for (int i = 0; i < 9; ++i) M[i] = (i % 4 == 0) ? 1.0 : 0.0;
        return;
    }
    XY c[4] = {};
    primariesXY(p, c);
    double P[9];
    for (int i = 0; i < 3; ++i) {
        P[0 + i] = c[i].x / c[i].y;
        P[3 + i] = 1.0;
        P[6 + i] = (1.0 - c[i].x - c[i].y) / c[i].y;
    }
    const double W[3] = {c[3].x / c[3].y, 1.0, (1.0 - c[3].x - c[3].y) / c[3].y};
    double Pi[9];
    invert3(P, Pi);
    double S[3];
    for (int i = 0; i < 3; ++i) S[i] = Pi[3 * i] * W[0] + Pi[3 * i + 1] * W[1] + Pi[3 * i + 2] * W[2];
    for (int r = 0; r < 3; ++r)
        for (int col = 0; col < 3; ++col) M[3 * r + col] = P[3 * r + col] * S[col];
}

double hlgInverseOetf(double e) {
    const double a = 0.17883277, b = 1.0 - 4.0 * a, c = 0.5 - a * std::log(4.0 * a);
    return e <= 0.5 ? e * e / 3.0 : (std::exp((e - c) / a) + b) / 12.0;
}

}  // namespace

Transfer parseTransfer(const std::string& s) {
    if (s == "bt1886" || s == "bt709" || s == "gamma24" || s == "smpte170m" || s == "bt470bg" || s == "bt2020-10" ||
        s == "bt2020-12")
        return Transfer::BT1886;
    if (s == "srgb" || s == "iec61966-2-1") return Transfer::SRGB;
    if (s == "pq" || s == "smpte2084") return Transfer::PQ;
    if (s == "hlg" || s == "arib-std-b67") return Transfer::HLG;
    if (s == "linear") return Transfer::Linear;
    if (s == "gamma26" || s == "dci") return Transfer::Gamma26;
    throw std::runtime_error("unknown transfer function: " + s);
}

Primaries parsePrimaries(const std::string& s) {
    if (s == "bt709" || s == "srgb") return Primaries::BT709;
    if (s == "bt2020") return Primaries::BT2020;
    if (s == "p3d65" || s == "smpte432" || s == "display-p3") return Primaries::P3D65;
    if (s == "dcip3" || s == "smpte431") return Primaries::DCIP3;
    if (s == "xyz") return Primaries::XYZ;
    throw std::runtime_error("unknown primaries: " + s);
}

const char* transferName(Transfer t) {
    switch (t) {
    case Transfer::BT1886: return "bt1886";
    case Transfer::SRGB: return "srgb";
    case Transfer::PQ: return "pq";
    case Transfer::HLG: return "hlg";
    case Transfer::Linear: return "linear";
    case Transfer::Gamma26: return "gamma26";
    }
    return "?";
}

const char* primariesName(Primaries p) {
    switch (p) {
    case Primaries::BT709: return "bt709";
    case Primaries::BT2020: return "bt2020";
    case Primaries::P3D65: return "p3d65";
    case Primaries::DCIP3: return "dcip3";
    case Primaries::XYZ: return "xyz";
    }
    return "?";
}

namespace {

// std::lround for v in [0, 65535]: v - trunc(v) is exact in float, so this rounds half up
// exactly like lround does for non-negative values, without the libm call.
inline uint16_t roundTo16(float v) {
    const int r = (int)v;
    return (uint16_t)(r + ((v - (float)r) >= 0.5f ? 1 : 0));
}

template <typename S>
void planarRowImpl(const FrameFormat& f, const S* src, uint16_t* rgbRow, int ay) {
    const int aw = f.activeW, ah = f.activeH, cw = f.chromaW(), ch = f.chromaH();
    const size_t uOff = (size_t)aw * ah, vOff = uOff + (size_t)cw * ch;
    // samples are LSB-aligned: 8-bit in bytes, 10/12/16-bit in 16-bit words
    const float maxv = (float)((1 << f.bits) - 1), scale = (float)(1 << (f.bits - 8));
    float crR, cbG, crG, cbB;
    f.matrix(crR, cbG, crG, cbB);
    // chroma row for this luma row (see color_yuv.comp)
    float cyp = !f.sy ? (float)ay : f.chroma == FrameFormat::Chroma::TopLeft ? ay * 0.5f : (ay - 0.5f) * 0.5f;
    cyp = std::clamp(cyp, 0.f, (float)(ch - 1));
    const int y0 = (int)cyp, y1 = std::min(y0 + 1, ch - 1);
    const float fy = cyp - y0;
    const S* yRow = src + (size_t)ay * aw;
    const S* u0 = src + uOff + (size_t)y0 * cw;
    const S* u1 = src + uOff + (size_t)y1 * cw;
    const S* v0 = src + vOff + (size_t)y0 * cw;
    const S* v1 = src + vOff + (size_t)y1 * cw;
    const bool center = f.chroma == FrameFormat::Chroma::Center;
    uint16_t* o = rgbRow + (size_t)f.ox * 3;
    for (int x = 0; x < aw; ++x, o += 3) {
        float cxp = !f.sx ? (float)x : center ? (x - 0.5f) * 0.5f : x * 0.5f;
        cxp = std::clamp(cxp, 0.f, (float)(cw - 1));
        const int x0 = (int)cxp, x1 = std::min(x0 + 1, cw - 1);
        const float fx = cxp - x0;
        auto bil = [&](const S* r0, const S* r1) {
            const float a = (float)r0[x0], b = (float)r0[x1];
            const float c = (float)r1[x0], d = (float)r1[x1];
            return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy;
        };
        float Y = (float)yRow[x], U = bil(u0, u1), V = bil(v0, v1);
        if (f.rgb) {  // planes G', B', R'
            const float rgbf[3] = {V / maxv, Y / maxv, U / maxv};
            for (int c = 0; c < 3; ++c) o[c] = roundTo16(std::clamp(rgbf[c], 0.f, 1.f) * 65535.f);
            continue;
        }
        if (f.fullRange) {
            Y /= maxv;
            U = (U - 128.f * scale) / maxv;
            V = (V - 128.f * scale) / maxv;
        } else {
            Y = (Y - 16.f * scale) / (219.f * scale);
            U = (U - 128.f * scale) / (224.f * scale);
            V = (V - 128.f * scale) / (224.f * scale);
        }
        const float rgbf[3] = {Y + crR * V, Y - cbG * U - crG * V, Y + cbB * U};
        for (int c = 0; c < 3; ++c) o[c] = roundTo16(std::clamp(rgbf[c], 0.f, 1.f) * 65535.f);
    }
}

}  // namespace

void planarToRgb48Row(const FrameFormat& f, const void* src, uint16_t* rgbRow, int y) {
    const int ay = y - f.oy;
    if (ay < 0 || ay >= f.activeH) {
        std::fill(rgbRow, rgbRow + (size_t)f.width * 3, 0);
        return;
    }
    std::fill(rgbRow, rgbRow + (size_t)f.ox * 3, 0);
    std::fill(rgbRow + (size_t)(f.ox + f.activeW) * 3, rgbRow + (size_t)f.width * 3, 0);
    if (f.bits > 8) planarRowImpl(f, (const uint16_t*)src, rgbRow, ay);
    else planarRowImpl(f, (const uint8_t*)src, rgbRow, ay);
}

void planarToRgb48(const FrameFormat& f, const void* src, uint16_t* rgb) {
    for (int y = 0; y < f.height; ++y) planarToRgb48Row(f, src, rgb + (size_t)y * f.width * 3, y);
}

double pqEotf(double e) {
    e = std::clamp(e, 0.0, 1.0);
    const double p = std::pow(e, 1.0 / kPqM2);
    return 10000.0 * std::pow(std::max(p - kPqC1, 0.0) / (kPqC2 - kPqC3 * p), 1.0 / kPqM1);
}

double pqInverseEotf(double nits) {
    const double y = std::pow(std::clamp(nits / 10000.0, 0.0, 1.0), kPqM1);
    return std::pow((kPqC1 + kPqC2 * y) / (1.0 + kPqC3 * y), kPqM2);
}

ToneMap parseToneMap(const std::string& s) {
    if (s == "signal" || s == "dcpomatic" || s == "default") return ToneMap::Signal;
    if (s == "shoulder") return ToneMap::Shoulder;
    if (s == "bt2390") return ToneMap::BT2390;
    if (s == "none" || s == "clip") return ToneMap::None;
    throw std::runtime_error("unknown tone map: " + s);
}

ToneNorm parseToneNorm(const std::string& s) {
    if (s == "channel" || s == "rgb") return ToneNorm::Channel;
    if (s == "maxrgb") return ToneNorm::MaxRGB;
    if (s == "luminance" || s == "y") return ToneNorm::Luminance;
    throw std::runtime_error("unknown tone map norm: " + s);
}

TargetGamut parseGamut(const std::string& s) {
    if (s == "p3d65" || s == "p3") return TargetGamut::P3D65;
    if (s == "dcip3") return TargetGamut::DCIP3;
    if (s == "none" || s == "off") return TargetGamut::None;
    throw std::runtime_error("unknown target gamut: " + s);
}

std::vector<float> readCubeLut(const std::string& path, int& size) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open LUT " + path);
    std::vector<float> data;
    size = 0;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::string key;
        ss >> key;
        if (key == "LUT_3D_SIZE") {
            ss >> size;
        } else if (key == "LUT_1D_SIZE") {
            throw std::runtime_error("1D .cube LUTs are not supported: " + path);
        } else if (key == "DOMAIN_MIN" || key == "DOMAIN_MAX") {
            float a, b, c;
            ss >> a >> b >> c;
            const float want = key == "DOMAIN_MIN" ? 0.f : 1.f;
            if (a != want || b != want || c != want) throw std::runtime_error("LUT domain must be 0..1: " + path);
        } else if (key == "TITLE") {
            continue;
        } else {
            std::istringstream vs(line);
            float r, g, b;
            if (vs >> r >> g >> b) {
                data.push_back(r);
                data.push_back(g);
                data.push_back(b);
            }
        }
    }
    if (size < 2 || data.size() != (size_t)size * size * size * 3)
        throw std::runtime_error("malformed 3D LUT (size / entry count) in " + path);
    return data;
}

}  // namespace gail

// The shared per-pixel maths (kernels/color_math.h) takes this converter's tables as
// arguments, as the CUDA kernels do.
#define CM_ARGS [[maybe_unused]] const float* cp, [[maybe_unused]] const float* eotf, [[maybe_unused]] const float* lut3,
#define CM_PASS cp, eotf, lut3,
#define CP(i) cp[i]
#define EOTF(i) eotf[i]
#define LUT3(i) lut3[i]
#include "kernels/color_math.h"

namespace gail {

namespace {
void invertM(const double m[9], double r[9]) { invert3(m, r); }
}  // namespace

ColorConverter::ColorConverter(const ColorParams& p) : p_(p), lut_(65536), params_(CP_COUNT, 0.f) {
    const bool hdr = p.transfer == Transfer::PQ || p.transfer == Transfer::HLG;
    const bool passthrough = p.primaries == Primaries::XYZ;
    for (int i = 0; i < 65536; ++i) {
        const double e = i / 65535.0;
        double nits = 0;
        switch (p.transfer) {
        case Transfer::BT1886: nits = std::pow(e, 2.4) * p.sdrWhite; break;
        case Transfer::SRGB:
            nits = (e <= 0.04045 ? e / 12.92 : std::pow((e + 0.055) / 1.055, 2.4)) * p.sdrWhite;
            break;
        case Transfer::Linear: nits = e * p.sdrWhite; break;
        case Transfer::Gamma26: nits = std::pow(e, 2.6) * kDciNorm; break;
        case Transfer::PQ:
            nits = p.toneMap == ToneMap::Signal ? std::pow(e, 2.4) * p.sdrWhite : pqEotf(e);
            break;
        case Transfer::HLG:
            if (p.toneMap == ToneMap::Signal) {
                nits = std::pow(e, 2.4) * p.sdrWhite;
                break;
            }
            // Scene light -> display light for a 1000 cd/m^2 reference display (system gamma 1.2,
            // applied per channel here as an approximation of the luminance-based OOTF).
            nits = 1000.0 * std::pow(hlgInverseOetf(e), 1.2);
            break;
        }
        lut_[i] = (float)nits;
    }
    double M[9];
    rgbToXyzMatrix(p.primaries, M);
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c) params_[CP_RGB2XYZ + 4 * r + c] = (float)M[3 * r + c];
    params_[CP_PEAK] = (float)p.cinemaPeak;

    if (hdr && !passthrough && p.toneMap == ToneMap::Shoulder) {
        const double gain = p.cinemaRefWhite / p.cinemaPeak;
        double knee = std::clamp(p.knee, 0.05, 0.99);
        const double vmax = gain * p.hdrPeak / p.hdrRefWhite;  // source peak on the output scale
        if (vmax <= 1.0) knee = 1.0;  // the whole source range already fits: no shoulder
        params_[CP_TONE] = 1.f;
        params_[CP_SRC_REF] = (float)p.hdrRefWhite;
        params_[CP_GAIN] = (float)gain;
        params_[CP_KNEE] = (float)knee;
        params_[CP_SHOULDER_T] = (float)std::max(1e-3, (vmax - knee) / (1.0 - knee));
    } else if (hdr && !passthrough && p.toneMap == ToneMap::BT2390 && p.hdrPeak > p.cinemaPeak) {
        params_[CP_TONE] = 2.f;
        params_[CP_PQ_SRCMAX] = (float)pqInverseEotf(p.hdrPeak);
        params_[CP_PQ_MAXLUM] = (float)(pqInverseEotf(p.cinemaPeak) / pqInverseEotf(p.hdrPeak));
        params_[CP_PQ_KS] = std::max(0.f, 1.5f * params_[CP_PQ_MAXLUM] - 0.5f);
    }

    params_[CP_TONE_NORM] = p.toneNorm == ToneNorm::Channel ? 2.f : p.toneNorm == ToneNorm::MaxRGB ? 1.f : 0.f;
    if (!passthrough) {
        // target gamut matrices: used by gamut compression and by the max(R,G,B) tone norm
        double T[9], Ti[9];
        rgbToXyzMatrix(p.gamut == TargetGamut::DCIP3 ? Primaries::DCIP3 : Primaries::P3D65, T);
        invertM(T, Ti);
        for (int i = 0; i < 9; ++i) {
            params_[CP_XYZ2TGT + i] = (float)(Ti[i] / p.cinemaPeak);
            params_[CP_TGT2XYZ + i] = (float)(T[i] * p.cinemaPeak);
        }
        if (p.gamut != TargetGamut::None) {
            params_[CP_GAMUT] = 1.f;
            params_[CP_GAMUT_THRESH] = (float)std::clamp(p.gamutThreshold, 0.0, 0.99);
        }
    }

    if (!p.lut3dData.empty()) {
        if (p.lut3dSize < 2 || p.lut3dData.size() != (size_t)p.lut3dSize * p.lut3dSize * p.lut3dSize * 3)
            throw std::runtime_error("3D LUT data does not match its size");
        lut3_ = p.lut3dData;
        lut3n_ = p.lut3dSize;
    } else if (!p.lut3d.empty()) {
        lut3_ = readCubeLut(p.lut3d, lut3n_);
    }
    params_[CP_LUT_SIZE] = (float)lut3n_;
}

std::string ColorConverter::describe() const {
    std::ostringstream o;
    if (lut3n_) {
        o << "3D LUT " << (p_.lut3d.empty() ? std::string("(app supplied)") : p_.lut3d) << " (" << lut3n_ << "^3)";
        return o.str();
    }
    if (p_.primaries == Primaries::XYZ) return "X'Y'Z' pass-through";
    o << transferName(p_.transfer) << " " << primariesName(p_.primaries);
    if ((p_.transfer == Transfer::PQ || p_.transfer == Transfer::HLG) && p_.toneMap == ToneMap::Signal)
        o << ", HDR signal as gamma 2.4 (as DCP-o-matic)";
    if (params_[CP_TONE] == 1.f)
        o << ", tone map shoulder (" << p_.hdrRefWhite << " -> " << p_.cinemaRefWhite << " cd/m2, knee " << p_.knee
          << ", source peak " << p_.hdrPeak << ")";
    else if (params_[CP_TONE] == 2.f)
        o << ", tone map BT.2390 (source peak " << p_.hdrPeak << ")";
    if (params_[CP_TONE] != 0.f) o << (p_.toneNorm == ToneNorm::Channel  ? " per channel"
                                     : p_.toneNorm == ToneNorm::MaxRGB ? " on max(R,G,B)"
                                                                       : " on luminance");
    o << ", gamut "
      << (params_[CP_GAMUT] ? (p_.gamut == TargetGamut::DCIP3 ? "DCI-P3" : "P3-D65") : std::string("unconstrained"));
    if (params_[CP_GAMUT]) o << " (soft from " << p_.gamutThreshold << ")";
    o << ", peak " << p_.cinemaPeak << " cd/m2";
    return o.str();
}

namespace {
inline void pixelToXYZ12(CM_ARGS uint16_t r, uint16_t g, uint16_t b, int out[3]) {
    float x, y, z;
    cm_rgb_to_xyz12(CM_PASS r / 65535.f, g / 65535.f, b / 65535.f, x, y, z);
    out[0] = (int)x;
    out[1] = (int)y;
    out[2] = (int)z;
}
}  // namespace

void ColorConverter::toXYZ12(uint16_t r, uint16_t g, uint16_t b, int out[3]) const {
    pixelToXYZ12(params_.data(), lut_.data(), lut3_.data(), r, g, b, out);
}

void ColorConverter::convertFrame(const uint16_t* rgb, int width, int height, float* planes[3],
                                  uint16_t* xyz12) const {
    const float* cp = params_.data();
    const float* eotf = lut_.data();
    const float* lut3 = lut3_.data();
    const size_t n = (size_t)width * height;
    for (size_t i = 0; i < n; ++i) {
        int c[3];
        pixelToXYZ12(CM_PASS rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2], c);
        if (xyz12) {
            xyz12[3 * i] = (uint16_t)c[0];
            xyz12[3 * i + 1] = (uint16_t)c[1];
            xyz12[3 * i + 2] = (uint16_t)c[2];
        }
        ictForward((float)(c[0] - 2048), (float)(c[1] - 2048), (float)(c[2] - 2048), planes[0][i], planes[1][i],
                   planes[2][i]);
    }
}

}  // namespace gail
