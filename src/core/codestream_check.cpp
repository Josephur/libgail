// SPDX-License-Identifier: MIT
#include "core/codestream_check.h"

#include <cmath>
#include <sstream>

namespace gail {

namespace {

struct Reader {
    const uint8_t* p;
    const uint8_t* end;
    bool fail = false;
    size_t pos(const uint8_t* base) const { return (size_t)(p - base); }
    uint32_t u8() {
        if (p >= end) {
            fail = true;
            return 0;
        }
        return *p++;
    }
    uint32_t u16() {
        uint32_t a = u8();
        return (a << 8) | u8();
    }
    uint32_t u32() {
        uint32_t a = u16();
        return (a << 16) | u16();
    }
};

std::string hex(uint32_t v) {
    char b[16];
    snprintf(b, sizeof b, "0x%X", v);
    return b;
}

}  // namespace

CheckReport checkDciCodestream(const uint8_t* data, size_t size, double fps) {
    CheckReport r;
    r.totalBytes = size;
    Reader rd{data, data + size};
    if (rd.u16() != 0xFF4F) {
        r.error("codestream does not start with SOC");
        return r;
    }
    if (rd.u16() != 0xFF51) {
        r.error("SIZ must immediately follow SOC");
        return r;
    }
    const uint32_t lsiz = rd.u16();
    if (lsiz != 47) r.error("Lsiz must be 47 for 3 components, found " + std::to_string(lsiz));
    r.rsiz = (int)rd.u16();
    r.width = (int)rd.u32();
    r.height = (int)rd.u32();
    const uint32_t xo = rd.u32(), yo = rd.u32(), tw = rd.u32(), th = rd.u32(), txo = rd.u32(), tyo = rd.u32();
    r.comps = (int)rd.u16();
    const bool fourk = r.width > 2048;
    if (fourk && r.rsiz != 4) r.error("Rsiz must be 4 (CINEMA4K) for 4K, found " + std::to_string(r.rsiz));
    if (!fourk && r.rsiz != 3) r.error("Rsiz must be 3 (CINEMA2K) for 2K, found " + std::to_string(r.rsiz));
    if (fourk ? (r.width > 4096 || r.height > 2160) : (r.width > 2048 || r.height > 1080))
        r.error("image exceeds DCI container: " + std::to_string(r.width) + "x" + std::to_string(r.height));
    if (xo || yo) r.error("image offset must be 0");
    if (tw != (uint32_t)r.width || th != (uint32_t)r.height) r.error("frame must be a single tile");
    if (txo || tyo) r.error("tile offset must be 0");
    if (r.comps != 3) r.error("Csiz must be 3");
    for (int c = 0; c < r.comps && c < 3; ++c) {
        const uint32_t s = rd.u8(), xr = rd.u8(), yr = rd.u8();
        r.depth = (int)(s & 0x7F) + 1;
        if (s != 11) r.error("component " + std::to_string(c) + ": Ssiz must be 11 (12-bit unsigned), found " + std::to_string(s));
        if (xr != 1 || yr != 1) r.error("component subsampling must be 1x1");
    }

    int nCOD = 0, nQCD = 0, nPOCmain = 0, nPOClate = 0;
    bool inMain = true, sawEOC = false;
    std::vector<uint32_t> tlmLengths;
    int expectedTP = fourk ? 6 : 3;
    int nextTP = 0;
    const int nres = fourk ? 7 : 6;
    const double maxFrame = std::rint(250e6 / (8 * fps)), riskyFrame = std::rint(230e6 / (8 * fps));
    const double maxTP = std::floor(200e6 / (8 * fps));

    // End of the marker segment whose length field (Lxxx, which counts itself) was just
    // read, or nullptr when the segment does not fit in the codestream: that is reported
    // and ends the parse, so no handler reads past `end` or walks backwards.
    auto segment = [&](uint32_t marker, size_t mpos, uint32_t len) -> const uint8_t* {
        if (len < 2 || (size_t)(rd.end - rd.p) < (size_t)len - 2) {
            r.error("marker " + hex(marker) + " at offset " + std::to_string(mpos) + ": segment length " +
                    std::to_string(len) + " runs past the end of the codestream");
            rd.fail = true;
            return nullptr;
        }
        return rd.p + len - 2;
    };

    while (!rd.fail && rd.p < rd.end) {
        const size_t mpos = rd.pos(data);
        const uint32_t m = rd.u16();
        if ((m >> 8) != 0xFF) {
            r.error("expected marker at offset " + std::to_string(mpos));
            break;
        }
        switch (m) {
        case 0xFF52: {  // COD
            ++nCOD;
            if (!inMain) r.error("COD in tile-part header");
            const uint32_t len = rd.u16();
            const uint8_t* next = segment(m, mpos, len);
            if (!next) break;
            const uint32_t scod = rd.u8();
            r.progression = (int)rd.u8();
            r.layers = (int)rd.u16();
            r.mct = (int)rd.u8();
            r.levels = (int)rd.u8();
            r.cblkW = 1 << (rd.u8() + 2);
            r.cblkH = 1 << (rd.u8() + 2);
            r.cblkStyle = (int)rd.u8();
            r.transform = (int)rd.u8();
            if (scod != 1) r.error("Scod must be 1 (precincts defined, no SOP/EPH), found " + hex(scod));
            if (r.progression != 4) r.error("progression order must be CPRL");
            if (r.layers != 1) r.error("exactly one quality layer is required");
            if (r.mct != 1) r.error("multiple component transform flag must be 1");
            if (r.levels != (fourk ? 6 : 5)) r.error("decomposition levels must be " + std::string(fourk ? "6" : "5"));
            if (r.cblkW != 32 || r.cblkH != 32) r.error("code-blocks must be 32x32");
            if (r.cblkStyle != 0) r.error("code-block style (mode switches) must be 0");
            if (r.transform != 0) r.error("wavelet must be 9/7 irreversible");
            if (len != (uint32_t)(12 + nres)) r.error("unexpected Lcod");
            for (int i = 0; i < nres; ++i) {
                const int pp = (int)rd.u8();
                r.precincts.push_back(pp);
                if (pp != (i == 0 ? 0x77 : 0x88)) r.error("precinct sizes must be 128 (LL) / 256 (others)");
            }
            rd.p = next;
            break;
        }
        case 0xFF5C: {  // QCD
            ++nQCD;
            const uint32_t len = rd.u16();
            const uint8_t* next = segment(m, mpos, len);
            if (!next) break;
            const uint32_t sq = rd.u8();
            r.guardBits = (int)(sq >> 5);
            r.qstyle = (int)(sq & 0x1F);
            if (r.qstyle != 2) r.error("quantisation must be scalar expounded");
            if (fourk && r.guardBits != 2) r.error("4K requires 2 guard bits");
            if (!fourk && r.guardBits != 1) r.error("2K requires 1 guard bit");
            const int nb = 3 * (fourk ? 6 : 5) + 1;
            if (len != (uint32_t)(3 + 2 * nb)) r.error("unexpected Lqcd");
            for (int i = 0; i < nb && rd.p + 1 < next; ++i) {
                const uint32_t spq = rd.u16();
                const int eps = (int)(spq >> 11);
                if (r.guardBits + eps - 1 > 30)
                    r.error("subband " + std::to_string(i) + " needs " + std::to_string(r.guardBits + eps - 1) +
                            " magnitude bitplanes (decoders support at most 30)");
            }
            rd.p = next;
            break;
        }
        case 0xFF5F: {  // POC
            if (inMain) ++nPOCmain;
            else ++nPOClate;
            const uint32_t len = rd.u16();
            const uint8_t* next = segment(m, mpos, len);
            if (!next) break;
            static const int expect[14] = {0, 0, 1, 6, 3, 4, 6, 0, 1, 7, 3, 4};
            if (len != 16) r.error("POC length must be 16");
            else {
                const int vals[12] = {(int)rd.u8(), (int)rd.u8(), (int)rd.u16(), (int)rd.u8(), (int)rd.u8(), (int)rd.u8(),
                                      (int)rd.u8(), (int)rd.u8(), (int)rd.u16(), (int)rd.u8(), (int)rd.u8(), (int)rd.u8()};
                for (int i = 0; i < 12; ++i)
                    if (vals[i] != expect[i]) r.error("POC values differ from the DCI 4K progression");
            }
            rd.p = next;
            break;
        }
        case 0xFF55: {  // TLM
            r.tlm = true;
            const uint32_t len = rd.u16();
            const uint8_t* next = segment(m, mpos, len);
            if (!next) break;
            rd.u8();  // Ztlm
            const uint32_t st = rd.u8();
            const int stBytes = (st >> 4) & 3, spBytes = ((st >> 6) & 1) ? 4 : 2;
            const int entry = (stBytes == 3 ? 0 : stBytes) + spBytes;
            const int n = (int)(len - 4) / entry;
            for (int i = 0; i < n; ++i) {
                for (int b = 0; b < (stBytes == 3 ? 0 : stBytes); ++b) rd.u8();
                tlmLengths.push_back(spBytes == 4 ? rd.u32() : rd.u16());
            }
            rd.p = next;
            break;
        }
        case 0xFF53: case 0xFF5D: case 0xFF64: {  // COC, QCC, COM
            const uint32_t len = rd.u16();
            const uint8_t* next = segment(m, mpos, len);
            if (!next) break;
            // Rcom = 1: Latin-1 text; the text is the len - 4 bytes up to `next` (in bounds)
            if (m == 0xFF64 && len >= 4 && inMain && rd.p[0] == 0 && rd.p[1] == 1)
                r.comment.assign((const char*)rd.p + 2, len - 4);
            rd.p = next;
            if (m == 0xFF53) r.warnings.push_back("COC marker present");
            break;
        }
        case 0xFF90: {  // SOT
            inMain = false;
            const uint8_t* sot = rd.p - 2;
            const uint32_t lsot = rd.u16();
            const uint32_t isot = rd.u16();
            const uint32_t psot = rd.u32();
            const uint32_t tpsot = rd.u8();
            const uint32_t tnsot = rd.u8();
            if (lsot != 10) r.error("Lsot must be 10");
            if (isot != 0) r.error("tile index must be 0");
            if ((int)tnsot != expectedTP) r.error("TNsot must be " + std::to_string(expectedTP) + ", found " + std::to_string(tnsot));
            if ((int)tpsot != nextTP) r.error("tile-parts out of order");
            ++nextTP;
            r.tilePartBytes.push_back(psot);
            if (psot > maxTP)
                r.error("tile-part " + std::to_string(tpsot) + " is " + std::to_string(psot) + " bytes, above the " +
                        std::to_string((size_t)maxTP) + " byte limit");
            const int comp = (int)tpsot % 3;
            r.compBytes[comp] += psot;
            if (psot < 14 || (size_t)psot > (size_t)(rd.end - sot)) {  // compared, never formed, past the end
                r.error("invalid Psot");
                rd.fail = true;
                break;
            }
            if (rd.u16() != 0xFF93) r.error("SOD must follow SOT");
            // scan packet data for false markers
            for (const uint8_t* q = rd.p; q + 1 < sot + psot; ++q)
                if (q[0] == 0xFF && q[1] >= 0x90) {
                    r.error("marker-like byte pair 0xFF" + hex(q[1]).substr(2) + " inside packet data at offset " +
                            std::to_string((size_t)(q - data)));
                    break;
                }
            rd.p = sot + psot;
            break;
        }
        case 0xFFD9:
            sawEOC = true;
            if (rd.p != rd.end) r.error("data after EOC");
            rd.p = rd.end;
            break;
        default:
            r.error("marker " + hex(m) + " is not allowed in DCI codestreams");
            rd.fail = true;
        }
    }
    r.tileParts = nextTP;
    r.pocCount = nPOCmain;
    if (rd.fail) r.error("truncated or malformed codestream");
    if (!sawEOC) r.error("missing EOC");
    if (nCOD != 1) r.error("exactly one COD required");
    if (nQCD != 1) r.error("exactly one QCD required");
    if (fourk && nPOCmain != 1) r.error("4K requires exactly one POC in the main header");
    if (!fourk && nPOCmain != 0) r.error("2K must not contain POC");
    if (nPOClate) r.error("POC must be in the main header");
    if (!r.tlm) r.error("TLM marker required");
    if (r.tileParts != expectedTP) r.error("expected " + std::to_string(expectedTP) + " tile-parts");
    if (r.tlm) {
        if (tlmLengths.size() != r.tilePartBytes.size()) r.error("TLM entry count does not match tile-parts");
        else
            for (size_t i = 0; i < tlmLengths.size(); ++i)
                if (tlmLengths[i] != r.tilePartBytes[i]) r.error("TLM length mismatch for tile-part " + std::to_string(i));
    }
    const double maxComp = std::floor(200e6 / (8 * fps));
    for (int c = 0; c < 3; ++c)
        if (r.compBytes[c] > maxComp)
            r.error("component " + std::to_string(c) + " uses " + std::to_string(r.compBytes[c]) + " bytes, above " +
                    std::to_string((size_t)maxComp));
    if ((double)size > maxFrame)
        r.error("frame is " + std::to_string(size) + " bytes, above the 250 Mbit/s limit (" + std::to_string((size_t)maxFrame) + ")");
    else if ((double)size > riskyFrame)
        r.warnings.push_back("frame is above 230 Mbit/s (libdcp NEARLY_INVALID_PICTURE_FRAME_SIZE_IN_BYTES)");
    return r;
}

std::string formatReport(const CheckReport& r, bool verbose) {
    std::ostringstream o;
    o << (r.ok ? "PASS" : "FAIL") << "  " << r.profile() << " " << r.width << "x" << r.height << "  " << r.totalBytes
      << " bytes";
    if (verbose) {
        o << "\n  Rsiz=" << r.rsiz << " comps=" << r.comps << " depth=" << r.depth << " levels=" << r.levels
          << " layers=" << r.layers << " progression=" << (r.progression == 4 ? "CPRL" : std::to_string(r.progression))
          << " mct=" << r.mct << " cblk=" << r.cblkW << "x" << r.cblkH << " style=" << r.cblkStyle
          << " wavelet=" << (r.transform == 0 ? "9/7" : "5/3") << " guard=" << r.guardBits << " tileparts=" << r.tileParts
          << " POC=" << r.pocCount << " TLM=" << (r.tlm ? "yes" : "no") << "\n  component bytes: " << r.compBytes[0]
          << " " << r.compBytes[1] << " " << r.compBytes[2];
        if (!r.comment.empty()) o << "\n  comment: " << r.comment;
    }
    for (auto& e : r.errors) o << "\n  error: " << e;
    for (auto& w : r.warnings) o << "\n  warning: " << w;
    return o.str();
}

}  // namespace gail
