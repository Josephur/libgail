// SPDX-License-Identifier: MIT
#include "core/codestream_parse.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "core/markers.h"

namespace gail {

namespace {

// Big-endian reads confined to [0, size): past the end they return 0 and set fail.
struct Reader {
    const uint8_t* base;
    size_t size;
    size_t pos = 0;
    bool fail = false;
    uint32_t u8() {
        if (pos >= size) {
            fail = true;
            return 0;
        }
        return base[pos++];
    }
    uint32_t u16() {
        const uint32_t a = u8();
        return (a << 8) | u8();
    }
    uint32_t u32() {
        const uint32_t a = u16();
        return (a << 16) | u16();
    }
    size_t left() const { return size - pos; }
};

std::string hex(uint32_t v) {
    char b[16];
    std::snprintf(b, sizeof b, "0x%04X", v);
    return b;
}

void notSupported(StreamHeader& h, const std::string& why) {
    if (!h.supported) return;  // keep the first reason
    h.supported = false;
    h.unsupported = why;
}

// The fields COD and COC share (A.6.1, A.6.2).
struct CodingStyle {
    int levels = 0, xcb = 0, ycb = 0, style = 0, transform = 0;
    bool userPrecincts = false;
    uint8_t precinct[kMaxLevels + 1] = {};
    bool read(Reader& r, bool precincts) {
        levels = (int)r.u8();
        xcb = (int)r.u8() + 2;
        ycb = (int)r.u8() + 2;
        style = (int)r.u8();
        transform = (int)r.u8();
        userPrecincts = precincts;
        if (levels > kMaxLevels) return false;
        for (int i = 0; i <= levels; ++i) precinct[i] = precincts ? (uint8_t)r.u8() : 0xFF;
        return !r.fail && xcb <= 10 && ycb <= 10 && xcb + ycb <= 12 && transform <= 1;
    }
    bool sameAs(const CodingStyle& o) const {
        if (levels != o.levels || xcb != o.xcb || ycb != o.ycb || style != o.style || transform != o.transform) return false;
        for (int i = 0; i <= levels; ++i)
            if (precinct[i] != o.precinct[i]) return false;
        return true;
    }
};

// The body of QCD and QCC (A.6.4, A.6.5).
struct Quantization {
    bool present = false;
    int style = 0, guard = 0, count = 0;
    int eps[kMaxBands] = {}, mu[kMaxBands] = {};
    bool read(Reader& r) {
        const uint32_t s = r.u8();
        style = (int)(s & 31);
        guard = (int)(s >> 5);
        count = 0;
        if (style == 0) {  // no quantization: one exponent per band, 5 bits
            while (r.left() > 0 && count < kMaxBands) {
                eps[count] = (int)(r.u8() >> 3);
                mu[count++] = 0;
            }
        } else if (style == 1 || style == 2) {  // scalar derived (one pair) or expounded
            while (r.left() >= 2 && count < kMaxBands) {
                const uint32_t v = r.u16();
                eps[count] = (int)(v >> 11);
                mu[count++] = (int)(v & 0x7FF);
            }
        } else {
            return false;
        }
        present = true;
        return !r.fail && r.left() == 0 && count >= 1 && (style != 1 || count == 1);
    }
};

int bandLevel(int levels, int band) { return band == 0 ? levels : levels - (1 + (band - 1) / 3) + 1; }

}  // namespace

bool parseCodestream(const uint8_t* data, size_t size, bool headerOnly, StreamHeader& h, std::string& error) {
    h = StreamHeader{};
    Reader rd{data, size};
    auto bad = [&](const std::string& what) {
        error = what;
        return false;
    };

    if (rd.u16() != marker::SOC || rd.fail) return bad("not a JPEG 2000 codestream (no SOC marker)");
    if (rd.u16() != marker::SIZ || rd.fail) return bad("SIZ marker missing after SOC");

    // A marker segment's bytes after its length field, as a reader of its own: a handler
    // cannot read past the segment, and the segment cannot run past the data.
    uint32_t segMarker = 0;
    auto segment = [&](uint32_t m, Reader& seg) {
        segMarker = m;
        const size_t at = rd.pos - 2;
        const uint32_t len = rd.u16();
        if (rd.fail || len < 2 || rd.left() < (size_t)len - 2) {
            error = "marker " + hex(m) + " at offset " + std::to_string(at) + ": segment runs past the end of the codestream";
            return false;
        }
        seg = Reader{data, rd.pos + len - 2, rd.pos};
        rd.pos += len - 2;
        return true;
    };
    auto shortSegment = [&] { return bad("marker " + hex(segMarker) + ": segment too short for its contents"); };

    Reader seg{data, 0};
    if (!segment(marker::SIZ, seg)) return false;
    h.rsiz = (int)seg.u16();
    const uint64_t xsiz = seg.u32(), ysiz = seg.u32(), xo = seg.u32(), yo = seg.u32();
    const uint64_t xt = seg.u32(), yt = seg.u32(), xto = seg.u32(), yto = seg.u32();
    const uint32_t csiz = seg.u16();
    if (seg.fail) return shortSegment();
    if (csiz < 1 || csiz > 16384 || xsiz <= xo || ysiz <= yo || xt == 0 || yt == 0 || xto > xo || yto > yo ||
        xto + xt <= xo || yto + yt <= yo)
        return bad("SIZ: impossible image or tile dimensions");
    if (xsiz - xo > 0x7FFFFFFFu || ysiz - yo > 0x7FFFFFFFu) return bad("SIZ: image too large");
    h.width = (int)(xsiz - xo);
    h.height = (int)(ysiz - yo);
    h.components = (int)csiz;
    const uint64_t tilesX = (xsiz - xto + xt - 1) / xt, tilesY = (ysiz - yto + yt - 1) / yt;
    h.tiles = tilesX * tilesY > 0x7FFFFFFFu ? 0x7FFFFFFF : (int)(tilesX * tilesY);
    bool plain12 = true, subsampled = false;
    for (uint32_t c = 0; c < csiz; ++c) {
        const uint32_t ssiz = seg.u8(), xr = seg.u8(), yr = seg.u8();
        if (seg.fail) return shortSegment();
        if (xr == 0 || yr == 0) return bad("SIZ: component subsampling of zero");
        if (c == 0) h.bitDepth = (int)(ssiz & 0x7F) + 1;
        plain12 = plain12 && ssiz == 11;
        subsampled = subsampled || xr != 1 || yr != 1;
    }
    h.highThroughput = (h.rsiz & 0x4000) != 0;
    if (h.highThroughput) notSupported(h, "high-throughput (Part 15) block coding");
    if (h.rsiz != 3 && h.rsiz != 4) notSupported(h, "not a DCI cinema profile (Rsiz " + std::to_string(h.rsiz) + ")");
    if (csiz != kNumComps) notSupported(h, std::to_string(csiz) + " components (3 needed)");
    if (!plain12) notSupported(h, "components are not 12-bit unsigned");
    if (subsampled) notSupported(h, "subsampled components");
    if (xo || yo || xto || yto) notSupported(h, "image or tile offset");
    if (h.tiles != 1) notSupported(h, "more than one tile");
    const bool fourk = h.rsiz == 4;
    if (h.width < 64 || h.height < 64 || (fourk ? (h.width > 4096 || h.height > 2160) : (h.width > 2048 || h.height > 1080)))
        notSupported(h, "image size " + std::to_string(h.width) + "x" + std::to_string(h.height) +
                            " outside the DCI container of its profile");

    // ---- the rest of the main header
    CodingStyle cod;
    bool haveCod = false;
    std::vector<CodingStyle> cocs;
    Quantization qcd, qcc[kNumComps];
    struct PocEntry {
        int rs, cs, lye, re, ce, order;
    };
    std::vector<PocEntry> poc;
    uint32_t m = 0;
    for (;;) {
        if (rd.left() < 2) return bad("codestream ends inside the main header");
        const size_t at = rd.pos;
        m = rd.u16();
        if ((m >> 8) != 0xFF) return bad("expected a marker at offset " + std::to_string(at));
        if (m == marker::SOT || m == marker::EOC) break;
        if (m == marker::SOC || m == marker::SOD || (m >= 0xFF30 && m <= 0xFF3F))
            return bad("marker " + hex(m) + " not allowed in the main header");
        if (!segment(m, seg)) return false;
        switch (m) {
        case marker::COD: {
            if (haveCod) return bad("more than one COD marker");
            haveCod = true;
            const uint32_t scod = seg.u8();
            h.progression = (int)seg.u8();
            h.layers = (int)seg.u16();
            h.mct = (int)seg.u8();
            if (!cod.read(seg, scod & 1) || h.progression > 4 || h.layers < 1) return bad("COD: invalid coding style");
            h.sop = (scod & 2) != 0;
            h.eph = (scod & 4) != 0;
            break;
        }
        case marker::COC: {
            if (csiz < 257) seg.u8();
            else seg.u16();
            const uint32_t scoc = seg.u8();
            CodingStyle c;
            if (!c.read(seg, scoc & 1)) return bad("COC: invalid coding style");
            cocs.push_back(c);
            break;
        }
        case marker::QCD:
            if (qcd.present) return bad("more than one QCD marker");
            if (!qcd.read(seg)) return bad("QCD: invalid quantization");
            break;
        case marker::QCC: {
            const uint32_t comp = csiz < 257 ? seg.u8() : seg.u16();
            Quantization q;
            if (!q.read(seg) || comp >= csiz) return bad("QCC: invalid quantization");
            if (comp < (uint32_t)kNumComps) qcc[comp] = q;
            break;
        }
        case marker::POC: {
            const size_t entry = csiz < 257 ? 7 : 9;
            if (seg.left() == 0 || seg.left() % entry) return bad("POC: invalid length");
            while (seg.left() > 0) {
                PocEntry e;
                e.rs = (int)seg.u8();
                e.cs = (int)(csiz < 257 ? seg.u8() : seg.u16());
                e.lye = (int)seg.u16();
                e.re = (int)seg.u8();
                e.ce = (int)(csiz < 257 ? seg.u8() : seg.u16());
                e.order = (int)seg.u8();
                if (csiz < 257 && e.ce == 0) e.ce = 256;
                poc.push_back(e);
            }
            break;
        }
        case 0xFF50:  // CAP: capabilities beyond Part 1
            h.highThroughput = true;
            notSupported(h, "high-throughput (Part 15) block coding");
            break;
        case marker::RGN: notSupported(h, "region of interest"); break;
        case marker::PPM: notSupported(h, "packed packet headers"); break;
        case marker::TLM:
        case marker::PLM:
        case marker::CRG:
        case marker::COM: break;  // nothing the decoder needs
        default: notSupported(h, "marker " + hex(m) + " in the main header"); break;
        }
    }
    if (!haveCod || !qcd.present) return bad("COD or QCD marker missing");

    h.levels = cod.levels;
    h.cblkLog2W = cod.xcb;
    h.cblkLog2H = cod.ycb;
    h.cblkStyle = cod.style;
    h.reversible = cod.transform == 1;
    if (h.reversible) notSupported(h, "reversible 5/3 wavelet");
    if (h.progression != 4) notSupported(h, "progression order " + std::to_string(h.progression) + " (CPRL needed)");
    if (h.layers != 1) notSupported(h, std::to_string(h.layers) + " quality layers (1 needed)");
    if (h.mct != 1) notSupported(h, "no multiple component transform");
    if (h.levels < 1 || h.levels > (fourk ? 6 : 5))
        notSupported(h, std::to_string(h.levels) + " decomposition levels");
    if (cod.xcb != kCblkLog2 || cod.ycb != kCblkLog2) notSupported(h, "code-blocks are not 32x32");
    if (cod.style != 0) notSupported(h, "code-block style " + hex((uint32_t)cod.style));
    bool dciPrecincts = cod.userPrecincts;
    for (int r = 0; r <= cod.levels && dciPrecincts; ++r) dciPrecincts = cod.precinct[r] == (r == 0 ? 0x77 : 0x88);
    if (!dciPrecincts) notSupported(h, "precinct sizes other than 128 (lowest resolution) and 256");
    for (const CodingStyle& c : cocs)
        if (!c.sameAs(cod)) notSupported(h, "per-component coding style");

    // Step sizes for every band of each component (E.1): expounded as signaled, derived
    // from the LL band's pair by its decomposition level
    h.guardBits = qcd.guard;
    const int nb = 3 * h.levels + 1;
    for (int c = 0; c < kNumComps; ++c) {
        const Quantization& q = qcc[c].present ? qcc[c] : qcd;
        if (q.guard != qcd.guard) notSupported(h, "per-component guard bits");
        if (q.style == 0) notSupported(h, "no quantization");
        if (q.style != 1 && q.count < nb) return bad("too few step sizes for the decomposition levels");
        for (int i = 0; i < nb; ++i) {
            int eps = q.style == 1 ? q.eps[0] - h.levels + bandLevel(h.levels, i) : q.eps[i];
            const int mu = q.style == 1 ? q.mu[0] : q.mu[i];
            if (eps < 0) return bad("derived step size exponent below zero");
            h.eps[c][i] = eps;
            h.mu[c][i] = mu;
            const int Mb = h.guardBits + eps - 1;
            if (Mb < 0 || Mb > 22) notSupported(h, "step size with " + std::to_string(Mb) + " bit-planes");
        }
    }

    // Packet order: all resolutions at once, or the POC's ranges of resolutions in turn
    // (A.6.6), accepted when each covers every component and they follow on from each other
    if (poc.empty()) {
        h.volumes.push_back({0, h.levels + 1});
    } else {
        int next = 0;
        for (const PocEntry& e : poc) {
            const int re = std::min(e.re, h.levels + 1);
            if (e.order != 4 || e.cs != 0 || e.ce < h.components || e.lye < h.layers || e.rs != next || re <= e.rs) {
                notSupported(h, "progression order change other than by resolution");
                break;
            }
            h.volumes.push_back({e.rs, re});
            next = re;
        }
        if (h.supported && next != h.levels + 1) notSupported(h, "progression order change leaves resolutions out");
        if (!h.supported) h.volumes.assign(1, {0, h.levels + 1});
    }
    if (headerOnly) return true;

    // ---- tile-parts (A.4.2): the packets of each lie between its SOD and its end
    while (m == marker::SOT) {
        const size_t sotPos = rd.pos - 2;
        const uint32_t lsot = rd.u16(), isot = rd.u16(), psot = rd.u32(), tpsot = rd.u8();
        rd.u8();  // TNsot
        if (rd.fail || lsot != 10) return bad("SOT at offset " + std::to_string(sotPos) + " is malformed");
        if (isot != 0) notSupported(h, "more than one tile");
        size_t tpEnd;
        if (psot == 0) {  // the last tile-part: up to the EOC
            tpEnd = size >= 2 && data[size - 2] == 0xFF && data[size - 1] == 0xD9 ? size - 2 : size;
        } else {
            if (psot < 14 || size - sotPos < psot) return bad("tile-part at offset " + std::to_string(sotPos) + " runs past the end");
            tpEnd = sotPos + psot;
        }
        for (;;) {
            if (tpEnd - rd.pos < 2 || rd.pos > tpEnd) return bad("tile-part header at offset " + std::to_string(sotPos) + " has no SOD");
            const size_t at = rd.pos;
            const uint32_t t = rd.u16();
            if (t == marker::SOD) break;
            if ((t >> 8) != 0xFF) return bad("expected a marker at offset " + std::to_string(at));
            if (!segment(t, seg)) return false;
            if (rd.pos > tpEnd) return bad("marker " + hex(t) + " runs past the end of its tile-part");
            if (t != marker::PLT && t != marker::COM) notSupported(h, "marker " + hex(t) + " in a tile-part header");
        }
        if (tpsot != h.tileParts.size()) notSupported(h, "tile-parts out of order");
        if (h.tileParts.size() >= 255) return bad("too many tile-parts");
        h.tileParts.push_back({(int)tpsot, rd.pos, tpEnd});
        rd.pos = tpEnd;
        if (rd.left() == 0) break;  // no EOC: tolerated, the data is complete
        if (rd.left() < 2) return bad("stray byte after the last tile-part");
        m = rd.u16();
        if (m != marker::SOT && m != marker::EOC) return bad("expected SOT or EOC at offset " + std::to_string(rd.pos - 2));
    }
    if (h.tileParts.empty()) return bad("no tile-parts");
    return true;
}

FrameLayout layoutFor(const StreamHeader& h) {
    if (!h.supported) throw std::runtime_error("unsupported codestream: " + h.unsupported);
    FrameLayout L = makeGeometry(h.width, h.height, h.rsiz == 4 ? Profile::Cinema4K : Profile::Cinema2K, h.levels, h.guardBits);
    for (int c = 0; c < kNumComps; ++c) setSignaledSteps(L, h.eps[c], h.mu[c], c);
    return L;
}

}  // namespace gail
