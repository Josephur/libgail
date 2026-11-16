// SPDX-License-Identifier: MIT
#include "core/t2_parse.h"

#include "core/bitio.h"
#include "core/markers.h"
#include "core/progression.h"
#include "core/tagtree.h"
#include "kernels/portable.h"

namespace gail {

namespace {

// Number of coding passes (B.10.6, Table B.4).
int readNumPasses(BitReader& br) {
    if (!br.getBit()) return 1;
    if (!br.getBit()) return 2;
    uint32_t v = br.getBits(2);
    if (v != 3) return 3 + (int)v;
    v = br.getBits(5);
    if (v != 31) return 6 + (int)v;
    return 37 + (int)br.getBits(7);
}

}  // namespace

bool parsePackets(const uint8_t* data, const StreamHeader& h, const FrameLayout& L, int maxRes, std::vector<CblkCode>& out,
                  std::string& error) {
    out.assign((size_t)L.numCblks(), CblkCode{});
    if (h.tileParts.empty()) {
        error = "no tile-parts";
        return false;
    }
    TagTree incl, imsb;
    size_t tp = 0, pos = h.tileParts[0].begin, end = h.tileParts[0].end;
    bool ok = true;
    auto fail = [&](const char* what, int c, int r, int prec) {
        if (!ok) return;
        ok = false;
        error = std::string(what) + " (component " + std::to_string(c) + ", resolution " + std::to_string(r) + ", precinct " +
                std::to_string(prec) + ")";
    };

    for (const ProgressionVolume& v : h.volumes) {
        if (v.rs > maxRes) break;  // later volumes hold only higher resolutions
        for (int c = 0; c < kNumComps && ok; ++c)
            forEachPacket(L, v, c, [&](int r, int prec) {
                if (!ok) return;
                while (pos >= end) {  // this tile-part is used up: the packet starts the next one
                    if (++tp >= h.tileParts.size()) return fail("the tile-parts end before the last packet", c, r, prec);
                    pos = h.tileParts[tp].begin;
                    end = h.tileParts[tp].end;
                }
                if (h.sop && end - pos >= 6 && data[pos] == 0xFF && data[pos + 1] == (marker::SOP & 0xFF)) pos += 6;

                const Precinct& P = L.resolution(c, r).precincts[prec];
                BitReader br(data + pos, end - pos);
                size_t body = 0;
                const bool present = br.getBit() != 0;
                for (int k = 0; present && k < P.nb; ++k) {
                    const PrecinctBand& pb = P.bands[k];
                    if (!pb.ncbx) continue;
                    const Subband& b = L.bands[pb.band];
                    incl.init(pb.ncbx, pb.ncby);
                    imsb.init(pb.ncbx, pb.ncby);
                    for (int j = 0; j < pb.ncby; ++j)
                        for (int i = 0; i < pb.ncbx; ++i) {
                            const int leaf = j * pb.ncbx + i;
                            if (!incl.decode(br, leaf, 1)) continue;  // not in this (the only) layer
                            // missing most significant bit-planes: at most all Mb of them
                            if (!imsb.decode(br, leaf, b.Mb + 1)) return fail("more missing bit-planes than the band has", c, r, prec);
                            const int nbps = b.Mb - imsb.value(leaf);
                            const int np = readNumPasses(br);
                            int lblock = 3;  // B.10.7.1
                            while (br.getBit())
                                if (++lblock > 24) return fail("code-block length field too wide", c, r, prec);
                            const uint32_t len = br.getBits(lblock + FIND_MSB((uint32_t)np));
                            if (br.overrun()) return fail("packet header runs past its tile-part", c, r, prec);
                            if (nbps < 1 || np > 3 * nbps - 2) return fail("more coding passes than the bit-planes allow", c, r, prec);
                            CblkCode& cb = out[(size_t)(b.cblkBase + (pb.cby0 + j) * b.ncbx + pb.cbx0 + i)];
                            cb.len = len;
                            cb.passes = (uint8_t)np;
                            cb.nbps = (uint8_t)nbps;
                            body += len;
                        }
                }
                const size_t hdr = br.finish();
                if (br.overrun() || hdr > end - pos) return fail("packet header runs past its tile-part", c, r, prec);
                pos += hdr;
                if (h.eph && end - pos >= 2 && data[pos] == 0xFF && data[pos + 1] == (marker::EPH & 0xFF)) pos += 2;
                if (body > end - pos) return fail("code-block data runs past its tile-part", c, r, prec);
                // the packet body: each included block's bytes, in the order of the header
                for (int k = 0; present && k < P.nb; ++k) {
                    const PrecinctBand& pb = P.bands[k];
                    if (!pb.ncbx) continue;
                    const Subband& b = L.bands[pb.band];
                    for (int j = 0; j < pb.ncby; ++j)
                        for (int i = 0; i < pb.ncbx; ++i) {
                            CblkCode& cb = out[(size_t)(b.cblkBase + (pb.cby0 + j) * b.ncbx + pb.cbx0 + i)];
                            if (!cb.passes) continue;
                            cb.offset = (uint32_t)pos;
                            pos += cb.len;
                            if (r > maxRes) cb = CblkCode{};  // read past, not wanted
                        }
                }
            });
    }
    return ok;
}

}  // namespace gail
