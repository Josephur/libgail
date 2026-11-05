// SPDX-License-Identifier: MIT
#include "core/t2.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>

#include "core/build_version.h"
#include "kernels/portable.h"

namespace gail {

namespace {

// Packet-header bit writer (B.10.1): bits fill each byte MSB first, and a byte that
// follows 0xFF carries only 7 bits (its MSB is the stuffed 0), so no marker code can
// appear in a header. With dst == nullptr it only counts bytes.
class BitWriter {
public:
    explicit BitWriter(std::vector<uint8_t>* dst) : dst_(dst) {}
    void putBit(uint32_t b) {
        acc_ = (acc_ << 1) | (b & 1u);
        if (++used_ == room_) emit(acc_);
    }
    // The n (0..32) low bits of v, MSB first, as many at a time as the current byte takes.
    void putBits(uint32_t v, int n) {
        while (n > 0) {
            const int take = std::min(n, room_ - used_);
            n -= take;
            acc_ = (acc_ << take) | ((v >> n) & ((1u << take) - 1u));
            used_ += take;
            if (used_ == room_) emit(acc_);
        }
    }
    void putZeros(int n) {
        for (; n > 24; n -= 24) putBits(0, 24);
        putBits(0, n);
    }
    // Pads the last byte with zeros. A header ending in 0xFF still owes the stuffed
    // byte that must follow it, so that is written as 0x00. An empty header is one
    // zero byte.
    size_t flush() {
        if (used_ > 0 || count_ == 0) emit(acc_ << (room_ - used_));
        if (last_ == 0xFF) emit(0);
        return count_;
    }

private:
    void emit(uint32_t byte) {
        last_ = byte & 0xFFu;
        if (dst_) dst_->push_back((uint8_t)last_);
        ++count_;
        room_ = last_ == 0xFF ? 7 : 8;
        acc_ = 0;
        used_ = 0;
    }
    std::vector<uint8_t>* dst_;
    uint32_t acc_ = 0, last_ = 0;
    int used_ = 0, room_ = 8;
    size_t count_ = 0;
};

class TagTree {
public:
    void init(int w, int h) {
        if (w == w_ && h == h_) {  // same shape as the last precinct band: keep the links
            for (Node& n : nodes_) {
                n.low = 0;
                n.known = false;
            }
            return;
        }
        w_ = w;
        h_ = h;
        nodes_.clear();
        int lw = w, lh = h;
        levels_.clear();
        int base = 0;
        while (true) {
            levels_.push_back({base, lw, lh});
            base += lw * lh;
            if (lw == 1 && lh == 1) break;
            lw = (lw + 1) / 2;
            lh = (lh + 1) / 2;
        }
        nodes_.assign(base, Node{});
        for (size_t l = 0; l + 1 < levels_.size(); ++l) {
            const Level& a = levels_[l];
            const Level& b = levels_[l + 1];
            for (int y = 0; y < a.h; ++y)
                for (int x = 0; x < a.w; ++x) nodes_[a.base + y * a.w + x].parent = b.base + (y / 2) * b.w + x / 2;
        }
    }
    void setLeaf(int i, int v) { nodes_[i].value = v; }
    void finalize() {
        // internal node = min of children
        for (size_t l = 1; l < levels_.size(); ++l)
            for (int i = 0; i < levels_[l].w * levels_[l].h; ++i) nodes_[levels_[l].base + i].value = 1 << 30;
        for (size_t i = 0; i < nodes_.size(); ++i) {
            const int p = nodes_[i].parent;
            if (p >= 0) nodes_[p].value = std::min(nodes_[p].value, nodes_[i].value);
        }
    }
    // Codes what the decoder learns about leaf's value from asking "is it < threshold?"
    // (B.10.2). Nodes are visited root first. Each node has already sent "value > n.low"
    // as n.low zeros, and its value is at least the bound reached at its parent. It
    // sends one 0 per further unit the value is known to exceed, up to the threshold,
    // and a single 1 the first time its exact value is reached below the threshold.
    void encode(BitWriter& bw, int leaf, int threshold) {
        int path[32];
        int depth = 0;
        for (int i = leaf; i >= 0; i = nodes_[i].parent) path[depth++] = i;
        int bound = 0;
        while (depth > 0) {
            Node& n = nodes_[path[--depth]];
            const int from = std::max(n.low, bound);
            const int to = std::min(n.value, threshold);
            if (to > from) bw.putZeros(to - from);
            n.low = std::max(from, to);
            if (n.low >= n.value && n.low < threshold && !n.known) {
                bw.putBit(1);
                n.known = true;
            }
            bound = n.low;
        }
    }

private:
    struct Node {
        int value = 0, low = 0, parent = -1;
        bool known = false;
    };
    struct Level {
        int base, w, h;
    };
    std::vector<Node> nodes_;
    std::vector<Level> levels_;
    int w_ = 0, h_ = 0;
};

int floorLog2(uint32_t v) { return FIND_MSB(v); }

void putNumPasses(BitWriter& bw, int n) {
    if (n == 1) bw.putBits(0, 1);
    else if (n == 2) bw.putBits(2, 2);
    else if (n <= 5) bw.putBits(0xC | (n - 3), 4);
    else if (n <= 36) bw.putBits(0x1E0 | (n - 6), 9);
    else if (n <= 164) bw.putBits(0xFF80 | (n - 37), 16);
    else throw std::runtime_error("too many passes");
}

struct PacketWriter {
    const FrameLayout& L;
    const Tier1Frame& t1;
    const uint8_t* passes;
    TagTree incl, imsb;

    // Appends the packet to dst (if non-null); returns its size in bytes.
    size_t write(int comp, int res, int prec, std::vector<uint8_t>* dst) {
        const Resolution& R = L.resolution(comp, res);
        const Precinct& P = R.precincts[prec];
        bool any = false;
        for (int k = 0; k < P.nb && !any; ++k) {
            const PrecinctBand& pb = P.bands[k];
            const Subband& b = L.bands[pb.band];
            for (int j = 0; j < pb.ncby && !any; ++j)
                for (int i = 0; i < pb.ncbx; ++i)
                    if (passes[b.cblkBase + (pb.cby0 + j) * b.ncbx + pb.cbx0 + i]) {
                        any = true;
                        break;
                    }
        }
        BitWriter bw(dst);
        if (!any) {
            bw.putBit(0);
            return bw.flush();
        }
        bw.putBit(1);
        size_t body = 0;
        for (int k = 0; k < P.nb; ++k) {
            const PrecinctBand& pb = P.bands[k];
            if (!pb.ncbx) continue;
            const Subband& b = L.bands[pb.band];
            incl.init(pb.ncbx, pb.ncby);
            imsb.init(pb.ncbx, pb.ncby);
            for (int j = 0; j < pb.ncby; ++j)
                for (int i = 0; i < pb.ncbx; ++i) {
                    const int ci = b.cblkBase + (pb.cby0 + j) * b.ncbx + pb.cbx0 + i;
                    incl.setLeaf(j * pb.ncbx + i, passes[ci] ? 0 : 1);
                    imsb.setLeaf(j * pb.ncbx + i, b.Mb - (int)t1.info[ci].nbps);
                }
            incl.finalize();
            imsb.finalize();
            for (int j = 0; j < pb.ncby; ++j)
                for (int i = 0; i < pb.ncbx; ++i) {
                    const int leaf = j * pb.ncbx + i;
                    const int ci = b.cblkBase + (pb.cby0 + j) * b.ncbx + pb.cbx0 + i;
                    incl.encode(bw, leaf, 1);
                    const int np = passes[ci];
                    if (!np) continue;
                    imsb.encode(bw, leaf, 999);
                    putNumPasses(bw, np);
                    const uint32_t len = t1.len(ci, np - 1);
                    int lblock = 3;
                    const int extra = floorLog2((uint32_t)np);
                    const int need = len ? floorLog2(len) + 1 : 1;
                    const int incr = std::max(0, need - (lblock + extra));
                    bw.putBits(((1u << incr) - 1u) << 1, incr + 1);  // incr ones, then 0
                    lblock += incr;
                    bw.putBits(len, lblock + extra);
                    body += len;
                }
        }
        const size_t hdr = bw.flush();
        if (dst) {
            size_t pos = dst->size();
            dst->resize(pos + body);
            uint8_t* out = dst->data();
            for (int k = 0; k < P.nb; ++k) {
                const PrecinctBand& pb = P.bands[k];
                if (!pb.ncbx) continue;
                const Subband& b = L.bands[pb.band];
                for (int j = 0; j < pb.ncby; ++j)
                    for (int i = 0; i < pb.ncbx; ++i) {
                        const int ci = b.cblkBase + (pb.cby0 + j) * b.ncbx + pb.cbx0 + i;
                        const int np = passes[ci];
                        if (!np) continue;
                        const uint32_t len = t1.len(ci, np - 1);
                        std::memcpy(out + pos, t1.data.data() + t1.info[ci].dataOffset, len);
                        pos += len;
                    }
            }
        }
        return hdr + body;
    }
};

void put16(std::vector<uint8_t>& v, uint32_t x) {
    v.push_back((uint8_t)(x >> 8));
    v.push_back((uint8_t)x);
}
void put32(std::vector<uint8_t>& v, uint32_t x) {
    put16(v, x >> 16);
    put16(v, x & 0xFFFF);
}

void writeMainHeader(const FrameLayout& L, std::vector<uint8_t>& v, int ntp) {
    const bool fourk = L.profile == Profile::Cinema4K;
    const int nres = L.numRes();
    put16(v, 0xFF4F);  // SOC
    put16(v, 0xFF51);  // SIZ
    put16(v, 38 + 3 * kNumComps);
    put16(v, fourk ? 4 : 3);  // Rsiz: CINEMA4K / CINEMA2K
    put32(v, L.width);
    put32(v, L.height);
    put32(v, 0);
    put32(v, 0);
    put32(v, L.width);  // one tile covering the image
    put32(v, L.height);
    put32(v, 0);
    put32(v, 0);
    put16(v, kNumComps);
    for (int c = 0; c < kNumComps; ++c) {
        v.push_back(kBitDepth - 1);
        v.push_back(1);
        v.push_back(1);
    }
    put16(v, 0xFF52);  // COD
    put16(v, 12 + nres);
    v.push_back(0x01);  // Scod: user-defined precincts
    v.push_back(4);     // CPRL
    put16(v, 1);        // layers
    v.push_back(1);     // multiple component transform (ICT)
    v.push_back((uint8_t)L.levels);
    v.push_back(kCblkLog2 - 2);
    v.push_back(kCblkLog2 - 2);
    v.push_back(0);  // code-block style: no mode switches
    v.push_back(0);  // 9/7 irreversible
    for (int r = 0; r < nres; ++r) v.push_back(r == 0 ? 0x77 : 0x88);
    put16(v, 0xFF5C);  // QCD
    const int nb = L.bandsPerComp();
    put16(v, 3 + 2 * nb);
    v.push_back((uint8_t)((L.guardBits << 5) | 2));  // scalar expounded
    for (int i = 0; i < nb; ++i) put16(v, (uint32_t)((L.bands[i].eps << 11) | L.bands[i].mu));
    if (fourk) {
        put16(v, 0xFF5F);  // POC: 2K resolutions first, then the 4K resolution
        put16(v, 2 + 7 * 2);
        const int NL = L.levels;
        const uint8_t e[2][7] = {{0, 0, 0, 1, (uint8_t)NL, 3, 4}, {(uint8_t)NL, 0, 0, 1, (uint8_t)(NL + 1), 3, 4}};
        for (auto& p : e) v.insert(v.end(), p, p + 7);
    }
    const char* label = codestreamLabel();
    const size_t labelLen = std::strlen(label);
    put16(v, 0xFF64);  // COM
    put16(v, (uint32_t)(4 + labelLen));
    put16(v, 1);  // Rcom: Latin-1 text
    v.insert(v.end(), label, label + labelLen);
    put16(v, 0xFF55);  // TLM (lengths patched later); buildCodestream relies on it being last
    put16(v, 4 + 5 * ntp);
    v.push_back(0);     // Ztlm
    v.push_back(0x50);  // Stlm: 8-bit Ttlm, 32-bit Ptlm
    for (int i = 0; i < ntp; ++i) {
        v.push_back(0);
        put32(v, 0);
    }
}

}  // namespace

const char* codestreamLabel() {
    static const std::string label = std::string("GAIL ") + buildVersion();
    return label.c_str();
}

void buildCodestream(const FrameLayout& L, const Tier1Frame& t1, const uint8_t* passes, std::vector<uint8_t>* out,
                     CodestreamSizes* sizes) {
    const bool fourk = L.profile == Profile::Cinema4K;
    const int NL = L.levels;
    const int ntp = fourk ? 6 : 3;
    PacketWriter pw{L, t1, passes, {}, {}};

    std::vector<uint8_t> hdr;
    writeMainHeader(L, hdr, ntp);
    const size_t tlmPos = hdr.size() - 5 * ntp;
    if (out) {
        out->clear();
        out->reserve(1400000);
        *out = hdr;
    }

    struct Poc {
        int rs, re;
    };
    const Poc pocs[2] = {{0, fourk ? NL : NL + 1}, {NL, NL + 1}};
    const int npoc = fourk ? 2 : 1;
    CodestreamSizes sz;
    sz.numTileParts = ntp;
    size_t total = hdr.size();
    for (int pi = 0; pi < npoc; ++pi) {
        const Poc poc = pocs[pi];
        int ystep = 1 << 30, xstep = 1 << 30;
        for (int r = poc.rs; r < poc.re; ++r) {
            const Resolution& R = L.resolution(0, r);
            xstep = std::min(xstep, 1 << (R.ppx + NL - r));
            ystep = std::min(ystep, 1 << (R.ppy + NL - r));
        }
        for (int c = 0; c < kNumComps; ++c) {
            const int tp = pi * 3 + c;
            size_t sotPos = 0;
            if (out) {
                sotPos = out->size();
                put16(*out, 0xFF90);
                put16(*out, 10);
                put16(*out, 0);
                put32(*out, 0);  // Psot patched below
                out->push_back((uint8_t)tp);
                out->push_back((uint8_t)ntp);
                put16(*out, 0xFF93);
            }
            size_t tpBytes = 14;  // SOT (12) + SOD (2)
            for (int y = 0; y < L.height; y += ystep)
                for (int x = 0; x < L.width; x += xstep)
                    for (int r = poc.rs; r < poc.re; ++r) {
                        const Resolution& R = L.resolution(c, r);
                        const int xr = 1 << (R.ppx + NL - r), yr = 1 << (R.ppy + NL - r);
                        if ((x & (xr - 1)) || (y & (yr - 1))) continue;
                        const int px = x / xr, py = y / yr;
                        if (px >= R.npx || py >= R.npy) continue;
                        tpBytes += pw.write(c, r, py * R.npx + px, out);
                    }
            if (out) {
                const uint32_t psot = (uint32_t)(out->size() - sotPos);
                if (psot != tpBytes) throw std::logic_error("tile-part size mismatch");
                for (int b = 0; b < 4; ++b) (*out)[sotPos + 6 + b] = (uint8_t)(psot >> (24 - 8 * b));
                for (int b = 0; b < 4; ++b) (*out)[tlmPos + 5 * tp + 1 + b] = (uint8_t)(psot >> (24 - 8 * b));
            }
            sz.tilePart[tp] = tpBytes;
            sz.comp[c] += tpBytes;
            total += tpBytes;
        }
    }
    total += 2;
    if (out) put16(*out, 0xFFD9);  // EOC
    sz.total = total;
    if (out && out->size() != total) throw std::logic_error("codestream size mismatch");
    if (sizes) *sizes = sz;
}

size_t emptyCodestreamBytes(const FrameLayout& L) {
    // A Tier-1 result with no passes anywhere: the packet writer never touches the (empty)
    // pass tables or data, it only writes the empty-packet bytes.
    Tier1Frame t1;
    t1.info.assign((size_t)L.numCblks(), CblkInfo{});
    const std::vector<uint8_t> passes((size_t)L.numCblks(), 0);
    CodestreamSizes sz;
    buildCodestream(L, t1, passes.data(), nullptr, &sz);
    return sz.total;
}

}  // namespace gail
