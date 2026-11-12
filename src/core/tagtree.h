// SPDX-License-Identifier: MIT
// Tag trees (ISO/IEC 15444-1 B.10.2): the hierarchical code packet headers use for the
// first layer a code-block appears in and for its number of missing bit-planes.
// Header-only so the packet writer, which codes them while rate control measures
// candidate codestreams, keeps them inline.
#pragma once
#include <algorithm>
#include <vector>

#include "core/bitio.h"

namespace gail {

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

    // Reads what encode() wrote for the same sequence of (leaf, threshold) calls. Returns
    // true once the leaf's value is known to be below the threshold; value(leaf) is then
    // exact. A decoding tree needs only init(): its values come from the bits.
    bool decode(BitReader& br, int leaf, int threshold) {
        int path[32];
        int depth = 0;
        for (int i = leaf; i >= 0; i = nodes_[i].parent) path[depth++] = i;
        int bound = 0;
        while (depth > 0) {
            Node& n = nodes_[path[--depth]];
            n.low = std::max(n.low, bound);
            while (!n.known && n.low < threshold) {
                if (br.getBit()) {
                    n.known = true;
                    n.value = n.low;
                } else {
                    ++n.low;
                }
            }
            bound = n.low;
        }
        const Node& l = nodes_[leaf];
        return l.known && l.value < threshold;
    }
    int value(int leaf) const { return nodes_[leaf].value; }

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

}  // namespace gail
