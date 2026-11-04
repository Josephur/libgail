// SPDX-License-Identifier: MIT
// Independent DCI compliance checker for JPEG 2000 codestreams (SMPTE ST 429-4 /
// ISO 15444-1 Amd.1 cinema profiles, plus the libdcp/ClairMeta verification rules).
#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gail {

struct CheckReport {
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
    // Parsed facts
    int rsiz = 0, width = 0, height = 0, comps = 0, depth = 0;
    int levels = 0, layers = 0, progression = -1, mct = -1, cblkW = 0, cblkH = 0, cblkStyle = -1, transform = -1;
    int guardBits = -1, qstyle = -1, tileParts = 0, pocCount = 0;
    bool tlm = false;
    std::string comment;  // text of the main header's COM marker, if any
    std::vector<int> precincts;
    size_t totalBytes = 0;
    size_t compBytes[3] = {0, 0, 0};
    std::vector<size_t> tilePartBytes;
    std::string profile() const { return rsiz == 3 ? "2K" : rsiz == 4 ? "4K" : "?"; }
    void error(const std::string& s) {
        ok = false;
        errors.push_back(s);
    }
};

CheckReport checkDciCodestream(const uint8_t* data, size_t size, double fps);
std::string formatReport(const CheckReport& r, bool verbose);

}  // namespace gail
