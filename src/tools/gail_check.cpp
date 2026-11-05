// SPDX-License-Identifier: MIT
// gail-check: verify JPEG 2000 codestreams against the DCI constraints.
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "core/codestream_check.h"

int main(int argc, char** argv) {
    double fps = 24;
    bool verbose = false;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--fps" && i + 1 < argc) fps = std::atof(argv[++i]);
        else if (a == "-v") verbose = true;
        else files.push_back(a);
    }
    if (files.empty()) {
        std::fprintf(stderr, "usage: gail-check [--fps N] [-v] FILE.j2c...\n");
        return 2;
    }
    int bad = 0;
    for (auto& f : files) {
        std::ifstream in(f, std::ios::binary);
        if (!in) {  // not a malformed codestream: say so instead of "does not start with SOC"
            std::printf("%s: cannot read the file\n", f.c_str());
            ++bad;
            continue;
        }
        std::vector<uint8_t> d((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        auto r = gail::checkDciCodestream(d.data(), d.size(), fps);
        if (!r.ok) ++bad;
        if (verbose || !r.ok) std::printf("%s: %s\n", f.c_str(), gail::formatReport(r, verbose).c_str());
    }
    std::printf("%zu files checked, %d non-compliant\n", files.size(), bad);
    return bad ? 1 : 0;
}
