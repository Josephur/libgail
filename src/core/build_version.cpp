// SPDX-License-Identifier: MIT
// The only file that includes the generated header, so a new version recompiles just this.
#include "core/build_version.h"

#include "gail_build_version.h"

namespace gail {

const char* buildVersion() { return GAIL_BUILD_VERSION; }

}  // namespace gail
