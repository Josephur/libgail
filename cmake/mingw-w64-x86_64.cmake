# SPDX-License-Identifier: MIT
# Cross-compile Windows x64 binaries (gail.dll, gail-enc.exe, tests) with MinGW-w64, the
# toolchain FFmpeg's Windows builds use. CUDA needs MSVC on Windows, so
# these builds have the Vulkan and CPU backends.
#
#   cmake -S . -B build-win -DCMAKE_TOOLCHAIN_FILE=cmake/mingw-w64-x86_64.cmake \
#         -DGAIL_VULKAN_INCLUDE_DIR=<dir containing vulkan/vulkan.h>
set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)
set(GAIL_MINGW_PREFIX x86_64-w64-mingw32 CACHE STRING "MinGW-w64 tool prefix")
set(CMAKE_C_COMPILER ${GAIL_MINGW_PREFIX}-gcc)
set(CMAKE_CXX_COMPILER ${GAIL_MINGW_PREFIX}-g++)
set(CMAKE_RC_COMPILER ${GAIL_MINGW_PREFIX}-windres)
set(CMAKE_FIND_ROOT_PATH /usr/${GAIL_MINGW_PREFIX})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)   # glslc runs on the build machine
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(GAIL_ENABLE_CUDA OFF CACHE STRING "" FORCE)
set(CMAKE_CROSSCOMPILING_EMULATOR wine CACHE STRING "run tests under Wine")
