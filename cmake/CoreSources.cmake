# SPDX-License-Identifier: MIT
# The library's internals that every backend builds on, relative to the source tree. The
# main build compiles them into gail_core; each CUDA backend plugin compiles them again
# with its own toolkit's host compiler (cmake/cuda_backend/CMakeLists.txt).
set(GAIL_CORE_SOURCES
  src/core/layout.cpp
  src/core/color.cpp
  src/core/dwt_cpu.cpp
  src/core/quant.cpp
  src/core/t1_ref.cpp
  src/core/t1_dec.cpp
  src/core/t1_dec_tables.cpp
  src/core/rate_control.cpp
  src/core/t2.cpp
  src/core/codestream_check.cpp
  src/core/codestream_parse.cpp
  src/core/t2_parse.cpp
  src/core/cpu_backend.cpp
  src/core/cpu_decode_backend.cpp
  src/core/pipeline.cpp
)
