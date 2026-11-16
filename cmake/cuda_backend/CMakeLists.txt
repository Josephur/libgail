# SPDX-License-Identifier: MIT
# Builds the CUDA backend (src/cuda/*.cu) with one CUDA toolkit. A CMake project has a
# single CUDA compiler, so the main CMakeLists.txt runs this project once per toolkit:
#
#   GAIL_SOURCE_DIR        the libgail source tree
#   GAIL_CUDA_NAME         gail_cuda13 or gail_cuda12
#   GAIL_CUDA_ARCHS        CUDA architectures to build for
#   GAIL_CUDA_PLUGIN_DIR   where the plugin goes (empty: no plugin)
#   GAIL_CUDA_OBJECT_DIR   where the backend's two object files go, for a static libgail
#                          to link in (empty: none), named <unit>${GAIL_CUDA_OBJECT_SUFFIX}
#
# The toolkit is CMAKE_CUDA_COMPILER / CUDAToolkit_ROOT as for any CUDA project.
cmake_minimum_required(VERSION 3.20)
project(gail_cuda_backend LANGUAGES CXX CUDA)

if(NOT GAIL_SOURCE_DIR OR NOT GAIL_CUDA_NAME OR NOT GAIL_CUDA_ARCHS)
  message(FATAL_ERROR "this project is built by the main CMakeLists.txt of libgail")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_RUNTIME_LIBRARY Static)  # no cudart DLL to ship; only the NVIDIA driver is needed
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(GAIL_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  endif()
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wno-missing-field-initializers>")
endif()

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(GAIL_CUDA_OPTIONS -O3 --use_fast_math -lineinfo)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13)
  # CUDA 12.9 says on every compile that Maxwell, Pascal and Volta are deprecated; they
  # are what this toolkit is used for.
  list(APPEND GAIL_CUDA_OPTIONS -Wno-deprecated-gpu-targets)
endif()

# Each translation unit is an object library of its own, so that its one object file can
# be named when it is linked into the plugin and copied for the static library.
set(GAIL_CUDA_UNITS cuda_backend cuda_decode)
foreach(unit ${GAIL_CUDA_UNITS})
  add_library(${unit} OBJECT ${GAIL_SOURCE_DIR}/src/cuda/${unit}.cu)
  set_target_properties(${unit} PROPERTIES CUDA_ARCHITECTURES "${GAIL_CUDA_ARCHS}" CUDA_SEPARABLE_COMPILATION OFF)
  target_include_directories(${unit} PRIVATE ${GAIL_SOURCE_DIR}/src)
  target_compile_definitions(${unit} PRIVATE GAIL_HAVE_CUDA=1)
  target_compile_options(${unit} PRIVATE ${GAIL_CUDA_OPTIONS})
endforeach()

string(REPLACE ";" " " GAIL_CUDA_ARCH_TEXT "${GAIL_CUDA_ARCHS}")

if(GAIL_CUDA_PLUGIN_DIR)
  include(${GAIL_SOURCE_DIR}/cmake/CoreSources.cmake)
  list(TRANSFORM GAIL_CORE_SOURCES PREPEND ${GAIL_SOURCE_DIR}/)
  add_library(gail_core STATIC ${GAIL_CORE_SOURCES})
  target_include_directories(gail_core PUBLIC ${GAIL_SOURCE_DIR}/src)
  target_link_libraries(gail_core PUBLIC Threads::Threads)

  add_library(${GAIL_CUDA_NAME} MODULE ${GAIL_SOURCE_DIR}/src/cuda/cuda_plugin.cpp)
  foreach(unit ${GAIL_CUDA_UNITS})
    target_sources(${GAIL_CUDA_NAME} PRIVATE $<TARGET_OBJECTS:${unit}>)
  endforeach()
  target_compile_definitions(${GAIL_CUDA_NAME} PRIVATE
    GAIL_HAVE_CUDA=1
    GAIL_CUDA_PLUGIN_BUILD=1
    GAIL_CUDA_PLUGIN_NAME="${GAIL_CUDA_NAME}"
    GAIL_CUDA_PLUGIN_TOOLKIT="${CMAKE_CUDA_COMPILER_VERSION}"
    GAIL_CUDA_PLUGIN_ARCHS="${GAIL_CUDA_ARCH_TEXT}")
  target_link_libraries(${GAIL_CUDA_NAME} PRIVATE gail_core CUDA::cudart_static)
  set_target_properties(${GAIL_CUDA_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${GAIL_CUDA_PLUGIN_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${GAIL_CUDA_PLUGIN_DIR})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # The plugin exports its entry point alone: the CUDA runtime inside it stays its own,
    # so two plugins with two runtimes can be loaded into one process.
    target_link_options(${GAIL_CUDA_NAME} PRIVATE -Wl,--exclude-libs,ALL -Wl,--no-undefined)
  endif()
endif()

if(GAIL_CUDA_OBJECT_DIR)
  set(copies)
  foreach(unit ${GAIL_CUDA_UNITS})
    set(copy ${GAIL_CUDA_OBJECT_DIR}/${unit}${GAIL_CUDA_OBJECT_SUFFIX})
    add_custom_command(OUTPUT ${copy}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${GAIL_CUDA_OBJECT_DIR}
      COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_OBJECTS:${unit}> ${copy}
      DEPENDS ${unit} $<TARGET_OBJECTS:${unit}>
      VERBATIM)
    list(APPEND copies ${copy})
  endforeach()
  add_custom_target(gail_cuda_objects ALL DEPENDS ${copies})
endif()
