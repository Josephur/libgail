# SPDX-License-Identifier: MIT
cmake_minimum_required(VERSION 3.20)
# The version lives in one place: the macros in gail.h, which scripts/release.py updates.
file(STRINGS include/gail/gail.h GAIL_VERSION_LINES REGEX "^#define GAIL_VERSION_(MAJOR|MINOR|PATCH) +[0-9]+")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS include/gail/gail.h)  # a release reconfigures
foreach(part MAJOR MINOR PATCH)
  string(REGEX MATCH "GAIL_VERSION_${part} +([0-9]+)" _ "${GAIL_VERSION_LINES}")
  set(GAIL_VERSION_${part} ${CMAKE_MATCH_1})
endforeach()
project(gail VERSION ${GAIL_VERSION_MAJOR}.${GAIL_VERSION_MINOR}.${GAIL_VERSION_PATCH} LANGUAGES C CXX
        DESCRIPTION "GAIL, GPU Accelerated Intraframe Library: a DCI JPEG 2000 encoder")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GAIL_ENABLE_VULKAN "Build the Vulkan compute backend" ON)
set(GAIL_ENABLE_CUDA AUTO CACHE STRING "Build the optional CUDA backend (AUTO, ON, OFF)")
option(GAIL_BUILD_SHARED "Build libgail as a shared library (libgail.so / gail.dll)" ON)
option(GAIL_BUILD_STATIC "Also build the static library (libgail.a / gail_static.lib)" ON)
option(GAIL_BUILD_TOOLS "Build gail-enc and the other command-line tools" ON)
option(GAIL_BUILD_TESTS "Build the tests" ON)
option(GAIL_NATIVE_ARCH "Optimise for this machine's CPU only (-march=native); not for distributed builds" OFF)
set(GAIL_VULKAN_INCLUDE_DIR "" CACHE PATH "Vulkan headers (only headers are needed; default: find_package(Vulkan))")

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(GAIL_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  endif()
  # (Vulkan's "sType first, rest zero" struct initialisers trip -Wmissing-field-initializers)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wno-missing-field-initializers>")
endif()

find_package(Threads REQUIRED)
if(MINGW)
  # self-contained Windows binaries: no libstdc++ / libgcc / winpthread DLLs to ship
  add_link_options(-static)
endif()
# The internals are static and position independent so they can be linked into the shared
# library, which exports only the GAIL_API functions.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

add_library(gail_core STATIC
  src/core/build_version.cpp
  src/core/layout.cpp
  src/core/color.cpp
  src/core/dwt_cpu.cpp
  src/core/quant.cpp
  src/core/t1_ref.cpp
  src/core/rate_control.cpp
  src/core/t2.cpp
  src/core/codestream_check.cpp
  src/core/cpu_backend.cpp
  src/core/pipeline.cpp
)
target_include_directories(gail_core PUBLIC src)
target_link_libraries(gail_core PUBLIC Threads::Threads)

# Refreshed on every build from git; the file only changes when the version does (cmake/BuildVersion.cmake)
find_package(Git QUIET)
set(GAIL_BUILD_VERSION_HEADER ${CMAKE_BINARY_DIR}/generated/gail_build_version.h)
add_custom_target(gail_build_version
  COMMAND ${CMAKE_COMMAND} -DGIT_EXECUTABLE=${GIT_EXECUTABLE} -DSOURCE_DIR=${CMAKE_SOURCE_DIR}
          -DBASE_VERSION=${PROJECT_VERSION} -DOUTPUT=${GAIL_BUILD_VERSION_HEADER}
          -P ${CMAKE_SOURCE_DIR}/cmake/BuildVersion.cmake
  BYPRODUCTS ${GAIL_BUILD_VERSION_HEADER}
  VERBATIM)
add_dependencies(gail_core gail_build_version)
target_include_directories(gail_core PRIVATE ${CMAKE_BINARY_DIR}/generated)

set(GAIL_BACKEND_LIBS)

if(GAIL_ENABLE_VULKAN)
  # Only the Vulkan headers are needed: the loader is opened at run time (vk_loader.h).
  if(GAIL_VULKAN_INCLUDE_DIR)
    set(GAIL_VK_INCLUDE ${GAIL_VULKAN_INCLUDE_DIR})
  else()
    find_package(Vulkan REQUIRED)
    set(GAIL_VK_INCLUDE ${Vulkan_INCLUDE_DIRS})
  endif()
  find_program(GLSLC glslc REQUIRED)
  set(SHADER_DIR ${CMAKE_SOURCE_DIR}/src/vulkan/shaders)
  set(SHADER_NAMES color color_yuv dwt_rows dwt_cols quantize t1_cm t1_mq compact scan order)
  set(SPV_HEADERS)
  foreach(name ${SHADER_NAMES})
    set(src ${SHADER_DIR}/${name}.comp)
    set(spv ${CMAKE_BINARY_DIR}/shaders/${name}.spv)
    set(hdr ${CMAKE_BINARY_DIR}/shaders/${name}_spv.h)
    file(GLOB_RECURSE SHADER_DEPS CONFIGURE_DEPENDS ${SHADER_DIR}/*.glsl ${CMAKE_SOURCE_DIR}/src/kernels/*.h)
    add_custom_command(
      OUTPUT ${hdr}
      COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/shaders
      COMMAND ${GLSLC} --target-env=vulkan1.3 -O -I${SHADER_DIR} -I${CMAKE_SOURCE_DIR}/src
              ${src} -o ${spv}
      COMMAND ${CMAKE_COMMAND} -DIN=${spv} -DOUT=${hdr} -DNAME=${name}_spv
              -P ${CMAKE_SOURCE_DIR}/cmake/embed_spv.cmake
      DEPENDS ${src} ${SHADER_DEPS} ${CMAKE_SOURCE_DIR}/cmake/embed_spv.cmake
      COMMENT "Compiling shader ${name}.comp")
    list(APPEND SPV_HEADERS ${hdr})
  endforeach()
  add_custom_target(gail_shaders DEPENDS ${SPV_HEADERS})

  add_library(gail_vulkan STATIC src/vulkan/vk_backend.cpp)
  add_dependencies(gail_vulkan gail_shaders)
  target_include_directories(gail_vulkan PRIVATE ${CMAKE_BINARY_DIR}/shaders)
  target_include_directories(gail_vulkan SYSTEM PUBLIC ${GAIL_VK_INCLUDE})
  target_link_libraries(gail_vulkan PUBLIC gail_core ${CMAKE_DL_LIBS})
  target_compile_definitions(gail_vulkan PUBLIC GAIL_HAVE_VULKAN=1)
  list(APPEND GAIL_BACKEND_LIBS gail_vulkan)
endif()

if(NOT GAIL_ENABLE_CUDA STREQUAL "OFF")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)  # no cudart DLL to ship; only the NVIDIA driver is needed
    set(GAIL_CUDA_ARCHS "75;86;89;120" CACHE STRING "CUDA architectures for the optional CUDA backend")
    add_library(gail_cuda STATIC src/cuda/cuda_backend.cu)
    set_target_properties(gail_cuda PROPERTIES CUDA_ARCHITECTURES "${GAIL_CUDA_ARCHS}")
    target_link_libraries(gail_cuda PUBLIC gail_core)
    target_compile_definitions(gail_cuda PUBLIC GAIL_HAVE_CUDA=1)
    target_compile_options(gail_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -lineinfo>)
    set_target_properties(gail_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
    list(APPEND GAIL_BACKEND_LIBS gail_cuda)
  elseif(GAIL_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "CUDA requested but no CUDA compiler found")
  endif()
endif()

# ---- libgail: include/gail/gail.h --------------------------------------------------------
set(GAIL_API_SOURCES src/api/gail_api.cpp src/api/devices.cpp)
set(GAIL_LIB_TARGETS)

function(gail_library target type)
  if(type STREQUAL "STATIC")
    # one self-contained archive: the internal objects go in, only system libraries are needed
    set(objs)
    foreach(lib gail_core ${GAIL_BACKEND_LIBS})
      list(APPEND objs $<TARGET_OBJECTS:${lib}>)
    endforeach()
    add_library(${target} STATIC ${GAIL_API_SOURCES} ${objs})
    target_link_libraries(${target} PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
    if(TARGET gail_cuda)
      target_link_libraries(${target} PUBLIC CUDA::cudart_static)
    endif()
    foreach(lib gail_core ${GAIL_BACKEND_LIBS})
      target_compile_definitions(${target} PRIVATE $<TARGET_PROPERTY:${lib},INTERFACE_COMPILE_DEFINITIONS>)
      target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()
  else()
    add_library(${target} ${type} ${GAIL_API_SOURCES})
    target_link_libraries(${target} PRIVATE gail_core ${GAIL_BACKEND_LIBS})
  endif()
  target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include> $<INSTALL_INTERFACE:include>
                                       PRIVATE src)
  list(APPEND GAIL_LIB_TARGETS ${target})
  set(GAIL_LIB_TARGETS ${GAIL_LIB_TARGETS} PARENT_SCOPE)
endfunction()

if(GAIL_BUILD_SHARED)
  gail_library(gail SHARED)
  set_target_properties(gail PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION ${PROJECT_VERSION_MAJOR})
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    # export only the gail_* API, versioned GAIL_1
    target_link_options(gail PRIVATE -Wl,--exclude-libs,ALL -Wl,--version-script=${CMAKE_SOURCE_DIR}/cmake/gail.map)
    set_target_properties(gail PROPERTIES LINK_DEPENDS ${CMAKE_SOURCE_DIR}/cmake/gail.map)
  elseif(APPLE)
    # the same for the dylib (hidden visibility alone still lets libc++ template RTTI out)
    target_link_options(gail PRIVATE "-Wl,-exported_symbol,_gail_*")
  endif()
endif()
if(GAIL_BUILD_STATIC)
  gail_library(gail_static STATIC)
  target_compile_definitions(gail_static PUBLIC GAIL_STATIC)
  if(NOT MSVC)  # MSVC's gail.lib is the DLL's import library
    set_target_properties(gail_static PROPERTIES OUTPUT_NAME gail)
  endif()
endif()
if(GAIL_BUILD_SHARED)
  set(GAIL_LINK gail)
else()
  set(GAIL_LINK gail_static)
endif()

# ---- tools -------------------------------------------------------------------------------
if(GAIL_BUILD_TOOLS OR GAIL_BUILD_TESTS)
  # file / FFmpeg / pattern frame sources used by the tools (not part of the library)
  add_library(gail_io STATIC src/core/frame_source.cpp)
  target_link_libraries(gail_io PUBLIC gail_core)
endif()

if(GAIL_BUILD_TOOLS)
  add_executable(gail-enc src/tools/gail_enc.cpp src/api/devices.cpp)
  target_link_libraries(gail-enc PRIVATE gail_io ${GAIL_BACKEND_LIBS})

  add_executable(gail-check src/tools/gail_check.cpp)
  target_link_libraries(gail-check PRIVATE gail_core)

  add_executable(gail-bench src/tools/gail_bench.cpp)
  target_link_libraries(gail-bench PRIVATE gail_io ${GAIL_BACKEND_LIBS})
endif()

# ---- tests -------------------------------------------------------------------------------
if(GAIL_BUILD_TESTS)
  enable_testing()
  add_executable(gail-tests tests/test_main.cpp)
  target_link_libraries(gail-tests PRIVATE gail_io ${GAIL_BACKEND_LIBS})
  add_test(NAME gail-tests COMMAND gail-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # public API, compiled as C against the installed-style header
  add_executable(gail-api-test tests/test_api.c)
  target_link_libraries(gail-api-test PRIVATE ${GAIL_LINK} Threads::Threads)
  add_test(NAME gail-api-test COMMAND gail-api-test)
endif()

# ---- install -----------------------------------------------------------------------------
include(GNUInstallDirs)
install(FILES include/gail/gail.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gail)
install(TARGETS ${GAIL_LIB_TARGETS} EXPORT gailTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT gailTargets NAMESPACE gail:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gail)
file(WRITE ${CMAKE_BINARY_DIR}/gailConfig.cmake "include(\"\${CMAKE_CURRENT_LIST_DIR}/gailTargets.cmake\")\n")
install(FILES ${CMAKE_BINARY_DIR}/gailConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gail)
if(GAIL_BUILD_TOOLS)
  install(TARGETS gail-enc gail-check RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# pkg-config: FFmpeg's configure finds the library this way
set(GAIL_PC_LIBS_PRIVATE "-lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS}")
if(CMAKE_DL_LIBS)
  set(GAIL_PC_LIBS_PRIVATE "-lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT} -l${CMAKE_DL_LIBS}")
endif()
if(MINGW)
  set(GAIL_PC_LIBS_PRIVATE "-lstdc++ -lpthread")
endif()
if(TARGET gail_cuda)
  # static linking of a CUDA build also needs NVIDIA's static runtime (the shared library has it inside)
  set(GAIL_PC_LIBS_PRIVATE "${GAIL_PC_LIBS_PRIVATE} -L${CUDAToolkit_LIBRARY_DIR} -lcudart_static -lrt")
endif()
configure_file(cmake/gail.pc.in ${CMAKE_BINARY_DIR}/gail.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/gail.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
