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
        DESCRIPTION "GAIL, GPU Accelerated Intraframe Library: a DCI JPEG 2000 encoder and decoder")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(GAIL_ENABLE_VULKAN "Build the Vulkan compute backend" ON)
set(GAIL_ENABLE_CUDA AUTO CACHE STRING "Build the optional CUDA backend (AUTO, ON, OFF)")
# The CUDA backend is built once per toolkit. With neither set, the one toolkit CMake finds
# (CMAKE_CUDA_COMPILER, CUDAToolkit_ROOT, CUDA_PATH, nvcc on the PATH) is used for its plugin.
set(GAIL_CUDA13_ROOT "" CACHE PATH "CUDA 13 toolkit for the gail_cuda13 plugin (Turing and newer GPUs)")
set(GAIL_CUDA12_ROOT "" CACHE PATH "CUDA 12 toolkit for the gail_cuda12 plugin (Maxwell, Pascal and Volta GPUs)")
set(GAIL_CUDA13_ARCHS "75;86;89;120" CACHE STRING "CUDA architectures of the gail_cuda13 plugin")
set(GAIL_CUDA12_ARCHS "50;52;60;61;70" CACHE STRING "CUDA architectures of the gail_cuda12 plugin")
set(GAIL_CUDA13_HOST_COMPILER "" CACHE FILEPATH "C++ compiler for the gail_cuda13 plugin (default: the project's)")
set(GAIL_CUDA12_HOST_COMPILER "" CACHE FILEPATH "C++ compiler for the gail_cuda12 plugin, when CUDA 12 does not take the project's")
set(GAIL_CUDA13_FLAGS "" CACHE STRING "Extra nvcc flags for the gail_cuda13 plugin")
set(GAIL_CUDA12_FLAGS "" CACHE STRING "Extra nvcc flags for the gail_cuda12 plugin")
option(GAIL_BUILD_SHARED "Build libgail as a shared library (libgail.so / gail.dll)" ON)
option(GAIL_BUILD_STATIC "Also build the static library (libgail.a / gail_static.lib)" ON)
option(GAIL_BUILD_TOOLS "Build gail-enc and the other command-line tools" ON)
option(GAIL_BUILD_TESTS "Build the tests" ON)
option(GAIL_NATIVE_ARCH "Optimize for this machine's CPU only (-march=native); not for distributed builds" OFF)
set(GAIL_VULKAN_INCLUDE_DIR "" CACHE PATH "Vulkan headers (only headers are needed; default: find_package(Vulkan))")

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/permissive->)
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  if(GAIL_NATIVE_ARCH)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=native>)
  endif()
  # (Vulkan's "sType first, rest zero" struct initializers trip -Wmissing-field-initializers)
  add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-Wall;-Wextra;-Wno-missing-field-initializers>")
endif()

include(GNUInstallDirs)
find_package(Threads REQUIRED)
if(MINGW)
  # self-contained Windows binaries: no libstdc++ / libgcc / winpthread DLLs to ship
  add_link_options(-static)
endif()
# The internals are static and position independent so they can be linked into the shared
# library, which exports only the GAIL_API functions.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(cmake/CoreSources.cmake)
add_library(gail_core STATIC src/core/build_version.cpp ${GAIL_CORE_SOURCES})
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
  set(SHADER_NAMES color color_yuv dwt_rows dwt_cols quantize t1_cm t1_mq compact scan order
                   dec_t1 dec_idwt_rows dec_idwt_cols dec_color)
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

  add_library(gail_vulkan STATIC src/vulkan/vk_backend.cpp src/vulkan/vk_decode.cpp)
  add_dependencies(gail_vulkan gail_shaders)
  target_include_directories(gail_vulkan PRIVATE ${CMAKE_BINARY_DIR}/shaders)
  target_include_directories(gail_vulkan SYSTEM PUBLIC ${GAIL_VK_INCLUDE})
  target_link_libraries(gail_vulkan PUBLIC gail_core ${CMAKE_DL_LIBS})
  target_compile_definitions(gail_vulkan PUBLIC GAIL_HAVE_VULKAN=1)
  list(APPEND GAIL_BACKEND_LIBS gail_vulkan)
endif()

# ---- CUDA backend -------------------------------------------------------------------------
# The backend is built once per CUDA toolkit: CUDA 13 no longer compiles for Maxwell, Pascal
# and Volta, so those GPUs get a build with CUDA 12 and Turing and newer one with CUDA 13.
# The shared library links neither. Each build is a plugin, a shared library with its own
# CUDA runtime inside, which libgail loads from its own directory at run time
# (src/cuda/cuda_plugin.h). The static library cannot carry plugins and links one backend in.
set(GAIL_CUDA_VERSIONS)  # the toolkits in use, of 13 and 12
if(NOT GAIL_ENABLE_CUDA STREQUAL "OFF")
  function(gail_nvcc_version nvcc out)
    execute_process(COMMAND ${nvcc} --version OUTPUT_VARIABLE text ERROR_QUIET)
    if(text MATCHES "release [0-9.]+, V([0-9]+\\.[0-9]+\\.[0-9]+)")
      set(${out} ${CMAKE_MATCH_1} PARENT_SCOPE)
    else()
      set(${out} "" PARENT_SCOPE)
    endif()
  endfunction()

  if(NOT GAIL_CUDA13_ROOT AND NOT GAIL_CUDA12_ROOT)
    # No toolkit was named: the one CMake finds builds the plugin it is for.
    if(CMAKE_CUDA_COMPILER)
      set(GAIL_NVCC ${CMAKE_CUDA_COMPILER})
    else()
      find_program(GAIL_NVCC nvcc HINTS ${CUDAToolkit_ROOT} ENV CUDAToolkit_ROOT ENV CUDA_PATH ENV CUDA_HOME
                   PATHS /usr/local/cuda /opt/cuda PATH_SUFFIXES bin)
    endif()
    if(GAIL_NVCC)
      gail_nvcc_version(${GAIL_NVCC} version)
      get_filename_component(root ${GAIL_NVCC} DIRECTORY)
      get_filename_component(root ${root} DIRECTORY)
      if(version VERSION_GREATER_EQUAL 13)
        set(GAIL_CUDA13_ROOT ${root})
      elseif(version VERSION_GREATER_EQUAL 12)
        set(GAIL_CUDA12_ROOT ${root})
      else()
        message(WARNING "${GAIL_NVCC} is CUDA ${version}; the CUDA backend needs CUDA 12 or 13")
      endif()
    endif()
  endif()
  foreach(ver 13 12)
    if(GAIL_CUDA${ver}_ROOT)
      set(GAIL_CUDA${ver}_NVCC ${GAIL_CUDA${ver}_ROOT}/bin/nvcc${CMAKE_EXECUTABLE_SUFFIX})
      if(NOT EXISTS ${GAIL_CUDA${ver}_NVCC})
        message(FATAL_ERROR "GAIL_CUDA${ver}_ROOT: no nvcc in ${GAIL_CUDA${ver}_ROOT}/bin")
      endif()
      gail_nvcc_version(${GAIL_CUDA${ver}_NVCC} GAIL_CUDA${ver}_VERSION)
      if(NOT GAIL_CUDA${ver}_VERSION MATCHES "^${ver}\\.")
        message(FATAL_ERROR "GAIL_CUDA${ver}_ROOT holds CUDA ${GAIL_CUDA${ver}_VERSION}, not CUDA ${ver}")
      endif()
      list(APPEND GAIL_CUDA_VERSIONS ${ver})
    endif()
  endforeach()
  if(NOT GAIL_CUDA_VERSIONS AND GAIL_ENABLE_CUDA STREQUAL "ON")
    message(FATAL_ERROR "CUDA requested but no CUDA 12 or 13 toolkit found (GAIL_CUDA13_ROOT, GAIL_CUDA12_ROOT)")
  endif()
endif()

if(GAIL_CUDA_VERSIONS)
  include(ExternalProject)
  get_property(multi_config GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
  # The static library's backend: CUDA 13 when both toolkits are there.
  list(GET GAIL_CUDA_VERSIONS 0 GAIL_CUDA_STATIC_VERSION)
  set(GAIL_CUDA_PLUGIN_FILES)
  set(GAIL_CUDA_OBJECTS)
  foreach(ver ${GAIL_CUDA_VERSIONS})
    set(name gail_cuda${ver})
    set(args
      -DGAIL_SOURCE_DIR:PATH=${CMAKE_SOURCE_DIR}
      -DGAIL_CUDA_NAME:STRING=${name}
      -DGAIL_CUDA_ARCHS:STRING=${GAIL_CUDA${ver}_ARCHS}
      -DGAIL_NATIVE_ARCH:BOOL=${GAIL_NATIVE_ARCH}
      -DCMAKE_BUILD_TYPE:STRING=${CMAKE_BUILD_TYPE}
      -DCMAKE_INSTALL_PREFIX:PATH=${CMAKE_BINARY_DIR}/${name}/install  # nothing is installed from there
      -DCMAKE_CUDA_COMPILER:FILEPATH=${GAIL_CUDA${ver}_NVCC}
      -DCUDAToolkit_ROOT:PATH=${GAIL_CUDA${ver}_ROOT}
      # _INIT, so that CMake still adds its own defaults for the platform (/EHsc with MSVC)
      -DCMAKE_CUDA_FLAGS_INIT:STRING=${GAIL_CUDA${ver}_FLAGS}
      -DCMAKE_CXX_FLAGS:STRING=${CMAKE_CXX_FLAGS})
    if(GAIL_CUDA${ver}_HOST_COMPILER)
      list(APPEND args -DCMAKE_CXX_COMPILER:FILEPATH=${GAIL_CUDA${ver}_HOST_COMPILER}
                       -DCMAKE_CUDA_HOST_COMPILER:FILEPATH=${GAIL_CUDA${ver}_HOST_COMPILER})
    else()
      list(APPEND args -DCMAKE_CXX_COMPILER:FILEPATH=${CMAKE_CXX_COMPILER})
    endif()
    if(CMAKE_MSVC_RUNTIME_LIBRARY)
      list(APPEND args -DCMAKE_MSVC_RUNTIME_LIBRARY:STRING=${CMAKE_MSVC_RUNTIME_LIBRARY})
    endif()
    set(byproducts)
    if(GAIL_BUILD_SHARED)
      # next to the gail library and the tools in the build tree
      list(APPEND args -DGAIL_CUDA_PLUGIN_DIR:PATH=${CMAKE_BINARY_DIR})
      set(file ${CMAKE_SHARED_MODULE_PREFIX}${name}${CMAKE_SHARED_MODULE_SUFFIX})
      if(multi_config)
        list(APPEND GAIL_CUDA_PLUGIN_FILES ${CMAKE_BINARY_DIR}/$<CONFIG>/${file})
      else()
        list(APPEND GAIL_CUDA_PLUGIN_FILES ${CMAKE_BINARY_DIR}/${file})
        list(APPEND byproducts ${CMAKE_BINARY_DIR}/${file})
      endif()
    endif()
    if(GAIL_BUILD_STATIC AND ver EQUAL GAIL_CUDA_STATIC_VERSION)
      set(objects ${CMAKE_BINARY_DIR}/${name}/objects)
      list(APPEND args -DGAIL_CUDA_OBJECT_DIR:PATH=${objects} -DGAIL_CUDA_OBJECT_SUFFIX:STRING=${CMAKE_CXX_OUTPUT_EXTENSION})
      foreach(unit cuda_backend cuda_decode)
        list(APPEND GAIL_CUDA_OBJECTS ${objects}/${unit}${CMAKE_CXX_OUTPUT_EXTENSION})
      endforeach()
      list(APPEND byproducts ${GAIL_CUDA_OBJECTS})
    endif()
    # BUILD_ALWAYS: the plugin's own build knows what a change to the sources touches.
    ExternalProject_Add(${name}_build
      SOURCE_DIR ${CMAKE_SOURCE_DIR}/cmake/cuda_backend
      PREFIX ${CMAKE_BINARY_DIR}/${name}
      BINARY_DIR ${CMAKE_BINARY_DIR}/${name}/build
      CMAKE_CACHE_ARGS ${args}
      BUILD_ALWAYS ON
      USES_TERMINAL_BUILD ON
      INSTALL_COMMAND ""
      BUILD_BYPRODUCTS ${byproducts})
    message(STATUS "CUDA backend: ${name} with CUDA ${GAIL_CUDA${ver}_VERSION} (${GAIL_CUDA${ver}_ROOT}), sm ${GAIL_CUDA${ver}_ARCHS}")
  endforeach()

  if(GAIL_BUILD_SHARED)
    # The library's side of the plugins. The tools have the library's internals linked in
    # and look for the plugins themselves; installed, they are in bin/ and the plugins in lib/.
    add_library(gail_cuda_host STATIC src/cuda/cuda_host.cpp)
    add_library(gail_cuda_host_tools STATIC src/cuda/cuda_host.cpp)
    foreach(host gail_cuda_host gail_cuda_host_tools)
      target_link_libraries(${host} PUBLIC gail_core ${CMAKE_DL_LIBS})
      target_compile_definitions(${host} PUBLIC GAIL_HAVE_CUDA=1)
      foreach(ver ${GAIL_CUDA_VERSIONS})
        add_dependencies(${host} gail_cuda${ver}_build)
      endforeach()
    endforeach()
    if(NOT WIN32)
      file(RELATIVE_PATH plugin_dir ${CMAKE_INSTALL_FULL_BINDIR} ${CMAKE_INSTALL_FULL_LIBDIR})
      target_compile_definitions(gail_cuda_host_tools PRIVATE GAIL_CUDA_TOOL_PLUGIN_DIR="${plugin_dir}")
    endif()
  endif()
  if(GAIL_BUILD_STATIC)
    set(CUDAToolkit_ROOT ${GAIL_CUDA${GAIL_CUDA_STATIC_VERSION}_ROOT})
    find_package(CUDAToolkit REQUIRED)
    string(REPLACE ";" " " arch_text "${GAIL_CUDA${GAIL_CUDA_STATIC_VERSION}_ARCHS}")
    set_source_files_properties(${GAIL_CUDA_OBJECTS} PROPERTIES EXTERNAL_OBJECT TRUE GENERATED TRUE)
    add_library(gail_cuda_builtin STATIC src/cuda/cuda_host.cpp src/cuda/cuda_plugin.cpp ${GAIL_CUDA_OBJECTS})
    add_dependencies(gail_cuda_builtin gail_cuda${GAIL_CUDA_STATIC_VERSION}_build)
    target_link_libraries(gail_cuda_builtin PUBLIC gail_core CUDA::cudart_static)
    target_compile_definitions(gail_cuda_builtin PUBLIC GAIL_HAVE_CUDA=1 PRIVATE
      GAIL_CUDA_BUILTIN=1
      GAIL_CUDA_PLUGIN_NAME="built in"
      GAIL_CUDA_PLUGIN_TOOLKIT="${GAIL_CUDA${GAIL_CUDA_STATIC_VERSION}_VERSION}"
      GAIL_CUDA_PLUGIN_ARCHS="${arch_text}")
  endif()
endif()

# ---- libgail: include/gail/gail.h --------------------------------------------------------
set(GAIL_API_SOURCES src/api/gail_api.cpp src/api/gail_decoder_api.cpp src/api/devices.cpp)
set(GAIL_LIB_TARGETS)

# The CUDA backend as each kind of binary has it: the shared library and the tools built
# with it load the plugins, the static library and the tools of a static-only build have
# one backend linked in.
set(GAIL_SHARED_BACKEND_LIBS ${GAIL_BACKEND_LIBS})
set(GAIL_STATIC_BACKEND_LIBS ${GAIL_BACKEND_LIBS})
set(GAIL_TOOL_BACKEND_LIBS ${GAIL_BACKEND_LIBS})
if(TARGET gail_cuda_host)
  list(APPEND GAIL_SHARED_BACKEND_LIBS gail_cuda_host)
  list(APPEND GAIL_TOOL_BACKEND_LIBS gail_cuda_host_tools)
endif()
if(TARGET gail_cuda_builtin)
  list(APPEND GAIL_STATIC_BACKEND_LIBS gail_cuda_builtin)
  if(NOT TARGET gail_cuda_host)
    list(APPEND GAIL_TOOL_BACKEND_LIBS gail_cuda_builtin)
  endif()
endif()

function(gail_library target type)
  if(type STREQUAL "STATIC")
    # one self-contained archive: the internal objects go in, only system libraries are needed
    set(objs)
    foreach(lib gail_core ${GAIL_STATIC_BACKEND_LIBS})
      list(APPEND objs $<TARGET_OBJECTS:${lib}>)
    endforeach()
    add_library(${target} STATIC ${GAIL_API_SOURCES} ${objs})
    target_link_libraries(${target} PUBLIC Threads::Threads ${CMAKE_DL_LIBS})
    if(TARGET gail_cuda_builtin)
      # the backend's object files come from its own build, which the objects above leave out
      target_sources(${target} PRIVATE ${GAIL_CUDA_OBJECTS})
      add_dependencies(${target} gail_cuda${GAIL_CUDA_STATIC_VERSION}_build)
      target_link_libraries(${target} PUBLIC CUDA::cudart_static)
    endif()
    foreach(lib gail_core ${GAIL_STATIC_BACKEND_LIBS})
      target_compile_definitions(${target} PRIVATE $<TARGET_PROPERTY:${lib},INTERFACE_COMPILE_DEFINITIONS>)
      target_include_directories(${target} PRIVATE $<TARGET_PROPERTY:${lib},INTERFACE_INCLUDE_DIRECTORIES>)
    endforeach()
  else()
    add_library(${target} ${type} ${GAIL_API_SOURCES})
    target_link_libraries(${target} PRIVATE gail_core ${GAIL_SHARED_BACKEND_LIBS})
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
  target_link_libraries(gail-enc PRIVATE gail_io ${GAIL_TOOL_BACKEND_LIBS})

  add_executable(gail-check src/tools/gail_check.cpp)
  target_link_libraries(gail-check PRIVATE gail_core)

  # the decoder's tool goes through the public API alone
  add_executable(gail-dec src/tools/gail_dec.cpp)
  target_link_libraries(gail-dec PRIVATE ${GAIL_LINK} Threads::Threads)

  add_executable(gail-bench src/tools/gail_bench.cpp)
  target_link_libraries(gail-bench PRIVATE gail_io ${GAIL_TOOL_BACKEND_LIBS})
endif()

# ---- tests -------------------------------------------------------------------------------
if(GAIL_BUILD_TESTS)
  enable_testing()
  add_executable(gail-tests tests/test_main.cpp)
  target_link_libraries(gail-tests PRIVATE gail_io ${GAIL_TOOL_BACKEND_LIBS})
  add_test(NAME gail-tests COMMAND gail-tests WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  if(TARGET gail_cuda_host)
    # Stand-ins for the CUDA plugins, for the loader's tests, which then need no GPU and no
    # CUDA runtime. Each is in a directory of its own under the name the loader looks for,
    # in test-plugins/ next to gail-tests.
    function(gail_test_plugin dir plugin mode)
      set(target gail_test_plugin_${dir})
      add_library(${target} MODULE tests/test_cuda_plugin.cpp)
      target_include_directories(${target} PRIVATE src)
      target_compile_definitions(${target} PRIVATE TEST_PLUGIN_NAME="${plugin}" TEST_PLUGIN_MODE=${mode})
      set_target_properties(${target} PROPERTIES OUTPUT_NAME ${plugin}
        LIBRARY_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:gail-tests>/test-plugins/${dir}
        RUNTIME_OUTPUT_DIRECTORY $<TARGET_FILE_DIR:gail-tests>/test-plugins/${dir})
      add_dependencies(gail-tests ${target})
    endfunction()
    gail_test_plugin(cuda13 gail_cuda13 0)     # modes: see tests/test_cuda_plugin.cpp
    gail_test_plugin(cuda12 gail_cuda12 0)
    gail_test_plugin(other_abi gail_cuda13 1)
    gail_test_plugin(declines gail_cuda12 2)
    gail_test_plugin(no_driver gail_cuda13 3)
    gail_test_plugin(busy gail_cuda12 4)
    target_compile_definitions(gail-tests PRIVATE GAIL_TEST_PLUGINS=1)
  endif()

  # public API, compiled as C against the installed-style header
  add_executable(gail-api-test tests/test_api.c)
  target_link_libraries(gail-api-test PRIVATE ${GAIL_LINK} Threads::Threads)
  if(NOT WIN32)
    target_link_libraries(gail-api-test PRIVATE m)
  endif()
  add_test(NAME gail-api-test COMMAND gail-api-test)
endif()

# ---- install -----------------------------------------------------------------------------
install(FILES include/gail/gail.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/gail)
install(TARGETS ${GAIL_LIB_TARGETS} EXPORT gailTargets
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(EXPORT gailTargets NAMESPACE gail:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gail)
file(WRITE ${CMAKE_BINARY_DIR}/gailConfig.cmake "include(\"\${CMAKE_CURRENT_LIST_DIR}/gailTargets.cmake\")\n")
install(FILES ${CMAKE_BINARY_DIR}/gailConfig.cmake DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/gail)
if(GAIL_BUILD_TOOLS)
  install(TARGETS gail-enc gail-dec gail-check RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()
if(GAIL_CUDA_PLUGIN_FILES)
  # the CUDA plugins go next to the gail shared library, the one place it looks for them
  if(WIN32)
    install(PROGRAMS ${GAIL_CUDA_PLUGIN_FILES} DESTINATION ${CMAKE_INSTALL_BINDIR})
  else()
    install(PROGRAMS ${GAIL_CUDA_PLUGIN_FILES} DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
endif()

# pkg-config: FFmpeg's configure finds the library this way
set(GAIL_PC_LIBS_PRIVATE "-lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT} ${CMAKE_DL_LIBS}")
if(CMAKE_DL_LIBS)
  set(GAIL_PC_LIBS_PRIVATE "-lstdc++ -lm ${CMAKE_THREAD_LIBS_INIT} -l${CMAKE_DL_LIBS}")
endif()
if(MINGW)
  set(GAIL_PC_LIBS_PRIVATE "-lstdc++ -lpthread")
endif()
if(TARGET gail_cuda_builtin)
  # static linking of a CUDA build also needs NVIDIA's static runtime (the plugins have it inside)
  set(GAIL_PC_LIBS_PRIVATE "${GAIL_PC_LIBS_PRIVATE} -L${CUDAToolkit_LIBRARY_DIR} -lcudart_static -lrt")
endif()
configure_file(cmake/gail.pc.in ${CMAKE_BINARY_DIR}/gail.pc @ONLY)
install(FILES ${CMAKE_BINARY_DIR}/gail.pc DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig)
