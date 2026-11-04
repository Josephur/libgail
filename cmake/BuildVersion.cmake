# SPDX-License-Identifier: MIT
# Writes GAIL_BUILD_VERSION to OUTPUT, leaving the file alone if it is already right.
#   clean build of the tagged release    1.1.0
#   3 commits after the last tag         1.1.0+3.gabc1234
#   with uncommitted changes             1.1.0+3.gabc1234.dirty
#   no git, or not GAIL's own repository 1.1.0
# Usage: cmake -DGIT_EXECUTABLE=... -DSOURCE_DIR=... -DBASE_VERSION=... -DOUTPUT=... -P BuildVersion.cmake

set(version "${BASE_VERSION}")

function(run_git out)
  execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${SOURCE_DIR}" ${ARGN}
                  OUTPUT_VARIABLE text RESULT_VARIABLE status ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(status EQUAL 0)
    set(${out} "${text}" PARENT_SCOPE)
  else()
    set(${out} "" PARENT_SCOPE)
  endif()
endfunction()

if(GIT_EXECUTABLE)
  # An unpacked source archive inside some other repository must not take that repository's version
  run_git(top rev-parse --show-toplevel)
  set(own_repo FALSE)
  if(top)
    file(REAL_PATH "${top}" top_real)
    file(REAL_PATH "${SOURCE_DIR}" source_real)
    if(top_real STREQUAL source_real)
      set(own_repo TRUE)
    endif()
  endif()

  if(own_repo)
    run_git(described describe --tags --long --dirty --abbrev=7 --match "v[0-9]*")
    if(described MATCHES "^v(.+)-([0-9]+)-(g[0-9a-f]+)(-dirty)?$")
      set(tag_version "${CMAKE_MATCH_1}")
      set(count "${CMAKE_MATCH_2}")
      set(hash "${CMAKE_MATCH_3}")
      set(dirty "${CMAKE_MATCH_4}")
    else()
      # No release tag yet: count every commit
      run_git(count rev-list --count HEAD)
      run_git(hash rev-parse --short=7 HEAD)
      set(hash "g${hash}")
      set(tag_version "")
      run_git(changes status --porcelain)
      set(dirty "${changes}")
    endif()

    if(count MATCHES "^[0-9]+$" AND NOT (tag_version STREQUAL BASE_VERSION AND count EQUAL 0 AND NOT dirty))
      set(version "${BASE_VERSION}+${count}.${hash}")
      if(dirty)
        string(APPEND version ".dirty")
      endif()
    endif()
  endif()
endif()

set(content "#define GAIL_BUILD_VERSION \"${version}\"\n")
set(current "")
if(EXISTS "${OUTPUT}")
  file(READ "${OUTPUT}" current)
endif()
if(NOT current STREQUAL content)
  file(WRITE "${OUTPUT}" "${content}")
  message(STATUS "GAIL version ${version}")
endif()
