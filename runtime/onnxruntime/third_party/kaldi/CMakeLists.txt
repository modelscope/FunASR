cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(kaldi)

add_library(kaldi-util STATIC
  base/kaldi-error.cc
  base/kaldi-math.cc
  util/kaldi-io.cc
  util/parse-options.cc
  util/simple-io-funcs.cc
  util/text-utils.cc
)
#target_link_libraries(kaldi-util PUBLIC utils)

add_library(kaldi-decoder STATIC
  lat/determinize-lattice-pruned.cc
  lat/lattice-functions.cc
  decoder/lattice-faster-decoder.cc
  decoder/lattice-faster-online-decoder.cc
)

if (WIN32)
target_link_libraries(kaldi-decoder PUBLIC kaldi-util)
else()
target_link_libraries(kaldi-decoder PUBLIC kaldi-util dl)
endif (WIN32)


if (WIN32)
  target_compile_definitions (kaldi-decoder PUBLIC GLOG_NO_ABBREVIATED_SEVERITIES)
endif (WIN32)

include_directories(${CMAKE_SOURCE_DIR}/build/third_party/glog)
include_directories(${CMAKE_SOURCE_DIR}/third_party/glog/src)
include_directories(${CMAKE_SOURCE_DIR}/third_party/gflags/src/include)
if(TRUE)
  # Arpa binary
  add_executable(arpa2fst
    lm/arpa-file-parser.cc
    lm/arpa-lm-compiler.cc
    lmbin/arpa2fst.cc
  )

if (WIN32)
target_link_libraries(arpa2fst PUBLIC kaldi-util fst)
else()
target_link_libraries(arpa2fst PUBLIC kaldi-util fst dl)
endif (WIN32)


  # FST tools binary
  set(FST_BINS
    fstaddselfloops
    fstdeterminizestar
    fstisstochastic
    fstminimizeencoded
    fsttablecompose
  )

  foreach(name IN LISTS FST_BINS)
    add_executable(${name}
      fstbin/${name}.cc
      fstext/kaldi-fst-io.cc
    )
if (WIN32)
    target_link_libraries(${name} PUBLIC kaldi-util fst)
else()
    target_link_libraries(${name} PUBLIC kaldi-util fst dl)
endif (WIN32)
  endforeach()
endif()
