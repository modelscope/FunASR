
include_directories(${PROJECT_SOURCE_DIR})
set(sources
  feature-fbank.cc
  feature-functions.cc
  feature-window.cc
  fftsg.c
  mel-computations.cc
  online-feature.cc
  rfft.cc
)

if(KALDI_NATIVE_FBANK_ENABLE_CHECK)
  list(APPEND sources log.cc)
endif()

add_library(kaldi-native-fbank-core ${sources})
if(KALDI_NATIVE_FBANK_ENABLE_CHECK)
  target_compile_definitions(kaldi-native-fbank-core PUBLIC KNF_ENABLE_CHECK=1)

  if(KNF_HAVE_EXECINFO_H)
    target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_EXECINFO_H=1)
  endif()

  if(KNF_HAVE_CXXABI_H)
    target_compile_definitions(kaldi-native-fbank-core PRIVATE KNF_HAVE_CXXABI_H=1)
  endif()
endif()

# We are using std::call_once() in log.h,which requires us to link with -pthread
if(NOT WIN32 AND KALDI_NATIVE_FBANK_ENABLE_CHECK)
  target_link_libraries(kaldi-native-fbank-core -pthread)
endif()

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  add_executable(test-online-fbank test-online-fbank.cc)
  target_link_libraries(test-online-fbank kaldi-native-fbank-core)
endif()

function(kaldi_native_fbank_add_test source)
  get_filename_component(name ${source} NAME_WE)
  add_executable(${name} "${source}")
  target_link_libraries(${name}
    PRIVATE
      kaldi-native-fbank-core
      gtest
      gtest_main
  )

  add_test(NAME "Test.${name}"
    COMMAND
    $<TARGET_FILE:${name}>
  )
endfunction()

# please sort the source files alphabetically
set(test_srcs
  # test-online-feature.cc
  test-log.cc
  test-rfft.cc
)

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  foreach(source IN LISTS test_srcs)
    kaldi_native_fbank_add_test(${source})
  endforeach()
endif()

install(TARGETS kaldi-native-fbank-core
  DESTINATION lib
)

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  install(TARGETS test-online-fbank
    DESTINATION bin
  )
endif()

file(MAKE_DIRECTORY
  DESTINATION
    ${PROJECT_BINARY_DIR}/include/kaldi-native-fbank/csrc
)
file(GLOB_RECURSE all_headers *.h)

file(COPY
  ${all_headers}
  DESTINATION
    ${PROJECT_BINARY_DIR}/include/kaldi-native-fbank/csrc
)

install(FILES ${all_headers}
  DESTINATION include/kaldi-native-fbank/csrc
)
