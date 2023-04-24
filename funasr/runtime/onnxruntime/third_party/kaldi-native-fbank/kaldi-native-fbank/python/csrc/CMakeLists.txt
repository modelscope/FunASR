pybind11_add_module(_kaldi_native_fbank
  feature-fbank.cc
  feature-window.cc
  kaldi-native-fbank.cc
  mel-computations.cc
  online-feature.cc
  utils.cc
)

if(APPLE)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGE_DIR
  )
  message(STATUS "PYTHON_SITE_PACKAGE_DIR: ${PYTHON_SITE_PACKAGE_DIR}")
  target_link_libraries(_kaldi_native_fbank PRIVATE "-Wl,-rpath,${PYTHON_SITE_PACKAGE_DIR}")
endif()

if(NOT WIN32)
  target_link_libraries(_kaldi_native_fbank PRIVATE "-Wl,-rpath,${kaldi_native_fbank_rpath_origin}/kaldi_native_fbank/lib")
endif()

target_link_libraries(_kaldi_native_fbank PRIVATE kaldi-native-fbank-core)

install(TARGETS _kaldi_native_fbank
  DESTINATION ../
)
