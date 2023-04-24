add_subdirectory(csrc)

if(KALDI_NATIVE_FBANK_BUILD_PYTHON)
  message(STATUS "Building Python")
  add_subdirectory(python)
else()
  message(STATUS "Disable building Python")
endif()
