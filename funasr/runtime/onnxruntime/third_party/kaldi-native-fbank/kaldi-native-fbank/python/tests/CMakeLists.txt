function(kaldi_native_fbank_add_py_test source)
  get_filename_component(name ${source} NAME_WE)
  set(name "${name}_py")

    message(STATUS "source: ${source}")

  add_test(NAME ${name}
    COMMAND
      "${PYTHON_EXECUTABLE}"
      "${CMAKE_CURRENT_SOURCE_DIR}/${source}"
  )

  get_filename_component(kaldi_native_fbank_path ${CMAKE_CURRENT_LIST_DIR} DIRECTORY)

  set_property(TEST ${name}
    PROPERTY ENVIRONMENT "PYTHONPATH=${kaldi_native_fbank_path}:$<TARGET_FILE_DIR:_kaldi_native_fbank>:$ENV{PYTHONPATH}"
  )
endfunction()

# please sort the files in alphabetic order
set(py_test_files
  test_frame_extraction_options.py
  test_mel_bank_options.py
  test_fbank_options.py
)

if(KALDI_NATIVE_FBANK_BUILD_TESTS)
  foreach(source IN LISTS py_test_files)
    kaldi_native_fbank_add_py_test(${source})
  endforeach()
endif()
