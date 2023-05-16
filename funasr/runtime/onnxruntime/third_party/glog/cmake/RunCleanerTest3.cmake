set (RUNS 3)

# Create the subdirectory required by this unit test.
file (MAKE_DIRECTORY ${TEST_DIR}/${TEST_SUBDIR})

foreach (iter RANGE 1 ${RUNS})
  execute_process (COMMAND ${LOGCLEANUP} -log_dir=${TEST_DIR}
    RESULT_VARIABLE _RESULT)

  if (NOT _RESULT EQUAL 0)
    message (FATAL_ERROR "Failed to run logcleanup_unittest (error: ${_RESULT})")
  endif (NOT _RESULT EQUAL 0)

  # Ensure the log files to have different modification timestamps such that
  # exactly one log file remains at the end. Otherwise all log files will be
  # retained.
  execute_process (COMMAND ${CMAKE_COMMAND} -E sleep 2)
endforeach (iter)

file (GLOB LOG_FILES ${TEST_DIR}/${TEST_SUBDIR}/test_cleanup_*.relativefoo)
list (LENGTH LOG_FILES NUM_FILES)

if (NOT NUM_FILES EQUAL 1)
  message (SEND_ERROR "Expected 1 log file in build directory ${TEST_DIR}${TEST_SUBDIR} but found ${NUM_FILES}")
endif (NOT NUM_FILES EQUAL 1)

# Remove the subdirectory required by this unit test.
file (REMOVE_RECURSE ${TEST_DIR}/${TEST_SUBDIR})
