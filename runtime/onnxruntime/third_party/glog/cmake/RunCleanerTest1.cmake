set (RUNS 3)

foreach (iter RANGE 1 ${RUNS})
  set (ENV{GOOGLE_LOG_DIR} ${TEST_DIR})
  execute_process (COMMAND ${LOGCLEANUP} RESULT_VARIABLE _RESULT)

  if (NOT _RESULT EQUAL 0)
    message (FATAL_ERROR "Failed to run logcleanup_unittest (error: ${_RESULT})")
  endif (NOT _RESULT EQUAL 0)

  # Ensure the log files to have different modification timestamps such that
  # exactly one log file remains at the end. Otherwise all log files will be
  # retained.
  execute_process (COMMAND ${CMAKE_COMMAND} -E sleep 1)
endforeach (iter)

file (GLOB LOG_FILES ${TEST_DIR}/*.foobar)
list (LENGTH LOG_FILES NUM_FILES)

if (NOT NUM_FILES EQUAL 1)
  message (SEND_ERROR "Expected 1 log file in log directory but found ${NUM_FILES}")
endif (NOT NUM_FILES EQUAL 1)
