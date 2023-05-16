# Create the build directory
execute_process (
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_BINARY_DIR}
  RESULT_VARIABLE _DIRECTORY_CREATED_SUCCEEDED
)

if (NOT _DIRECTORY_CREATED_SUCCEEDED EQUAL 0)
  message (FATAL_ERROR "Failed to create build directory")
endif (NOT _DIRECTORY_CREATED_SUCCEEDED EQUAL 0)

file (WRITE ${INITIAL_CACHE} "${CACHEVARS}")
