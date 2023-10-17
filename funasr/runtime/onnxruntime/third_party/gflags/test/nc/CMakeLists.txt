## gflags negative compilation tests

cmake_minimum_required (VERSION 2.8.12 FATAL_ERROR)

if (NOT TEST_NAME)
  message (FATAL_ERROR "Missing TEST_NAME CMake flag")
endif ()
string (TOUPPER ${TEST_NAME} TEST_NAME_UPPER)

project (gflags_${TEST_NAME})

find_package (gflags REQUIRED)
include_directories ("${CMAKE_CURRENT_SOURCE_DIR}/..")
add_definitions (-DTEST_${TEST_NAME_UPPER})
add_executable (gflags_${TEST_NAME} gflags_nc.cc)
target_link_libraries(gflags_${TEST_NAME} gflags)
