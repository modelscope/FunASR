find_package(Threads REQUIRED)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
set(BUILD_MOCK ON CACHE BOOL "" FORCE)
set(CMAKE_POLICY_DEFAULT_CMP0048 NEW)

add_subdirectory(
  "${CMAKE_CURRENT_SOURCE_DIR}/gtest-1.11.0"
  "${CMAKE_CURRENT_BINARY_DIR}/prefix")

include_directories(SYSTEM "${CMAKE_CURRENT_SOURCE_DIR}/gtest-1.11.0/googletest/include")

set(test-new-api-pattern "new-api/*.cpp")
set(test-source-pattern "*.cpp" "integration/*.cpp" "node/*.cpp")
if (CMAKE_VERSION VERSION_GREATER 3.11)
  list(INSERT test-new-api-pattern 0 CONFIGURE_DEPENDS)
  list(INSERT test-source-pattern 0 CONFIGURE_DEPENDS)
endif()

file(GLOB test-new-api-sources ${test-new-api-pattern})
file(GLOB test-sources ${test-source-pattern})

add_executable(yaml-cpp-tests "")
target_sources(yaml-cpp-tests
  PRIVATE
    ${test-new-api-sources}
    ${test-sources})
target_include_directories(yaml-cpp-tests
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/integration
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/src)
target_compile_options(yaml-cpp-tests
  PRIVATE
    $<$<CXX_COMPILER_ID:Clang>:-Wno-c99-extensions -Wno-variadic-macros -Wno-sign-compare>
    $<$<CXX_COMPILER_ID:GNU>:-Wno-variadic-macros -Wno-sign-compare>)
target_link_libraries(yaml-cpp-tests
  PRIVATE
    Threads::Threads
    yaml-cpp
    gmock)

set_property(TARGET yaml-cpp-tests PROPERTY CXX_STANDARD_REQUIRED ON)
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set_target_properties(yaml-cpp-tests PROPERTIES CXX_STANDARD 11)
endif()


add_test(NAME yaml-cpp::test COMMAND yaml-cpp-tests)

if (build-windows-dll)
  add_custom_command(
    TARGET yaml-cpp-tests
    POST_BUILD COMMAND ${CMAKE_COMMAND} -E
    copy_if_different "$<TARGET_FILE:yaml-cpp>" "$<TARGET_FILE_DIR:yaml-cpp-tests>")
endif()
