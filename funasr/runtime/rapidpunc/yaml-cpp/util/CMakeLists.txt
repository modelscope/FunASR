add_executable(yaml-cpp-sandbox sandbox.cpp)
add_executable(yaml-cpp-parse parse.cpp)
add_executable(yaml-cpp-read read.cpp)

target_link_libraries(yaml-cpp-sandbox PRIVATE yaml-cpp)
target_link_libraries(yaml-cpp-parse PRIVATE yaml-cpp)
target_link_libraries(yaml-cpp-read PRIVATE yaml-cpp)

set_property(TARGET yaml-cpp-sandbox PROPERTY OUTPUT_NAME sandbox)
set_property(TARGET yaml-cpp-parse PROPERTY OUTPUT_NAME parse)
set_property(TARGET yaml-cpp-read PROPERTY OUTPUT_NAME read)

set_target_properties(yaml-cpp-sandbox
  PROPERTIES
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME sandbox)

set_target_properties(yaml-cpp-parse
  PROPERTIES
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME parse)

set_target_properties(yaml-cpp-read
  PROPERTIES
    CXX_STANDARD_REQUIRED ON
    OUTPUT_NAME read)

if (NOT DEFINED CMAKE_CXX_STANDARD)
  set_target_properties(yaml-cpp-sandbox yaml-cpp-parse yaml-cpp-read
    PROPERTIES
      CXX_STANDARD 11)
endif()
