FILE(GLOB HEADER_FILES ../include/fst/*.h)

if(WIN32)
add_library(fst STATIC
  compat.cc
  flags.cc
  fst-types.cc
  fst.cc
  mapped-file.cc
  properties.cc
  symbol-table.cc
  symbol-table-ops.cc
  util.cc
  weight.cc
  ${HEADER_FILES}
)
else()
add_library(fst
  compat.cc
  flags.cc
  fst-types.cc
  fst.cc
  mapped-file.cc
  properties.cc
  symbol-table.cc
  symbol-table-ops.cc
  util.cc
  weight.cc
  ${HEADER_FILES}
)
endif()

set_target_properties(fst PROPERTIES
  SOVERSION "${SOVERSION}"
)

include_directories(${CMAKE_SOURCE_DIR}/build/third_party/glog)
include_directories(${CMAKE_SOURCE_DIR}/third_party/glog/src)
link_directories(${CMAKE_SOURCE_DIR}/build/third_party/glog)

include_directories(${CMAKE_SOURCE_DIR}/third_party/gflags/src/include)
link_directories(${CMAKE_SOURCE_DIR}/build/third_party/gflags)

target_link_libraries(fst PUBLIC glog gflags)
add_dependencies(fst glog gflags)

install(TARGETS fst
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib)
