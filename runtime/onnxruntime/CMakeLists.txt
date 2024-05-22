cmake_minimum_required(VERSION 3.16)

project(FunASROnnx)

option(ENABLE_GLOG "Whether to build glog" ON)
option(ENABLE_FST "Whether to build openfst" ON) # ITN need openfst compiled

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
    message("Big endian system")
else()
    message("Little endian system")
endif()

# json
include(FetchContent)
if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/json/ChangeLog.md )
FetchContent_Declare(json
  URL   https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/json
)

FetchContent_MakeAvailable(json)
endif()

# for onnxruntime
IF(WIN32)
    file(REMOVE ${PROJECT_SOURCE_DIR}/third_party/glog/src/config.h 
                ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/export.h 
                ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/logging.h 
                ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/raw_logging.h 
                ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/stl_logging.h 
                ${PROJECT_SOURCE_DIR}/third_party/glog/src/glog/vlog_is_on.h)
ELSE()
    link_directories(${ONNXRUNTIME_DIR}/lib)
    link_directories(${FFMPEG_DIR}/lib)
endif()
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${PROJECT_SOURCE_DIR}/third_party/kaldi-native-fbank)
include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/jieba/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/jieba/include/limonp/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/kaldi)
include_directories(${PROJECT_SOURCE_DIR}/third_party/json/include)

if(ENABLE_GLOG)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/glog/src)
    set(BUILD_TESTING OFF)
    add_subdirectory(third_party/glog)
    include_directories(${glog_BINARY_DIR})
    
endif()

if(ENABLE_FST)
    # fst depend on glog and gflags
    include_directories(${PROJECT_SOURCE_DIR}/third_party/gflags)
    add_subdirectory(third_party/gflags)
    include_directories(${gflags_BINARY_DIR}/include)
    
    # the following openfst if cloned from https://github.com/kkm000/openfst.git
    # with some patch to fix the make errors. 
    add_subdirectory(third_party/openfst)
    include_directories(${openfst_SOURCE_DIR}/src/include)
    if(WIN32)
    include_directories(${openfst_SOURCE_DIR}/src/lib)
    endif()

endif()

add_subdirectory(third_party/yaml-cpp)
add_subdirectory(third_party/kaldi-native-fbank/kaldi-native-fbank/csrc)
add_subdirectory(third_party/kaldi)
add_subdirectory(src)
add_subdirectory(bin)
