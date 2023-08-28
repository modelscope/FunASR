cmake_minimum_required(VERSION 3.16)

project(FunASRWebscoket) 

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_WEBSOCKET "Whether to build websocket server" ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
option(ENABLE_GLOG "Whether to build glog" ON)
option(ENABLE_FST "Whether to build openfst" ON) # ITN need openfst compiled
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
 
if(ENABLE_WEBSOCKET)
  # cmake_policy(SET CMP0135 NEW)
  include(FetchContent)

  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/websocket )
    FetchContent_Declare(websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
      GIT_TAG 0.8.2
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/websocket
      )
    
    FetchContent_MakeAvailable(websocketpp)
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/websocket)
   
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/asio )
    FetchContent_Declare(asio
      URL   https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-24-0.tar.gz
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/asio
    )
    
    FetchContent_MakeAvailable(asio)
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/asio/asio/include)
 
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/json )
    FetchContent_Declare(json
      URL   https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/json
    )
    
    FetchContent_MakeAvailable(json)
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/json/include)

endif()

# Include generated *.pb.h files
link_directories(${ONNXRUNTIME_DIR}/lib)
link_directories(${FFMPEG_DIR}/lib)

if(ENABLE_GLOG)
    include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src)
    set(BUILD_TESTING OFF)
    add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog glog)
    include_directories(${glog_BINARY_DIR})

endif()

if(ENABLE_FST)
    # fst depend on glog and gflags
    include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/gflags)
    add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/gflags gflags)
    include_directories(${gflags_BINARY_DIR}/include)
      
    # the following openfst if cloned from https://github.com/kkm000/openfst.git
    # with some patch to fix the make errors. 
    add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/openfst openfst)
    include_directories(${openfst_SOURCE_DIR}/src/include)
    
endif()


include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank)

add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp yaml-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank/kaldi-native-fbank/csrc csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/src src)

# install openssl first apt-get install libssl-dev
find_package(OpenSSL REQUIRED)

message("CXX_FLAGS "${CMAKE_CXX_FLAGS})
# 获取项目中所有包含文件夹的路径
get_property(includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
# 遍历并输出每个包含文件夹的路径
foreach(include ${includes})
  message("Include directory: ${include}")
endforeach()

add_subdirectory(bin)
