cmake_minimum_required(VERSION 3.10)

project(FunASRWebscoket) 

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


option(ENABLE_WEBSOCKET "Whether to build websocket server" ON)
 
if(ENABLE_WEBSOCKET)
  cmake_policy(SET CMP0135 NEW)

  include(FetchContent)
  FetchContent_Declare(websocketpp
  GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
    GIT_TAG 0.8.2
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/websocket
    )
  
  FetchContent_MakeAvailable(websocketpp)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/websocket)
   

  FetchContent_Declare(asio
     URL   https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-24-0.tar.gz
   SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/asio
  )
  
  FetchContent_MakeAvailable(asio)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/asio/asio/include)
 
  FetchContent_Declare(json
     URL   https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
   SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/json
  )
  
  FetchContent_MakeAvailable(json)
  include_directories(${PROJECT_SOURCE_DIR}/third_party/json/include)
 
 

endif()

# Include generated *.pb.h files
link_directories(${ONNXRUNTIME_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank)

add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp yaml-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank/kaldi-native-fbank/csrc csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/src src)

include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog)
set(BUILD_TESTING OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog glog)
 

add_executable(websocketmain "websocketmain.cpp" "websocketsrv.cpp")
add_executable(websocketclient "websocketclient.cpp")

target_link_libraries(websocketclient PUBLIC funasr)
target_link_libraries(websocketmain PUBLIC funasr)
