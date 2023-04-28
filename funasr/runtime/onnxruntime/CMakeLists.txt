cmake_minimum_required(VERSION 3.16)

project(FunASROnnx)

option(ENABLE_GLOG "Whether to build glog" ON)
option(ENABLE_WEBSOCKET "Whether to build websocket server" ON)
# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
    message("Big endian system")
else()
    message("Little endian system")
endif()

# for onnxruntime
IF(WIN32)
	if(CMAKE_CL_64)
		link_directories(${ONNXRUNTIME_DIR}\\lib)
	else()
		add_definitions(-D_WIN_X86)
	endif()
ELSE()
    link_directories(${ONNXRUNTIME_DIR}/lib)
endif()

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
  add_subdirectory(third_party/websocket)

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

include_directories(${PROJECT_SOURCE_DIR}/third_party/kaldi-native-fbank)
include_directories(${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include)

add_subdirectory(third_party/yaml-cpp)
add_subdirectory(third_party/kaldi-native-fbank/kaldi-native-fbank/csrc)
add_subdirectory(src)

if(ENABLE_GLOG)
    include_directories(${PROJECT_SOURCE_DIR}/third_party/glog)
    set(BUILD_TESTING OFF)
    add_subdirectory(third_party/glog)
endif()


