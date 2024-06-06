cmake_minimum_required(VERSION 3.16)

project(FunASRWebscoket) 

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(ENABLE_WEBSOCKET "Whether to build websocket server" ON)
option(ENABLE_PORTAUDIO "Whether to build portaudio" ON)
option(ENABLE_GLOG "Whether to build glog" ON)
option(ENABLE_FST "Whether to build openfst" ON) # ITN need openfst compiled
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(GPU "Whether to build with GPU" OFF)

if(WIN32)
  file(REMOVE ${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src/config.h 
    ${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src/glog/export.h 
    ${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src/glog/logging.h 
    ${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src/glog/raw_logging.h 
    ${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src/glog/stl_logging.h 
    ${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog/src/glog/vlog_is_on.h)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fPIC")
endif()

if(GPU)
    add_definitions(-DUSE_GPU)
    set(TORCH_DIR "/usr/local/lib/python3.8/dist-packages/torch")
    set(TORCH_BLADE_DIR "/usr/local/lib/python3.8/dist-packages/torch_blade")
    include_directories(${TORCH_DIR}/include)
    include_directories(${TORCH_DIR}/include/torch/csrc/api/include)
    link_directories(${TORCH_DIR}/lib)
    link_directories(${TORCH_BLADE_DIR})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC -D_GLIBCXX_USE_CXX11_ABI=0")
endif()
 
if(ENABLE_WEBSOCKET)
  # cmake_policy(SET CMP0135 NEW)
  include(FetchContent)

  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/websocket/websocketpp )
    FetchContent_Declare(websocketpp
    GIT_REPOSITORY https://github.com/zaphoyd/websocketpp.git
      GIT_TAG 0.8.2
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/websocket
      )
    
    FetchContent_MakeAvailable(websocketpp)
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/websocket)
   
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/asio/asio )
    FetchContent_Declare(asio
      URL   https://github.com/chriskohlhoff/asio/archive/refs/tags/asio-1-24-0.tar.gz
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/asio
    )
    
    FetchContent_MakeAvailable(asio)
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/asio/asio/include)
 
  if(NOT EXISTS ${PROJECT_SOURCE_DIR}/third_party/json/ChangeLog.md )
    FetchContent_Declare(json
      URL   https://github.com/nlohmann/json/archive/refs/tags/v3.11.2.tar.gz
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/json
    )
    
    FetchContent_MakeAvailable(json)
  endif()
  include_directories(${PROJECT_SOURCE_DIR}/third_party/json/include)

endif()

if(ENABLE_PORTAUDIO)
  include(FetchContent)

  set(portaudio_URL  "http://files.portaudio.com/archives/pa_stable_v190700_20210406.tgz")
  set(portaudio_URL2 "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/dep_libs/pa_stable_v190700_20210406.tgz")
  set(portaudio_HASH "SHA256=47efbf42c77c19a05d22e627d42873e991ec0c1357219c0d74ce6a2948cb2def")

  FetchContent_Declare(portaudio
    URL
      ${portaudio_URL}
      ${portaudio_URL2}
    URL_HASH          ${portaudio_HASH}
  )

  FetchContent_GetProperties(portaudio)
  if(NOT portaudio_POPULATED)
    message(STATUS "Downloading portaudio from ${portaudio_URL}")
    FetchContent_Populate(portaudio)
  endif()
  message(STATUS "portaudio is downloaded to ${portaudio_SOURCE_DIR}")
  message(STATUS "portaudio's binary dir is ${portaudio_BINARY_DIR}")

  add_subdirectory(${portaudio_SOURCE_DIR} ${portaudio_BINARY_DIR} EXCLUDE_FROM_ALL)
  if(NOT WIN32)
    target_compile_options(portaudio PRIVATE "-Wno-deprecated-declarations")
  else()
    install(TARGETS portaudio DESTINATION ..)
  endif()

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
    if(WIN32)
    include_directories(${openfst_SOURCE_DIR}/src/lib)
    endif() 
endif()


include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/src)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/jieba/include)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/jieba/include/limonp/include)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi)

add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp yaml-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank/kaldi-native-fbank/csrc csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/src src)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi kaldi)

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
