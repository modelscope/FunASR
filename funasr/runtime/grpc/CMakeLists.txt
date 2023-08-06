# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
# Reserved. MIT License  (https://opensource.org/licenses/MIT)
#
# 2023 by burkliu(刘柏基) liubaiji@xverse.cn

cmake_minimum_required(VERSION 3.10)

project(ASR C CXX)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_VERBOSE_MAKEFILE on)

include(common.cmake)

# Proto file
get_filename_component(rg_proto ../python/grpc/proto/paraformer.proto ABSOLUTE)
get_filename_component(rg_proto_path ${rg_proto} PATH)

# Generated sources
set(rg_proto_srcs ${CMAKE_CURRENT_BINARY_DIR}/paraformer.pb.cc)
set(rg_proto_hdrs ${CMAKE_CURRENT_BINARY_DIR}/paraformer.pb.h)
set(rg_grpc_srcs ${CMAKE_CURRENT_BINARY_DIR}/paraformer.grpc.pb.cc)
set(rg_grpc_hdrs ${CMAKE_CURRENT_BINARY_DIR}/paraformer.grpc.pb.h)
add_custom_command(
  OUTPUT ${rg_proto_srcs} ${rg_proto_hdrs} ${rg_grpc_srcs} ${rg_grpc_hdrs}
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS --grpc_out ${CMAKE_CURRENT_BINARY_DIR}
    --cpp_out ${CMAKE_CURRENT_BINARY_DIR}
    -I ${rg_proto_path}
    --plugin=protoc-gen-grpc=${_GRPC_CPP_PLUGIN_EXECUTABLE}
    ${rg_proto}
  DEPENDS ${rg_proto})

# Include generated *.pb.h files
include_directories(${CMAKE_CURRENT_BINARY_DIR})

link_directories(${ONNXRUNTIME_DIR}/lib)
link_directories(${FFMPEG_DIR}/lib)

include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp/include/)
include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank)

add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/yaml-cpp yaml-cpp)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/kaldi-native-fbank/kaldi-native-fbank/csrc csrc)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/src src)

include_directories(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog)
set(BUILD_TESTING OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/../onnxruntime/third_party/glog glog)

# rg_grpc_proto
add_library(rg_grpc_proto
  ${rg_grpc_srcs}
  ${rg_grpc_hdrs}
  ${rg_proto_srcs}
  ${rg_proto_hdrs})

target_link_libraries(rg_grpc_proto
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})

add_executable(paraformer-server
  paraformer-server.cc)
target_link_libraries(paraformer-server
  rg_grpc_proto
  funasr
  ${EXTRA_LIBS}
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF})
