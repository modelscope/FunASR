
file(GLOB files1 "*.cpp")
set(files ${files1})

message("files: "${files})

if(WIN32)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
endif()

add_library(funasr SHARED ${files})

if(WIN32)
    set(EXTRA_LIBS yaml-cpp csrc kaldi-decoder fst glog gflags avutil avcodec avformat swresample onnxruntime)
    include_directories(${ONNXRUNTIME_DIR}/include)
    include_directories(${FFMPEG_DIR}/include)
    target_link_directories(funasr PUBLIC ${ONNXRUNTIME_DIR}/lib)
    target_link_directories(funasr PUBLIC ${FFMPEG_DIR}/lib)
    target_compile_definitions(funasr PUBLIC -D_FUNASR_API_EXPORT -DNOMINMAX -DYAML_CPP_DLL)
else()
    set(EXTRA_LIBS pthread yaml-cpp csrc kaldi-decoder fst glog gflags avutil avcodec avformat swresample)
    include_directories(${ONNXRUNTIME_DIR}/include)
    include_directories(${FFMPEG_DIR}/include)
endif()

#message("CXX_FLAGS "${CMAKE_CXX_FLAGS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/third_party)
target_link_libraries(funasr PUBLIC onnxruntime ${EXTRA_LIBS})
