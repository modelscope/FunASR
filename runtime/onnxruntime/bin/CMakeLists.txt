include_directories(${CMAKE_SOURCE_DIR}/include)

if(WIN32)
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/execution-charset:utf-8>")
add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/source-charset:utf-8>")
include_directories(${ONNXRUNTIME_DIR}/include)
include_directories(${FFMPEG_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party)
SET(RELATION_SOURCE "../src/resample.cpp" "../src/util.cpp" "../src/alignedmem.cpp" "../src/encode_converter.cpp")
endif()

add_executable(funasr-onnx-offline "funasr-onnx-offline.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-offline PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-offline PUBLIC funasr)

add_executable(funasr-onnx-offline-vad "funasr-onnx-offline-vad.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-offline-vad PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-offline-vad PUBLIC funasr)

add_executable(funasr-onnx-online-vad "funasr-onnx-online-vad.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-online-vad PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-online-vad PUBLIC funasr)

add_executable(funasr-onnx-online-asr "funasr-onnx-online-asr.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-online-asr PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-online-asr PUBLIC funasr)

add_executable(funasr-onnx-offline-punc "funasr-onnx-offline-punc.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-offline-punc PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-offline-punc PUBLIC funasr)

add_executable(funasr-onnx-online-punc "funasr-onnx-online-punc.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-online-punc PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-online-punc PUBLIC funasr)

add_executable(funasr-onnx-offline-rtf "funasr-onnx-offline-rtf.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-offline-rtf PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-offline-rtf PUBLIC funasr)

add_executable(funasr-onnx-2pass "funasr-onnx-2pass.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-2pass PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-2pass PUBLIC funasr)

add_executable(funasr-onnx-2pass-rtf "funasr-onnx-2pass-rtf.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-2pass-rtf PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-2pass-rtf PUBLIC funasr)

add_executable(funasr-onnx-online-rtf "funasr-onnx-online-rtf.cpp" ${RELATION_SOURCE})
target_link_options(funasr-onnx-online-rtf PRIVATE "-Wl,--no-as-needed")
target_link_libraries(funasr-onnx-online-rtf PUBLIC funasr)

# include_directories(${FFMPEG_DIR}/include)
# add_executable(ff "ffmpeg.cpp")
# target_link_libraries(ff PUBLIC avutil avcodec avformat swresample)
