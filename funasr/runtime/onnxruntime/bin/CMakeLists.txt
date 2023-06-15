include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(funasr-onnx-offline "funasr-onnx-offline.cpp")
target_link_libraries(funasr-onnx-offline PUBLIC funasr)

add_executable(funasr-onnx-offline-vad "funasr-onnx-offline-vad.cpp")
target_link_libraries(funasr-onnx-offline-vad PUBLIC funasr)

add_executable(funasr-onnx-online-vad "funasr-onnx-online-vad.cpp")
target_link_libraries(funasr-onnx-online-vad PUBLIC funasr)

add_executable(funasr-onnx-offline-punc "funasr-onnx-offline-punc.cpp")
target_link_libraries(funasr-onnx-offline-punc PUBLIC funasr)

add_executable(funasr-onnx-offline-rtf "funasr-onnx-offline-rtf.cpp")
target_link_libraries(funasr-onnx-offline-rtf PUBLIC funasr)
