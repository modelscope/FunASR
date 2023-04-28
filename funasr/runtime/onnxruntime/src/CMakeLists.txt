
file(GLOB files1 "*.cpp")
file(GLOB files2 "*.cc")

set(files ${files1} ${files2})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_library(funasr ${files})

if(WIN32)
    set(EXTRA_LIBS pthread yaml-cpp csrc glog)
    if(CMAKE_CL_64)
        target_link_directories(funasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x64)
    else()
        target_link_directories(funasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x86)
    endif()
    target_include_directories(funasr PUBLIC ${CMAKE_SOURCE_DIR}/win/include )
    
    target_compile_definitions(funasr PUBLIC -D_FUNASR_API_EXPORT)
else()
    set(EXTRA_LIBS pthread yaml-cpp csrc glog )
    include_directories(${ONNXRUNTIME_DIR}/include)
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(funasr PUBLIC onnxruntime ${EXTRA_LIBS})

add_executable(funasr-onnx-offline "funasr-onnx-offline.cpp")
add_executable(funasr-onnx-offline-rtf "funasr-onnx-offline-rtf.cpp")
target_link_libraries(funasr-onnx-offline PUBLIC funasr)
target_link_libraries(funasr-onnx-offline-rtf PUBLIC funasr)