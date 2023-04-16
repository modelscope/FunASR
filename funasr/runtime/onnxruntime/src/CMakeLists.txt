
file(GLOB files1 "*.cpp")
file(GLOB files2 "*.cc")
file(GLOB files4 "paraformer/*.cpp")

set(files ${files1} ${files2} ${files3} ${files4})

# message("${files}")

add_library(funasr ${files})

if(WIN32)

        set(EXTRA_LIBS libfftw3f-3 yaml-cpp)
        if(CMAKE_CL_64)
            target_link_directories(funasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x64)
        else()
            target_link_directories(funasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x86)
        endif()
        target_include_directories(funasr PUBLIC ${CMAKE_SOURCE_DIR}/win/include )
        
        target_compile_definitions(funasr PUBLIC -D_FUNASR_API_EXPORT)
else()

    set(EXTRA_LIBS fftw3f pthread yaml-cpp)
    target_include_directories(funasr PUBLIC "/usr/local/opt/fftw/include")
    target_link_directories(funasr PUBLIC "/usr/local/opt/fftw/lib")

    target_include_directories(funasr PUBLIC "/usr/local/opt/openblas/include")
    target_link_directories(funasr PUBLIC "/usr/local/opt/openblas/lib")

    target_include_directories(funasr PUBLIC "/usr/include")
    target_link_directories(funasr PUBLIC "/usr/lib64")

    target_include_directories(funasr PUBLIC  ${FFTW3F_INCLUDE_DIR})
    target_link_directories(funasr PUBLIC ${FFTW3F_LIBRARY_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)    
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(funasr PUBLIC onnxruntime ${EXTRA_LIBS})



