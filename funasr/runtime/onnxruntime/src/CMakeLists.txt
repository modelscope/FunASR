
file(GLOB files1 "*.cpp")
file(GLOB files4 "paraformer/*.cpp")

set(files ${files1} ${files2} ${files3} ${files4})

# message("${files}")

add_library(rapidasr ${files})

if(WIN32)

        set(EXTRA_LIBS libfftw3f-3 webrtcvad)
        if(CMAKE_CL_64)
            target_link_directories(rapidasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x64)
        else()
            target_link_directories(rapidasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x86)
        endif()
        target_include_directories(rapidasr PUBLIC ${CMAKE_SOURCE_DIR}/win/include )
        
        target_compile_definitions(rapidasr PUBLIC -D_RPASR_API_EXPORT)
else()

    set(EXTRA_LIBS fftw3f webrtcvad pthread)
    target_include_directories(rapidasr PUBLIC "/usr/local/opt/fftw/include")
    target_link_directories(rapidasr PUBLIC "/usr/local/opt/fftw/lib")

    target_include_directories(rapidasr PUBLIC "/usr/local/opt/openblas/include")
    target_link_directories(rapidasr PUBLIC "/usr/local/opt/openblas/lib")

    target_include_directories(rapidasr PUBLIC "/usr/include")
    target_link_directories(rapidasr PUBLIC "/usr/lib64")

    target_include_directories(rapidasr PUBLIC  ${FFTW3F_INCLUDE_DIR})
    target_link_directories(rapidasr PUBLIC ${FFTW3F_LIBRARY_DIR})
    include_directories(${ONNXRUNTIME_DIR}/include)    
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(rapidasr PUBLIC onnxruntime ${EXTRA_LIBS})



