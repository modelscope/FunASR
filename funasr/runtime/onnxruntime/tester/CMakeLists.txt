

if(WIN32)
    if(CMAKE_CL_64)
        link_directories( ${CMAKE_SOURCE_DIR}/win/lib/x64 )
    else()
        link_directories( ${CMAKE_SOURCE_DIR}/win/lib/x86 )
    endif()
endif()

set(EXTRA_LIBS rapidasr)


include_directories(${CMAKE_SOURCE_DIR}/include)
set(EXECNAME "tester")

add_executable(${EXECNAME} "tester.cpp")
target_link_libraries(${EXECNAME} PUBLIC onnxruntime ${EXTRA_LIBS})


