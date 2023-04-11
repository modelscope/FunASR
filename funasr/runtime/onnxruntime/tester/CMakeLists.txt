

if(WIN32)
    if(CMAKE_CL_64)
        link_directories( ${CMAKE_SOURCE_DIR}/win/lib/x64 )
    else()
        link_directories( ${CMAKE_SOURCE_DIR}/win/lib/x86 )
    endif()
endif()

set(EXTRA_LIBS funasr)


include_directories(${CMAKE_SOURCE_DIR}/include)
set(EXECNAME "tester")
set(EXECNAMERTF "tester_rtf")

add_executable(${EXECNAME} "tester.cpp")
target_link_libraries(${EXECNAME} PUBLIC ${EXTRA_LIBS})

add_executable(${EXECNAMERTF} "tester_rtf.cpp")
target_link_libraries(${EXECNAMERTF} PUBLIC ${EXTRA_LIBS})

