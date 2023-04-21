cmake_minimum_required(VERSION 3.19)
project(qmpunc)

add_subdirectory(yaml-cpp)



include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include yaml-cpp/include ${CMAKE_INSTALL_PREFIX}/include) 

link_directories(yaml-cpp ${CMAKE_INSTALL_PREFIX}/lib ${CMAKE_INSTALL_PREFIX}/lib64)

set(APP_NAME rapidpunc_tester)
set(LIB_NAME rapidpunc)


if(WIN32)

add_definitions(-D_WIN32)

else()


endif()

set(MAIN_SRC "sources/tokenizer.cpp" "sources/punc_infer.cpp"  "sources/constdef.cpp" "sources/libpuncapi.cpp" )

add_library(${LIB_NAME} SHARED  ${MAIN_SRC})

target_compile_definitions(${LIB_NAME} PUBLIC -D_QMPUC_API_EXPORT)
target_link_libraries( ${LIB_NAME} PUBLIC  yaml-cpp onnxruntime )

add_executable(${APP_NAME} sources/tester.cpp )


target_link_libraries( ${APP_NAME} PRIVATE  ${LIB_NAME})