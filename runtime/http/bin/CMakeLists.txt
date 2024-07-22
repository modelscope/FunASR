

if(WIN32)
  include_directories(${ONNXRUNTIME_DIR}/include)
  include_directories(${FFMPEG_DIR}/include)
  include_directories(${OPENSSL_ROOT_DIR}//include)
  link_directories(${OPENSSL_ROOT_DIR}/lib)
  add_definitions(-D_WEBSOCKETPP_CPP11_RANDOM_DEVICE_)
  add_definitions(-D_WEBSOCKETPP_CPP11_TYPE_TRAITS_)
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/bigobj>")
  add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
  SET(RELATION_SOURCE "../../onnxruntime/src/resample.cpp" "../../onnxruntime/src/util.cpp" "../../onnxruntime/src/alignedmem.cpp" "../../onnxruntime/src/encode_converter.cpp")
endif()


find_package(ZLIB REQUIRED)

file(GLOB SRC_FILES "*.cpp")
add_executable(funasr-http-server ${SRC_FILES} ${RELATION_SOURCE})
 

 
target_link_libraries(funasr-http-server PUBLIC funasr ${OPENSSL_CRYPTO_LIBRARY} ${OPENSSL_SSL_LIBRARY})
