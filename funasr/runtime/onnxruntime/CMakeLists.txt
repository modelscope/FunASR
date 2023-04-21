cmake_minimum_required(VERSION 3.10)

project(FunASRonnx)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD 14 CACHE STRING "The C++ version to be used.")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(TestBigEndian)
test_big_endian(BIG_ENDIAN)
if(BIG_ENDIAN)
    message("Big endian system")
else()
    message("Little endian system")
endif()

# for onnxruntime
IF(WIN32)
	if(CMAKE_CL_64)
		link_directories(${ONNXRUNTIME_DIR}\\lib)
	else()
		add_definitions(-D_WIN_X86)
	endif()
ELSE()
    link_directories(${ONNXRUNTIME_DIR}/lib)
endif()

include_directories(${PROJECT_SOURCE_DIR}/kaldi-native-fbank)

add_subdirectory("./third_party/yaml-cpp")
add_subdirectory(kaldi-native-fbank/kaldi-native-fbank/csrc)
add_subdirectory(src)
