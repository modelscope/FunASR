cmake_minimum_required(VERSION 3.10)

#-DONNXRUNTIME_DIR=D:\thirdpart\onnxruntime
project(FastASR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

#option(FASTASR_BUILD_PYTHON_MODULE "build python module, using FastASR in Python" OFF)

add_subdirectory("./third_party/webrtc")
add_subdirectory(src)
add_subdirectory(tester)
