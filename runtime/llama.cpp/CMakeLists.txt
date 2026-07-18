# Standalone, CI-friendly build for the FunASR llama.cpp runtime.
#
# Fetches a pinned llama.cpp (which provides ggml + llama) and builds the runtime
# binaries against it — no manual copying into a llama.cpp checkout, no hardcoded
# paths. Linux / macOS / Windows, x64 / arm64. Static libs -> self-contained binaries.
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j               # binaries land in build/bin/
#
# Build against a local checkout instead of fetching:
#   -DFETCHCONTENT_SOURCE_DIR_LLAMA=/path/to/llama.cpp
cmake_minimum_required(VERSION 3.16)
project(funasr-llamacpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
    add_compile_definitions(NOMINMAX _USE_MATH_DEFINES)   # min/max macros + M_PI on MSVC
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)   # static deps -> self-contained binaries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

include(FetchContent)
if(POLICY CMP0169)
    cmake_policy(SET CMP0169 OLD)
endif()
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        8086439a4cea94c71a5dfb8fe4ad1546aebd640f)
FetchContent_GetProperties(llama)
if(NOT llama_POPULATED)
    FetchContent_Populate(llama)
    if(GGML_CUDA_FORCE_CUBLAS)
        set(_funasr_ggml_cuda_cmake "${llama_SOURCE_DIR}/ggml/src/ggml-cuda/CMakeLists.txt")
        file(READ "${_funasr_ggml_cuda_cmake}" _funasr_ggml_cuda_text)
        string(REPLACE
"    file(GLOB   SRCS \"template-instances/fattn-mma*.cu\")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
    file(GLOB   SRCS \"template-instances/mmq*.cu\")
    list(APPEND GGML_SOURCES_CUDA ${SRCS})
"
"    if (NOT GGML_CUDA_FORCE_CUBLAS)
        file(GLOB   SRCS \"template-instances/fattn-mma*.cu\")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
        file(GLOB   SRCS \"template-instances/mmq*.cu\")
        list(APPEND GGML_SOURCES_CUDA ${SRCS})
    endif()
"
            _funasr_ggml_cuda_text "${_funasr_ggml_cuda_text}")
        file(WRITE "${_funasr_ggml_cuda_cmake}" "${_funasr_ggml_cuda_text}")
        message(WARNING "FunASR CUDA release: skipping ggml CUDA MMQ/fattn-mma template sources for GGML_CUDA_FORCE_CUBLAS")
    endif()
    add_subdirectory(${llama_SOURCE_DIR} ${llama_BINARY_DIR})
endif()

find_package(Threads REQUIRED)
set(FUNASR_COMMON ${CMAKE_CURRENT_SOURCE_DIR}/funasr-common)

function(funasr_add name src)
    add_executable(${name} ${src})
    target_include_directories(${name} PRIVATE ${FUNASR_COMMON})
    target_link_libraries(${name} PRIVATE ${ARGN} Threads::Threads)
    install(TARGETS ${name} RUNTIME DESTINATION bin)
endfunction()

funasr_add(llama-funasr-cli fun-asr-nano/funasr-cli/funasr-cli.cpp llama ggml)
funasr_add(llama-funasr-encoder fun-asr-nano/funasr-encoder/funasr-encoder.cpp ggml)
funasr_add(llama-funasr-embd fun-asr-nano/funasr-embd/funasr-embd.cpp llama ggml)
funasr_add(llama-funasr-sensevoice sensevoice/funasr-sensevoice/funasr-sensevoice.cpp ggml)
funasr_add(llama-funasr-paraformer paraformer/funasr-paraformer/funasr-paraformer.cpp ggml)
funasr_add(llama-funasr-vad funasr-vad/funasr-vad.cpp ggml)
