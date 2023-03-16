

if(WIN32)
    add_definitions(-DWEBRTC_WIN)
else()
    add_definitions(-DWEBRTC_POSIX)
endif()


include_directories("..")

file(GLOB_RECURSE files "*.c" "rtc_base/checks.cc")

message("${files}")

add_library(webrtcvad ${files})
