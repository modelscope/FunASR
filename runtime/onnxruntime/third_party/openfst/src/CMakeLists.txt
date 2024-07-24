
#-DHAVE_CONFIG_H -I./../include -fno-exceptions -funsigned-char -std=c++11 -MT symbol-table.lo -MD -MP -MF .deps/symbol-table.Tpo -c symbol-table.cc  -fno-common -DPIC -o .libs/symbol-table.o

include_directories(./include/)
install(DIRECTORY include/ DESTINATION include/
        FILES_MATCHING PATTERN "*.h")

add_subdirectory(lib)

if(HAVE_SCRIPT)
  add_subdirectory(script)
endif(HAVE_SCRIPT)

if(HAVE_BIN)
  add_subdirectory(bin)
endif(HAVE_BIN)

add_subdirectory(extensions)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(test)
endif(BUILD_TESTING)
