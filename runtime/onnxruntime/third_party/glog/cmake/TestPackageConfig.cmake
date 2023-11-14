if (GENERATOR_TOOLSET)
  list (APPEND _ADDITIONAL_ARGS -T ${GENERATOR_TOOLSET})
endif (GENERATOR_TOOLSET)

if (GENERATOR_PLATFORM)
  list (APPEND _ADDITIONAL_ARGS -A ${GENERATOR_PLATFORM})
endif (GENERATOR_PLATFORM)

# Run CMake
execute_process (
  # Capture the PATH environment variable content set during project generation
  # stage. This is required because later during the build stage the PATH is
  # modified again (e.g., for MinGW AppVeyor CI builds) by adding back the
  # directory containing git.exe. Incidentally, the Git installation directory
  # also contains sh.exe which causes MinGW Makefile generation to fail.
  COMMAND ${CMAKE_COMMAND} -E env PATH=${PATH}
  ${CMAKE_COMMAND} -C ${INITIAL_CACHE}
    -G ${GENERATOR}
    ${_ADDITIONAL_ARGS}
    -DCMAKE_FIND_PACKAGE_NO_PACKAGE_REGISTRY=ON
    -DCMAKE_FIND_PACKAGE_NO_SYSTEM_PACKAGE_REGISTRY=ON
    -Dglog_ROOT=${PACKAGE_DIR}
    -S ${SOURCE_DIR}
    -B ${TEST_BINARY_DIR}
  RESULT_VARIABLE _GENERATE_SUCCEEDED
)

if (NOT _GENERATE_SUCCEEDED EQUAL 0)
  message (FATAL_ERROR "Failed to generate project files using CMake")
endif (NOT _GENERATE_SUCCEEDED EQUAL 0)
