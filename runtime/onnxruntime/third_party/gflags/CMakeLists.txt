## CMake configuration file of gflags project
##
## This CMakeLists.txt defines some gflags specific configuration variables
## using the "gflags_define" utility macro. The default values of these variables
## can be overridden either on the CMake command-line using the -D option of
## the cmake command or in a super-project which includes the gflags source
## tree by setting the GFLAGS_<varname> CMake variables before adding the
## gflags source directory via CMake's "add_subdirectory" command. Only when
## the non-cached variable GFLAGS_IS_SUBPROJECT has a value equivalent to FALSE,
## these configuration variables are added to the CMake cache so they can be
## edited in the CMake GUI. By default, GFLAGS_IS_SUBPROJECT is set to TRUE when
## the CMAKE_SOURCE_DIR is not identical to the directory of this CMakeLists.txt
## file, i.e., the top-level directory of the gflags project source tree.
##
## When this project is a subproject (GFLAGS_IS_SUBPROJECT is TRUE), the default
## settings are such that only the static single-threaded library is built without
## installation of the gflags files. The "gflags::gflags" target is in this case an ALIAS
## library target for the "gflags_nothreads_static" library target. Targets which
## depend on the gflags library should link to the "gflags::gflags" library target.
##
## Example CMakeLists.txt of user project which requires separate gflags installation:
##   cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
##
##   project(Foo)
##
##   find_package(gflags REQUIRED)
##
##   add_executable(foo src/foo.cc)
##   target_link_libraries(foo gflags::gflags)
##
## Example CMakeLists.txt of user project which requires separate single-threaded static gflags installation:
##   cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
##
##   project(Foo)
##
##   find_package(gflags COMPONENTS nothreads_static)
##
##   add_executable(foo src/foo.cc)
##   target_link_libraries(foo gflags::gflags)
##
## Example CMakeLists.txt of super-project which contains gflags source tree:
##   cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
##
##   project(Foo)
##
##   add_subdirectory(gflags)
##
##   add_executable(foo src/foo.cc)
##   target_link_libraries(foo gflags::gflags)
##
## Variables to configure the source files:
## - GFLAGS_IS_A_DLL
## - GFLAGS_NAMESPACE
## - GFLAGS_ATTRIBUTE_UNUSED
## - GFLAGS_INTTYPES_FORMAT
##
## Variables to configure the build:
## - GFLAGS_SOVERSION
## - GFLAGS_BUILD_SHARED_LIBS
## - GFLAGS_BUILD_STATIC_LIBS
## - GFLAGS_BUILD_gflags_LIB
## - GFLAGS_BUILD_gflags_nothreads_LIB
## - GFLAGS_BUILD_TESTING
## - GFLAGS_BUILD_PACKAGING
##
## Variables to configure the installation:
## - GFLAGS_INCLUDE_DIR
## - GFLAGS_LIBRARY_INSTALL_DIR or LIB_INSTALL_DIR or LIB_SUFFIX
## - GFLAGS_INSTALL_HEADERS
## - GFLAGS_INSTALL_SHARED_LIBS
## - GFLAGS_INSTALL_STATIC_LIBS

cmake_minimum_required (VERSION 3.0.2 FATAL_ERROR)

if (POLICY CMP0042)
  cmake_policy (SET CMP0042 NEW)
endif ()

if (POLICY CMP0048)
  cmake_policy (SET CMP0048 NEW)
endif ()

# ----------------------------------------------------------------------------
# includes
include ("${CMAKE_CURRENT_SOURCE_DIR}/cmake/utils.cmake")

# ----------------------------------------------------------------------------
# package information
set (PACKAGE_NAME        "gflags")
set (PACKAGE_VERSION     "2.2.2")
set (PACKAGE_STRING      "${PACKAGE_NAME} ${PACKAGE_VERSION}")
set (PACKAGE_TARNAME     "${PACKAGE_NAME}-${PACKAGE_VERSION}")
set (PACKAGE_BUGREPORT   "https://github.com/gflags/gflags/issues")
set (PACKAGE_DESCRIPTION "A commandline flags library that allows for distributed flags.")
set (PACKAGE_URL         "http://gflags.github.io/gflags")

project (${PACKAGE_NAME} VERSION ${PACKAGE_VERSION} LANGUAGES CXX)
if (CMAKE_VERSION VERSION_LESS 3.4)
  # C language still needed because the following required CMake modules
  # (or their dependencies, respectively) are not correctly handling
  # the case where only CXX is enabled
  # - CheckTypeSize.cmake (fixed in CMake 3.1, cf. https://cmake.org/Bug/view.php?id=14056)
  # - FindThreads.cmake   (fixed in CMake 3.4, cf. https://cmake.org/Bug/view.php?id=14905)
  enable_language (C)
endif ()

version_numbers (
  ${PACKAGE_VERSION}
    PACKAGE_VERSION_MAJOR
    PACKAGE_VERSION_MINOR
    PACKAGE_VERSION_PATCH
)

# shared library ABI version number, can be overridden by package maintainers
# using -DGFLAGS_SOVERSION=XXX on the command-line
if (GFLAGS_SOVERSION)
  set (PACKAGE_SOVERSION "${GFLAGS_SOVERSION}")
else ()
  # TODO: Change default SOVERSION back to PACKAGE_VERSION_MAJOR with the
  #       next increase of major version number (i.e., 3.0.0 -> SOVERSION 3)
  #       The <major>.<minor> SOVERSION should be used for the 2.x releases
  #       versions only which temporarily broke the API by changing the default
  #       namespace from "google" to "gflags".
  set (PACKAGE_SOVERSION "${PACKAGE_VERSION_MAJOR}.${PACKAGE_VERSION_MINOR}")
endif ()

# when gflags is included as subproject (e.g., as Git submodule/subtree) in the source
# tree of a project that uses it, no variables should be added to the CMake cache;
# users may set the non-cached variable GFLAGS_IS_SUBPROJECT before add_subdirectory(gflags)
if (NOT DEFINED GFLAGS_IS_SUBPROJECT)
  if ("^${CMAKE_SOURCE_DIR}$" STREQUAL "^${PROJECT_SOURCE_DIR}$")
    set (GFLAGS_IS_SUBPROJECT FALSE)
  else ()
    set (GFLAGS_IS_SUBPROJECT TRUE)
  endif ()
endif ()

# prefix for package variables in CMake configuration file
string (TOUPPER "${PACKAGE_NAME}" PACKAGE_PREFIX)

# convert file path on Windows with back slashes to path with forward slashes
# otherwise this causes an issue with the cmake_install.cmake script
file (TO_CMAKE_PATH "${CMAKE_INSTALL_PREFIX}" CMAKE_INSTALL_PREFIX)

# ----------------------------------------------------------------------------
# options

# maintain binary backwards compatibility with gflags library version <= 2.0,
# but at the same time enable the use of the preferred new "gflags" namespace
gflags_define (STRING NAMESPACE "Name(s) of library namespace (separate multiple options by semicolon)" "google;${PACKAGE_NAME}" "${PACKAGE_NAME}")
gflags_property (NAMESPACE ADVANCED TRUE)
set (GFLAGS_NAMESPACE_SECONDARY "${NAMESPACE}")
list (REMOVE_DUPLICATES GFLAGS_NAMESPACE_SECONDARY)
if (NOT GFLAGS_NAMESPACE_SECONDARY)
  message (FATAL_ERROR "GFLAGS_NAMESPACE must be set to one (or more) valid C++ namespace identifier(s separated by semicolon \";\").")
endif ()
foreach (ns IN LISTS GFLAGS_NAMESPACE_SECONDARY)
  if (NOT ns MATCHES "^[a-zA-Z][a-zA-Z0-9_]*$")
    message (FATAL_ERROR "GFLAGS_NAMESPACE contains invalid namespace identifier: ${ns}")
  endif ()
endforeach ()
list (GET       GFLAGS_NAMESPACE_SECONDARY 0 GFLAGS_NAMESPACE)
list (REMOVE_AT GFLAGS_NAMESPACE_SECONDARY 0)

# cached build options when gflags is not a subproject, otherwise non-cached CMake variables
# usage: gflags_define(BOOL <name> <doc> <default> [<subproject default>])
gflags_define (BOOL BUILD_SHARED_LIBS          "Request build of shared libraries."                                       OFF OFF)
gflags_define (BOOL BUILD_STATIC_LIBS          "Request build of static libraries (default if BUILD_SHARED_LIBS is OFF)." OFF ON)
gflags_define (BOOL BUILD_gflags_LIB           "Request build of the multi-threaded gflags library."                      ON  OFF)
gflags_define (BOOL BUILD_gflags_nothreads_LIB "Request build of the single-threaded gflags library."                     ON  ON)
gflags_define (BOOL BUILD_PACKAGING            "Enable build of distribution packages using CPack."                       OFF OFF)
gflags_define (BOOL BUILD_TESTING              "Enable build of the unit tests and their execution using CTest."          OFF OFF)
gflags_define (BOOL INSTALL_HEADERS            "Request installation of headers and other development files."             ON  OFF)
gflags_define (BOOL INSTALL_SHARED_LIBS        "Request installation of shared libraries."                                ON  ON)
gflags_define (BOOL INSTALL_STATIC_LIBS        "Request installation of static libraries."                                ON  OFF)
gflags_define (BOOL REGISTER_BUILD_DIR         "Request entry of build directory in CMake's package registry."            OFF OFF)
gflags_define (BOOL REGISTER_INSTALL_PREFIX    "Request entry of installed package in CMake's package registry."          ON  OFF)

gflags_property (BUILD_STATIC_LIBS   ADVANCED TRUE)
gflags_property (INSTALL_HEADERS     ADVANCED TRUE)
gflags_property (INSTALL_SHARED_LIBS ADVANCED TRUE)
gflags_property (INSTALL_STATIC_LIBS ADVANCED TRUE)

if (NOT GFLAGS_IS_SUBPROJECT)
  foreach (varname IN ITEMS CMAKE_INSTALL_PREFIX)
    gflags_property (${varname} ADVANCED FALSE)
  endforeach ()
  foreach (varname IN ITEMS CMAKE_CONFIGURATION_TYPES CMAKE_OSX_ARCHITECTURES CMAKE_OSX_DEPLOYMENT_TARGET CMAKE_OSX_SYSROOT)
    gflags_property (${varname} ADVANCED TRUE)
  endforeach ()
  if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CXX_FLAGS)
    gflags_set (CMAKE_BUILD_TYPE Release)
  endif ()
  if (CMAKE_CONFIGURATION_TYPES)
    gflags_property (CMAKE_BUILD_TYPE STRINGS "${CMAKE_CONFIGURATION_TYPES}")
  endif ()
endif () # NOT GFLAGS_IS_SUBPROJECT

if (NOT BUILD_SHARED_LIBS AND NOT BUILD_STATIC_LIBS)
  set (BUILD_STATIC_LIBS ON)
endif ()
if (NOT BUILD_gflags_LIB AND NOT BUILD_gflags_nothreads_LIB)
  message (FATAL_ERROR "At least one of [GFLAGS_]BUILD_gflags_LIB and [GFLAGS_]BUILD_gflags_nothreads_LIB must be ON.")
endif ()

gflags_define (STRING INCLUDE_DIR "Name of include directory of installed header files relative to CMAKE_INSTALL_PREFIX/include/" "${PACKAGE_NAME}")
gflags_property (INCLUDE_DIR ADVANCED TRUE)
file (TO_CMAKE_PATH "${INCLUDE_DIR}" INCLUDE_DIR)
if (IS_ABSOLUTE INCLUDE_DIR)
  message (FATAL_ERROR "[GFLAGS_]INCLUDE_DIR must be a path relative to CMAKE_INSTALL_PREFIX/include/")
endif ()
if (INCLUDE_DIR MATCHES "^\\.\\.[/\\]")
  message (FATAL_ERROR "[GFLAGS_]INCLUDE_DIR must not start with parent directory reference (../)")
endif ()
set (GFLAGS_INCLUDE_DIR "${INCLUDE_DIR}")

# ----------------------------------------------------------------------------
# system checks
include (CheckTypeSize)
include (CheckIncludeFileCXX)
include (CheckCXXSymbolExists)

if (WIN32 AND NOT CYGWIN)
  set (OS_WINDOWS 1)
else ()
  set (OS_WINDOWS 0)
endif ()

if (MSVC)
  set (HAVE_SYS_TYPES_H 1)
  set (HAVE_STDDEF_H    1) # used by CheckTypeSize module
  set (HAVE_UNISTD_H    0)
  set (HAVE_SYS_STAT_H  1)
  set (HAVE_SHLWAPI_H   1)
  if (MSVC_VERSION VERSION_LESS 1600)
    check_include_file_cxx ("stdint.h" HAVE_STDINT_H)
    bool_to_int (HAVE_STDINT_H)  # used in #if directive
  else ()
    set (HAVE_STDINT_H 1)
  endif ()
  if (MSVC_VERSION VERSION_LESS 1800)
    check_include_file_cxx ("inttypes.h" HAVE_INTTYPES_H)
    bool_to_int (HAVE_INTTYPES_H)  # used in #if directive
  else ()
    set (HAVE_INTTYPES_H 1)
  endif ()
else ()
  foreach (fname IN ITEMS unistd stdint inttypes sys/types sys/stat fnmatch)
    string (TOUPPER "${fname}" FNAME)
    string (REPLACE "/" "_" FNAME "${FNAME}")
    if (NOT HAVE_${FNAME}_H)
      check_include_file_cxx ("${fname}.h" HAVE_${FNAME}_H)
    endif ()
  endforeach ()
  if (NOT HAVE_FNMATCH_H AND OS_WINDOWS)
    check_include_file_cxx ("shlwapi.h" HAVE_SHLWAPI_H)
  endif ()
  # the following are used in #if directives not #ifdef
  bool_to_int (HAVE_STDINT_H)
  bool_to_int (HAVE_SYS_TYPES_H)
  bool_to_int (HAVE_INTTYPES_H)
endif ()

gflags_define (STRING INTTYPES_FORMAT "Format of integer types: \"C99\" (uint32_t), \"BSD\" (u_int32_t), \"VC7\" (__int32)" "")
gflags_property (INTTYPES_FORMAT STRINGS "C99;BSD;VC7")
gflags_property (INTTYPES_FORMAT ADVANCED TRUE)
if (NOT INTTYPES_FORMAT)
  set (TYPES uint32_t u_int32_t)
  if (MSVC)
    list (INSERT TYPES 0 __int32)
  endif ()
  foreach (type IN LISTS TYPES)
    check_type_size (${type} ${type} LANGUAGE CXX)
    if (HAVE_${type})
      break ()
    endif ()
  endforeach ()
  if (HAVE_uint32_t)
    gflags_set (INTTYPES_FORMAT C99)
  elseif (HAVE_u_int32_t)
    gflags_set (INTTYPES_FORMAT BSD)
  elseif (HAVE___int32)
    gflags_set (INTTYPES_FORMAT VC7)
  else ()
    gflags_property (INTTYPES_FORMAT ADVANCED FALSE)
    message (FATAL_ERROR "Do not know how to define a 32-bit integer quantity on your system!"
                         " Neither uint32_t, u_int32_t, nor __int32 seem to be available."
                         " Set [GFLAGS_]INTTYPES_FORMAT to either C99, BSD, or VC7 and try again.")
  endif ()
endif ()
# use of special characters in strings to circumvent bug #0008226
if ("^${INTTYPES_FORMAT}$" STREQUAL "^WIN$")
  gflags_set (INTTYPES_FORMAT VC7)
endif ()
if (NOT INTTYPES_FORMAT MATCHES "^(C99|BSD|VC7)$")
  message (FATAL_ERROR "Invalid value for [GFLAGS_]INTTYPES_FORMAT! Choose one of \"C99\", \"BSD\", or \"VC7\"")
endif ()
set (GFLAGS_INTTYPES_FORMAT "${INTTYPES_FORMAT}")
set (GFLAGS_INTTYPES_FORMAT_C99 0)
set (GFLAGS_INTTYPES_FORMAT_BSD 0)
set (GFLAGS_INTTYPES_FORMAT_VC7 0)
set ("GFLAGS_INTTYPES_FORMAT_${INTTYPES_FORMAT}" 1)

if (MSVC)
  set (HAVE_strtoll 0)
  set (HAVE_strtoq  0)
else ()
  check_cxx_symbol_exists (strtoll stdlib.h HAVE_STRTOLL)
  if (NOT HAVE_STRTOLL)
    check_cxx_symbol_exists (strtoq stdlib.h HAVE_STRTOQ)
  endif ()
endif ()

if (BUILD_gflags_LIB)
  set (CMAKE_THREAD_PREFER_PTHREAD TRUE)
  find_package (Threads)
  if (Threads_FOUND AND CMAKE_USE_PTHREADS_INIT)
    set (HAVE_PTHREAD 1)
    check_type_size (pthread_rwlock_t RWLOCK LANGUAGE CXX)
  else ()
    set (HAVE_PTHREAD 0)
  endif ()
  if (UNIX AND NOT HAVE_PTHREAD)
    if (CMAKE_HAVE_PTHREAD_H)
      set (what "library")
    else ()
      set (what ".h file")
    endif ()
    message (FATAL_ERROR "Could not find pthread${what}. Check the log file"
                         "\n\t${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeError.log"
                         "\nor disable the build of the multi-threaded gflags library (BUILD_gflags_LIB=OFF).")
  endif ()
else ()
  set (HAVE_PTHREAD 0)
endif ()

# ----------------------------------------------------------------------------
# source files - excluding root subdirectory and/or .in suffix
set (PUBLIC_HDRS
  "gflags.h"
  "gflags_declare.h"
  "gflags_completions.h"
)

if (GFLAGS_NAMESPACE_SECONDARY)
  set (INCLUDE_GFLAGS_NS_H "// Import gflags library symbols into alternative/deprecated namespace(s)")
  foreach (ns IN LISTS GFLAGS_NAMESPACE_SECONDARY)
    string (TOUPPER "${ns}" NS)
    set (gflags_ns_h "${PROJECT_BINARY_DIR}/include/${GFLAGS_INCLUDE_DIR}/gflags_${ns}.h")
    configure_file ("${PROJECT_SOURCE_DIR}/src/gflags_ns.h.in" "${gflags_ns_h}" @ONLY)
    list (APPEND PUBLIC_HDRS "${gflags_ns_h}")
    set (INCLUDE_GFLAGS_NS_H "${INCLUDE_GFLAGS_NS_H}\n#include \"gflags_${ns}.h\"")
  endforeach ()
else ()
  set (INCLUDE_GFLAGS_NS_H)
endif ()

set (PRIVATE_HDRS
  "defines.h"
  "config.h"
  "util.h"
  "mutex.h"
)

set (GFLAGS_SRCS
  "gflags.cc"
  "gflags_reporting.cc"
  "gflags_completions.cc"
)

if (OS_WINDOWS)
  list (APPEND PRIVATE_HDRS "windows_port.h")
  list (APPEND GFLAGS_SRCS  "windows_port.cc")
endif ()

# ----------------------------------------------------------------------------
# configure source files
if (NOT DEFINED GFLAGS_ATTRIBUTE_UNUSED)
  if (CMAKE_COMPILER_IS_GNUCXX)
    set (GFLAGS_ATTRIBUTE_UNUSED "__attribute((unused))")
  else ()
    set (GFLAGS_ATTRIBUTE_UNUSED)
  endif ()
endif ()

# whenever we build a shared library (DLL on Windows), configure the public
# headers of the API for use of this shared library rather than the optionally
# also build statically linked library; users can override GFLAGS_DLL_DECL
# in particular, this done by setting the INTERFACE_COMPILE_DEFINITIONS of
# static libraries to include an empty definition for GFLAGS_DLL_DECL
if (NOT DEFINED GFLAGS_IS_A_DLL)
  if (BUILD_SHARED_LIBS)
    set (GFLAGS_IS_A_DLL 1)
  else ()
    set (GFLAGS_IS_A_DLL 0)
  endif ()
endif ()

configure_headers (PUBLIC_HDRS  ${PUBLIC_HDRS})
configure_sources (PRIVATE_HDRS ${PRIVATE_HDRS})
configure_sources (GFLAGS_SRCS  ${GFLAGS_SRCS})

# ----------------------------------------------------------------------------
# output directories
if (NOT GFLAGS_IS_SUBPROJECT)
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY "bin")
  set (CMAKE_LIBRARY_OUTPUT_DIRECTORY "lib")
  set (CMAKE_ARCHIVE_OUTPUT_DIRECTORY "lib")
endif ()
# Set postfixes for generated libraries based on buildtype.
set(CMAKE_RELEASE_POSTFIX "")
set(CMAKE_DEBUG_POSTFIX "_debug")

# ----------------------------------------------------------------------------
# installation directories
if (OS_WINDOWS)
  set (RUNTIME_INSTALL_DIR "bin")
  set (LIBRARY_INSTALL_DIR "lib")
  set (INCLUDE_INSTALL_DIR "include")
  set (CONFIG_INSTALL_DIR  "lib/cmake/${PACKAGE_NAME}")
  set (PKGCONFIG_INSTALL_DIR)
else ()
  set (RUNTIME_INSTALL_DIR bin)
  # The LIB_INSTALL_DIR and LIB_SUFFIX variables are used by the Fedora
  # package maintainers. Also package maintainers of other distribution
  # packages need to be able to specify the name of the library directory.
  if (NOT GFLAGS_LIBRARY_INSTALL_DIR AND LIB_INSTALL_DIR)
    set (GFLAGS_LIBRARY_INSTALL_DIR "${LIB_INSTALL_DIR}")
  endif ()
  gflags_define (PATH LIBRARY_INSTALL_DIR "Directory of installed libraries, e.g., \"lib64\"" "lib${LIB_SUFFIX}")
  gflags_property (LIBRARY_INSTALL_DIR ADVANCED TRUE)
  set (INCLUDE_INSTALL_DIR include)
  set (CONFIG_INSTALL_DIR  ${LIBRARY_INSTALL_DIR}/cmake/${PACKAGE_NAME})
  set (PKGCONFIG_INSTALL_DIR ${LIBRARY_INSTALL_DIR}/pkgconfig)
endif ()

# ----------------------------------------------------------------------------
# add library targets
set (TARGETS)
# static vs. shared
foreach (TYPE IN ITEMS STATIC SHARED)
  if (BUILD_${TYPE}_LIBS)
    string (TOLOWER "${TYPE}" type)
    # whether or not targets are a DLL
    if (OS_WINDOWS AND "^${TYPE}$" STREQUAL "^SHARED$")
      set (GFLAGS_IS_A_DLL 1)
    else ()
      set (GFLAGS_IS_A_DLL 0)
    endif ()
    # filename suffix for static libraries on Windows
    if (OS_WINDOWS AND "^${TYPE}$" STREQUAL "^STATIC$")
      set (type_suffix "_${type}")
    else ()
      set (type_suffix "")
    endif ()
    # multi-threaded vs. single-threaded
    foreach (opts IN ITEMS "" _nothreads)
      if (BUILD_gflags${opts}_LIB)
        set (target_name "gflags${opts}_${type}")
        add_library (${target_name} ${TYPE} ${GFLAGS_SRCS} ${PRIVATE_HDRS} ${PUBLIC_HDRS})
        set_target_properties (${target_name} PROPERTIES
          OUTPUT_NAME "gflags${opts}${type_suffix}"
          VERSION     "${PACKAGE_VERSION}"
          SOVERSION   "${PACKAGE_SOVERSION}"
        )
        set (include_dirs "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>")
        if (INSTALL_HEADERS)
          list (APPEND include_dirs "$<INSTALL_INTERFACE:${INCLUDE_INSTALL_DIR}>")
        endif ()
        target_include_directories (${target_name}
          PUBLIC  "${include_dirs}"
          PRIVATE "${PROJECT_SOURCE_DIR}/src;${PROJECT_BINARY_DIR}/include/${GFLAGS_INCLUDE_DIR}"
        )
        target_compile_definitions (${target_name} PUBLIC GFLAGS_IS_A_DLL=${GFLAGS_IS_A_DLL})
        if (opts MATCHES "nothreads")
          target_compile_definitions (${target_name} PRIVATE NO_THREADS)
        elseif (CMAKE_USE_PTHREADS_INIT)
          target_link_libraries (${target_name} ${CMAKE_THREAD_LIBS_INIT})
        endif ()
        if (HAVE_SHLWAPI_H)
          target_link_libraries (${target_name} shlwapi.lib)
        endif ()
        list (APPEND TARGETS ${target_name})
        # add convenience make target for build of both shared and static libraries
        if (NOT GFLAGS_IS_SUBPROJECT)
          if (NOT TARGET gflags${opts})
            add_custom_target (gflags${opts})
          endif ()
          add_dependencies (gflags${opts} ${target_name})
        endif ()
      endif ()
    endforeach ()
  endif ()
endforeach ()

# add ALIAS target for use in super-project, prefer static over shared, single-threaded over multi-threaded
if (GFLAGS_IS_SUBPROJECT)
  foreach (type IN ITEMS static shared)
    foreach (opts IN ITEMS "_nothreads" "")
      if (TARGET gflags${opts}_${type})
        # Define "gflags" alias for super-projects treating targets of this library as part of their own project
        # (also for backwards compatibility with gflags 2.2.1 which only defined this alias)
        add_library (gflags ALIAS gflags${opts}_${type})
        # Define "gflags::gflags" alias for projects that support both find_package(gflags) and add_subdirectory(gflags)
        add_library (gflags::gflags ALIAS gflags${opts}_${type})
        break ()
      endif ()
    endforeach ()
    if (TARGET gflags::gflags)
       break ()
    endif ()
  endforeach ()
endif ()

# ----------------------------------------------------------------------------
# installation rules
set (EXPORT_NAME ${PACKAGE_NAME}-targets)
file (RELATIVE_PATH INSTALL_PREFIX_REL2CONFIG_DIR "${CMAKE_INSTALL_PREFIX}/${CONFIG_INSTALL_DIR}" "${CMAKE_INSTALL_PREFIX}")
configure_file (cmake/config.cmake.in  "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-config-install.cmake" @ONLY)
configure_file (cmake/version.cmake.in "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-config-version.cmake" @ONLY)

if (BUILD_SHARED_LIBS AND INSTALL_SHARED_LIBS)
  foreach (opts IN ITEMS "" _nothreads)
    if (BUILD_gflags${opts}_LIB)
      install (TARGETS gflags${opts}_shared
               EXPORT ${EXPORT_NAME}
               RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
               LIBRARY DESTINATION ${LIBRARY_INSTALL_DIR}
               ARCHIVE DESTINATION ${LIBRARY_INSTALL_DIR}
      )
    endif ()
  endforeach ()
endif ()
if (BUILD_STATIC_LIBS AND INSTALL_STATIC_LIBS)
  foreach (opts IN ITEMS "" _nothreads)
    if (BUILD_gflags${opts}_LIB)
      install (TARGETS gflags${opts}_static
               EXPORT ${EXPORT_NAME}
               RUNTIME DESTINATION ${RUNTIME_INSTALL_DIR}
               LIBRARY DESTINATION ${LIBRARY_INSTALL_DIR}
               ARCHIVE DESTINATION ${LIBRARY_INSTALL_DIR}
      )
    endif ()
  endforeach ()
endif ()

if (INSTALL_HEADERS)
  install (FILES ${PUBLIC_HDRS} DESTINATION ${INCLUDE_INSTALL_DIR}/${GFLAGS_INCLUDE_DIR})
  install (
    FILES "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-config-install.cmake"
    RENAME ${PACKAGE_NAME}-config.cmake
    DESTINATION ${CONFIG_INSTALL_DIR}
  )
  install (
    FILES "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-config-version.cmake"
    DESTINATION ${CONFIG_INSTALL_DIR}
  )
  install (
    EXPORT ${EXPORT_NAME}
    NAMESPACE ${PACKAGE_NAME}::
    DESTINATION ${CONFIG_INSTALL_DIR}
  )
  install (
    EXPORT ${EXPORT_NAME}
    FILE ${PACKAGE_NAME}-nonamespace-targets.cmake
    DESTINATION ${CONFIG_INSTALL_DIR}
  )
  if (UNIX)
    install (PROGRAMS src/gflags_completions.sh DESTINATION ${RUNTIME_INSTALL_DIR})
  endif ()
endif ()

if (PKGCONFIG_INSTALL_DIR)
  configure_file ("cmake/package.pc.in" "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}.pc" @ONLY)
  install (FILES "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}.pc" DESTINATION "${PKGCONFIG_INSTALL_DIR}")
endif ()

# ----------------------------------------------------------------------------
# support direct use of build tree
set (INSTALL_PREFIX_REL2CONFIG_DIR .)
export (
  TARGETS ${TARGETS}
  NAMESPACE ${PACKAGE_NAME}::
  FILE "${PROJECT_BINARY_DIR}/${EXPORT_NAME}.cmake"
)
export (
  TARGETS ${TARGETS}
  FILE "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-nonamespace-targets.cmake"
)
if (REGISTER_BUILD_DIR)
  export (PACKAGE ${PACKAGE_NAME})
endif ()
if (REGISTER_INSTALL_PREFIX)
  register_gflags_package(${CONFIG_INSTALL_DIR})
endif ()
configure_file (cmake/config.cmake.in "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-config.cmake" @ONLY)

# ----------------------------------------------------------------------------
# testing - MUST follow the generation of the build tree config file
if (BUILD_TESTING)
  include (CTest)
  add_subdirectory (test)
endif ()

# ----------------------------------------------------------------------------
# packaging
if (BUILD_PACKAGING)

  if (NOT BUILD_SHARED_LIBS AND NOT INSTALL_HEADERS)
    message (WARNING "Package will contain static libraries without headers!"
                     "\nRecommended options for generation of runtime package:"
                     "\n  BUILD_SHARED_LIBS=ON"
                     "\n  BUILD_STATIC_LIBS=OFF"
                     "\n  INSTALL_HEADERS=OFF"
                     "\n  INSTALL_SHARED_LIBS=ON"
                     "\nRecommended options for generation of development package:"
                     "\n  BUILD_SHARED_LIBS=ON"
                     "\n  BUILD_STATIC_LIBS=ON"
                     "\n  INSTALL_HEADERS=ON"
                     "\n  INSTALL_SHARED_LIBS=ON"
                     "\n  INSTALL_STATIC_LIBS=ON"
    )
  endif ()

  # default package generators
  if (APPLE)
    set (PACKAGE_GENERATOR        "PackageMaker")
    set (PACKAGE_SOURCE_GENERATOR "TGZ;ZIP")
  elseif (UNIX)
    set (PACKAGE_GENERATOR        "DEB;RPM")
    set (PACKAGE_SOURCE_GENERATOR "TGZ;ZIP")
  else ()
    set (PACKAGE_GENERATOR        "ZIP")
    set (PACKAGE_SOURCE_GENERATOR "ZIP")
  endif ()

  # used package generators
  set (CPACK_GENERATOR        "${PACKAGE_GENERATOR}"        CACHE STRING "List of binary package generators (CPack).")
  set (CPACK_SOURCE_GENERATOR "${PACKAGE_SOURCE_GENERATOR}" CACHE STRING "List of source package generators (CPack).")
  mark_as_advanced (CPACK_GENERATOR CPACK_SOURCE_GENERATOR)

  # some package generators (e.g., PackageMaker) do not allow .md extension
  configure_file ("${CMAKE_CURRENT_LIST_DIR}/README.md" "${CMAKE_CURRENT_BINARY_DIR}/README.txt" COPYONLY)

  # common package information
  set (CPACK_PACKAGE_VENDOR              "Andreas Schuh")
  set (CPACK_PACKAGE_CONTACT             "google-gflags@googlegroups.com")
  set (CPACK_PACKAGE_NAME                "${PACKAGE_NAME}")
  set (CPACK_PACKAGE_VERSION             "${PACKAGE_VERSION}")
  set (CPACK_PACKAGE_VERSION_MAJOR       "${PACKAGE_VERSION_MAJOR}")
  set (CPACK_PACKAGE_VERSION_MINOR       "${PACKAGE_VERSION_MINOR}")
  set (CPACK_PACKAGE_VERSION_PATCH       "${PACKAGE_VERSION_PATCH}")
  set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "${PACKAGE_DESCRIPTION}")
  set (CPACK_RESOURCE_FILE_WELCOME       "${CMAKE_CURRENT_BINARY_DIR}/README.txt")
  set (CPACK_RESOURCE_FILE_LICENSE       "${CMAKE_CURRENT_LIST_DIR}/COPYING.txt")
  set (CPACK_PACKAGE_DESCRIPTION_FILE    "${CMAKE_CURRENT_BINARY_DIR}/README.txt")
  set (CPACK_INSTALL_PREFIX              "${CMAKE_INSTALL_PREFIX}")
  set (CPACK_OUTPUT_FILE_PREFIX          packages)
  set (CPACK_PACKAGE_RELOCATABLE         TRUE)
  set (CPACK_MONOLITHIC_INSTALL          TRUE)

  # RPM package information -- used in cmake/package.cmake.in also for DEB
  set (CPACK_RPM_PACKAGE_GROUP   "Development/Libraries")
  set (CPACK_RPM_PACKAGE_LICENSE "BSD")
  set (CPACK_RPM_PACKAGE_URL     "${PACKAGE_URL}")
  set (CPACK_RPM_CHANGELOG_FILE  "${CMAKE_CURRENT_LIST_DIR}/ChangeLog.txt")

  if (INSTALL_HEADERS)
    set (CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_LIST_DIR}/doc/index.html")
  else ()
    set (CPACK_RESOURCE_FILE_README "${CMAKE_CURRENT_LIST_DIR}/cmake/README_runtime.txt")
  endif ()

  # system/architecture
  if (WINDOWS)
    if (CMAKE_CL_64)
      set (CPACK_SYSTEM_NAME "win64")
    else ()
      set (CPACK_SYSTEM_NAME "win32")
    endif ()
    set (CPACK_PACKAGE_ARCHITECTURE)
  elseif (APPLE)
    set (CPACK_PACKAGE_ARCHITECTURE darwin)
  else ()
    string (TOLOWER "${CMAKE_SYSTEM_NAME}" CPACK_SYSTEM_NAME)
    if (CMAKE_CXX_FLAGS MATCHES "-m32")
      set (CPACK_PACKAGE_ARCHITECTURE i386)
    else ()
      execute_process (
        COMMAND         dpkg --print-architecture
        RESULT_VARIABLE RV
        OUTPUT_VARIABLE CPACK_PACKAGE_ARCHITECTURE
      )
      if (RV EQUAL 0)
	      string (STRIP "${CPACK_PACKAGE_ARCHITECTURE}" CPACK_PACKAGE_ARCHITECTURE)
      else ()
        execute_process (COMMAND uname -m OUTPUT_VARIABLE CPACK_PACKAGE_ARCHITECTURE)
        if (CPACK_PACKAGE_ARCHITECTURE MATCHES "x86_64")
	        set (CPACK_PACKAGE_ARCHITECTURE amd64)
        else ()
          set (CPACK_PACKAGE_ARCHITECTURE i386)
        endif ()
      endif ()
    endif ()
  endif ()

  # source package settings
  set (CPACK_SOURCE_TOPLEVEL_TAG      "source")
  set (CPACK_SOURCE_PACKAGE_FILE_NAME "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}")
  set (CPACK_SOURCE_IGNORE_FILES      "/\\\\.git/;\\\\.swp$;\\\\.#;/#;\\\\.*~;cscope\\\\.*;/[Bb]uild[.+-_a-zA-Z0-9]*/")

  # default binary package settings
  set (CPACK_INCLUDE_TOPLEVEL_DIRECTORY TRUE)
  set (CPACK_PACKAGE_FILE_NAME          "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CPACK_SYSTEM_NAME}")
  if (CPACK_PACKAGE_ARCHITECTURE)
    set (CPACK_PACKAGE_FILE_NAME "${CPACK_PACKAGE_FILE_NAME}-${CPACK_PACKAGE_ARCHITECTURE}")
  endif ()

  # generator specific configuration file
  #
  # allow package maintainers to use their own configuration file
  # $ cmake -DCPACK_PROJECT_CONFIG_FILE:FILE=/path/to/package/config
  if (NOT CPACK_PROJECT_CONFIG_FILE)
    configure_file (
      "${CMAKE_CURRENT_LIST_DIR}/cmake/package.cmake.in"
      "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-package.cmake" @ONLY
    )
    set (CPACK_PROJECT_CONFIG_FILE "${PROJECT_BINARY_DIR}/${PACKAGE_NAME}-package.cmake")
  endif ()

  include (CPack)

endif () # BUILD_PACKAGING

if (NOT GFLAGS_IS_SUBPROJECT AND NOT TARGET uninstall)
  configure_file (
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" @ONLY
  )
  add_custom_target(uninstall COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake")
endif ()
