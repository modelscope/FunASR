cmake_minimum_required (VERSION 3.16)
project (glog
  VERSION 0.7.0
  DESCRIPTION "C++ implementation of the Google logging module"
  HOMEPAGE_URL https://github.com/google/glog
  LANGUAGES CXX
)

set (CPACK_PACKAGE_NAME glog)
set (CPACK_PACKAGE_DESCRIPTION_SUMMARY "Google logging library")
set (CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set (CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set (CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set (CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include (CheckCXXCompilerFlag)
include (CheckCXXSourceCompiles)
include (CheckCXXSourceRuns)
include (CheckCXXSymbolExists)
include (CheckFunctionExists)
include (CheckIncludeFileCXX)
include (CheckLibraryExists)
include (CheckStructHasMember)
include (CheckTypeSize)
include (CMakeDependentOption)
include (CMakePackageConfigHelpers)
include (CMakePushCheckState)
include (CPack)
include (CTest)
include (DetermineGflagsNamespace)
include (GenerateExportHeader)
include (GetCacheVariables)
include (GNUInstallDirs)

option (BUILD_SHARED_LIBS "Build shared libraries" ON)
option (PRINT_UNSYMBOLIZED_STACK_TRACES
  "Print file offsets in traces instead of symbolizing" OFF)
option (WITH_GFLAGS "Use gflags" ON)
option (WITH_GTEST "Use Google Test" ON)
option (WITH_PKGCONFIG "Enable pkg-config support" ON)
option (WITH_SYMBOLIZE "Enable symbolize module" ON)
option (WITH_THREADS "Enable multithreading support" ON)
option (WITH_TLS "Enable Thread Local Storage (TLS) support" ON)
option (WITH_UNWIND "Enable libunwind support" ON)

cmake_dependent_option (WITH_GMOCK "Use Google Mock" ON WITH_GTEST OFF)

set (WITH_FUZZING none CACHE STRING "Fuzzing engine")
set_property (CACHE WITH_FUZZING PROPERTY STRINGS none libfuzzer ossfuzz)

if (NOT WITH_UNWIND)
  set (CMAKE_DISABLE_FIND_PACKAGE_Unwind ON)
endif (NOT WITH_UNWIND)

if (NOT WITH_GTEST)
  set (CMAKE_DISABLE_FIND_PACKAGE_GTest ON)
endif (NOT WITH_GTEST)

if (NOT WITH_THREADS)
  set (CMAKE_DISABLE_FIND_PACKAGE_Threads ON)
endif (NOT WITH_THREADS)

set (CMAKE_C_VISIBILITY_PRESET hidden)
set (CMAKE_CXX_VISIBILITY_PRESET hidden)
set (CMAKE_POSITION_INDEPENDENT_CODE ON)
set (CMAKE_VISIBILITY_INLINES_HIDDEN ON)

set (CMAKE_DEBUG_POSTFIX d)
set (CMAKE_THREAD_PREFER_PTHREAD 1)

find_package (GTest NO_MODULE)

if (GTest_FOUND)
  set (HAVE_LIB_GTEST 1)
endif (GTest_FOUND)

if (WITH_GMOCK AND TARGET GTest::gmock)
  set (HAVE_LIB_GMOCK 1)
endif (WITH_GMOCK AND TARGET GTest::gmock)

if (WITH_GFLAGS)
  find_package (gflags 2.2.2)

  if (gflags_FOUND)
    set (HAVE_LIB_GFLAGS 1)
    determine_gflags_namespace (gflags_NAMESPACE)
  endif (gflags_FOUND)
endif (WITH_GFLAGS)

find_package (Threads)
find_package (Unwind)

if (Unwind_FOUND)
  set (HAVE_LIB_UNWIND 1)
else (Unwind_FOUND)
  # Check whether linking actually succeeds. ARM toolchains of LLVM unwind
  # implementation do not necessarily provide the _Unwind_Backtrace function
  # which causes the previous check to succeed but the linking to fail.
  check_cxx_symbol_exists (_Unwind_Backtrace unwind.h HAVE__UNWIND_BACKTRACE)
  check_cxx_symbol_exists (_Unwind_GetIP unwind.h HAVE__UNWIND_GETIP)
endif (Unwind_FOUND)

check_include_file_cxx (dlfcn.h HAVE_DLFCN_H)
check_include_file_cxx (glob.h HAVE_GLOB_H)
check_include_file_cxx (memory.h HAVE_MEMORY_H)
check_include_file_cxx (pwd.h HAVE_PWD_H)
check_include_file_cxx (strings.h HAVE_STRINGS_H)
check_include_file_cxx (sys/stat.h HAVE_SYS_STAT_H)
check_include_file_cxx (sys/syscall.h HAVE_SYS_SYSCALL_H)
check_include_file_cxx (sys/time.h HAVE_SYS_TIME_H)
check_include_file_cxx (sys/types.h HAVE_SYS_TYPES_H)
check_include_file_cxx (sys/utsname.h HAVE_SYS_UTSNAME_H)
check_include_file_cxx (sys/wait.h HAVE_SYS_WAIT_H)
check_include_file_cxx (syscall.h HAVE_SYSCALL_H)
check_include_file_cxx (syslog.h HAVE_SYSLOG_H)
check_include_file_cxx (ucontext.h HAVE_UCONTEXT_H)
check_include_file_cxx (unistd.h HAVE_UNISTD_H)

check_type_size (mode_t HAVE_MODE_T LANGUAGE CXX)
check_type_size (ssize_t HAVE_SSIZE_T LANGUAGE CXX)

check_function_exists (dladdr HAVE_DLADDR)
check_function_exists (fcntl HAVE_FCNTL)
check_function_exists (pread HAVE_PREAD)
check_function_exists (pwrite HAVE_PWRITE)
check_function_exists (sigaction HAVE_SIGACTION)
check_function_exists (sigaltstack HAVE_SIGALTSTACK)

check_cxx_symbol_exists (backtrace execinfo.h HAVE_EXECINFO_BACKTRACE)
check_cxx_symbol_exists (backtrace_symbols execinfo.h
  HAVE_EXECINFO_BACKTRACE_SYMBOLS)

# NOTE gcc does not fail if you pass a non-existent -Wno-* option as an
# argument. However, it will happily fail if you pass the corresponding -W*
# option. So, we check whether options that disable warnings exist by testing
# the availability of the corresponding option that enables the warning. This
# eliminates the need to check for compiler for several (mainly Clang) options.

check_cxx_compiler_flag (-Wdeprecated HAVE_NO_DEPRECATED)
check_cxx_compiler_flag (-Wunnamed-type-template-args
    HAVE_NO_UNNAMED_TYPE_TEMPLATE_ARGS)

cmake_push_check_state (RESET)

if (Threads_FOUND)
  set (CMAKE_REQUIRED_LIBRARIES Threads::Threads)
endif (Threads_FOUND)

check_cxx_symbol_exists (pthread_threadid_np "pthread.h" HAVE_PTHREAD_THREADID_NP)
cmake_pop_check_state ()

# NOTE: Cannot use check_function_exists here since >=vc-14.0 can define
# snprintf as an inline function
check_cxx_symbol_exists (snprintf cstdio HAVE_SNPRINTF)

check_library_exists (dbghelp UnDecorateSymbolName "" HAVE_DBGHELP)

check_cxx_source_compiles ("
#include <cstdlib>
static void foo(void) __attribute__ ((unused));
int main(void) { return 0; }
" HAVE___ATTRIBUTE__)

check_cxx_source_compiles ("
#include <cstdlib>
static void foo(void) __attribute__ ((visibility(\"default\")));
int main(void) { return 0; }
" HAVE___ATTRIBUTE__VISIBILITY_DEFAULT)

check_cxx_source_compiles ("
#include <cstdlib>
static void foo(void) __attribute__ ((visibility(\"hidden\")));
int main(void) { return 0; }
" HAVE___ATTRIBUTE__VISIBILITY_HIDDEN)

check_cxx_source_compiles ("
int main(void) { if (__builtin_expect(0, 0)) return 1; return 0; }
" HAVE___BUILTIN_EXPECT)

check_cxx_source_compiles ("
int main(void)
{
  int a; if (__sync_val_compare_and_swap(&a, 0, 1)) return 1; return 0;
}
" HAVE___SYNC_VAL_COMPARE_AND_SWAP)

if (Threads_FOUND)
  cmake_push_check_state (RESET)
  set (CMAKE_REQUIRED_LIBRARIES Threads::Threads)
  check_cxx_source_compiles ("
#define _XOPEN_SOURCE 500
#include <pthread.h>
int main(void)
{
  pthread_rwlock_t l;
  pthread_rwlock_init(&l, NULL);
  pthread_rwlock_rdlock(&l);
  return 0;
}
  " HAVE_RWLOCK)
  cmake_pop_check_state ()
endif (Threads_FOUND)

check_cxx_source_compiles ("
__declspec(selectany) int a;
int main(void) { return 0; }
" HAVE___DECLSPEC)

if (WITH_TLS)
  set (GLOG_THREAD_LOCAL_STORAGE 1)
endif (WITH_TLS)

set (_PC_FIELDS
  "gregs[REG_PC]"
  "gregs[REG_EIP]"
  "gregs[REG_RIP]"
  "sc_ip"
  "uc_regs->gregs[PT_NIP]"
  "gregs[R15]"
  "arm_pc"
  "mc_eip"
  "mc_rip"
  "__gregs[REG_EIP]"
  "__gregs[REG_RIP]"
  "ss.eip"
  "__ss.__eip"
  "ss.rip"
  "__ss.__rip"
  "ss.srr0"
  "__ss.__srr0"
)

set (_PC_HEADERS ucontext.h signal.h)

if (HAVE_UCONTEXT_H AND NOT PC_FROM_UCONTEXT)
  foreach (_PC_FIELD ${_PC_FIELDS})
    foreach (_PC_HEADER ${_PC_HEADERS})
      set (_TMP
      ${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/uctfield.cpp)
      file (WRITE ${_TMP} "
#define _GNU_SOURCE 1
#include <${_PC_HEADER}>
int main(void)
{
  ucontext_t u;
  return u.${_PC_FIELD} == 0;
}
")
      try_compile (HAVE_PC_FROM_UCONTEXT ${CMAKE_CURRENT_BINARY_DIR} ${_TMP}
        COMPILE_DEFINITIONS _GNU_SOURCE=1)

      if (HAVE_PC_FROM_UCONTEXT)
        set (PC_FROM_UCONTEXT ${_PC_FIELD} CACHE)
      endif (HAVE_PC_FROM_UCONTEXT)
    endforeach (_PC_HEADER)
  endforeach (_PC_FIELD)
endif  (HAVE_UCONTEXT_H AND NOT PC_FROM_UCONTEXT)

set (GOOGLE_NAMESPACE google)
set (_START_GOOGLE_NAMESPACE_ "namespace ${GOOGLE_NAMESPACE} {")
set (_END_GOOGLE_NAMESPACE_ "}")
set (ac_cv_have_glog_export 1)

if (HAVE_LIB_GFLAGS)
  set (ac_cv_have_libgflags 1)
else (HAVE_LIB_GFLAGS)
  set (ac_cv_have_libgflags 0)
endif (HAVE_LIB_GFLAGS)

if (HAVE_SYS_TYPES_H)
  set (ac_cv_have_systypes_h 1)
else (HAVE_SYS_TYPES_H)
  set (ac_cv_have_systypes_h 0)
endif (HAVE_SYS_TYPES_H)

if (HAVE_SSIZE_T)
  set (ac_cv_have_ssize_t 1)
else (HAVE_SSIZE_T)
  set (ac_cv_have_ssize_t 0)
endif (HAVE_SSIZE_T)

if (HAVE_MODE_T)
  set (ac_cv_have_mode_t 1)
else (HAVE_MODE_T)
  set (ac_cv_have_mode_t 0)
endif (HAVE_MODE_T)

if (HAVE_UNISTD_H)
  set (ac_cv_have_unistd_h 1)
else (HAVE_UNISTD_H)
  set (ac_cv_have_unistd_h 0)
endif (HAVE_UNISTD_H)

set (ac_google_namespace ${GOOGLE_NAMESPACE})
set (ac_google_end_namespace ${_END_GOOGLE_NAMESPACE_})
set (ac_google_start_namespace ${_START_GOOGLE_NAMESPACE_})

if (HAVE___ATTRIBUTE__)
  set (ac_cv___attribute___noinline "__attribute__((noinline))")
  set (ac_cv___attribute___printf_4_5 "__attribute__((__format__(__printf__, 4, 5)))")
elseif (HAVE___DECLSPEC)
  #set (ac_cv___attribute___noinline "__declspec(noinline)")
endif (HAVE___ATTRIBUTE__)

if (HAVE___BUILTIN_EXPECT)
  set (ac_cv_have___builtin_expect 1)
else (HAVE___BUILTIN_EXPECT)
  set (ac_cv_have___builtin_expect 0)
endif (HAVE___BUILTIN_EXPECT)

if (HAVE_EXECINFO_BACKTRACE AND HAVE_EXECINFO_BACKTRACE_SYMBOLS)
  set (HAVE_STACKTRACE 1)
endif (HAVE_EXECINFO_BACKTRACE AND HAVE_EXECINFO_BACKTRACE_SYMBOLS)

if (WITH_SYMBOLIZE)
  if (WIN32 OR CYGWIN)
    cmake_push_check_state (RESET)
    set (CMAKE_REQUIRED_LIBRARIES DbgHelp)

    check_cxx_source_runs ([=[
    #include <windows.h>
    #include <dbghelp.h>
    #include <cstdlib>

    void foobar() { }

    int main()
    {
        HANDLE process = GetCurrentProcess();

        if (!SymInitialize(process, NULL, TRUE))
            return EXIT_FAILURE;

        char buf[sizeof(SYMBOL_INFO) + MAX_SYM_NAME];
        SYMBOL_INFO *symbol = reinterpret_cast<SYMBOL_INFO *>(buf);
        symbol->SizeOfStruct = sizeof(SYMBOL_INFO);
        symbol->MaxNameLen = MAX_SYM_NAME;

        void* const pc = reinterpret_cast<void*>(&foobar);
        BOOL ret = SymFromAddr(process, reinterpret_cast<DWORD64>(pc), 0, symbol);

        return ret ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    ]=] HAVE_SYMBOLIZE)

    cmake_pop_check_state ()

    if (HAVE_SYMBOLIZE)
      set (HAVE_STACKTRACE 1)
    endif (HAVE_SYMBOLIZE)
  elseif (UNIX)
    cmake_push_check_state (RESET)
    check_cxx_source_compiles ([=[
    int main()
    {
      #ifndef __ELF__
      #error __ELF__ not defined
      #endif
    }
    ]=] HAVE_SYMBOLIZE)
    cmake_pop_check_state ()
  elseif (APPLE AND HAVE_DLADDR)
    set (HAVE_SYMBOLIZE 1)
  endif (WIN32 OR CYGWIN)
endif (WITH_SYMBOLIZE)

# CMake manages symbolize availability. The definition is necessary only when
# building the library.
add_compile_definitions (GLOG_NO_SYMBOLIZE_DETECTION)

check_cxx_source_compiles ("
#include <cstdlib>
#include <ctime>
int main()
{
    time_t timep;
    struct tm result;
    localtime_r(&timep, &result);
    return EXIT_SUCCESS;
}
" HAVE_LOCALTIME_R)

set (SIZEOF_VOID_P ${CMAKE_SIZEOF_VOID_P})

if (WITH_THREADS AND Threads_FOUND)
  if (CMAKE_USE_PTHREADS_INIT)
    set (HAVE_PTHREAD 1)
  endif (CMAKE_USE_PTHREADS_INIT)
else (WITH_THREADS AND Threads_FOUND)
  set (NO_THREADS 1)
endif (WITH_THREADS AND Threads_FOUND)

# fopen/open on Cygwin can not handle unix-type paths like /home/....
# therefore we translate TEST_SRC_DIR to windows-path.
if (CYGWIN)
  execute_process (COMMAND cygpath.exe -m ${CMAKE_CURRENT_SOURCE_DIR}
                   OUTPUT_STRIP_TRAILING_WHITESPACE
                   OUTPUT_VARIABLE TEST_SRC_DIR)
  set (TEST_SRC_DIR \"${TEST_SRC_DIR}\")
else (CYGWIN)
  set (TEST_SRC_DIR \"${CMAKE_CURRENT_SOURCE_DIR}\")
endif (CYGWIN)

configure_file (src/config.h.cmake.in config.h)
configure_file (src/glog/logging.h.in glog/logging.h @ONLY)
configure_file (src/glog/raw_logging.h.in glog/raw_logging.h @ONLY)
configure_file (src/glog/stl_logging.h.in glog/stl_logging.h @ONLY)
configure_file (src/glog/vlog_is_on.h.in glog/vlog_is_on.h @ONLY)

add_compile_options ($<$<AND:$<BOOL:${HAVE_NO_UNNAMED_TYPE_TEMPLATE_ARGS}>,$<NOT:$<CXX_COMPILER_ID:GNU>>>:-Wno-unnamed-type-template-args>)

set (_glog_CMake_BINDIR ${CMAKE_INSTALL_BINDIR})
set (_glog_CMake_INCLUDE_DIR ${CMAKE_INSTALL_INCLUDEDIR})
set (_glog_CMake_LIBDIR ${CMAKE_INSTALL_LIBDIR})
set (_glog_CMake_INSTALLDIR ${_glog_CMake_LIBDIR}/cmake/glog)

set (_glog_CMake_DIR glog/cmake)
set (_glog_CMake_DATADIR ${CMAKE_INSTALL_DATAROOTDIR}/${_glog_CMake_DIR})
set (_glog_BINARY_CMake_DATADIR
  ${CMAKE_CURRENT_BINARY_DIR}/${_glog_CMake_DATADIR})

# Add additional CMake find modules here.
set (_glog_CMake_MODULES)

if (Unwind_FOUND)
  # Copy the module only if libunwind is actually used.
  list (APPEND _glog_CMake_MODULES ${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindUnwind.cmake)
endif (Unwind_FOUND)

# Generate file name for each module in the binary directory
foreach (_file ${_glog_CMake_MODULES})
  get_filename_component (_module "${_file}" NAME)

  list (APPEND _glog_BINARY_CMake_MODULES
    ${_glog_BINARY_CMake_DATADIR}/${_module})
endforeach (_file)

if (_glog_CMake_MODULES)
  # Copy modules to binary directory during the build
  add_custom_command (OUTPUT ${_glog_BINARY_CMake_MODULES}
    COMMAND ${CMAKE_COMMAND} -E make_directory
    ${_glog_BINARY_CMake_DATADIR}
    COMMAND ${CMAKE_COMMAND} -E copy ${_glog_CMake_MODULES}
    ${_glog_BINARY_CMake_DATADIR}
    DEPENDS ${_glog_CMake_MODULES}
    COMMENT "Copying find modules..."
  )
endif (_glog_CMake_MODULES)

set (GLOG_PUBLIC_H
  ${CMAKE_CURRENT_BINARY_DIR}/glog/export.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/raw_logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/stl_logging.h
  ${CMAKE_CURRENT_BINARY_DIR}/glog/vlog_is_on.h
  src/glog/log_severity.h
  src/glog/platform.h
)

set (GLOG_SRCS
  ${GLOG_PUBLIC_H}
  src/base/commandlineflags.h
  src/base/googleinit.h
  src/base/mutex.h
  src/demangle.cc
  src/demangle.h
  src/logging.cc
  src/raw_logging.cc
  src/symbolize.cc
  src/symbolize.h
  src/utilities.cc
  src/utilities.h
  src/vlog_is_on.cc
)

if (HAVE_PTHREAD OR WIN32 OR CYGWIN)
  list (APPEND GLOG_SRCS src/signalhandler.cc)
endif (HAVE_PTHREAD OR WIN32 OR CYGWIN)

if (CYGWIN OR WIN32)
  list (APPEND GLOG_SRCS
    src/windows/port.cc
    src/windows/port.h
  )
endif (CYGWIN OR WIN32)

add_library (glog_internal OBJECT
  ${_glog_BINARY_CMake_MODULES}
  ${GLOG_SRCS}
)
target_compile_features (glog_internal PUBLIC $<TARGET_PROPERTY:glog,COMPILE_FEATURES>)

add_library (glog
  $<TARGET_OBJECTS:glog_internal>
)
target_compile_features (glog PUBLIC cxx_std_14)

add_library (glog::glog ALIAS glog)

set (glog_libraries_options_for_static_linking)

if (Unwind_FOUND)
  target_link_libraries (glog PRIVATE unwind::unwind)
  set (glog_libraries_options_for_static_linking "${glog_libraries_options_for_static_linking} -lunwind")
  set (Unwind_DEPENDENCY "find_dependency (Unwind ${Unwind_VERSION})")
endif (Unwind_FOUND)

if (HAVE_DBGHELP)
  target_link_libraries (glog PRIVATE dbghelp)
  set (glog_libraries_options_for_static_linking "${glog_libraries_options_for_static_linking} -ldbghelp")
endif (HAVE_DBGHELP)

if (HAVE_PTHREAD)
  target_link_libraries (glog PRIVATE ${CMAKE_THREAD_LIBS_INIT})

  if (CMAKE_THREAD_LIBS_INIT)
    set (glog_libraries_options_for_static_linking "${glog_libraries_options_for_static_linking} ${CMAKE_THREAD_LIBS_INIT}")
  endif (CMAKE_THREAD_LIBS_INIT)
endif (HAVE_PTHREAD)

if (gflags_FOUND)
  # Prefer the gflags target that uses double colon convention
  if (TARGET gflags::gflags)
    target_link_libraries (glog PUBLIC gflags::gflags)
  else (TARGET gflags::gflags)
    target_link_libraries (glog PUBLIC gflags)
  endif (TARGET gflags::gflags)

  set (glog_libraries_options_for_static_linking "${glog_libraries_options_for_static_linking} -lgflags")
endif (gflags_FOUND)

if (ANDROID)
  target_link_libraries (glog PRIVATE log)
  set (glog_libraries_options_for_static_linking "${glog_libraries_options_for_static_linking} -llog")
endif (ANDROID)

set_target_properties (glog PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties (glog PROPERTIES SOVERSION 1)

if (CYGWIN OR WIN32)
  target_compile_definitions (glog PUBLIC GLOG_NO_ABBREVIATED_SEVERITIES)
endif (CYGWIN OR WIN32)

set_target_properties (glog PROPERTIES PUBLIC_HEADER "${GLOG_PUBLIC_H}")

target_include_directories (glog BEFORE PUBLIC
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>"
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>"
  "$<INSTALL_INTERFACE:${_glog_CMake_INCLUDE_DIR}>"
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (CYGWIN OR WIN32)
  target_include_directories (glog_internal PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/windows>"
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/windows)

  target_include_directories (glog PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/windows>"
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/windows)
endif (CYGWIN OR WIN32)

set_target_properties (glog PROPERTIES DEFINE_SYMBOL GOOGLE_GLOG_IS_A_DLL)

target_include_directories (glog_internal PUBLIC
  $<TARGET_PROPERTY:glog,INCLUDE_DIRECTORIES>)
target_compile_definitions (glog_internal PUBLIC
  $<TARGET_PROPERTY:glog,COMPILE_DEFINITIONS>
  PRIVATE GOOGLE_GLOG_IS_A_DLL)

generate_export_header (glog
  EXPORT_MACRO_NAME GLOG_EXPORT
  EXPORT_FILE_NAME ${CMAKE_CURRENT_BINARY_DIR}/glog/export.h)

string (STRIP "${glog_libraries_options_for_static_linking}" glog_libraries_options_for_static_linking)

if (WITH_PKGCONFIG)
  set (VERSION ${PROJECT_VERSION})
  set (prefix ${CMAKE_INSTALL_PREFIX})
  set (exec_prefix ${CMAKE_INSTALL_FULL_BINDIR})
  set (libdir ${CMAKE_INSTALL_FULL_LIBDIR})
  set (includedir ${CMAKE_INSTALL_FULL_INCLUDEDIR})

  configure_file (
    "${PROJECT_SOURCE_DIR}/libglog.pc.in"
    "${PROJECT_BINARY_DIR}/libglog.pc"
    @ONLY
  )

  unset (VERSION)
  unset (prefix)
  unset (exec_prefix)
  unset (libdir)
  unset (includedir)
endif (WITH_PKGCONFIG)

# Unit testing

if (NOT WITH_FUZZING STREQUAL "none")
  add_executable (fuzz_demangle
    src/fuzz_demangle.cc
  )

  if (WITH_FUZZING STREQUAL "ossfuzz")
    set (LIB_FUZZING_ENGINE $ENV{LIB_FUZZING_ENGINE})
    target_link_libraries (fuzz_demangle PRIVATE glog ${LIB_FUZZING_ENGINE})
  elseif (WITH_FUZZING STREQUAL "libfuzzer")
    target_compile_options (fuzz_demangle PRIVATE -fsanitize=fuzzer)
    target_link_libraries (fuzz_demangle PRIVATE glog)
  else (WITH_FUZZING STREQUAL "libfuzzer")
    message (FATAL_ERROR "Unsupported fuzzing engine ${WITH_FUZZING}")
  endif (WITH_FUZZING STREQUAL "ossfuzz")
endif (NOT WITH_FUZZING STREQUAL "none")

if (BUILD_TESTING)
  add_library (glogtest STATIC
    $<TARGET_OBJECTS:glog_internal>
  )

  target_include_directories (glogtest PUBLIC
    $<TARGET_PROPERTY:glog,INCLUDE_DIRECTORIES>)
  target_compile_definitions (glogtest PUBLIC
    $<TARGET_PROPERTY:glog,COMPILE_DEFINITIONS> GLOG_STATIC_DEFINE)
  target_link_libraries (glogtest PUBLIC
    $<TARGET_PROPERTY:glog,LINK_LIBRARIES>)

  set (_GLOG_TEST_LIBS glogtest)

  if (HAVE_LIB_GTEST)
    list (APPEND _GLOG_TEST_LIBS GTest::gtest)
  endif (HAVE_LIB_GTEST)

  if (HAVE_LIB_GMOCK)
    list (APPEND _GLOG_TEST_LIBS GTest::gmock)
  endif (HAVE_LIB_GMOCK)

  add_executable (logging_unittest
    src/logging_unittest.cc
  )

  target_link_libraries (logging_unittest PRIVATE ${_GLOG_TEST_LIBS})

  add_executable (stl_logging_unittest
    src/stl_logging_unittest.cc
  )

  target_link_libraries (stl_logging_unittest PRIVATE ${_GLOG_TEST_LIBS})

  if (HAVE_NO_DEPRECATED)
    set_property (TARGET stl_logging_unittest APPEND PROPERTY COMPILE_OPTIONS
      -Wno-deprecated)
  endif (HAVE_NO_DEPRECATED)

  if (HAVE_EXT_SLIST)
    target_compile_definitions (stl_logging_unittest PRIVATE
      GLOG_STL_LOGGING_FOR_EXT_SLIST)
  endif (HAVE_EXT_SLIST)

  if (HAVE_SYMBOLIZE)
    add_executable (symbolize_unittest
      src/symbolize_unittest.cc
    )

    target_link_libraries (symbolize_unittest PRIVATE ${_GLOG_TEST_LIBS})
  endif (HAVE_SYMBOLIZE)

  add_executable (demangle_unittest
    src/demangle_unittest.cc
  )

  target_link_libraries (demangle_unittest PRIVATE ${_GLOG_TEST_LIBS})

  if (HAVE_STACKTRACE)
    add_executable (stacktrace_unittest
      src/stacktrace_unittest.cc
    )

    target_link_libraries (stacktrace_unittest PRIVATE ${_GLOG_TEST_LIBS})
  endif (HAVE_STACKTRACE)

  add_executable (utilities_unittest
    src/utilities_unittest.cc
  )

  target_link_libraries (utilities_unittest PRIVATE ${_GLOG_TEST_LIBS})

  if (HAVE_STACKTRACE AND HAVE_SYMBOLIZE)
    add_executable (signalhandler_unittest
      src/signalhandler_unittest.cc
    )

    target_link_libraries (signalhandler_unittest PRIVATE ${_GLOG_TEST_LIBS})
  endif (HAVE_STACKTRACE AND HAVE_SYMBOLIZE)

  add_test (NAME demangle COMMAND demangle_unittest)
  add_test (NAME logging COMMAND logging_unittest)

  set_tests_properties (logging PROPERTIES TIMEOUT 30)
  # MacOS diff is not deterministic: use the output to determine whether the
  # test passed.
  set_tests_properties (logging PROPERTIES PASS_REGULAR_EXPRESSION ".*\nPASS\n.*")

  # FIXME: Skip flaky test
  set_tests_properties (logging PROPERTIES SKIP_REGULAR_EXPRESSION
    "Check failed: time_ns within LogTimes::LOG_PERIOD_TOL_NS of LogTimes::LOG_PERIOD_NS")

  if (APPLE)
    # FIXME: Skip flaky test
    set_property (TEST logging APPEND PROPERTY SKIP_REGULAR_EXPRESSION
      "unexpected new.*PASS\nTest with golden file failed. We'll try to show the diff:")
  endif (APPLE)

  if (TARGET signalhandler_unittest)
    add_test (NAME signalhandler COMMAND signalhandler_unittest)
  endif (TARGET signalhandler_unittest)

  if (TARGET stacktrace_unittest)
    add_test (NAME stacktrace COMMAND stacktrace_unittest)
    set_tests_properties (stacktrace PROPERTIES TIMEOUT 30)
  endif (TARGET stacktrace_unittest)

  add_test (NAME stl_logging COMMAND stl_logging_unittest)

  if (TARGET symbolize_unittest)
    add_test (NAME symbolize COMMAND symbolize_unittest)

    # FIXME: Skip flaky test when compiled in C++20 mode
    set_tests_properties (symbolize PROPERTIES SKIP_REGULAR_EXPRESSION
      [=[Check failed: streq\("nonstatic_func"\, TrySymbolize\(\(void \*\)\(&nonstatic_func\)\)\)]=])
  endif (TARGET symbolize_unittest)

  if (HAVE_LIB_GMOCK)
    add_executable (mock-log_unittest
      src/mock-log_unittest.cc
      src/mock-log.h
    )

    target_link_libraries (mock-log_unittest PRIVATE ${_GLOG_TEST_LIBS})

    add_test (NAME mock-log COMMAND mock-log_unittest)
  endif (HAVE_LIB_GMOCK)

  # Generate an initial cache

  get_cache_variables (_CACHEVARS)

  set (_INITIAL_CACHE
    ${CMAKE_CURRENT_BINARY_DIR}/test_package_config/glog_package_config_initial_cache.cmake)

  # Package config test

  add_test (NAME cmake_package_config_init COMMAND ${CMAKE_COMMAND}
    -DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_package_config
    -DINITIAL_CACHE=${_INITIAL_CACHE}
    -DCACHEVARS=${_CACHEVARS}
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestInitPackageConfig.cmake
  )

  add_test (NAME cmake_package_config_generate COMMAND ${CMAKE_COMMAND}
    -DGENERATOR=${CMAKE_GENERATOR}
    -DGENERATOR_PLATFORM=${CMAKE_GENERATOR_PLATFORM}
    -DGENERATOR_TOOLSET=${CMAKE_GENERATOR_TOOLSET}
    -DINITIAL_CACHE=${_INITIAL_CACHE}
    -DPACKAGE_DIR=${CMAKE_CURRENT_BINARY_DIR}
    -DPATH=$ENV{PATH}
    -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/src/package_config_unittest/working_config
    -DTEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/test_package_config/working_config
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/TestPackageConfig.cmake
  )

  add_test (NAME cmake_package_config_build COMMAND
    ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR}/test_package_config/working_config
                     --config $<CONFIG>
  )

  add_test (NAME cmake_package_config_cleanup COMMAND ${CMAKE_COMMAND} -E
    remove_directory
    ${CMAKE_CURRENT_BINARY_DIR}/test_package_config
  )

  # Fixtures setup
  set_tests_properties (cmake_package_config_init PROPERTIES FIXTURES_SETUP
    cmake_package_config)
  set_tests_properties (cmake_package_config_generate PROPERTIES FIXTURES_SETUP
    cmake_package_config_working)

  # Fixtures cleanup
  set_tests_properties (cmake_package_config_cleanup PROPERTIES FIXTURES_CLEANUP
    cmake_package_config)

  # Fixture requirements
  set_tests_properties (cmake_package_config_generate PROPERTIES
    FIXTURES_REQUIRED cmake_package_config)
  set_tests_properties (cmake_package_config_build PROPERTIES
    FIXTURES_REQUIRED "cmake_package_config;cmake_package_config_working")

  add_executable (cleanup_immediately_unittest
    src/cleanup_immediately_unittest.cc)

  target_link_libraries (cleanup_immediately_unittest PRIVATE ${_GLOG_TEST_LIBS})

  add_executable (cleanup_with_absolute_prefix_unittest
    src/cleanup_with_absolute_prefix_unittest.cc)

  target_link_libraries (cleanup_with_absolute_prefix_unittest PRIVATE ${_GLOG_TEST_LIBS})

  add_executable (cleanup_with_relative_prefix_unittest
    src/cleanup_with_relative_prefix_unittest.cc)

  target_link_libraries (cleanup_with_relative_prefix_unittest PRIVATE ${_GLOG_TEST_LIBS})

  set (CLEANUP_LOG_DIR ${CMAKE_CURRENT_BINARY_DIR}/cleanup_tests)

  add_test (NAME cleanup_init COMMAND
    ${CMAKE_COMMAND} -E make_directory ${CLEANUP_LOG_DIR})
  add_test (NAME cleanup_logdir COMMAND
    ${CMAKE_COMMAND} -E remove_directory ${CLEANUP_LOG_DIR})
  add_test (NAME cleanup_immediately COMMAND
    ${CMAKE_COMMAND}
    -DLOGCLEANUP=$<TARGET_FILE:cleanup_immediately_unittest>
    # NOTE The trailing slash is important
    -DTEST_DIR=${CLEANUP_LOG_DIR}/
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCleanerTest1.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test (NAME cleanup_with_absolute_prefix COMMAND
    ${CMAKE_COMMAND}
    -DLOGCLEANUP=$<TARGET_FILE:cleanup_with_absolute_prefix_unittest>
    -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCleanerTest2.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  add_test (NAME cleanup_with_relative_prefix COMMAND
    ${CMAKE_COMMAND}
    -DLOGCLEANUP=$<TARGET_FILE:cleanup_with_relative_prefix_unittest>
    -DTEST_DIR=${CMAKE_CURRENT_BINARY_DIR}/
    -DTEST_SUBDIR=test_subdir/
    -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/RunCleanerTest3.cmake
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # Fixtures setup
  set_tests_properties (cleanup_init PROPERTIES FIXTURES_SETUP logcleanuptest)
  ## Fixtures cleanup
  set_tests_properties (cleanup_logdir PROPERTIES FIXTURES_CLEANUP logcleanuptest)
  # Fixture requirements
  set_tests_properties (cleanup_immediately PROPERTIES FIXTURES_REQUIRED logcleanuptest)
  set_tests_properties (cleanup_with_absolute_prefix PROPERTIES FIXTURES_REQUIRED logcleanuptest)
  set_tests_properties (cleanup_with_relative_prefix PROPERTIES FIXTURES_REQUIRED logcleanuptest)
endif (BUILD_TESTING)

install (TARGETS glog
  EXPORT glog-targets
  RUNTIME DESTINATION ${_glog_CMake_BINDIR}
  PUBLIC_HEADER DESTINATION ${_glog_CMake_INCLUDE_DIR}/glog
  LIBRARY DESTINATION ${_glog_CMake_LIBDIR}
  ARCHIVE DESTINATION ${_glog_CMake_LIBDIR})

if (WITH_PKGCONFIG)
  install (
    FILES "${PROJECT_BINARY_DIR}/libglog.pc"
    DESTINATION "${_glog_CMake_LIBDIR}/pkgconfig"
  )
endif (WITH_PKGCONFIG)

set (glog_CMake_VERSION 3.0)

if (gflags_FOUND)
  # Ensure clients locate only the package config and not third party find
  # modules having the same name. This avoid cmake_policy PUSH/POP errors.
  if (CMAKE_VERSION VERSION_LESS 3.9)
    set (gflags_DEPENDENCY "find_dependency (gflags ${gflags_VERSION})")
  else (CMAKE_VERSION VERSION_LESS 3.9)
    # Passing additional find_package arguments to find_dependency is possible
    # starting with CMake 3.9.
    set (glog_CMake_VERSION 3.9)
    set (gflags_DEPENDENCY "find_dependency (gflags ${gflags_VERSION} NO_MODULE)")
  endif (CMAKE_VERSION VERSION_LESS 3.9)
endif (gflags_FOUND)

configure_package_config_file (glog-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/glog-config.cmake
  INSTALL_DESTINATION ${_glog_CMake_INSTALLDIR}
  NO_CHECK_REQUIRED_COMPONENTS_MACRO)

write_basic_package_version_file (
  ${CMAKE_CURRENT_BINARY_DIR}/glog-config-version.cmake
  COMPATIBILITY SameMajorVersion)

export (TARGETS glog NAMESPACE glog:: FILE glog-targets.cmake)
export (PACKAGE glog)

get_filename_component (_PREFIX "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)

# Directory containing the find modules relative to the config install
# directory.
file (RELATIVE_PATH glog_REL_CMake_MODULES
  ${_PREFIX}/${_glog_CMake_INSTALLDIR}
  ${_PREFIX}/${_glog_CMake_DATADIR}/glog-modules.cmake)

get_filename_component (glog_REL_CMake_DATADIR ${glog_REL_CMake_MODULES}
  DIRECTORY)

set (glog_FULL_CMake_DATADIR
  ${CMAKE_CURRENT_BINARY_DIR}/${_glog_CMake_DATADIR})

configure_file (glog-modules.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/glog-modules.cmake @ONLY)

install (CODE
"
set (glog_FULL_CMake_DATADIR \"\\\${CMAKE_CURRENT_LIST_DIR}/${glog_REL_CMake_DATADIR}\")
set (glog_DATADIR_DESTINATION ${_glog_CMake_INSTALLDIR})

if (NOT IS_ABSOLUTE ${_glog_CMake_INSTALLDIR})
  set (glog_DATADIR_DESTINATION \"\${CMAKE_INSTALL_PREFIX}/\${glog_DATADIR_DESTINATION}\")
endif (NOT IS_ABSOLUTE ${_glog_CMake_INSTALLDIR})

configure_file (\"${CMAKE_CURRENT_SOURCE_DIR}/glog-modules.cmake.in\"
  \"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/glog-modules.cmake\" @ONLY)
file (INSTALL
  \"${CMAKE_CURRENT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CMakeTmp/glog-modules.cmake\"
  DESTINATION
  \"\${glog_DATADIR_DESTINATION}\")
"
  COMPONENT Development
)

install (FILES
  ${CMAKE_CURRENT_BINARY_DIR}/glog-config.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/glog-config-version.cmake
  DESTINATION ${_glog_CMake_INSTALLDIR})

# Find modules in share/glog/cmake
install (DIRECTORY ${_glog_BINARY_CMake_DATADIR}
  DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/glog
  COMPONENT Development
  FILES_MATCHING PATTERN "*.cmake"
)

install (EXPORT glog-targets NAMESPACE glog:: DESTINATION
  ${_glog_CMake_INSTALLDIR})
