// Note: This header file is only used internally. It is not part of public interface!

#ifndef GFLAGS_CONFIG_H_
#define GFLAGS_CONFIG_H_


// ---------------------------------------------------------------------------
// System checks

// CMake build configuration is written to defines.h file, unused by Bazel build
#if !defined(GFLAGS_BAZEL_BUILD)
#  include "defines.h"
#endif

// gcc requires this to get PRId64, etc.
#if defined(HAVE_INTTYPES_H) && !defined(__STDC_FORMAT_MACROS)
#  define __STDC_FORMAT_MACROS 1
#endif

// ---------------------------------------------------------------------------
// Path separator
#ifndef PATH_SEPARATOR
#  ifdef OS_WINDOWS
#    define PATH_SEPARATOR  '\\'
#  else
#    define PATH_SEPARATOR  '/'
#  endif
#endif

// ---------------------------------------------------------------------------
// Windows

// Always export symbols when compiling a shared library as this file is only
// included by internal modules when building the gflags library itself.
// The gflags_declare.h header file will set it to import these symbols otherwise.
#ifndef GFLAGS_DLL_DECL
#  if GFLAGS_IS_A_DLL && defined(_MSC_VER)
#    define GFLAGS_DLL_DECL __declspec(dllexport)
#  elif defined(__GNUC__) && __GNUC__ >= 4
#    define GFLAGS_DLL_DECL __attribute__((visibility("default")))
#  else
#    define GFLAGS_DLL_DECL
#  endif
#endif
// Flags defined by the gflags library itself must be exported
#ifndef GFLAGS_DLL_DEFINE_FLAG
#  define GFLAGS_DLL_DEFINE_FLAG GFLAGS_DLL_DECL
#endif

#ifdef OS_WINDOWS
// The unittests import the symbols of the shared gflags library
#  if GFLAGS_IS_A_DLL && defined(_MSC_VER)
#    define GFLAGS_DLL_DECL_FOR_UNITTESTS __declspec(dllimport)
#  endif
#  include "windows_port.h"
#endif


#endif // GFLAGS_CONFIG_H_
