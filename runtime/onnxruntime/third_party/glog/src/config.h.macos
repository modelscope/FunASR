#ifndef GLOG_CONFIG_H
#define GLOG_CONFIG_H

/* Namespace for Google classes */
#define GOOGLE_NAMESPACE google

/* Define if you have the `dladdr' function */
#define HAVE_DLADDR

/* Define if you have the `snprintf' function */
#define HAVE_SNPRINTF

/* Define to 1 if you have the <dlfcn.h> header file. */
#define HAVE_DLFCN_H

/* Define if you have the `backtrace' function in <execinfo.h> */
#define HAVE_EXECINFO_BACKTRACE

/* Define if you have the `backtrace_symbols' function in <execinfo.h> */
#define HAVE_EXECINFO_BACKTRACE_SYMBOLS

/* Define if you have the `fcntl' function */
#define HAVE_FCNTL

/* Define to 1 if you have the <glob.h> header file. */
#define HAVE_GLOB_H

/* Define to 1 if you have the `pthread' library (-lpthread). */
/* #undef HAVE_LIBPTHREAD */

/* define if you have google gflags library */
/* #undef HAVE_LIB_GFLAGS */

/* define if you have google gmock library */
/* #undef HAVE_LIB_GMOCK */

/* define if you have google gtest library */
/* #undef HAVE_LIB_GTEST */

/* define if you have dbghelp library */
/* #undef HAVE_DBGHELP */

/* define if you have libunwind */
/* #undef HAVE_LIB_UNWIND */

/* Define to 1 if you have the <memory.h> header file. */
#define HAVE_MEMORY_H

/* define to disable multithreading support. */
/* #undef NO_THREADS */

/* Define if you have the 'pread' function */
#define HAVE_PREAD

/* Define if you have POSIX threads libraries and header files. */
#define HAVE_PTHREAD

/* Define to 1 if you have the <pwd.h> header file. */
#define HAVE_PWD_H

/* Define if you have the 'pwrite' function */
#define HAVE_PWRITE

/* define if the compiler implements pthread_rwlock_* */
#define HAVE_RWLOCK

/* Define if you have the 'sigaction' function */
#define HAVE_SIGACTION

/* Define if you have the `sigaltstack' function */
#define HAVE_SIGALTSTACK

/* Define to 1 if you have the <strings.h> header file. */
#define HAVE_STRINGS_H

/* Define to 1 if you have the <syscall.h> header file. */
/* #undef HAVE_SYSCALL_H */

/* Define to 1 if you have the <syslog.h> header file. */
#define HAVE_SYSLOG_H

/* Define to 1 if you have the <sys/stat.h> header file. */
#define HAVE_SYS_STAT_H

/* Define to 1 if you have the <sys/syscall.h> header file. */
#define HAVE_SYS_SYSCALL_H

/* Define to 1 if you have the <sys/time.h> header file. */
#define HAVE_SYS_TIME_H

/* Define to 1 if you have the <sys/types.h> header file. */
#define HAVE_SYS_TYPES_H 1

/* Define to 1 if you have the <sys/ucontext.h> header file. */
/* #undef HAVE_SYS_UCONTEXT_H */

/* Define to 1 if you have the <sys/utsname.h> header file. */
#define HAVE_SYS_UTSNAME_H

/* Define to 1 if you have the <sys/wait.h> header file. */
#define HAVE_SYS_WAIT_H

/* Define to 1 if you have the <ucontext.h> header file. */
/* #undef HAVE_UCONTEXT_H */

/* Define to 1 if you have the <unistd.h> header file. */
#define HAVE_UNISTD_H 1

/* Define if you linking to _Unwind_Backtrace is possible. */
#define HAVE__UNWIND_BACKTRACE

/* Define if you linking to _Unwind_GetIP is possible. */
#define HAVE__UNWIND_GETIP

/* define if your compiler has __attribute__ */
#define HAVE___ATTRIBUTE__

/* define if your compiler has __builtin_expect */
#define HAVE___BUILTIN_EXPECT 1

/* define if your compiler has __sync_val_compare_and_swap */
#define HAVE___SYNC_VAL_COMPARE_AND_SWAP

/* define if symbolize support is available */
/* #undef HAVE_SYMBOLIZE */

/* define if localtime_r is available in time.h */
#define HAVE_LOCALTIME_R

/* define if gmtime_r is available in time.h */
/* #undef HAVE_GMTIME_R */

/* Define to the sub-directory in which libtool stores uninstalled libraries.
   */
/* #undef LT_OBJDIR */

/* How to access the PC from a struct ucontext */
/* #undef PC_FROM_UCONTEXT */

/* define if we should print file offsets in traces instead of symbolizing. */
/* #undef PRINT_UNSYMBOLIZED_STACK_TRACES */

/* Define to necessary symbol if this constant uses a non-standard name on
   your system. */
/* #undef PTHREAD_CREATE_JOINABLE */

/* The size of `void *', as computed by sizeof. */
#define SIZEOF_VOID_P 8

/* location of source code */
#define TEST_SRC_DIR "/Users/qiuwei/Documents/邱威/语音识别-asr/to_alibaba-damo-academy/FunASR/funasr/runtime/onnxruntime/third_party/glog"

/* Define if thread-local storage is enabled. */
#define GLOG_THREAD_LOCAL_STORAGE

#ifdef GLOG_BAZEL_BUILD

/* TODO(rodrigoq): remove this workaround once bazel#3979 is resolved:
 * https://github.com/bazelbuild/bazel/issues/3979 */
#define _START_GOOGLE_NAMESPACE_ namespace GOOGLE_NAMESPACE {

#define _END_GOOGLE_NAMESPACE_ }

#else

/* Stops putting the code inside the Google namespace */
#define _END_GOOGLE_NAMESPACE_ }

/* Puts following code inside the Google namespace */
#define _START_GOOGLE_NAMESPACE_ namespace google {

#endif

/* Replacement for deprecated syscall(SYS_gettid) on macOS. */
#define HAVE_PTHREAD_THREADID_NP 1

#endif  // GLOG_CONFIG_H
