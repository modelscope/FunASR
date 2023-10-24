// base/kaldi-utils.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;
//                      Saarland University;  Karel Vesely;  Yanmin Qian

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#ifndef KALDI_BASE_KALDI_UTILS_H_
#define KALDI_BASE_KALDI_UTILS_H_ 1

#if _MSC_VER
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN 1
#endif
#ifndef NOMINMAX
#define NOMINMAX 1
#endif
#include <windows.h>
#endif

#ifdef _MSC_VER
#include <stdio.h>
#define unlink _unlink
#else
#include <unistd.h>
#endif

#include <limits>
#include <string>

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4056 4305 4800 4267 4996 4756 4661)
#if _MSC_VER < 1400
#define __restrict__
#else
#define __restrict__ __restrict
#endif
#endif

#if defined(_MSC_VER)
#  define KALDI_MEMALIGN(align, size, pp_orig) \
  (*(pp_orig) = _aligned_malloc(size, align))
#  define KALDI_MEMALIGN_FREE(x) _aligned_free(x)
#elif defined(__CYGWIN__)
#  define KALDI_MEMALIGN(align, size, pp_orig) \
  (*(pp_orig) = aligned_alloc(align, size))
#  define KALDI_MEMALIGN_FREE(x) free(x)
#else
#  define KALDI_MEMALIGN(align, size, pp_orig) \
     (!posix_memalign(pp_orig, align, size) ? *(pp_orig) : NULL)
#  define KALDI_MEMALIGN_FREE(x) free(x)
#endif

#ifdef __ICC
#pragma warning(disable: 383)  // ICPC remark we don't want.
#pragma warning(disable: 810)  // ICPC remark we don't want.
#pragma warning(disable: 981)  // ICPC remark we don't want.
#pragma warning(disable: 1418)  // ICPC remark we don't want.
#pragma warning(disable: 444)  // ICPC remark we don't want.
#pragma warning(disable: 869)  // ICPC remark we don't want.
#pragma warning(disable: 1287)  // ICPC remark we don't want.
#pragma warning(disable: 279)  // ICPC remark we don't want.
#pragma warning(disable: 981)  // ICPC remark we don't want.
#endif


namespace kaldi {


// CharToString prints the character in a human-readable form, for debugging.
std::string CharToString(const char &c);


inline int MachineIsLittleEndian() {
  int check = 1;
  return (*reinterpret_cast<char*>(&check) != 0);
}

// This function kaldi::Sleep() provides a portable way
// to sleep for a possibly fractional
// number of seconds.  On Windows it's only accurate to microseconds.
void Sleep(double seconds);
}

#define KALDI_SWAP8(a) do { \
  int t = (reinterpret_cast<char*>(&a))[0];\
          (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[7];\
          (reinterpret_cast<char*>(&a))[7]=t;\
      t = (reinterpret_cast<char*>(&a))[1];\
          (reinterpret_cast<char*>(&a))[1]=(reinterpret_cast<char*>(&a))[6];\
          (reinterpret_cast<char*>(&a))[6]=t;\
      t = (reinterpret_cast<char*>(&a))[2];\
          (reinterpret_cast<char*>(&a))[2]=(reinterpret_cast<char*>(&a))[5];\
          (reinterpret_cast<char*>(&a))[5]=t;\
      t = (reinterpret_cast<char*>(&a))[3];\
          (reinterpret_cast<char*>(&a))[3]=(reinterpret_cast<char*>(&a))[4];\
          (reinterpret_cast<char*>(&a))[4]=t;} while (0)
#define KALDI_SWAP4(a) do { \
  int t = (reinterpret_cast<char*>(&a))[0];\
          (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[3];\
          (reinterpret_cast<char*>(&a))[3]=t;\
      t = (reinterpret_cast<char*>(&a))[1];\
          (reinterpret_cast<char*>(&a))[1]=(reinterpret_cast<char*>(&a))[2];\
          (reinterpret_cast<char*>(&a))[2]=t;} while (0)
#define KALDI_SWAP2(a) do { \
  int t = (reinterpret_cast<char*>(&a))[0];\
          (reinterpret_cast<char*>(&a))[0]=(reinterpret_cast<char*>(&a))[1];\
          (reinterpret_cast<char*>(&a))[1]=t;} while (0)


///\brief Declare deleted copy constructor and copy assignment operator=().
///
/// Use this macro in the \e public part of a class declaration in a header
/// file, next to its constructors, so that uncopyability of the type is
/// clearly readable. Place a semicolon after it.
///\param type The exact enclosing class name.
#define KALDI_DISALLOW_COPY_AND_ASSIGN(type)    \
  type(const type&) = delete;                   \
  type& operator=(const type&) = delete

#if __cplusplus >= 201703L
#define KALDI_COMPILE_TIME_ASSERT static_assert
#else
#define KALDI_COMPILE_TIME_ASSERT(b) static_assert((b), #b)
#endif

#define KALDI_ASSERT_IS_INTEGER_TYPE(I) \
  KALDI_COMPILE_TIME_ASSERT(std::numeric_limits<I>::is_specialized \
                            && std::numeric_limits<I>::is_integer)

#define KALDI_ASSERT_IS_FLOATING_TYPE(F) \
  KALDI_COMPILE_TIME_ASSERT(std::numeric_limits<F>::is_specialized \
                            && !std::numeric_limits<F>::is_integer)

#if defined(_MSC_VER)
#define KALDI_STRCASECMP _stricmp
#elif defined(__CYGWIN__)
#include <strings.h>
#define KALDI_STRCASECMP strcasecmp
#else
#define KALDI_STRCASECMP strcasecmp
#endif
#ifdef _MSC_VER
#  define KALDI_STRTOLL(cur_cstr, end_cstr) _strtoi64(cur_cstr, end_cstr, 10);
#else
#  define KALDI_STRTOLL(cur_cstr, end_cstr) strtoll(cur_cstr, end_cstr, 10);
#endif

#endif  // KALDI_BASE_KALDI_UTILS_H_
