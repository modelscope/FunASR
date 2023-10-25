// base/kaldi-types.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Jan Silovsky;  Yanmin Qian

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

#ifndef KALDI_BASE_KALDI_TYPES_H_
#define KALDI_BASE_KALDI_TYPES_H_ 1

namespace kaldi {
// TYPEDEFS ..................................................................
#if (KALDI_DOUBLEPRECISION != 0)
typedef double  BaseFloat;
#else
typedef float   BaseFloat;
#endif
}

#ifdef _MSC_VER
#include <basetsd.h>
#define ssize_t SSIZE_T
#endif

// we can do this a different way if some platform
// we find in the future lacks stdint.h
#include <stdint.h>

// for discussion on what to do if you need compile kaldi
// without OpenFST, see the bottom of this this file
#include <fst/types.h>

namespace kaldi {
  using ::int16;
  using ::int32;
  using ::int64;
  using ::uint16;
  using ::uint32;
  using ::uint64;
  typedef float   float32;
  typedef double double64;
}  // end namespace kaldi

// In a theoretical case you decide compile Kaldi without the OpenFST
// comment the previous namespace statement and uncomment the following
/*
namespace kaldi {
  typedef int8_t   int8;
  typedef int16_t  int16;
  typedef int32_t  int32;
  typedef int64_t  int64;

  typedef uint8_t  uint8;
  typedef uint16_t uint16;
  typedef uint32_t uint32;
  typedef uint64_t uint64;
  typedef float    float32;
  typedef double   double64;
}  // end namespace kaldi
*/

#endif  // KALDI_BASE_KALDI_TYPES_H_
