// matrix/kaldi-blas.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
#ifndef KALDI_MATRIX_KALDI_BLAS_H_
#define KALDI_MATRIX_KALDI_BLAS_H_

// This file handles the #includes for BLAS, LAPACK and so on.
// It manipulates the declarations into a common format that kaldi can handle.
// However, the kaldi code will check whether HAVE_ATLAS is defined as that
// code is called a bit differently from CLAPACK that comes from other sources.

// There are three alternatives:
//   (i) you have ATLAS, which includes the ATLAS implementation of CBLAS
//   plus a subset of CLAPACK (but with clapack_ in the function declarations).
//   In this case, define HAVE_ATLAS and make sure the relevant directories are
//   in the include path.

//   (ii) you have CBLAS (some implementation thereof) plus CLAPACK.
//   In this case, define HAVE_CLAPACK.
//   [Since CLAPACK depends on BLAS, the presence of BLAS is implicit].

//  (iii) you have the MKL library, which includes CLAPACK and CBLAS.

// Note that if we are using ATLAS, no Svd implementation is supplied,
// so we define HAVE_Svd to be zero and this directs our implementation to
// supply its own "by hand" implementation which is based on TNT code.




#if (defined(HAVE_CLAPACK) && (defined(HAVE_ATLAS) || defined(HAVE_MKL))) \
    || (defined(HAVE_ATLAS) && defined(HAVE_MKL))
#error "Do not define more than one of HAVE_CLAPACK, HAVE_ATLAS and HAVE_MKL"
#endif

#ifdef HAVE_ATLAS
  extern "C" {
    #include <cblas.h>
    #include <clapack.h>
  }
#elif defined(HAVE_CLAPACK)
  #ifdef __APPLE__
    #ifndef __has_extension
    #define __has_extension(x) 0
    #endif
    #define vImage_Utilities_h
    #define vImage_CVUtilities_h
    #include <Accelerate/Accelerate.h>
    typedef __CLPK_integer          integer;
    typedef __CLPK_logical          logical;
    typedef __CLPK_real             real;
    typedef __CLPK_doublereal       doublereal;
    typedef __CLPK_complex          complex;
    typedef __CLPK_doublecomplex    doublecomplex;
    typedef __CLPK_ftnlen           ftnlen;
  #else
    extern "C" {
      // May be in /usr/[local]/include if installed; else this uses the one
      // from the tools/CLAPACK_include directory.
      #include <cblas.h>
      #include <f2c.h>
      #include <clapack.h>  

      // get rid of macros from f2c.h -- these are dangerous.
      #undef abs
      #undef dabs
      #undef min
      #undef max
      #undef dmin
      #undef dmax
      #undef bit_test
      #undef bit_clear
      #undef bit_set
    }
  #endif
#elif defined(HAVE_MKL)
  extern "C" {
    #include <mkl.h>
  }
#elif defined(HAVE_OPENBLAS)
  // getting cblas.h and lapacke.h from <openblas-install-dir>/.
  // putting in "" not <> to search -I before system libraries.
  #include "cblas.h"
  #include "lapacke.h"
  #undef I
  #undef complex
  // get rid of macros from f2c.h -- these are dangerous.
  #undef abs
  #undef dabs
  #undef min
  #undef max
  #undef dmin
  #undef dmax
  #undef bit_test
  #undef bit_clear
  #undef bit_set
#else
  #error "You need to define (using the preprocessor) either HAVE_CLAPACK or HAVE_ATLAS or HAVE_MKL (but not more than one)"  
#endif

#ifdef HAVE_OPENBLAS
typedef int KaldiBlasInt; // try int.
#endif
#ifdef HAVE_CLAPACK
typedef integer KaldiBlasInt;
#endif
#ifdef HAVE_MKL
typedef MKL_INT KaldiBlasInt;
#endif

#ifdef HAVE_ATLAS
// in this case there is no need for KaldiBlasInt-- this typedef is only needed
// for Svd code which is not included in ATLAS (we re-implement it).
#endif


#endif  // KALDI_MATRIX_KALDI_BLAS_H_
