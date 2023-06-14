// matrix/matrix-common.h

// Copyright 2009-2011  Microsoft Corporation

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
#ifndef KALDI_MATRIX_MATRIX_COMMON_H_
#define KALDI_MATRIX_MATRIX_COMMON_H_

// This file contains some #includes, forward declarations
// and typedefs that are needed by all the main header
// files in this directory.

#include "base/kaldi-common.h"

namespace kaldi {
// this enums equal to CblasTrans and CblasNoTrans constants from CBLAS library
// we are writing them as literals because we don't want to include here matrix/kaldi-blas.h,
// which puts many symbols into global scope (like "real") via the header f2c.h 
typedef enum {
  kTrans    = 112, // = CblasTrans
  kNoTrans  = 111  // = CblasNoTrans
} MatrixTransposeType;

typedef enum {
  kSetZero,
  kUndefined,
  kCopyData
} MatrixResizeType;


typedef enum {
  kDefaultStride,
  kStrideEqualNumCols,
} MatrixStrideType;

typedef enum {
  kTakeLower,
  kTakeUpper,
  kTakeMean,
  kTakeMeanAndCheck
} SpCopyType;

template<typename Real> class VectorBase;
template<typename Real> class Vector;
template<typename Real> class SubVector;
template<typename Real> class MatrixBase;
template<typename Real> class SubMatrix;
template<typename Real> class Matrix;
template<typename Real> class SpMatrix;
template<typename Real> class TpMatrix;
template<typename Real> class PackedMatrix;
template<typename Real> class SparseMatrix;

// these are classes that won't be defined in this
// directory; they're mostly needed for friend declarations.
template<typename Real> class CuMatrixBase;
template<typename Real> class CuSubMatrix;
template<typename Real> class CuMatrix;
template<typename Real> class CuVectorBase;
template<typename Real> class CuSubVector;
template<typename Real> class CuVector;
template<typename Real> class CuPackedMatrix;
template<typename Real> class CuSpMatrix;
template<typename Real> class CuTpMatrix;
template<typename Real> class CuSparseMatrix;

class CompressedMatrix;
class GeneralMatrix;

/// This class provides a way for switching between double and float types.
template<typename T> class OtherReal { };  // useful in reading+writing routines
                                           // to switch double and float.
/// A specialized class for switching from float to double.
template<> class OtherReal<float> {
 public:
  typedef double Real;
};
/// A specialized class for switching from double to float.
template<> class OtherReal<double> {
 public:
  typedef float Real;
};


typedef int32 MatrixIndexT;
typedef int32 SignedMatrixIndexT;
typedef uint32 UnsignedMatrixIndexT;

// If you want to use size_t for the index type, do as follows instead:
//typedef size_t MatrixIndexT;
//typedef ssize_t SignedMatrixIndexT;
//typedef size_t UnsignedMatrixIndexT;

}



#endif  // KALDI_MATRIX_MATRIX_COMMON_H_
