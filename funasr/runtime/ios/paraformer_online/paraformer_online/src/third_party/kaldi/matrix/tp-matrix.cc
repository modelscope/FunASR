// matrix/tp-matrix.cc

// Copyright 2009-2011  Ondrej Glembek;  Lukas Burget;  Microsoft Corporation
//                      Saarland University;  Yanmin Qian;   Haihua Xu

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

#include "matrix/tp-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/cblas-wrappers.h"


namespace kaldi {

#ifndef HAVE_ATLAS
template<typename Real>
void TpMatrix<Real>::Invert() {
  // these are CLAPACK types
  KaldiBlasInt result;
  KaldiBlasInt rows = static_cast<int>(this->num_rows_);

  // clapack call
  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  clapack_Xtptri(&rows, this->data_, &result);

  if (result < 0) {
    KALDI_ERR << "Call to CLAPACK stptri_ function failed";
  } else if (result > 0) {
    KALDI_ERR << "Matrix is singular";
  }
}
#else
template<typename Real>
void TpMatrix<Real>::Invert() {
  // ATLAS doesn't implement triangular matrix inversion in packed
  // format, so we temporarily put in non-packed format.
  Matrix<Real> tmp(*this);
  int rows = static_cast<int>(this->num_rows_);
  
  // ATLAS call.  It's really row-major ordering and a lower triangular matrix,
  // but there is some weirdness with Fortran-style indexing that we need to
  // take account of, so everything gets swapped.
  int result = clapack_Xtrtri( rows, tmp.Data(), tmp.Stride());
  // Let's hope ATLAS has the same return value conventions as clapack.
  // I couldn't find any documentation online.
  if (result < 0) {
    KALDI_ERR << "Call to ATLAS strtri function failed";
  } else if (result > 0) {
    KALDI_ERR << "Matrix is singular";
  }
  (*this).CopyFromMat(tmp);
}
#endif

template<typename Real>
Real TpMatrix<Real>::Determinant() {
  double   det = 1.0;
  for (MatrixIndexT i = 0; i<this->NumRows(); i++) {
    det *= (*this)(i, i);
  }
  return static_cast<Real>(det);
}


template<typename Real>
void TpMatrix<Real>::Swap(TpMatrix<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->num_rows_, other->num_rows_);
}


template<typename Real>
void TpMatrix<Real>::Cholesky(const SpMatrix<Real> &orig) {
  KALDI_ASSERT(orig.NumRows() == this->NumRows());
  MatrixIndexT n = this->NumRows();
  this->SetZero();
  Real *data = this->data_, *jdata = data;  // start of j'th row of matrix.
  const Real *orig_jdata = orig.Data(); // start of j'th row of matrix.
  for (MatrixIndexT j = 0; j < n; j++, jdata += j, orig_jdata += j) {
    Real *kdata = data; // start of k'th row of matrix.
    Real d(0.0);
    for (MatrixIndexT k = 0; k < j; k++, kdata += k) {
      Real s = cblas_Xdot(k, kdata, 1, jdata, 1);
      // (*this)(j, k) = s = (orig(j, k) - s)/(*this)(k, k);
      jdata[k] = s = (orig_jdata[k] - s)/kdata[k];
      d = d + s*s;
    }
    // d = orig(j, j) - d;
    d = orig_jdata[j] - d;
    
    if (d >= 0.0) {
      // (*this)(j, j) = std::sqrt(d);
      jdata[j] = std::sqrt(d);
    } else {
      KALDI_WARN << "Cholesky decomposition failed. Maybe matrix "
          "is not positive definite. Throwing error";
      throw std::runtime_error("Cholesky decomposition failed.");
    }
  }
}

template<typename Real>
void TpMatrix<Real>::CopyFromMat(const MatrixBase<Real> &M,
                                 MatrixTransposeType Trans) {
  if (Trans == kNoTrans) {
    KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
    MatrixIndexT D = this->NumRows();
    const Real *in_i = M.Data();
    MatrixIndexT stride = M.Stride();
    Real *out_i = this->data_;
    for (MatrixIndexT i = 0; i < D; i++, in_i += stride, out_i += i)
      for (MatrixIndexT j = 0; j <= i; j++)
        out_i[j] = in_i[j];
  } else {
    KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
    MatrixIndexT D = this->NumRows();
    const Real *in_i = M.Data();
    MatrixIndexT stride = M.Stride();
    Real *out_i = this->data_;
    for (MatrixIndexT i = 0; i < D; i++, in_i++, out_i += i) {
      for (MatrixIndexT j = 0; j <= i; j++)
        out_i[j] = in_i[stride*j];
    }
  }
}


template class TpMatrix<float>;
template class TpMatrix<double>;

}  // namespace kaldi


