// matrix/sp-matrix.cc

// Copyright 2009-2011  Lukas Burget;  Ondrej Glembek;  Microsoft Corporation
//                      Saarland University;   Petr Schwarz;   Yanmin Qian;
//                      Haihua Xu

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

#include <limits>

#include "matrix/sp-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/matrix-functions.h"
#include "matrix/cblas-wrappers.h"

namespace kaldi {

// ****************************************************************************
// Returns the log-determinant if +ve definite, else KALDI_ERR.
// ****************************************************************************
template<typename Real>
Real SpMatrix<Real>::LogPosDefDet() const {
  TpMatrix<Real> chol(this->NumRows());
  double det = 0.0;
  double diag;
  chol.Cholesky(*this);  // Will throw exception if not +ve definite!

  for (MatrixIndexT i = 0; i < this->NumRows(); i++) {
    diag = static_cast<double>(chol(i, i));
    det += kaldi::Log(diag);
  }
  return static_cast<Real>(2*det);
}


template<typename Real>
void SpMatrix<Real>::Swap(SpMatrix<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->num_rows_, other->num_rows_);
}

template<typename Real>
void SpMatrix<Real>::SymPosSemiDefEig(VectorBase<Real> *s,
                                      MatrixBase<Real> *P,
                                      Real tolerance) const {
  Eig(s, P);
  Real max = s->Max(), min = s->Min();
  KALDI_ASSERT(-min <= tolerance * max);
  s->ApplyFloor(0.0);
}

template<typename Real>
Real SpMatrix<Real>::MaxAbsEig() const {
  Vector<Real> s(this->NumRows());
  this->Eig(&s, static_cast<MatrixBase<Real>*>(NULL));
  return std::max(s.Max(), -s.Min());
}

// returns true if positive definite--uses cholesky.
template<typename Real>
bool SpMatrix<Real>::IsPosDef() const {
  MatrixIndexT D = (*this).NumRows();
  KALDI_ASSERT(D > 0);
  try {
    TpMatrix<Real> C(D);
    C.Cholesky(*this);
    for (MatrixIndexT r = 0; r < D; r++)
      if (C(r, r) == 0.0) return false;
    return true;
  }
  catch(...) {  // not positive semidefinite.
    return false;
  }
}

template<typename Real>
void SpMatrix<Real>::ApplyPow(Real power) {
  if (power == 1) return;  // can do nothing.
  MatrixIndexT D = this->NumRows();
  KALDI_ASSERT(D > 0);
  Matrix<Real> U(D, D);
  Vector<Real> l(D);
  (*this).SymPosSemiDefEig(&l, &U);

  Vector<Real> l_copy(l);
  try {
    l.ApplyPow(power * 0.5);
  }
  catch(...) {
    KALDI_ERR << "Error taking power " << (power * 0.5) << " of vector "
              << l_copy;
  }
  U.MulColsVec(l);
  (*this).AddMat2(1.0, U, kNoTrans, 0.0);
}

template<typename Real>
void SpMatrix<Real>::CopyFromMat(const MatrixBase<Real> &M,
                                 SpCopyType copy_type) {
  KALDI_ASSERT(this->NumRows() == M.NumRows() && M.NumRows() == M.NumCols());
  MatrixIndexT D = this->NumRows();

  switch (copy_type) {
    case kTakeMeanAndCheck:
      {
        Real good_sum = 0.0, bad_sum = 0.0;
        for (MatrixIndexT i = 0; i < D; i++) {
          for (MatrixIndexT j = 0; j < i; j++) {
            Real a = M(i, j), b = M(j, i), avg = 0.5*(a+b), diff = 0.5*(a-b);
            (*this)(i, j) = avg;
            good_sum += std::abs(avg);
            bad_sum += std::abs(diff);
          }
          good_sum += std::abs(M(i, i));
          (*this)(i, i) = M(i, i);
        }
        if (bad_sum > 0.01 * good_sum) {
          KALDI_ERR << "SpMatrix::Copy(), source matrix is not symmetric: "
                    << bad_sum <<  ">" << good_sum;
        }
        break;
      }
    case kTakeMean:
      {
        for (MatrixIndexT i = 0; i < D; i++) {
          for (MatrixIndexT j = 0; j < i; j++) {
            (*this)(i, j) = 0.5*(M(i, j) + M(j, i));
          }
          (*this)(i, i) = M(i, i);
        }
        break;
      }
    case kTakeLower:
      { // making this one a bit more efficient.
        const Real *src = M.Data();
        Real *dest = this->data_;
        MatrixIndexT stride = M.Stride();
        for (MatrixIndexT i = 0; i < D; i++) {
          for (MatrixIndexT j = 0; j <= i; j++)
            dest[j] = src[j];
          dest += i + 1;
          src += stride;
        }
      }
      break;
    case kTakeUpper:
      for (MatrixIndexT i = 0; i < D; i++)
        for (MatrixIndexT j = 0; j <= i; j++)
          (*this)(i, j) = M(j, i);
      break;
    default:
      KALDI_ASSERT("Invalid argument to SpMatrix::CopyFromMat");
  }
}

template<typename Real>
Real SpMatrix<Real>::Trace() const {
  const Real *data = this->data_;
  MatrixIndexT num_rows = this->num_rows_;
  Real ans = 0.0;
  for (int32 i = 1; i <= num_rows; i++, data += i)
    ans += *data;
  return ans;
}

// diagonal update, this <-- this + diag(v)
template<typename Real>
template<typename OtherReal>
void  SpMatrix<Real>::AddDiagVec(const Real alpha, const VectorBase<OtherReal> &v) {
  int32 num_rows = this->num_rows_;
  KALDI_ASSERT(num_rows == v.Dim() && num_rows > 0);
  const OtherReal *src = v.Data();
  Real *dst = this->data_;
  if (alpha == 1.0)
    for (int32 i = 1; i <= num_rows; i++, src++, dst += i)
      *dst += *src;
  else
    for (int32 i = 1; i <= num_rows; i++, src++, dst += i)
      *dst += alpha * *src;
}

// instantiate the template above.
template
void SpMatrix<float>::AddDiagVec(const float alpha,
                                 const VectorBase<double> &v);

template
void SpMatrix<double>::AddDiagVec(const double alpha,
                                  const VectorBase<float> &v);

template
void SpMatrix<float>::AddDiagVec(const float alpha,
                                 const VectorBase<float> &v);

template
void SpMatrix<double>::AddDiagVec(const double alpha,
                                  const VectorBase<double> &v);

template<>
template<>
void SpMatrix<double>::AddVec2(const double alpha, const VectorBase<double> &v);

#ifndef HAVE_ATLAS
template<typename Real>
void SpMatrix<Real>::Invert(Real *logdet, Real *det_sign, bool need_inverse) {
  // these are CLAPACK types
  KaldiBlasInt   result;
  KaldiBlasInt   rows = static_cast<int>(this->num_rows_);
  KaldiBlasInt*  p_ipiv = new KaldiBlasInt[rows];
  Real *p_work;  // workspace for the lapack function
  void *temp;
  if ((p_work = static_cast<Real*>(
          KALDI_MEMALIGN(16, sizeof(Real) * rows, &temp))) == NULL) {
    delete[] p_ipiv;
    throw std::bad_alloc();
  }
#ifdef HAVE_OPENBLAS
  memset(p_work, 0, sizeof(Real) * rows); // gets rid of a probably
  // spurious Valgrind warning about jumps depending upon uninitialized values.
#endif


  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  clapack_Xsptrf(&rows, this->data_, p_ipiv, &result);


  KALDI_ASSERT(result >= 0 && "Call to CLAPACK ssptrf_ called with wrong arguments");

  if (result > 0) {  // Singular...
    if (det_sign) *det_sign = 0;
    if (logdet) *logdet = -std::numeric_limits<Real>::infinity();
    if (need_inverse) KALDI_ERR << "CLAPACK stptrf_ : factorization failed";
  } else {  // Not singular.. compute log-determinant if needed.
    if (logdet != NULL || det_sign != NULL) {
      Real prod = 1.0, log_prod = 0.0;
      int sign = 1;
      for (int i = 0; i < (int)this->num_rows_; i++) {
        if (p_ipiv[i] > 0) {  // not a 2x2 block...
          // if (p_ipiv[i] != i+1) sign *= -1;  // row swap.
          Real diag = (*this)(i, i);
          prod *= diag;
        } else {  // negative: 2x2 block. [we are in first of the two].
          i++;  // skip over the first of the pair.
          // each 2x2 block...
          Real diag1 = (*this)(i, i), diag2 = (*this)(i-1, i-1),
              offdiag = (*this)(i, i-1);
          Real thisdet = diag1*diag2 - offdiag*offdiag;
          // thisdet == determinant of 2x2 block.
          // The following line is more complex than it looks: there are 2 offsets of
          // 1 that cancel.
          prod *= thisdet;
        }
        if (i == (int)(this->num_rows_-1) || fabs(prod) < 1.0e-10 || fabs(prod) > 1.0e+10) {
          if (prod < 0) { prod = -prod; sign *= -1; }
          log_prod += kaldi::Log(std::abs(prod));
          prod = 1.0;
        }
      }
      if (logdet != NULL) *logdet = log_prod;
      if (det_sign != NULL) *det_sign = sign;
    }
  }
  if (!need_inverse) {
    delete [] p_ipiv;
    KALDI_MEMALIGN_FREE(p_work);
    return;  // Don't need what is computed next.
  }
  // NOTE: Even though "U" is for upper, lapack assumes column-wise storage
  // of the data. We have a row-wise storage, therefore, we need to "invert"
  clapack_Xsptri(&rows, this->data_, p_ipiv, p_work, &result);

  KALDI_ASSERT(result >=0 &&
               "Call to CLAPACK ssptri_ called with wrong arguments");

  if (result != 0) {
    KALDI_ERR << "CLAPACK ssptrf_ : Matrix is singular";
  }

  delete [] p_ipiv;
  KALDI_MEMALIGN_FREE(p_work);
}
#else
// in the ATLAS case, these are not implemented using a library and we back off to something else.
template<typename Real>
void SpMatrix<Real>::Invert(Real *logdet, Real *det_sign, bool need_inverse) {
  Matrix<Real> M(this->NumRows(), this->NumCols());
  M.CopyFromSp(*this);
  M.Invert(logdet, det_sign, need_inverse);
  if (need_inverse)
    for (MatrixIndexT i = 0; i < this->NumRows(); i++)
      for (MatrixIndexT j = 0; j <= i; j++)
        (*this)(i, j) = M(i, j);
}
#endif

template<typename Real>
void SpMatrix<Real>::InvertDouble(Real *logdet, Real *det_sign,
                                  bool inverse_needed) {
  SpMatrix<double> dmat(*this);
  double logdet_tmp, det_sign_tmp;
  dmat.Invert(logdet ? &logdet_tmp : NULL,
              det_sign ? &det_sign_tmp : NULL,
              inverse_needed);
  if (logdet) *logdet = logdet_tmp;
  if (det_sign) *det_sign = det_sign_tmp;
  (*this).CopyFromSp(dmat);
}



double TraceSpSp(const SpMatrix<double> &A, const SpMatrix<double> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  const double *Aptr = A.Data();
  const double *Bptr = B.Data();
  MatrixIndexT R = A.NumRows();
  MatrixIndexT RR = (R * (R + 1)) / 2;
  double all_twice = 2.0 * cblas_Xdot(RR, Aptr, 1, Bptr, 1);
  // "all_twice" contains twice the vector-wise dot-product... this is
  // what we want except the diagonal elements are represented
  // twice.
  double diag_once = 0.0;
  for (MatrixIndexT row_plus_two = 2; row_plus_two <= R + 1; row_plus_two++) {
    diag_once += *Aptr * *Bptr;
    Aptr += row_plus_two;
    Bptr += row_plus_two;
  }
  return all_twice - diag_once;
}


float TraceSpSp(const SpMatrix<float> &A, const SpMatrix<float> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  const float *Aptr = A.Data();
  const float *Bptr = B.Data();
  MatrixIndexT R = A.NumRows();
  MatrixIndexT RR = (R * (R + 1)) / 2;
  float all_twice = 2.0 * cblas_Xdot(RR, Aptr, 1, Bptr, 1);
  // "all_twice" contains twice the vector-wise dot-product... this is
  // what we want except the diagonal elements are represented
  // twice.
  float diag_once = 0.0;
  for (MatrixIndexT row_plus_two = 2; row_plus_two <= R + 1; row_plus_two++) {
    diag_once += *Aptr * *Bptr;
    Aptr += row_plus_two;
    Bptr += row_plus_two;
  }
  return all_twice - diag_once;
}


template<typename Real, typename OtherReal>
Real TraceSpSp(const SpMatrix<Real> &A, const SpMatrix<OtherReal> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows());
  Real ans = 0.0;
  const Real *Aptr = A.Data();
  const OtherReal *Bptr = B.Data();
  MatrixIndexT row, col, R = A.NumRows();
  for (row = 0; row < R; row++) {
    for (col = 0; col < row; col++)
      ans += 2.0 * *(Aptr++) * *(Bptr++);
    ans += *(Aptr++) * *(Bptr++);  // Diagonal.
  }
  return ans;
}

template
float TraceSpSp<float, double>(const SpMatrix<float> &A, const SpMatrix<double> &B);

template
double TraceSpSp<double, float>(const SpMatrix<double> &A, const SpMatrix<float> &B);


template<typename Real>
Real TraceSpMat(const SpMatrix<Real> &A, const MatrixBase<Real> &B) {
  KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols() &&
               "KALDI_ERR: TraceSpMat: arguments have mismatched dimension");
  MatrixIndexT R = A.NumRows();
  Real ans = (Real)0.0;
  const Real *Aptr = A.Data(), *Bptr = B.Data();
  MatrixIndexT bStride = B.Stride();
  for (MatrixIndexT r = 0;r < R;r++) {
    for (MatrixIndexT c = 0;c < r;c++) {
      // ans += A(r, c) * (B(r, c) + B(c, r));
      ans += *(Aptr++) * (Bptr[r*bStride + c] + Bptr[c*bStride + r]);
    }
    // ans += A(r, r) * B(r, r);
    ans += *(Aptr++) * Bptr[r*bStride + r];
  }
  return ans;
}

template
float TraceSpMat(const SpMatrix<float> &A, const MatrixBase<float> &B);

template
double TraceSpMat(const SpMatrix<double> &A, const MatrixBase<double> &B);


template<typename Real>
Real TraceMatSpMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                   const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                   MatrixTransposeType transC) {
  KALDI_ASSERT((transA == kTrans?A.NumCols():A.NumRows()) ==
               (transC == kTrans?C.NumRows():C.NumCols()) &&
               (transA == kTrans?A.NumRows():A.NumCols()) == B.NumRows() &&
               (transC == kTrans?C.NumCols():C.NumRows()) == B.NumRows() &&
               "TraceMatSpMat: arguments have wrong dimension.");
  Matrix<Real> tmp(B.NumRows(), B.NumRows());
  tmp.AddMatMat(1.0, C, transC, A, transA, 0.0);  // tmp = C * A.
  return TraceSpMat(B, tmp);
}

template
float TraceMatSpMat(const MatrixBase<float> &A, MatrixTransposeType transA,
                    const SpMatrix<float> &B, const MatrixBase<float> &C,
                    MatrixTransposeType transC);
template
double TraceMatSpMat(const MatrixBase<double> &A, MatrixTransposeType transA,
                     const SpMatrix<double> &B, const MatrixBase<double> &C,
                     MatrixTransposeType transC);

template<typename Real>
Real TraceMatSpMatSp(const MatrixBase<Real> &A, MatrixTransposeType transA,
                     const SpMatrix<Real> &B, const MatrixBase<Real> &C,
                     MatrixTransposeType transC, const SpMatrix<Real> &D) {
  KALDI_ASSERT((transA == kTrans ?A.NumCols():A.NumRows() == D.NumCols()) &&
               (transA == kTrans ? A.NumRows():A.NumCols() == B.NumRows()) &&
               (transC == kTrans ? A.NumCols():A.NumRows() == B.NumCols()) &&
               (transC == kTrans ? A.NumRows():A.NumCols() == D.NumRows()) &&
               "KALDI_ERR: TraceMatSpMatSp: arguments have mismatched dimension.");
  // Could perhaps optimize this more depending on dimensions of quantities.
  Matrix<Real> tmpAB(transA == kTrans ? A.NumCols():A.NumRows(), B.NumCols());
  tmpAB.AddMatSp(1.0, A, transA, B, 0.0);
  Matrix<Real> tmpCD(transC == kTrans ? C.NumCols():C.NumRows(), D.NumCols());
  tmpCD.AddMatSp(1.0, C, transC, D, 0.0);
  return TraceMatMat(tmpAB, tmpCD, kNoTrans);
}

template
float TraceMatSpMatSp(const MatrixBase<float> &A, MatrixTransposeType transA,
                      const SpMatrix<float> &B, const MatrixBase<float> &C,
                      MatrixTransposeType transC, const SpMatrix<float> &D);
template
double TraceMatSpMatSp(const MatrixBase<double> &A, MatrixTransposeType transA,
                       const SpMatrix<double> &B, const MatrixBase<double> &C,
                       MatrixTransposeType transC, const SpMatrix<double> &D);


template<typename Real>
bool SpMatrix<Real>::IsDiagonal(Real cutoff) const {
  MatrixIndexT R = this->NumRows();
  Real bad_sum = 0.0, good_sum = 0.0;
  for (MatrixIndexT i = 0; i < R; i++) {
    for (MatrixIndexT j = 0; j <= i; j++) {
      if (i == j)
        good_sum += std::abs((*this)(i, j));
      else
        bad_sum += std::abs((*this)(i, j));
    }
  }
  return (!(bad_sum > good_sum * cutoff));
}

template<typename Real>
bool SpMatrix<Real>::IsUnit(Real cutoff) const {
  MatrixIndexT R = this->NumRows();
  Real max = 0.0;  // max error
  for (MatrixIndexT i = 0; i < R; i++)
    for (MatrixIndexT j = 0; j <= i; j++)
      max = std::max(max, static_cast<Real>(std::abs((*this)(i, j) -
                                                     (i == j ? 1.0 : 0.0))));
  return (max <= cutoff);
}

template<typename Real>
bool SpMatrix<Real>::IsTridiagonal(Real cutoff) const {
  MatrixIndexT R = this->NumRows();
  Real max_abs_2diag = 0.0, max_abs_offdiag = 0.0;
  for (MatrixIndexT i = 0; i < R; i++)
    for (MatrixIndexT j = 0; j <= i; j++) {
      if (j+1 < i)
        max_abs_offdiag = std::max(max_abs_offdiag,
                                   std::abs((*this)(i, j)));
      else
        max_abs_2diag = std::max(max_abs_2diag,
                                 std::abs((*this)(i, j)));
    }
  return (max_abs_offdiag <= cutoff * max_abs_2diag);
}

template<typename Real>
bool SpMatrix<Real>::IsZero(Real cutoff) const {
  if (this->num_rows_ == 0) return true;
  return (this->Max() <= cutoff && this->Min() >= -cutoff);
}

template<typename Real>
Real SpMatrix<Real>::FrobeniusNorm() const {
  Real sum = 0.0;
  MatrixIndexT R = this->NumRows();
  for (MatrixIndexT i = 0; i < R; i++) {
    for (MatrixIndexT j = 0; j < i; j++)
      sum += (*this)(i, j) * (*this)(i, j) * 2;
    sum += (*this)(i, i) * (*this)(i, i);
  }
  return std::sqrt(sum);
}

template<typename Real>
bool SpMatrix<Real>::ApproxEqual(const SpMatrix<Real> &other, float tol) const {
  if (this->NumRows() != other.NumRows())
    KALDI_ERR << "SpMatrix::AproxEqual, size mismatch, "
              << this->NumRows() << " vs. " << other.NumRows();
  SpMatrix<Real> tmp(*this);
  tmp.AddSp(-1.0, other);
  return (tmp.FrobeniusNorm() <= tol * std::max(this->FrobeniusNorm(), other.FrobeniusNorm()));
}

// function Floor: A = Floor(B, alpha * C) ... see tutorial document.
template<typename Real>
int SpMatrix<Real>::ApplyFloor(const SpMatrix<Real> &C, Real alpha,
                               bool verbose) {
  MatrixIndexT dim = this->NumRows();
  int nfloored = 0;
  KALDI_ASSERT(C.NumRows() == dim);
  KALDI_ASSERT(alpha > 0);
  TpMatrix<Real> L(dim);
  L.Cholesky(C);
  L.Scale(std::sqrt(alpha));  // equivalent to scaling C by alpha.
  TpMatrix<Real> LInv(L);
  LInv.Invert();

  SpMatrix<Real> D(dim);
  {  // D = L^{-1} * (*this) * L^{-T}
    Matrix<Real> LInvFull(LInv);
    D.AddMat2Sp(1.0, LInvFull, kNoTrans, (*this), 0.0);
  }

  Vector<Real> l(dim);
  Matrix<Real> U(dim, dim);

  D.Eig(&l, &U);

  if (verbose) {
    KALDI_LOG << "ApplyFloor: flooring following diagonal to 1: " << l;
  }
  for (MatrixIndexT i = 0; i < l.Dim(); i++) {
    if (l(i) < 1.0) {
      nfloored++;
      l(i) = 1.0;
    }
  }
  l.ApplyPow(0.5);
  U.MulColsVec(l);
  D.AddMat2(1.0, U, kNoTrans, 0.0);
  {  // D' := U * diag(l') * U^T ... l'=floor(l, 1)
    Matrix<Real> LFull(L);
    (*this).AddMat2Sp(1.0, LFull, kNoTrans, D, 0.0);  // A := L * D' * L^T
  }
  return nfloored;
}

template<typename Real>
Real SpMatrix<Real>::LogDet(Real *det_sign) const {
  Real log_det;
  SpMatrix<Real> tmp(*this);
  // false== output not needed (saves some computation).
  tmp.Invert(&log_det, det_sign, false);
  return log_det;
}


template<typename Real>
int SpMatrix<Real>::ApplyFloor(Real floor) {
  MatrixIndexT Dim = this->NumRows();
  int nfloored = 0;
  Vector<Real> s(Dim);
  Matrix<Real> P(Dim, Dim);
  (*this).Eig(&s, &P);
  for (MatrixIndexT i = 0; i < Dim; i++) {
    if (s(i) < floor) {
      nfloored++;
      s(i) = floor;
    }
  }
  (*this).AddMat2Vec(1.0, P, kNoTrans, s, 0.0);
  return nfloored;
}

template<typename Real>
MatrixIndexT SpMatrix<Real>::LimitCond(Real maxCond, bool invert) {  // e.g. maxCond = 1.0e+05.
  MatrixIndexT Dim = this->NumRows();
  Vector<Real> s(Dim);
  Matrix<Real> P(Dim, Dim);
  (*this).SymPosSemiDefEig(&s, &P);
  KALDI_ASSERT(maxCond > 1);
  Real floor = s.Max() / maxCond;
  if (floor < 0) floor = 0;
  if (floor < 1.0e-40) {
    KALDI_WARN << "LimitCond: limiting " << floor << " to 1.0e-40";
    floor = 1.0e-40;
  }
  MatrixIndexT nfloored = 0;
  for (MatrixIndexT i = 0; i < Dim; i++) {
    if (s(i) <= floor) nfloored++;
    if (invert)
      s(i) = 1.0 / std::sqrt(std::max(s(i), floor));
    else
      s(i) = std::sqrt(std::max(s(i), floor));
  }
  P.MulColsVec(s);
  (*this).AddMat2(1.0, P, kNoTrans, 0.0);  // (*this) = P*P^T.  ... (*this) = P * floor(s) * P^T  ... if P was original P.
  return nfloored;
}

void SolverOptions::Check() const {
  KALDI_ASSERT(K>10 && eps<1.0e-10);
}

template<> double SolveQuadraticProblem(const SpMatrix<double> &H,
                                        const VectorBase<double> &g,
                                        const SolverOptions &opts,
                                        VectorBase<double> *x) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  opts.Check();
  MatrixIndexT dim = x->Dim();
  if (H.IsZero(0.0)) {
    KALDI_WARN << "Zero quadratic term in quadratic vector problem for "
               << opts.name << ": leaving it unchanged.";
    return 0.0;
  }
  if (opts.diagonal_precondition) {
    // We can re-cast the problem with a diagonal preconditioner to
    // make H better-conditioned.
    Vector<double> H_diag(dim);
    H_diag.CopyDiagFromSp(H);
    H_diag.ApplyFloor(std::numeric_limits<double>::min() * 1.0E+3);
    Vector<double> H_diag_sqrt(H_diag);
    H_diag_sqrt.ApplyPow(0.5);
    Vector<double> H_diag_inv_sqrt(H_diag_sqrt);
    H_diag_inv_sqrt.InvertElements();
    Vector<double> x_scaled(*x);
    x_scaled.MulElements(H_diag_sqrt);
    Vector<double> g_scaled(g);
    g_scaled.MulElements(H_diag_inv_sqrt);
    SpMatrix<double> H_scaled(dim);
    H_scaled.AddVec2Sp(1.0, H_diag_inv_sqrt, H, 0.0);
    double ans;
    SolverOptions new_opts(opts);
    new_opts.diagonal_precondition = false;
    ans = SolveQuadraticProblem(H_scaled, g_scaled, new_opts, &x_scaled);
    x->CopyFromVec(x_scaled);
    x->MulElements(H_diag_inv_sqrt);
    return ans;
  }
  Vector<double> gbar(g);
  if (opts.optimize_delta) gbar.AddSpVec(-1.0, H, *x, 1.0);  // gbar = g - H x
  Matrix<double> U(dim, dim);
  Vector<double> l(dim);
  H.SymPosSemiDefEig(&l, &U);  // does svd H = U L V^T and checks that H == U L U^T to within a tolerance.
  // floor l.
  double f = std::max(static_cast<double>(opts.eps), l.Max() / opts.K);
  MatrixIndexT nfloored = 0;
  for (MatrixIndexT i = 0; i < dim; i++) {  // floor l.
    if (l(i) < f) {
      nfloored++;
      l(i) = f;
    }
  }
  if (nfloored != 0 && opts.print_debug_output) {
    KALDI_LOG << "Solving quadratic problem for " << opts.name
              << ": floored " << nfloored<< " eigenvalues. ";
  }
  Vector<double> tmp(dim);
  tmp.AddMatVec(1.0, U, kTrans, gbar, 0.0);  // tmp = U^T \bar{g}
  tmp.DivElements(l);  // divide each element of tmp by l: tmp = \tilde{L}^{-1} U^T \bar{g}
  Vector<double> delta(dim);
  delta.AddMatVec(1.0, U, kNoTrans, tmp, 0.0);  // delta = U tmp = U \tilde{L}^{-1} U^T \bar{g}
  Vector<double> &xhat(tmp);
  xhat.CopyFromVec(delta);
  if (opts.optimize_delta) xhat.AddVec(1.0, *x);  // xhat = x + delta.
  double auxf_before = VecVec(g, *x) - 0.5 * VecSpVec(*x, H, *x),
         auxf_after = VecVec(g, xhat) - 0.5 * VecSpVec(xhat, H, xhat);
  if (auxf_after < auxf_before) {  // Reject change.
    if (auxf_after < auxf_before - 1.0e-10 && opts.print_debug_output)
      KALDI_WARN << "Optimizing vector auxiliary function for "
                 << opts.name<< ": auxf decreased " << auxf_before
                 << " to "  << auxf_after <<  ", change is "
                 << (auxf_after-auxf_before);
    return 0.0;
  } else {
    x->CopyFromVec(xhat);
    return auxf_after - auxf_before;
  }
}

template<> float SolveQuadraticProblem(const SpMatrix<float> &H,
                                       const VectorBase<float> &g,
                                       const SolverOptions &opts,
                                       VectorBase<float> *x) {
  KALDI_ASSERT(H.NumRows() == g.Dim() && g.Dim() == x->Dim() && x->Dim() != 0);
  SpMatrix<double> Hd(H);
  Vector<double> gd(g);
  Vector<double> xd(*x);
  float ans = static_cast<float>(SolveQuadraticProblem(Hd, gd, opts, &xd));
  x->CopyFromVec(xd);
  return ans;
}

// Maximizes the auxiliary function   Q(x) = tr(M^T SigmaInv Y) - 0.5 tr(SigmaInv M Q M^T).
// Like a numerically stable version of   M := Y Q^{-1}.
template<typename Real>
Real
SolveQuadraticMatrixProblem(const SpMatrix<Real> &Q,
                            const MatrixBase<Real> &Y,
                            const SpMatrix<Real> &SigmaInv,
                            const SolverOptions &opts,
                            MatrixBase<Real> *M) {
  KALDI_ASSERT(Q.NumRows() == M->NumCols() &&
               SigmaInv.NumRows() == M->NumRows() && Y.NumRows() == M->NumRows()
               && Y.NumCols() == M->NumCols() && M->NumCols() != 0);
  opts.Check();
  MatrixIndexT rows = M->NumRows(), cols = M->NumCols();
  if (Q.IsZero(0.0)) {
    KALDI_WARN << "Zero quadratic term in quadratic matrix problem for "
               << opts.name << ": leaving it unchanged.";
    return 0.0;
  }

  if (opts.diagonal_precondition) {
    // We can re-cast the problem with a diagonal preconditioner in the space
    // of Q (columns of M).  Helps to improve the condition of Q.
    Vector<Real> Q_diag(cols);
    Q_diag.CopyDiagFromSp(Q);
    Q_diag.ApplyFloor(std::numeric_limits<Real>::min() * 1.0E+3);
    Vector<Real> Q_diag_sqrt(Q_diag);
    Q_diag_sqrt.ApplyPow(0.5);
    Vector<Real> Q_diag_inv_sqrt(Q_diag_sqrt);
    Q_diag_inv_sqrt.InvertElements();
    Matrix<Real> M_scaled(*M);
    M_scaled.MulColsVec(Q_diag_sqrt);
    Matrix<Real> Y_scaled(Y);
    Y_scaled.MulColsVec(Q_diag_inv_sqrt);
    SpMatrix<Real> Q_scaled(cols);
    Q_scaled.AddVec2Sp(1.0, Q_diag_inv_sqrt, Q, 0.0);
    Real ans;
    SolverOptions new_opts(opts);
    new_opts.diagonal_precondition = false;
    ans = SolveQuadraticMatrixProblem(Q_scaled, Y_scaled, SigmaInv,
                                      new_opts, &M_scaled);
    M->CopyFromMat(M_scaled);
    M->MulColsVec(Q_diag_inv_sqrt);
    return ans;
  }

  Matrix<Real> Ybar(Y);
  if (opts.optimize_delta) {
    Matrix<Real> Qfull(Q);
    Ybar.AddMatMat(-1.0, *M, kNoTrans, Qfull, kNoTrans, 1.0);
  } // Ybar = Y - M Q.
  Matrix<Real> U(cols, cols);
  Vector<Real> l(cols);
  Q.SymPosSemiDefEig(&l, &U);  // does svd Q = U L V^T and checks that Q == U L U^T to within a tolerance.
  // floor l.
  Real f = std::max<Real>(static_cast<Real>(opts.eps), l.Max() / opts.K);
  MatrixIndexT nfloored = 0;
  for (MatrixIndexT i = 0; i < cols; i++) {  // floor l.
    if (l(i) < f) { nfloored++; l(i) = f; }
  }
  if (nfloored != 0 && opts.print_debug_output)
    KALDI_LOG << "Solving matrix problem for " << opts.name
              << ": floored " << nfloored << " eigenvalues. ";
  Matrix<Real> tmpDelta(rows, cols);
  tmpDelta.AddMatMat(1.0, Ybar, kNoTrans, U, kNoTrans, 0.0);  // tmpDelta = Ybar * U.
  l.InvertElements(); KALDI_ASSERT(1.0/l.Max() != 0);  // check not infinite.  eps should take care of this.
  tmpDelta.MulColsVec(l);  // tmpDelta = Ybar * U * \tilde{L}^{-1}

  Matrix<Real> Delta(rows, cols);
  Delta.AddMatMat(1.0, tmpDelta, kNoTrans, U, kTrans, 0.0);  // Delta = Ybar * U * \tilde{L}^{-1} * U^T

  Real auxf_before, auxf_after;
  SpMatrix<Real> MQM(rows);
  Matrix<Real> &SigmaInvY(tmpDelta);
  { Matrix<Real> SigmaInvFull(SigmaInv);  SigmaInvY.AddMatMat(1.0, SigmaInvFull, kNoTrans, Y, kNoTrans, 0.0); }
  {  // get auxf_before.      Q(x) = tr(M^T SigmaInv Y) - 0.5 tr(SigmaInv M Q M^T).
    MQM.AddMat2Sp(1.0, *M, kNoTrans, Q, 0.0);
    auxf_before = TraceMatMat(*M, SigmaInvY, kaldi::kTrans) - 0.5*TraceSpSp(SigmaInv, MQM);
  }

  Matrix<Real> Mhat(Delta);
  if (opts.optimize_delta) Mhat.AddMat(1.0, *M);  // Mhat = Delta + M.

  {  // get auxf_after.
    MQM.AddMat2Sp(1.0, Mhat, kNoTrans, Q, 0.0);
    auxf_after = TraceMatMat(Mhat, SigmaInvY, kaldi::kTrans) - 0.5*TraceSpSp(SigmaInv, MQM);
  }

  if (auxf_after < auxf_before) {
    if (auxf_after < auxf_before - 1.0e-10)
      KALDI_WARN << "Optimizing matrix auxiliary function for "
                 << opts.name << ", auxf decreased "
                 << auxf_before << " to " << auxf_after << ", change is "
                 << (auxf_after-auxf_before);
    return 0.0;
  } else {
    M->CopyFromMat(Mhat);
    return auxf_after - auxf_before;
  }
}

template<typename Real>
Real SolveDoubleQuadraticMatrixProblem(const MatrixBase<Real> &G,
                                       const SpMatrix<Real> &P1,
                                       const SpMatrix<Real> &P2,
                                       const SpMatrix<Real> &Q1,
                                       const SpMatrix<Real> &Q2,
                                       const SolverOptions &opts,
                                       MatrixBase<Real> *M) {
  KALDI_ASSERT(Q1.NumRows() == M->NumCols() && P1.NumRows() == M->NumRows() &&
               G.NumRows() == M->NumRows() && G.NumCols() == M->NumCols() &&
               M->NumCols() != 0 && Q2.NumRows() == M->NumCols() &&
               P2.NumRows() == M->NumRows());
  MatrixIndexT rows = M->NumRows(), cols = M->NumCols();
  // The following check should not fail as we stipulate P1, P2 and one of Q1
  // or Q2 must be +ve def and other Q1 or Q2 must be +ve semidef.
  TpMatrix<Real> LInv(rows);
  LInv.Cholesky(P1);
  LInv.Invert();  // Will throw exception if fails.
  SpMatrix<Real> S(rows);
  Matrix<Real> LInvFull(LInv);
  S.AddMat2Sp(1.0, LInvFull, kNoTrans, P2, 0.0);  // S := L^{-1} P_2 L^{-T}
  Matrix<Real> U(rows, rows);
  Vector<Real> d(rows);
  S.SymPosSemiDefEig(&d, &U);
  Matrix<Real> T(rows, rows);
  T.AddMatMat(1.0, U, kTrans, LInvFull, kNoTrans, 0.0);  // T := U^T * L^{-1}

#ifdef KALDI_PARANOID  // checking mainly for errors in the code or math.
  {
    SpMatrix<Real> P1Trans(rows);
    P1Trans.AddMat2Sp(1.0, T, kNoTrans, P1, 0.0);
    KALDI_ASSERT(P1Trans.IsUnit(0.01));
  }
  {
    SpMatrix<Real> P2Trans(rows);
    P2Trans.AddMat2Sp(1.0, T, kNoTrans, P2, 0.0);
    KALDI_ASSERT(P2Trans.IsDiagonal(0.01));
  }
#endif

  Matrix<Real> TInv(T);
  TInv.Invert();
  Matrix<Real> Gdash(rows, cols);
  Gdash.AddMatMat(1.0, T, kNoTrans, G, kNoTrans, 0.0);  // G' = T G
  Matrix<Real> MdashOld(rows, cols);
  MdashOld.AddMatMat(1.0, TInv, kTrans, *M, kNoTrans, 0.0);  // M' = T^{-T} M
  Matrix<Real> MdashNew(MdashOld);
  Real objf_impr = 0.0;
  for (MatrixIndexT n = 0; n < rows; n++) {
    SpMatrix<Real> Qsum(Q1);
    Qsum.AddSp(d(n), Q2);
    SubVector<Real> mdash_n = MdashNew.Row(n);
    SubVector<Real> gdash_n = Gdash.Row(n);

    Matrix<Real> QsumInv(Qsum);
    try {
      QsumInv.Invert();
      Real old_objf = VecVec(mdash_n, gdash_n)
          - 0.5 * VecSpVec(mdash_n, Qsum, mdash_n);
      mdash_n.AddMatVec(1.0, QsumInv, kNoTrans, gdash_n, 0.0); // m'_n := g'_n * (Q_1 + d_n Q_2)^{-1}
      Real new_objf = VecVec(mdash_n, gdash_n)
          - 0.5 * VecSpVec(mdash_n, Qsum, mdash_n);
      if (new_objf < old_objf) {
        if (new_objf < old_objf - 1.0e-05) {
          KALDI_WARN << "In double quadratic matrix problem: objective "
              "function decreasing during optimization of " << opts.name
              << ", " << old_objf << "->" << new_objf << ", change is "
              << (new_objf - old_objf);
          KALDI_ERR << "Auxiliary function decreasing."; // Will be caught.
        } else {  // Reset to old value, didn't improve (very close to optimum).
          MdashNew.Row(n).CopyFromVec(MdashOld.Row(n));
        }
      }
      objf_impr += new_objf - old_objf;
    }
    catch (...) {
      KALDI_WARN << "Matrix inversion or optimization failed during double "
          "quadratic problem, solving for" << opts.name
          << ": trying more stable approach.";
      objf_impr += SolveQuadraticProblem(Qsum, gdash_n, opts, &mdash_n);
    }
  }
  M->AddMatMat(1.0, T, kTrans, MdashNew, kNoTrans, 0.0); // M := T^T M'.
  return objf_impr;
}

// rank-one update, this <-- this + alpha V V'
template<>
template<>
void SpMatrix<float>::AddVec2(const float alpha, const VectorBase<float> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
  cblas_Xspr(v.Dim(), alpha, v.Data(), 1,
             this->data_);
}

template<class Real>
void SpMatrix<Real>::AddVec2Sp(const Real alpha, const VectorBase<Real> &v,
                               const SpMatrix<Real> &S, const Real beta) {
  KALDI_ASSERT(v.Dim() == this->NumRows() && S.NumRows() == this->NumRows());
  const Real *Sdata = S.Data();
  const Real *vdata = v.Data();
  Real *data = this->data_;
  MatrixIndexT dim = this->num_rows_;
  for (MatrixIndexT r = 0; r < dim; r++)
    for (MatrixIndexT c = 0; c <= r; c++, Sdata++, data++)
      *data = beta * *data + alpha * vdata[r] * vdata[c] * *Sdata;
}


// rank-one update, this <-- this + alpha V V'
template<>
template<>
void SpMatrix<double>::AddVec2(const double alpha, const VectorBase<double> &v) {
  KALDI_ASSERT(v.Dim() == num_rows_);
  cblas_Xspr(v.Dim(), alpha, v.Data(), 1, data_);
}


template<typename Real>
template<typename OtherReal>
void SpMatrix<Real>::AddVec2(const Real alpha, const VectorBase<OtherReal> &v) {
  KALDI_ASSERT(v.Dim() == this->NumRows());
  Real *data = this->data_;
  const OtherReal *v_data = v.Data();
  MatrixIndexT nr = this->num_rows_;
  for (MatrixIndexT i = 0; i < nr; i++)
    for (MatrixIndexT j = 0; j <= i; j++, data++)
      *data += alpha * v_data[i] * v_data[j];
}

// instantiate the template above.
template
void SpMatrix<float>::AddVec2(const float alpha, const VectorBase<double> &v);
template
void SpMatrix<double>::AddVec2(const double alpha, const VectorBase<float> &v);


template<typename Real>
Real VecSpVec(const VectorBase<Real> &v1, const SpMatrix<Real> &M,
              const VectorBase<Real> &v2) {
  MatrixIndexT D = M.NumRows();
  KALDI_ASSERT(v1.Dim() == D && v1.Dim() == v2.Dim());
  Vector<Real> tmp_vec(D);
  cblas_Xspmv(D, 1.0, M.Data(), v1.Data(), 1, 0.0, tmp_vec.Data(), 1);
  return VecVec(tmp_vec, v2);
}

template
float VecSpVec(const VectorBase<float> &v1, const SpMatrix<float> &M,
               const VectorBase<float> &v2);
template
double VecSpVec(const VectorBase<double> &v1, const SpMatrix<double> &M,
                const VectorBase<double> &v2);


template<typename Real>
void SpMatrix<Real>::AddMat2Sp(
    const Real alpha, const MatrixBase<Real> &M,
    MatrixTransposeType transM, const SpMatrix<Real> &A, const Real beta) {
  if (transM == kNoTrans) {
    KALDI_ASSERT(M.NumCols() == A.NumRows() && M.NumRows() == this->num_rows_);
  } else {
    KALDI_ASSERT(M.NumRows() == A.NumRows() && M.NumCols() == this->num_rows_);
  }
  Vector<Real> tmp_vec(A.NumRows());
  Real *tmp_vec_data = tmp_vec.Data();
  SpMatrix<Real> tmp_A;
  const Real *p_A_data = A.Data();
  Real *p_row_data = this->Data();
  MatrixIndexT M_other_dim = (transM == kNoTrans ? M.NumCols() : M.NumRows()),
      M_same_dim = (transM == kNoTrans ? M.NumRows() : M.NumCols()),
      M_stride = M.Stride(), dim = this->NumRows();
  KALDI_ASSERT(M_same_dim == dim);

  const Real *M_data = M.Data();

  if (this->Data() <= A.Data() + A.SizeInBytes() &&
      this->Data() + this->SizeInBytes() >= A.Data()) {
    // Matrices A and *this overlap. Make copy of A
    tmp_A.Resize(A.NumRows());
    tmp_A.CopyFromSp(A);
    p_A_data = tmp_A.Data();
  }

  if (transM == kNoTrans) {
    for (MatrixIndexT r = 0; r < dim; r++, p_row_data += r) {
      cblas_Xspmv(A.NumRows(), 1.0, p_A_data, M.RowData(r), 1, 0.0, tmp_vec_data, 1);
      cblas_Xgemv(transM, r+1, M_other_dim, alpha, M_data, M_stride,
                  tmp_vec_data, 1, beta, p_row_data, 1);
    }
  } else {
    for (MatrixIndexT r = 0; r < dim; r++, p_row_data += r) {
      cblas_Xspmv(A.NumRows(), 1.0, p_A_data, M.Data() + r, M.Stride(), 0.0, tmp_vec_data, 1);
      cblas_Xgemv(transM, M_other_dim, r+1, alpha, M_data, M_stride,
                  tmp_vec_data, 1, beta, p_row_data, 1);
    }
  }
}

template<typename Real>
void SpMatrix<Real>::AddSmat2Sp(
    const Real alpha, const MatrixBase<Real> &M,
    MatrixTransposeType transM, const SpMatrix<Real> &A,
    const Real beta) {
  KALDI_ASSERT((transM == kNoTrans && M.NumCols() == A.NumRows()) ||
               (transM == kTrans && M.NumRows() == A.NumRows()));
  if (transM == kNoTrans) {
    KALDI_ASSERT(M.NumCols() == A.NumRows() && M.NumRows() == this->num_rows_);
  } else {
    KALDI_ASSERT(M.NumRows() == A.NumRows() && M.NumCols() == this->num_rows_);
  }
  MatrixIndexT Adim = A.NumRows(), dim = this->num_rows_;

  Matrix<Real> temp_A(A); // represent A as full matrix.
  Matrix<Real> temp_MA(dim, Adim);
  temp_MA.AddSmatMat(1.0, M, transM, temp_A, kNoTrans, 0.0);

  // Next-- we want to do *this = alpha * temp_MA * M^T + beta * *this.
  // To make it sparse vector multiplies, since M is sparse, we'd like
  // to do: for each column c, (*this column c) += temp_MA * (M^T's column c.)
  // [ignoring the alpha and beta here.]
  // It's not convenient to process columns in the symmetric
  // packed format because they don't have a constant stride.  However,
  // we can use the fact that temp_MA * M is symmetric, to just assign
  // each row of *this instead of each column.
  // So the final iteration is:
  // for i = 0... dim-1,
  //   [the i'th row of *this] = beta * [the i'th row of *this] + alpha *
  //                               temp_MA * [the i'th column of M].
  // Of course, we only process the first 0 ... i elements of this row,
  // as that's all that are kept in the symmetric packed format.

  Matrix<Real> temp_this(*this);
  Real *data = this->data_;
  const Real *Mdata = M.Data(), *MAdata = temp_MA.Data();
  MatrixIndexT temp_MA_stride = temp_MA.Stride(), Mstride = M.Stride();

  if (transM == kNoTrans) {
    // The column of M^T corresponds to the rows of the supplied matrix.
    for (MatrixIndexT i = 0; i < dim; i++, data += i) {
      MatrixIndexT num_rows = i + 1, num_cols = Adim;
      Xgemv_sparsevec(kNoTrans, num_rows, num_cols, alpha, MAdata,
                      temp_MA_stride, Mdata + (i * Mstride), 1, beta, data, 1);
    }
  } else {
    // The column of M^T corresponds to the columns of the supplied matrix.
    for (MatrixIndexT i = 0; i < dim; i++, data += i) {
      MatrixIndexT num_rows = i + 1, num_cols = Adim;
      Xgemv_sparsevec(kNoTrans, num_rows, num_cols, alpha, MAdata,
                      temp_MA_stride, Mdata + i, Mstride, beta, data, 1);
    }
  }
}

template<typename Real>
void SpMatrix<Real>::AddMat2Vec(const Real alpha,
                                const MatrixBase<Real> &M,
                                MatrixTransposeType transM,
                                const VectorBase<Real> &v,
                                const Real beta) {
  this->Scale(beta);
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows() &&
                M.NumCols() == v.Dim()) ||
               (transM == kTrans && this->NumRows() == M.NumCols() &&
                M.NumRows() == v.Dim()));

  if (transM == kNoTrans) {
    const Real *Mdata = M.Data(), *vdata = v.Data();
    Real *data = this->data_;
    MatrixIndexT dim = this->NumRows(), mcols = M.NumCols(),
        mstride = M.Stride();
    for (MatrixIndexT col = 0; col < mcols; col++, vdata++, Mdata += 1)
      cblas_Xspr(dim, *vdata*alpha, Mdata, mstride, data);
  } else {
    const Real *Mdata = M.Data(), *vdata = v.Data();
    Real *data = this->data_;
    MatrixIndexT dim = this->NumRows(), mrows = M.NumRows(),
        mstride = M.Stride();
    for (MatrixIndexT row = 0; row < mrows; row++, vdata++, Mdata += mstride)
      cblas_Xspr(dim, *vdata*alpha, Mdata, 1, data);
  }
}

template<typename Real>
void SpMatrix<Real>::AddMat2(const Real alpha, const MatrixBase<Real> &M,
                             MatrixTransposeType transM, const Real beta)  {
  KALDI_ASSERT((transM == kNoTrans && this->NumRows() == M.NumRows())
               || (transM == kTrans && this->NumRows() == M.NumCols()));

  // Cblas has no function *sprk (i.e. symmetric packed rank-k update), so we
  // use as temporary storage a regular matrix of which we only access its lower
  // triangle

  MatrixIndexT this_dim = this->NumRows(),
      m_other_dim = (transM == kNoTrans ? M.NumCols() : M.NumRows());

  if (this_dim == 0) return;
  if (alpha == 0.0) {
    if (beta != 1.0) this->Scale(beta);
    return;
  }

  Matrix<Real> temp_mat(*this); // wastefully copies upper triangle too, but this
  // doesn't dominate O(N) time.

  // This function call is hard-coded to update the lower triangle.
  cblas_Xsyrk(transM, this_dim, m_other_dim, alpha, M.Data(),
              M.Stride(), beta, temp_mat.Data(), temp_mat.Stride());

  this->CopyFromMat(temp_mat, kTakeLower);
}

template<typename Real>
void SpMatrix<Real>::AddTp2Sp(const Real alpha, const TpMatrix<Real> &T,
                              MatrixTransposeType transM, const SpMatrix<Real> &A,
                              const Real beta) {
  Matrix<Real> Tmat(T);
  AddMat2Sp(alpha, Tmat, transM, A, beta);
}

template<typename Real>
void SpMatrix<Real>::AddVecVec(const Real alpha, const VectorBase<Real> &v,
                               const VectorBase<Real> &w) {
  int32 dim = this->NumRows();
  KALDI_ASSERT(dim == v.Dim() && dim == w.Dim() && dim > 0);
  cblas_Xspr2(dim, alpha, v.Data(), 1, w.Data(), 1, this->data_);
}


template<typename Real>
void SpMatrix<Real>::AddTp2(const Real alpha, const TpMatrix<Real> &T,
                            MatrixTransposeType transM, const Real beta) {
  Matrix<Real> Tmat(T);
  AddMat2(alpha, Tmat, transM, beta);
}


// Explicit instantiation of the class.
// This needs to be after the definition of all the class member functions.

template class SpMatrix<float>;
template class SpMatrix<double>;


template<typename Real>
Real TraceSpSpLower(const SpMatrix<Real> &A, const SpMatrix<Real> &B) {
  MatrixIndexT adim = A.NumRows();
  KALDI_ASSERT(adim == B.NumRows());
  MatrixIndexT dim = (adim*(adim+1))/2;
  return cblas_Xdot(dim, A.Data(), 1, B.Data(), 1);
}
// Instantiate the template above.
template
double TraceSpSpLower(const SpMatrix<double> &A, const SpMatrix<double> &B);
template
float TraceSpSpLower(const SpMatrix<float> &A, const SpMatrix<float> &B);

// Instantiate the template above.
template float SolveQuadraticMatrixProblem(const SpMatrix<float> &Q,
                                           const MatrixBase<float> &Y,
                                           const SpMatrix<float> &SigmaInv,
                                           const SolverOptions &opts,
                                           MatrixBase<float> *M);
template double SolveQuadraticMatrixProblem(const SpMatrix<double> &Q,
                                            const MatrixBase<double> &Y,
                                            const SpMatrix<double> &SigmaInv,
                                            const SolverOptions &opts,
                                            MatrixBase<double> *M);

// Instantiate the template above.
template float SolveDoubleQuadraticMatrixProblem(
    const MatrixBase<float> &G,
    const SpMatrix<float> &P1,
    const SpMatrix<float> &P2,
    const SpMatrix<float> &Q1,
    const SpMatrix<float> &Q2,
    const SolverOptions &opts,
    MatrixBase<float> *M);

template double SolveDoubleQuadraticMatrixProblem(
    const MatrixBase<double> &G,
    const SpMatrix<double> &P1,
    const SpMatrix<double> &P2,
    const SpMatrix<double> &Q1,
    const SpMatrix<double> &Q2,
    const SolverOptions &opts,
    MatrixBase<double> *M);



} // namespace kaldi
