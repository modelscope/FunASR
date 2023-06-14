// matrix/kaldi-matrix.cc

// Copyright 2009-2011   Lukas Burget;  Ondrej Glembek;  Go Vivace Inc.;
//                       Microsoft Corporation;  Saarland University;
//                       Yanmin Qian;  Petr Schwarz;  Jan Silovsky;
//                       Haihua Xu
//           2017        Shiyin Kang

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

#include "matrix/kaldi-matrix.h"
#include "matrix/sp-matrix.h"
#include "matrix/jama-svd.h"
#include "matrix/jama-eig.h"
#include "matrix/compressed-matrix.h"
#include "matrix/sparse-matrix.h"

static_assert(int(kaldi::kNoTrans) == int(CblasNoTrans) && int(kaldi::kTrans) == int(CblasTrans), 
    "kaldi::kNoTrans and kaldi::kTrans must be equal to the appropriate CBLAS library constants!");

namespace kaldi {

template<typename Real>
void MatrixBase<Real>::Invert(Real *log_det, Real *det_sign,
                              bool inverse_needed) {
  KALDI_ASSERT(num_rows_ == num_cols_);
  if (num_rows_ == 0) {
    if (det_sign) *det_sign = 1;
    if (log_det) *log_det = 0.0;
    return;
  }
#ifndef HAVE_ATLAS
  KaldiBlasInt *pivot = new KaldiBlasInt[num_rows_];
  KaldiBlasInt M = num_rows_;
  KaldiBlasInt N = num_cols_;
  KaldiBlasInt LDA = stride_;
  KaldiBlasInt result = -1;
  KaldiBlasInt l_work = std::max<KaldiBlasInt>(1, N);
  Real *p_work;
  void *temp;
  if ((p_work = static_cast<Real*>(
          KALDI_MEMALIGN(16, sizeof(Real)*l_work, &temp))) == NULL) {
    delete[] pivot;
    throw std::bad_alloc();
  }

  clapack_Xgetrf2(&M, &N, data_, &LDA, pivot, &result);
  const int pivot_offset = 1;
#else
  int *pivot = new int[num_rows_];
  int result;
  clapack_Xgetrf(num_rows_, num_cols_, data_, stride_, pivot, &result);
  const int pivot_offset = 0;
#endif
  KALDI_ASSERT(result >= 0 && "Call to CLAPACK sgetrf_ or ATLAS clapack_sgetrf "
               "called with wrong arguments");
  if (result > 0) {
    if (inverse_needed) {
      KALDI_ERR << "Cannot invert: matrix is singular";
    } else {
      if (log_det) *log_det = -std::numeric_limits<Real>::infinity();
      if (det_sign) *det_sign = 0;
      delete[] pivot;
#ifndef HAVE_ATLAS
      KALDI_MEMALIGN_FREE(p_work);
#endif
      return;
    }
  }
  if (det_sign != NULL) {
    int sign = 1;
    for (MatrixIndexT i = 0; i < num_rows_; i++)
      if (pivot[i] != static_cast<int>(i) + pivot_offset) sign *= -1;
    *det_sign = sign;
  }
  if (log_det != NULL || det_sign != NULL) {  // Compute log determinant.
    if (log_det != NULL) *log_det = 0.0;
    Real prod = 1.0;
    for (MatrixIndexT i = 0; i < num_rows_; i++) {
      prod *= (*this)(i, i);
      if (i == num_rows_ - 1 || std::fabs(prod) < 1.0e-10 ||
          std::fabs(prod) > 1.0e+10) {
        if (log_det != NULL) *log_det += Log(std::fabs(prod));
        if (det_sign != NULL) *det_sign *= (prod > 0 ? 1.0 : -1.0);
        prod = 1.0;
      }
    }
  }
#ifndef HAVE_ATLAS
  if (inverse_needed) clapack_Xgetri2(&M, data_, &LDA, pivot, p_work, &l_work,
                              &result);
  delete[] pivot;
  KALDI_MEMALIGN_FREE(p_work);
#else
  if (inverse_needed)
    clapack_Xgetri(num_rows_, data_, stride_, pivot, &result);
  delete [] pivot;
#endif
  KALDI_ASSERT(result == 0 && "Call to CLAPACK sgetri_ or ATLAS clapack_sgetri "
               "called with wrong arguments");
}

template<>
template<>
void MatrixBase<float>::AddVecVec(const float alpha,
                                  const VectorBase<float> &a,
                                  const VectorBase<float> &rb) {
  KALDI_ASSERT(a.Dim() == num_rows_ && rb.Dim() == num_cols_);
  cblas_Xger(a.Dim(), rb.Dim(), alpha, a.Data(), 1, rb.Data(),
             1, data_, stride_);
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::AddVecVec(const Real alpha,
                                 const VectorBase<OtherReal> &a,
                                 const VectorBase<OtherReal> &b) {
  KALDI_ASSERT(a.Dim() == num_rows_ && b.Dim() == num_cols_);
  if (num_rows_ * num_cols_ > 100) { // It's probably worth it to allocate
    // temporary vectors of the right type and use BLAS.
    Vector<Real> temp_a(a), temp_b(b);
    cblas_Xger(num_rows_, num_cols_, alpha, temp_a.Data(), 1,
               temp_b.Data(), 1, data_, stride_);
  } else {
    const OtherReal *a_data = a.Data(), *b_data = b.Data();
    Real *row_data = data_;
    for (MatrixIndexT i = 0; i < num_rows_; i++, row_data += stride_) {
      BaseFloat alpha_ai = alpha * a_data[i];
      for (MatrixIndexT j = 0; j < num_cols_; j++)
        row_data[j] += alpha_ai * b_data[j];
    }
  }
}

// instantiate the template above.
template
void MatrixBase<float>::AddVecVec(const float alpha,
                                  const VectorBase<double> &a,
                                  const VectorBase<double> &b);
template
void MatrixBase<double>::AddVecVec(const double alpha,
                                   const VectorBase<float> &a,
                                   const VectorBase<float> &b);

template<>
template<>
void MatrixBase<double>::AddVecVec(const double alpha,
                                   const VectorBase<double> &a,
                                   const VectorBase<double> &rb) {
  KALDI_ASSERT(a.Dim() == num_rows_ && rb.Dim() == num_cols_);
  if (num_rows_ == 0) return;
  cblas_Xger(a.Dim(), rb.Dim(), alpha, a.Data(), 1, rb.Data(),
             1, data_, stride_);
}

template<typename Real>
void MatrixBase<Real>::AddMatMat(const Real alpha,
                                  const MatrixBase<Real>& A,
                                  MatrixTransposeType transA,
                                  const MatrixBase<Real>& B,
                                  MatrixTransposeType transB,
                                  const Real beta) {
  KALDI_ASSERT((transA == kNoTrans && transB == kNoTrans && A.num_cols_ == B.num_rows_ && A.num_rows_ == num_rows_ && B.num_cols_ == num_cols_)
               || (transA == kTrans && transB == kNoTrans && A.num_rows_ == B.num_rows_ && A.num_cols_ == num_rows_ && B.num_cols_ == num_cols_)
               || (transA == kNoTrans && transB == kTrans && A.num_cols_ == B.num_cols_ && A.num_rows_ == num_rows_ && B.num_rows_ == num_cols_)
               || (transA == kTrans && transB == kTrans && A.num_rows_ == B.num_cols_ && A.num_cols_ == num_rows_ && B.num_rows_ == num_cols_));
  KALDI_ASSERT(&A !=  this && &B != this);
  if (num_rows_ == 0) return;
  cblas_Xgemm(alpha, transA, A.data_, A.num_rows_, A.num_cols_, A.stride_,
              transB, B.data_, B.stride_, beta, data_, num_rows_, num_cols_, stride_);

}

template<typename Real>
void MatrixBase<Real>::SetMatMatDivMat(const MatrixBase<Real>& A,
                                       const MatrixBase<Real>& B,
                                       const MatrixBase<Real>& C) {
  KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
  KALDI_ASSERT(A.NumRows() == C.NumRows() && A.NumCols() == C.NumCols());
  for (int32 r = 0; r < A.NumRows(); r++) { // each frame...
    for (int32 c = 0; c < A.NumCols(); c++) {
      BaseFloat i = C(r, c), o = B(r, c), od = A(r, c),
          id;
      if (i != 0.0) {
        id = od * (o / i); /// o / i is either zero or "scale".
      } else {
        id = od; /// Just imagine the scale was 1.0.  This is somehow true in
        /// expectation; anyway, this case should basically never happen so it doesn't
        /// really matter.
      }
      (*this)(r, c) = id;
    }
  }
}


template<typename Real>
void MatrixBase<Real>::CopyLowerToUpper() {
  KALDI_ASSERT(num_rows_ == num_cols_);
  Real *data = data_;
  MatrixIndexT num_rows = num_rows_, stride = stride_;
  for (int32 i = 0; i < num_rows; i++)
    for (int32 j = 0; j < i; j++)
      data[j * stride + i ] = data[i * stride + j];
}


template<typename Real>
void MatrixBase<Real>::CopyUpperToLower() {
  KALDI_ASSERT(num_rows_ == num_cols_);
  Real *data = data_;
  MatrixIndexT num_rows = num_rows_, stride = stride_;
  for (int32 i = 0; i < num_rows; i++)
    for (int32 j = 0; j < i; j++)
      data[i * stride + j] = data[j * stride + i];
}

template<typename Real>
void MatrixBase<Real>::SymAddMat2(const Real alpha,
                                  const MatrixBase<Real> &A,
                                  MatrixTransposeType transA,
                                  Real beta) {
  KALDI_ASSERT(num_rows_ == num_cols_ &&
               ((transA == kNoTrans && A.num_rows_ == num_rows_) ||
                (transA == kTrans && A.num_cols_ == num_cols_)));
  KALDI_ASSERT(A.data_ != data_);
  if (num_rows_ == 0) return;

  /// When the matrix dimension(this->num_rows_) is not less than 56
  /// and the transpose type transA == kTrans, the cblas_Xsyrk(...)
  /// function will produce NaN in the output. This is a bug in the
  /// ATLAS library. To overcome this, the AddMatMat function, which calls
  /// cblas_Xgemm(...) rather than cblas_Xsyrk(...), is used in this special
  /// sitation.
  /// Wei Shi: Note this bug is observerd for single precision matrix
  /// on a 64-bit machine
#ifdef HAVE_ATLAS
  if (transA == kTrans && num_rows_ >= 56) {
    this->AddMatMat(alpha, A, kTrans, A, kNoTrans, beta);
    return;
  }
#endif // HAVE_ATLAS

  MatrixIndexT A_other_dim = (transA == kNoTrans ? A.num_cols_ : A.num_rows_);

  // This function call is hard-coded to update the lower triangle.
  cblas_Xsyrk(transA, num_rows_, A_other_dim, alpha, A.Data(),
              A.Stride(), beta, this->data_, this->stride_);
}


template<typename Real>
void MatrixBase<Real>::AddMatSmat(const Real alpha,
                                  const MatrixBase<Real> &A,
                                  MatrixTransposeType transA,
                                  const MatrixBase<Real> &B,
                                  MatrixTransposeType transB,
                                  const Real beta) {
  KALDI_ASSERT((transA == kNoTrans && transB == kNoTrans && A.num_cols_ == B.num_rows_ && A.num_rows_ == num_rows_ && B.num_cols_ == num_cols_)
               || (transA == kTrans && transB == kNoTrans && A.num_rows_ == B.num_rows_ && A.num_cols_ == num_rows_ && B.num_cols_ == num_cols_)
               || (transA == kNoTrans && transB == kTrans && A.num_cols_ == B.num_cols_ && A.num_rows_ == num_rows_ && B.num_rows_ == num_cols_)
               || (transA == kTrans && transB == kTrans && A.num_rows_ == B.num_cols_ && A.num_cols_ == num_rows_ && B.num_rows_ == num_cols_));
  KALDI_ASSERT(&A !=  this && &B != this);

  // We iterate over the columns of B.

  MatrixIndexT Astride = A.stride_, Bstride = B.stride_, stride = this->stride_,
      Arows = A.num_rows_, Acols = A.num_cols_;
  Real *data = this->data_, *Adata = A.data_, *Bdata = B.data_;
  MatrixIndexT num_cols = this->num_cols_;
  if (transB == kNoTrans) {
    // Iterate over the columns of *this and of B.
    for (MatrixIndexT c = 0; c < num_cols; c++) {
      // for each column of *this, do
      // [this column] = [alpha * A * this column of B] + [beta * this column]
      Xgemv_sparsevec(transA, Arows, Acols, alpha, Adata, Astride,
                      Bdata + c, Bstride, beta, data + c, stride);
    }
  } else {
    // Iterate over the columns of *this and the rows of B.
    for (MatrixIndexT c = 0; c < num_cols; c++) {
      // for each column of *this, do
      // [this column] = [alpha * A * this row of B] + [beta * this column]
      Xgemv_sparsevec(transA, Arows, Acols, alpha, Adata, Astride,
                      Bdata + (c * Bstride), 1, beta, data + c, stride);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::AddSmatMat(const Real alpha,
                                  const MatrixBase<Real> &A,
                                  MatrixTransposeType transA,
                                  const MatrixBase<Real> &B,
                                  MatrixTransposeType transB,
                                  const Real beta) {
  KALDI_ASSERT((transA == kNoTrans && transB == kNoTrans && A.num_cols_ == B.num_rows_ && A.num_rows_ == num_rows_ && B.num_cols_ == num_cols_)
               || (transA == kTrans && transB == kNoTrans && A.num_rows_ == B.num_rows_ && A.num_cols_ == num_rows_ && B.num_cols_ == num_cols_)
               || (transA == kNoTrans && transB == kTrans && A.num_cols_ == B.num_cols_ && A.num_rows_ == num_rows_ && B.num_rows_ == num_cols_)
               || (transA == kTrans && transB == kTrans && A.num_rows_ == B.num_cols_ && A.num_cols_ == num_rows_ && B.num_rows_ == num_cols_));
  KALDI_ASSERT(&A !=  this && &B != this);

  MatrixIndexT Astride = A.stride_, Bstride = B.stride_, stride = this->stride_,
      Brows = B.num_rows_, Bcols = B.num_cols_;
  MatrixTransposeType invTransB = (transB == kTrans ? kNoTrans : kTrans);
  Real *data = this->data_, *Adata = A.data_, *Bdata = B.data_;
  MatrixIndexT num_rows = this->num_rows_;
  if (transA == kNoTrans) {
    // Iterate over the rows of *this and of A.
    for (MatrixIndexT r = 0; r < num_rows; r++) {
      // for each row of *this, do
      // [this row] = [alpha * (this row of A) * B^T] + [beta * this row]
      Xgemv_sparsevec(invTransB, Brows, Bcols, alpha, Bdata, Bstride,
                      Adata + (r * Astride), 1, beta, data + (r * stride), 1);
    }
  } else {
    // Iterate over the rows of *this and the columns of A.
    for (MatrixIndexT r = 0; r < num_rows; r++) {
      // for each row of *this, do
      // [this row] = [alpha * (this column of A) * B^T] + [beta * this row]
      Xgemv_sparsevec(invTransB, Brows, Bcols, alpha, Bdata, Bstride,
                      Adata + r, Astride, beta, data + (r * stride), 1);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::AddSpSp(const Real alpha, const SpMatrix<Real> &A_in,
                                const SpMatrix<Real> &B_in, const Real beta) {
  MatrixIndexT sz = num_rows_;
  KALDI_ASSERT(sz == num_cols_ && sz == A_in.NumRows() && sz == B_in.NumRows());

  Matrix<Real> A(A_in), B(B_in);
  // CblasLower or CblasUpper would work below as symmetric matrix is copied
  // fully (to save work, we used the matrix constructor from SpMatrix).
  // CblasLeft means A is on the left: C <-- alpha A B + beta C
  if (sz == 0) return;
  cblas_Xsymm(alpha, sz, A.data_, A.stride_, B.data_, B.stride_, beta, data_, stride_);
}

template<typename Real>
void MatrixBase<Real>::AddMat(const Real alpha, const MatrixBase<Real>& A,
                               MatrixTransposeType transA) {
  if (&A == this) {
    if (transA == kNoTrans) {
      Scale(alpha + 1.0);
    } else {
      KALDI_ASSERT(num_rows_ == num_cols_ && "AddMat: adding to self (transposed): not symmetric.");
      Real *data = data_;
      if (alpha == 1.0) {  // common case-- handle separately.
        for (MatrixIndexT row = 0; row < num_rows_; row++) {
          for (MatrixIndexT col = 0; col < row; col++) {
            Real *lower = data + (row * stride_) + col, *upper = data + (col
                                                                          * stride_) + row;
            Real sum = *lower + *upper;
            *lower = *upper = sum;
          }
          *(data + (row * stride_) + row) *= 2.0;  // diagonal.
        }
      } else {
        for (MatrixIndexT row = 0; row < num_rows_; row++) {
          for (MatrixIndexT col = 0; col < row; col++) {
            Real *lower = data + (row * stride_) + col, *upper = data + (col
                                                                          * stride_) + row;
            Real lower_tmp = *lower;
            *lower += alpha * *upper;
            *upper += alpha * lower_tmp;
          }
          *(data + (row * stride_) + row) *= (1.0 + alpha);  // diagonal.
        }
      }
    }
  } else {
    int aStride = (int) A.stride_, stride = stride_;
    Real *adata = A.data_, *data = data_;
    if (transA == kNoTrans) {
      KALDI_ASSERT(A.num_rows_ == num_rows_ && A.num_cols_ == num_cols_);
      if (num_rows_ == 0) return;
      for (MatrixIndexT row = 0; row < num_rows_; row++, adata += aStride,
               data += stride) {
        cblas_Xaxpy(num_cols_, alpha, adata, 1, data, 1);
      }
    } else {
      KALDI_ASSERT(A.num_cols_ == num_rows_ && A.num_rows_ == num_cols_);
      if (num_rows_ == 0) return;
      for (MatrixIndexT row = 0; row < num_rows_; row++, adata++, data += stride)
        cblas_Xaxpy(num_cols_, alpha, adata, aStride, data, 1);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::AddSmat(Real alpha, const SparseMatrix<Real> &A,
                               MatrixTransposeType trans) {
  if (trans == kNoTrans) {
    KALDI_ASSERT(NumRows() == A.NumRows());
    KALDI_ASSERT(NumCols() == A.NumCols());
    MatrixIndexT a_num_rows = A.NumRows();
    for (MatrixIndexT i = 0; i < a_num_rows; ++i) {
      const SparseVector<Real> &row = A.Row(i);
      MatrixIndexT num_elems = row.NumElements();
      for (MatrixIndexT id = 0; id < num_elems; ++id) {
        (*this)(i, row.GetElement(id).first) += alpha
            * row.GetElement(id).second;
      }
    }
  } else {
    KALDI_ASSERT(NumRows() == A.NumCols());
    KALDI_ASSERT(NumCols() == A.NumRows());
    MatrixIndexT a_num_rows = A.NumRows();
    for (MatrixIndexT i = 0; i < a_num_rows; ++i) {
      const SparseVector<Real> &row = A.Row(i);
      MatrixIndexT num_elems = row.NumElements();
      for (MatrixIndexT id = 0; id < num_elems; ++id) {
        (*this)(row.GetElement(id).first, i) += alpha
            * row.GetElement(id).second;
      }
    }
  }
}

template<typename Real>
void MatrixBase<Real>::AddSmatMat(Real alpha, const SparseMatrix<Real> &A,
                                  MatrixTransposeType transA,
                                  const MatrixBase<Real> &B, Real beta) {
  if (transA == kNoTrans) {
    KALDI_ASSERT(NumRows() == A.NumRows());
    KALDI_ASSERT(NumCols() == B.NumCols());
    KALDI_ASSERT(A.NumCols() == B.NumRows());

    this->Scale(beta);
    MatrixIndexT a_num_rows = A.NumRows(),
        this_num_cols = this->NumCols();
    for (MatrixIndexT i = 0; i < a_num_rows; ++i) {
      Real *this_row_i = this->RowData(i);
      const SparseVector<Real> &A_row_i = A.Row(i);
      MatrixIndexT num_elems = A_row_i.NumElements();
      for (MatrixIndexT e = 0; e < num_elems; ++e) {
        const std::pair<MatrixIndexT, Real> &p = A_row_i.GetElement(e);
        MatrixIndexT k = p.first;
        Real alpha_A_ik = alpha * p.second;
        const Real *b_row_k = B.RowData(k);
        cblas_Xaxpy(this_num_cols, alpha_A_ik, b_row_k, 1,
                    this_row_i, 1);
        //for (MatrixIndexT j = 0; j < this_num_cols; ++j)
        //  this_row_i[j] += alpha_A_ik * b_row_k[j];
      }
    }
  } else {
    KALDI_ASSERT(NumRows() == A.NumCols());
    KALDI_ASSERT(NumCols() == B.NumCols());
    KALDI_ASSERT(A.NumRows() == B.NumRows());

    this->Scale(beta);
    Matrix<Real> buf(NumRows(), NumCols(), kSetZero);
    MatrixIndexT a_num_rows = A.NumRows(),
        this_num_cols = this->NumCols();
    for (int k = 0; k < a_num_rows; ++k) {
      const Real *b_row_k = B.RowData(k);
      const SparseVector<Real> &A_row_k = A.Row(k);
      MatrixIndexT num_elems = A_row_k.NumElements();
      for (MatrixIndexT e = 0; e < num_elems; ++e) {
        const std::pair<MatrixIndexT, Real> &p = A_row_k.GetElement(e);
        MatrixIndexT i = p.first;
        Real alpha_A_ki = alpha * p.second;
        Real *this_row_i = this->RowData(i);
        cblas_Xaxpy(this_num_cols, alpha_A_ki, b_row_k, 1,
                    this_row_i, 1);
        //for (MatrixIndexT j = 0; j < this_num_cols; ++j)
        // this_row_i[j] += alpha_A_ki * b_row_k[j];
      }
    }
  }
}

template<typename Real>
void MatrixBase<Real>::AddMatSmat(Real alpha, const MatrixBase<Real> &A,
                                  const SparseMatrix<Real> &B,
                                  MatrixTransposeType transB, Real beta) {
  if (transB == kNoTrans) {
    KALDI_ASSERT(NumRows() == A.NumRows());
    KALDI_ASSERT(NumCols() == B.NumCols());
    KALDI_ASSERT(A.NumCols() == B.NumRows());

    this->Scale(beta);
    MatrixIndexT b_num_rows = B.NumRows(),
        this_num_rows = this->NumRows();
    // Iterate over the rows of sparse matrix B and columns of A.
    for (MatrixIndexT k = 0; k < b_num_rows; ++k) {
      const SparseVector<Real> &B_row_k = B.Row(k);
      MatrixIndexT num_elems = B_row_k.NumElements();
      const Real *a_col_k = A.Data() + k;
      for (MatrixIndexT e = 0; e < num_elems; ++e) {
        const std::pair<MatrixIndexT, Real> &p = B_row_k.GetElement(e);
        MatrixIndexT j = p.first;
        Real alpha_B_kj = alpha * p.second;
        Real *this_col_j = this->Data() + j;
        // Add to entire 'j'th column of *this at once using cblas_Xaxpy.
        // pass stride to write a colmun as matrices are stored in row major order.
        cblas_Xaxpy(this_num_rows, alpha_B_kj, a_col_k, A.stride_,
                    this_col_j, this->stride_);
        //for (MatrixIndexT i = 0; i < this_num_rows; ++i)
        // this_col_j[i*this->stride_] +=  alpha_B_kj * a_col_k[i*A.stride_];
      }
    }
  } else {
    KALDI_ASSERT(NumRows() == A.NumRows());
    KALDI_ASSERT(NumCols() == B.NumRows());
    KALDI_ASSERT(A.NumCols() == B.NumCols());

    this->Scale(beta);
    MatrixIndexT b_num_rows = B.NumRows(),
        this_num_rows = this->NumRows();
    // Iterate over the rows of sparse matrix B and columns of *this.
    for (MatrixIndexT j = 0; j < b_num_rows; ++j) {
      const SparseVector<Real> &B_row_j = B.Row(j);
      MatrixIndexT num_elems = B_row_j.NumElements();
      Real *this_col_j = this->Data() + j;
      for (MatrixIndexT e = 0; e < num_elems; ++e) {
        const std::pair<MatrixIndexT, Real> &p = B_row_j.GetElement(e);
        MatrixIndexT k = p.first;
        Real alpha_B_jk = alpha * p.second;
        const Real *a_col_k = A.Data() + k;
        // Add to entire 'j'th column of *this at once using cblas_Xaxpy.
        // pass stride to write a column as matrices are stored in row major order.
        cblas_Xaxpy(this_num_rows, alpha_B_jk, a_col_k, A.stride_,
                    this_col_j, this->stride_);
        //for (MatrixIndexT i = 0; i < this_num_rows; ++i) 
        // this_col_j[i*this->stride_] +=  alpha_B_jk * a_col_k[i*A.stride_];
      }
    }
  }
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::AddSp(const Real alpha, const SpMatrix<OtherReal> &S) {
  KALDI_ASSERT(S.NumRows() == NumRows() && S.NumRows() == NumCols());
  Real *data = data_; const OtherReal *sdata = S.Data();
  MatrixIndexT num_rows = NumRows(), stride = Stride();
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    for (MatrixIndexT j = 0; j < i; j++, sdata++) {
      data[i*stride + j] += alpha * *sdata;
      data[j*stride + i] += alpha * *sdata;
    }
    data[i*stride + i] += alpha * *sdata++;
  }
}

// instantiate the template above.
template
void MatrixBase<float>::AddSp(const float alpha, const SpMatrix<float> &S);
template
void MatrixBase<double>::AddSp(const double alpha, const SpMatrix<double> &S);
template
void MatrixBase<float>::AddSp(const float alpha, const SpMatrix<double> &S);
template
void MatrixBase<double>::AddSp(const double alpha, const SpMatrix<float> &S);


template<typename Real>
void MatrixBase<Real>::AddDiagVecMat(
    const Real alpha, const VectorBase<Real> &v,
    const MatrixBase<Real> &M,
    MatrixTransposeType transM,
    Real beta) {
  if (beta != 1.0) this->Scale(beta);

  if (transM == kNoTrans) {
    KALDI_ASSERT(SameDim(*this, M));
  } else {
    KALDI_ASSERT(M.NumRows() == NumCols() && M.NumCols() == NumRows());
  }
  KALDI_ASSERT(v.Dim() == this->NumRows());

  MatrixIndexT M_row_stride = M.Stride(), M_col_stride = 1, stride = stride_,
      num_rows = num_rows_, num_cols = num_cols_;
  if (transM == kTrans) std::swap(M_row_stride, M_col_stride);
  Real *data = data_;
  const Real *Mdata = M.Data(), *vdata = v.Data();
  if (num_rows_ == 0) return;
  for (MatrixIndexT i = 0; i < num_rows; i++, data += stride, Mdata += M_row_stride, vdata++)
    cblas_Xaxpy(num_cols, alpha * *vdata, Mdata, M_col_stride, data, 1);
}

template<typename Real>
void MatrixBase<Real>::AddMatDiagVec(
    const Real alpha,
    const MatrixBase<Real> &M, MatrixTransposeType transM,
    VectorBase<Real> &v,
    Real beta) {

  if (beta != 1.0) this->Scale(beta);

  if (transM == kNoTrans) {
    KALDI_ASSERT(SameDim(*this, M));
  } else {
    KALDI_ASSERT(M.NumRows() == NumCols() && M.NumCols() == NumRows());
  }
  KALDI_ASSERT(v.Dim() == this->NumCols());

  MatrixIndexT M_row_stride = M.Stride(),
               M_col_stride = 1,
               stride = stride_,
               num_rows = num_rows_,
               num_cols = num_cols_;

  if (transM == kTrans)
    std::swap(M_row_stride, M_col_stride);

  Real *data = data_;
  const Real *Mdata = M.Data(), *vdata = v.Data();
  if (num_rows_ == 0) return;
  for (MatrixIndexT i = 0; i < num_rows; i++){
      for(MatrixIndexT j = 0; j < num_cols; j ++ ){
          data[i*stride + j] += alpha * vdata[j] * Mdata[i*M_row_stride + j*M_col_stride];
      }
  }
}

template<typename Real>
void MatrixBase<Real>::AddMatMatElements(const Real alpha,
                                         const MatrixBase<Real>& A,
                                         const MatrixBase<Real>& B,
                                         const Real beta) {
    KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
    KALDI_ASSERT(A.NumRows() == NumRows() && A.NumCols() == NumCols());
    Real *data = data_;
    const Real *dataA = A.Data();
    const Real *dataB = B.Data();

    for (MatrixIndexT i = 0; i < num_rows_; i++) {
        for (MatrixIndexT j = 0; j < num_cols_; j++) {
            data[j] = beta*data[j] + alpha*dataA[j]*dataB[j];
        }
        data += Stride();
        dataA += A.Stride();
        dataB += B.Stride();
    }
}

#if !defined(HAVE_ATLAS) && !defined(USE_KALDI_SVD)
// ****************************************************************************
// ****************************************************************************
template<typename Real>
void MatrixBase<Real>::LapackGesvd(VectorBase<Real> *s, MatrixBase<Real> *U_in,
                                   MatrixBase<Real> *V_in) {
  KALDI_ASSERT(s != NULL && U_in != this && V_in != this);

  Matrix<Real> tmpU, tmpV;
  if (U_in == NULL) tmpU.Resize(this->num_rows_, 1);  // work-space if U_in empty.
  if (V_in == NULL) tmpV.Resize(1, this->num_cols_);  // work-space if V_in empty.

  /// Impementation notes:
  /// Lapack works in column-order, therefore the dimensions of *this are
  /// swapped as well as the U and V matrices.

  KaldiBlasInt M   = num_cols_;
  KaldiBlasInt N   = num_rows_;
  KaldiBlasInt LDA = Stride();

  KALDI_ASSERT(N>=M);  // NumRows >= columns.

  if (U_in) {
    KALDI_ASSERT((int)U_in->num_rows_ == N && (int)U_in->num_cols_ == M);
  }
  if (V_in) {
    KALDI_ASSERT((int)V_in->num_rows_ == M && (int)V_in->num_cols_ == M);
  }
  KALDI_ASSERT((int)s->Dim() == std::min(M, N));

  MatrixBase<Real> *U = (U_in ? U_in : &tmpU);
  MatrixBase<Real> *V = (V_in ? V_in : &tmpV);

  KaldiBlasInt V_stride      = V->Stride();
  KaldiBlasInt U_stride      = U->Stride();

  // Original LAPACK recipe
  // KaldiBlasInt l_work = std::max(std::max<long int>
  //   (1, 3*std::min(M, N)+std::max(M, N)), 5*std::min(M, N))*2;
  KaldiBlasInt l_work = -1;
  Real   work_query;
  KaldiBlasInt result;

  // query for work space
  char *u_job = const_cast<char*>(U_in ? "s" : "N");  // "s" == skinny, "N" == "none."
  char *v_job = const_cast<char*>(V_in ? "s" : "N");  // "s" == skinny, "N" == "none."
  clapack_Xgesvd(v_job, u_job,
                 &M, &N, data_, &LDA,
                 s->Data(),
                 V->Data(), &V_stride,
                 U->Data(), &U_stride,
                 &work_query, &l_work,
                 &result);

  KALDI_ASSERT(result >= 0 && "Call to CLAPACK dgesvd_ called with wrong arguments");

  l_work = static_cast<KaldiBlasInt>(work_query);
  Real *p_work;
  void *temp;
  if ((p_work = static_cast<Real*>(
          KALDI_MEMALIGN(16, sizeof(Real)*l_work, &temp))) == NULL)
    throw std::bad_alloc();

  // perform svd
  clapack_Xgesvd(v_job, u_job,
                 &M, &N, data_, &LDA,
                 s->Data(),
                 V->Data(), &V_stride,
                 U->Data(), &U_stride,
                 p_work, &l_work,
                 &result);

  KALDI_ASSERT(result >= 0 && "Call to CLAPACK dgesvd_ called with wrong arguments");

  if (result != 0) {
    KALDI_WARN << "CLAPACK sgesvd_ : some weird convergence not satisfied";
  }
  KALDI_MEMALIGN_FREE(p_work);
}

#endif

// Copy constructor.  Copies data to newly allocated memory.
template<typename Real>
Matrix<Real>::Matrix (const MatrixBase<Real> & M,
                      MatrixTransposeType trans/*=kNoTrans*/)
    : MatrixBase<Real>() {
  if (trans == kNoTrans) {
    Resize(M.num_rows_, M.num_cols_);
    this->CopyFromMat(M);
  } else {
    Resize(M.num_cols_, M.num_rows_);
    this->CopyFromMat(M, kTrans);
  }
}

// Copy constructor.  Copies data to newly allocated memory.
template<typename Real>
Matrix<Real>::Matrix (const Matrix<Real> & M):
    MatrixBase<Real>() {
  Resize(M.num_rows_, M.num_cols_);
  this->CopyFromMat(M);
}

/// Copy constructor from another type.
template<typename Real>
template<typename OtherReal>
Matrix<Real>::Matrix(const MatrixBase<OtherReal> & M,
                     MatrixTransposeType trans) : MatrixBase<Real>() {
  if (trans == kNoTrans) {
    Resize(M.NumRows(), M.NumCols());
    this->CopyFromMat(M);
  } else {
    Resize(M.NumCols(), M.NumRows());
    this->CopyFromMat(M, kTrans);
  }
}

// Instantiate this constructor for float->double and double->float.
template
Matrix<float>::Matrix(const MatrixBase<double> & M,
                      MatrixTransposeType trans);
template
Matrix<double>::Matrix(const MatrixBase<float> & M,
                       MatrixTransposeType trans);

template<typename Real>
inline void Matrix<Real>::Init(const MatrixIndexT rows,
                               const MatrixIndexT cols,
                               const MatrixStrideType stride_type) {
  if (rows * cols == 0) {
    KALDI_ASSERT(rows == 0 && cols == 0);
    this->num_rows_ = 0;
    this->num_cols_ = 0;
    this->stride_ = 0;
    this->data_ = NULL;
    return;
  }
  KALDI_ASSERT(rows > 0 && cols > 0);
  MatrixIndexT skip, stride;
  size_t size;
  void *data;  // aligned memory block
  void *temp;  // memory block to be really freed

  // compute the size of skip and real cols
  skip = ((16 / sizeof(Real)) - cols % (16 / sizeof(Real)))
      % (16 / sizeof(Real));
  stride = cols + skip;
  size = static_cast<size_t>(rows) * static_cast<size_t>(stride)
      * sizeof(Real);

  // allocate the memory and set the right dimensions and parameters
  if (NULL != (data = KALDI_MEMALIGN(16, size, &temp))) {
    MatrixBase<Real>::data_        = static_cast<Real *> (data);
    MatrixBase<Real>::num_rows_      = rows;
    MatrixBase<Real>::num_cols_      = cols;
    MatrixBase<Real>::stride_  = (stride_type == kDefaultStride ? stride : cols);
  } else {
    throw std::bad_alloc();
  }
}

template<typename Real>
void Matrix<Real>::Resize(const MatrixIndexT rows,
                          const MatrixIndexT cols,
                          MatrixResizeType resize_type,
                          MatrixStrideType stride_type) {
  // the next block uses recursion to handle what we have to do if
  // resize_type == kCopyData.
  if (resize_type == kCopyData) {
    if (this->data_ == NULL || rows == 0) resize_type = kSetZero;  // nothing to copy.
    else if (rows == this->num_rows_ && cols == this->num_cols_ &&
	     (stride_type == kDefaultStride || this->stride_ == this->num_cols_)) { return; } // nothing to do.
    else {
      // set tmp to a matrix of the desired size; if new matrix
      // is bigger in some dimension, zero it.
      MatrixResizeType new_resize_type =
          (rows > this->num_rows_ || cols > this->num_cols_) ? kSetZero : kUndefined;
      Matrix<Real> tmp(rows, cols, new_resize_type, stride_type);
      MatrixIndexT rows_min = std::min(rows, this->num_rows_),
          cols_min = std::min(cols, this->num_cols_);
      tmp.Range(0, rows_min, 0, cols_min).
          CopyFromMat(this->Range(0, rows_min, 0, cols_min));
      tmp.Swap(this);
      // and now let tmp go out of scope, deleting what was in *this.
      return;
    }
  }
  // At this point, resize_type == kSetZero or kUndefined.

  if (MatrixBase<Real>::data_ != NULL) {
    if (rows == MatrixBase<Real>::num_rows_
        && cols == MatrixBase<Real>::num_cols_) {
      if (resize_type == kSetZero)
        this->SetZero();
      return;
    }
    else
      Destroy();
  }
  Init(rows, cols, stride_type);
  if (resize_type == kSetZero) MatrixBase<Real>::SetZero();
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromMat(const MatrixBase<OtherReal> &M,
                                   MatrixTransposeType Trans) {
  if (sizeof(Real) == sizeof(OtherReal) &&
      static_cast<const void*>(M.Data()) ==
      static_cast<const void*>(this->Data())) {
    // CopyFromMat called on same data.  Nothing to do (except sanity checks).
    KALDI_ASSERT(Trans == kNoTrans && M.NumRows() == NumRows() &&
                 M.NumCols() == NumCols() && M.Stride() == Stride());
    return;
  }
  if (Trans == kNoTrans) {
    KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == M.NumCols());
    for (MatrixIndexT i = 0; i < num_rows_; i++)
      (*this).Row(i).CopyFromVec(M.Row(i));
  } else {
    KALDI_ASSERT(num_cols_ == M.NumRows() && num_rows_ == M.NumCols());
    int32 this_stride = stride_, other_stride = M.Stride();
    Real *this_data = data_;
    const OtherReal *other_data = M.Data();
    for (MatrixIndexT i = 0; i < num_rows_; i++)
      for (MatrixIndexT j = 0; j < num_cols_; j++)
        this_data[i * this_stride + j] = other_data[j * other_stride + i];
  }
}

// template instantiations.
template
void MatrixBase<float>::CopyFromMat(const MatrixBase<double> & M,
                                    MatrixTransposeType Trans);
template
void MatrixBase<double>::CopyFromMat(const MatrixBase<float> & M,
                                     MatrixTransposeType Trans);
template
void MatrixBase<float>::CopyFromMat(const MatrixBase<float> & M,
                                    MatrixTransposeType Trans);
template
void MatrixBase<double>::CopyFromMat(const MatrixBase<double> & M,
                                     MatrixTransposeType Trans);

// Specialize the template for CopyFromSp for float, float.
template<>
template<>
void MatrixBase<float>::CopyFromSp(const SpMatrix<float> & M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
  MatrixIndexT num_rows = num_rows_, stride = stride_;
  const float *Mdata = M.Data();
  float *row_data = data_, *col_data = data_;
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    cblas_scopy(i+1, Mdata, 1, row_data, 1); // copy to the row.
    cblas_scopy(i, Mdata, 1, col_data, stride); // copy to the column.
    Mdata += i+1;
    row_data += stride;
    col_data += 1;
  }
}

// Specialize the template for CopyFromSp for double, double.
template<>
template<>
void MatrixBase<double>::CopyFromSp(const SpMatrix<double> & M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
  MatrixIndexT num_rows = num_rows_, stride = stride_;
  const double *Mdata = M.Data();
  double *row_data = data_, *col_data = data_;
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    cblas_dcopy(i+1, Mdata, 1, row_data, 1); // copy to the row.
    cblas_dcopy(i, Mdata, 1, col_data, stride); // copy to the column.
    Mdata += i+1;
    row_data += stride;
    col_data += 1;
  }
}


template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromSp(const SpMatrix<OtherReal> & M) {
  KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
  // MORE EFFICIENT IF LOWER TRIANGULAR!  Reverse code otherwise.
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < i; j++) {
      (*this)(j, i)  = (*this)(i, j) = M(i, j);
    }
    (*this)(i, i) = M(i, i);
  }
}

// Instantiate this function
template
void MatrixBase<float>::CopyFromSp(const SpMatrix<double> & M);
template
void MatrixBase<double>::CopyFromSp(const SpMatrix<float> & M);


template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyFromTp(const TpMatrix<OtherReal> & M,
                                  MatrixTransposeType Trans) {
  if (Trans == kNoTrans) {
    KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
    SetZero();
    Real *out_i = data_;
    const OtherReal *in_i = M.Data();
    for (MatrixIndexT i = 0; i < num_rows_; i++, out_i += stride_, in_i += i) {
      for (MatrixIndexT j = 0; j <= i; j++)
        out_i[j] = in_i[j];
    }
  } else {
    SetZero();
    KALDI_ASSERT(num_rows_ == M.NumRows() && num_cols_ == num_rows_);
    MatrixIndexT stride = stride_;
    Real *out_i = data_;
    const OtherReal *in_i = M.Data();
    for (MatrixIndexT i = 0; i < num_rows_; i++, out_i ++, in_i += i) {
      for (MatrixIndexT j = 0; j <= i; j++)
        out_i[j*stride] = in_i[j];
    }
  }
}

template
void MatrixBase<float>::CopyFromTp(const TpMatrix<float> & M,
                                   MatrixTransposeType trans);
template
void MatrixBase<float>::CopyFromTp(const TpMatrix<double> & M,
                                   MatrixTransposeType trans);
template
void MatrixBase<double>::CopyFromTp(const TpMatrix<float> & M,
                                    MatrixTransposeType trans);
template
void MatrixBase<double>::CopyFromTp(const TpMatrix<double> & M,
                                    MatrixTransposeType trans);


template<typename Real>
void MatrixBase<Real>::CopyRowsFromVec(const VectorBase<Real> &rv) {
  if (rv.Dim() == num_rows_*num_cols_) {
    if (stride_ == num_cols_) {
      // one big copy operation.
      const Real *rv_data = rv.Data();
      std::memcpy(data_, rv_data, sizeof(Real)*num_rows_*num_cols_);
    } else {
      const Real *rv_data = rv.Data();
      for (MatrixIndexT r = 0; r < num_rows_; r++) {
        Real *row_data = RowData(r);
        for (MatrixIndexT c = 0; c < num_cols_; c++) {
          row_data[c] = rv_data[c];
        }
        rv_data += num_cols_;
      }
    }
  } else if (rv.Dim() == num_cols_) {
    const Real *rv_data = rv.Data();
    for (MatrixIndexT r = 0; r < num_rows_; r++)
      std::memcpy(RowData(r), rv_data, sizeof(Real)*num_cols_);
  } else {
    KALDI_ERR << "Wrong sized arguments";
  }
}

template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::CopyRowsFromVec(const VectorBase<OtherReal> &rv) {
  if (rv.Dim() == num_rows_*num_cols_) {
    const OtherReal *rv_data = rv.Data();
    for (MatrixIndexT r = 0; r < num_rows_; r++) {
      Real *row_data = RowData(r);
      for (MatrixIndexT c = 0; c < num_cols_; c++) {
        row_data[c] = static_cast<Real>(rv_data[c]);
      }
      rv_data += num_cols_;
    }
  } else if (rv.Dim() == num_cols_) {
    const OtherReal *rv_data = rv.Data();
    Real *first_row_data = RowData(0);
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      first_row_data[c] = rv_data[c];
    for (MatrixIndexT r = 1; r < num_rows_; r++)
      std::memcpy(RowData(r), first_row_data, sizeof(Real)*num_cols_);
  } else {
    KALDI_ERR << "Wrong sized arguments.";
  }
}


template
void MatrixBase<float>::CopyRowsFromVec(const VectorBase<double> &rv);
template
void MatrixBase<double>::CopyRowsFromVec(const VectorBase<float> &rv);

template<typename Real>
void MatrixBase<Real>::CopyColsFromVec(const VectorBase<Real> &rv) {
  if (rv.Dim() == num_rows_*num_cols_) {
    const Real *v_inc_data = rv.Data();
    Real *m_inc_data = data_;

    for (MatrixIndexT c = 0; c < num_cols_; c++) {
      for (MatrixIndexT r = 0; r < num_rows_; r++) {
        m_inc_data[r * stride_] = v_inc_data[r];
      }
      v_inc_data += num_rows_;
      m_inc_data ++;
    }
  } else if (rv.Dim() == num_rows_) {
    const Real *v_inc_data = rv.Data();
    Real *m_inc_data = data_;
    for (MatrixIndexT r = 0; r < num_rows_; r++) {
      Real value = *(v_inc_data++);
      for (MatrixIndexT c = 0; c < num_cols_; c++)
        m_inc_data[c] = value;
      m_inc_data += stride_;
    }
  } else {
    KALDI_ERR << "Wrong size of arguments.";
  }
}


template<typename Real>
void MatrixBase<Real>::CopyRowFromVec(const VectorBase<Real> &rv, const MatrixIndexT row) {
  KALDI_ASSERT(rv.Dim() == num_cols_ &&
               static_cast<UnsignedMatrixIndexT>(row) <
               static_cast<UnsignedMatrixIndexT>(num_rows_));

  const Real *rv_data = rv.Data();
  Real *row_data = RowData(row);

  std::memcpy(row_data, rv_data, num_cols_ * sizeof(Real));
}

template<typename Real>
void MatrixBase<Real>::CopyDiagFromVec(const VectorBase<Real> &rv) {
  KALDI_ASSERT(rv.Dim() == std::min(num_cols_, num_rows_));
  const Real *rv_data = rv.Data(), *rv_end = rv_data + rv.Dim();
  Real *my_data = this->Data();
  for (; rv_data != rv_end; rv_data++, my_data += (this->stride_+1))
    *my_data = *rv_data;
}

template<typename Real>
void MatrixBase<Real>::CopyColFromVec(const VectorBase<Real> &rv,
                                      const MatrixIndexT col) {
  KALDI_ASSERT(rv.Dim() == num_rows_ &&
               static_cast<UnsignedMatrixIndexT>(col) <
               static_cast<UnsignedMatrixIndexT>(num_cols_));

  const Real *rv_data = rv.Data();
  Real *col_data = data_ + col;

  for (MatrixIndexT r = 0; r < num_rows_; r++)
    col_data[r * stride_] = rv_data[r];
}



template<typename Real>
void Matrix<Real>::RemoveRow(MatrixIndexT i) {
  KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
               static_cast<UnsignedMatrixIndexT>(MatrixBase<Real>::num_rows_)
               && "Access out of matrix");
  for (MatrixIndexT j = i + 1; j <  MatrixBase<Real>::num_rows_; j++)
    MatrixBase<Real>::Row(j-1).CopyFromVec( MatrixBase<Real>::Row(j));
  MatrixBase<Real>::num_rows_--;
}

template<typename Real>
void Matrix<Real>::Destroy() {
  // we need to free the data block if it was defined
  if (NULL != MatrixBase<Real>::data_)
    KALDI_MEMALIGN_FREE( MatrixBase<Real>::data_);
  MatrixBase<Real>::data_ = NULL;
  MatrixBase<Real>::num_rows_ = MatrixBase<Real>::num_cols_
      = MatrixBase<Real>::stride_ = 0;
}



template<typename Real>
void MatrixBase<Real>::MulElements(const MatrixBase<Real> &a) {
  KALDI_ASSERT(a.NumRows() == num_rows_ && a.NumCols() == num_cols_);

  if (num_cols_ == stride_ && num_cols_ == a.stride_) {
    mul_elements(num_rows_ * num_cols_, a.data_, data_);
  } else {
    MatrixIndexT a_stride = a.stride_, stride = stride_;
    Real *data = data_, *a_data = a.data_;
    for (MatrixIndexT i = 0; i < num_rows_; i++) {
      mul_elements(num_cols_, a_data, data);
      a_data += a_stride;
      data += stride;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::DivElements(const MatrixBase<Real> &a) {
  KALDI_ASSERT(a.NumRows() == num_rows_ && a.NumCols() == num_cols_);
  MatrixIndexT i;
  MatrixIndexT j;

  for (i = 0; i < num_rows_; i++) {
    for (j = 0; j < num_cols_; j++) {
      (*this)(i, j) /= a(i, j);
    }
  }
}

template<typename Real>
Real MatrixBase<Real>::Sum() const {
  double sum = 0.0;

  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < num_cols_; j++) {
      sum += (*this)(i, j);
    }
  }

  return (Real)sum;
}

template<typename Real> void MatrixBase<Real>::Max(const MatrixBase<Real> &A) {
  KALDI_ASSERT(A.NumRows() == NumRows() && A.NumCols() == NumCols());
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    Real *row_data = RowData(row);
    const Real *other_row_data = A.RowData(row);
    MatrixIndexT num_cols = num_cols_;
    for (MatrixIndexT col = 0; col < num_cols; col++) {
      row_data[col] = std::max(row_data[col],
                               other_row_data[col]);
    }
  }
}

template<typename Real> void MatrixBase<Real>::Min(const MatrixBase<Real> &A) {
  KALDI_ASSERT(A.NumRows() == NumRows() && A.NumCols() == NumCols());
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    Real *row_data = RowData(row);
    const Real *other_row_data = A.RowData(row);
    MatrixIndexT num_cols = num_cols_;
    for (MatrixIndexT col = 0; col < num_cols; col++) {
      row_data[col] = std::min(row_data[col],
                               other_row_data[col]);
    }
  }
}


template<typename Real> void MatrixBase<Real>::Scale(Real alpha) {
  if (alpha == 1.0) return;
  if (num_rows_ == 0) return;
  if (num_cols_ == stride_) {
    cblas_Xscal(static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_),
                alpha, data_,1);
  } else {
    Real *data = data_;
    for (MatrixIndexT i = 0; i < num_rows_; ++i, data += stride_) {
      cblas_Xscal(num_cols_, alpha, data,1);
    }
  }
}

template<typename Real>  // scales each row by scale[i].
void MatrixBase<Real>::MulRowsVec(const VectorBase<Real> &scale) {
  KALDI_ASSERT(scale.Dim() == num_rows_);
  MatrixIndexT M = num_rows_, N = num_cols_;

  for (MatrixIndexT i = 0; i < M; i++) {
    Real this_scale = scale(i);
    for (MatrixIndexT j = 0; j < N; j++) {
      (*this)(i, j) *= this_scale;
    }
  }
}


template<typename Real>
void MatrixBase<Real>::MulRowsGroupMat(const MatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumRows() == this->NumRows() &&
               this->NumCols() % src.NumCols() == 0);
  int32 group_size = this->NumCols() / src.NumCols(),
      num_groups = this->NumCols() / group_size,
      num_rows = this->NumRows();

  for (MatrixIndexT i = 0; i < num_rows; i++) {
    Real *data = this->RowData(i);
    for (MatrixIndexT j = 0; j < num_groups; j++, data += group_size) {
      Real scale = src(i, j);
      cblas_Xscal(group_size, scale, data, 1);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::GroupPnormDeriv(const MatrixBase<Real> &input,
                                       const MatrixBase<Real> &output,
                                       Real power) {
  KALDI_ASSERT(input.NumCols() == this->NumCols() && input.NumRows() == this->NumRows());
  KALDI_ASSERT(this->NumCols() % output.NumCols() == 0 &&
               this->NumRows() == output.NumRows());

  int group_size = this->NumCols() / output.NumCols(),
    num_rows = this->NumRows(), num_cols = this->NumCols();

  if (power == 1.0) {
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      for (MatrixIndexT j = 0; j < num_cols; j++) {
        Real input_val = input(i, j);
        (*this)(i, j) = (input_val == 0 ? 0 : (input_val > 0 ? 1 : -1));
      }
    }
  } else if (power == std::numeric_limits<Real>::infinity()) {
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      for (MatrixIndexT j = 0; j < num_cols; j++) {
        Real output_val = output(i, j / group_size), input_val = input(i, j);
        if (output_val == 0)
          (*this)(i, j) = 0;
        else
          (*this)(i, j) = (std::abs(input_val) == output_val ? 1.0 : 0.0)
              * (input_val >= 0 ? 1 : -1);
      }
    }
  } else {
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      for (MatrixIndexT j = 0; j < num_cols; j++) {
        Real output_val = output(i, j / group_size),
          input_val = input(i, j);
        if (output_val == 0)
          (*this)(i, j) = 0;
         else
            (*this)(i, j) = pow(std::abs(input_val), power - 1) *
              pow(output_val, 1 - power) * (input_val >= 0 ? 1 : -1) ;
      }
    }
  }
}

template<typename Real>
void MatrixBase<Real>::GroupMaxDeriv(const MatrixBase<Real> &input,
                                     const MatrixBase<Real> &output) {
  KALDI_ASSERT(input.NumCols() == this->NumCols() &&
              input.NumRows() == this->NumRows());
  KALDI_ASSERT(this->NumCols() % output.NumCols() == 0 &&
               this->NumRows() == output.NumRows());

  int group_size = this->NumCols() / output.NumCols(),
      num_rows = this->NumRows(), num_cols = this->NumCols();

  for (MatrixIndexT i = 0; i < num_rows; i++) {
    for (MatrixIndexT j = 0; j < num_cols; j++) {
      Real input_val = input(i, j);
      Real output_val = output(i, j / group_size);
      (*this)(i, j) = (input_val == output_val ? 1 : 0);
    }
  }
}

template<typename Real>  // scales each column by scale[i].
void MatrixBase<Real>::MulColsVec(const VectorBase<Real> &scale) {
  KALDI_ASSERT(scale.Dim() == num_cols_);
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < num_cols_; j++) {
      Real this_scale = scale(j);
      (*this)(i, j) *= this_scale;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::SetZero() {
  if (num_cols_ == stride_)
    memset(data_, 0, sizeof(Real)*num_rows_*num_cols_);
  else
    for (MatrixIndexT row = 0; row < num_rows_; row++)
      memset(data_ + row*stride_, 0, sizeof(Real)*num_cols_);
}

template<typename Real>
void MatrixBase<Real>::Set(Real value) {
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    for (MatrixIndexT col = 0; col < num_cols_; col++) {
      (*this)(row, col) = value;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::SetUnit() {
  SetZero();
  for (MatrixIndexT row = 0; row < std::min(num_rows_, num_cols_); row++)
    (*this)(row, row) = 1.0;
}

template<typename Real>
void MatrixBase<Real>::SetRandn() {
  kaldi::RandomState rstate;
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    Real *row_data = this->RowData(row);
    MatrixIndexT nc = (num_cols_ % 2 == 1) ? num_cols_ - 1 : num_cols_;
    for (MatrixIndexT col = 0; col < nc; col += 2) {
      kaldi::RandGauss2(row_data + col, row_data + col + 1, &rstate);
    }
    if (nc != num_cols_) row_data[nc] = static_cast<Real>(kaldi::RandGauss(&rstate));
  }
}

template<typename Real>
void MatrixBase<Real>::SetRandUniform() {
  kaldi::RandomState rstate;
  for (MatrixIndexT row = 0; row < num_rows_; row++) {
    Real *row_data = this->RowData(row);
    for (MatrixIndexT col = 0; col < num_cols_; col++, row_data++) {
      *row_data = static_cast<Real>(kaldi::RandUniform(&rstate));
    }
  }
}

template<typename Real>
void MatrixBase<Real>::Write(std::ostream &os, bool binary) const {
  if (!os.good()) {
    KALDI_ERR << "Failed to write matrix to stream: stream not good";
  }
  if (binary) {  // Use separate binary and text formats,
    // since in binary mode we need to know if it's float or double.
    std::string my_token = (sizeof(Real) == 4 ? "FM" : "DM");

    WriteToken(os, binary, my_token);
    {
      int32 rows = this->num_rows_;  // make the size 32-bit on disk.
      int32 cols = this->num_cols_;
      KALDI_ASSERT(this->num_rows_ == (MatrixIndexT) rows);
      KALDI_ASSERT(this->num_cols_ == (MatrixIndexT) cols);
      WriteBasicType(os, binary, rows);
      WriteBasicType(os, binary, cols);
    }
    if (Stride() == NumCols())
      os.write(reinterpret_cast<const char*> (Data()), sizeof(Real)
               * static_cast<size_t>(num_rows_) * static_cast<size_t>(num_cols_));
    else
      for (MatrixIndexT i = 0; i < num_rows_; i++)
        os.write(reinterpret_cast<const char*> (RowData(i)), sizeof(Real)
                 * num_cols_);
    if (!os.good()) {
      KALDI_ERR << "Failed to write matrix to stream";
    }
  } else {  // text mode.
    if (num_cols_ == 0) {
      os << " [ ]\n";
    } else {
      os << " [";
      for (MatrixIndexT i = 0; i < num_rows_; i++) {
        os << "\n  ";
        for (MatrixIndexT j = 0; j < num_cols_; j++)
          os << (*this)(i, j) << " ";
      }
      os << "]\n";
    }
  }
}


template<typename Real>
void MatrixBase<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    Matrix<Real> tmp(num_rows_, num_cols_);
    tmp.Read(is, binary, false);  // read without adding.
    if (tmp.num_rows_ != this->num_rows_ || tmp.num_cols_ != this->num_cols_)
      KALDI_ERR << "MatrixBase::Read, size mismatch "
                << this->num_rows_ << ", " << this->num_cols_
                << " vs. " << tmp.num_rows_ << ", " << tmp.num_cols_;
    this->AddMat(1.0, tmp);
    return;
  }
  // now assume add == false.

  //  In order to avoid rewriting this, we just declare a Matrix and
  // use it to read the data, then copy.
  Matrix<Real> tmp;
  tmp.Read(is, binary, false);
  if (tmp.NumRows() != NumRows() || tmp.NumCols() != NumCols()) {
    KALDI_ERR << "MatrixBase<Real>::Read, size mismatch "
              << NumRows() << " x " << NumCols() << " versus "
              << tmp.NumRows() << " x " << tmp.NumCols();
  }
  CopyFromMat(tmp);
}


template<typename Real>
void Matrix<Real>::Read(std::istream & is, bool binary, bool add) {
  if (add) {
    Matrix<Real> tmp;
    tmp.Read(is, binary, false);  // read without adding.
    if (this->num_rows_ == 0) this->Resize(tmp.num_rows_, tmp.num_cols_);
    else {
      if (this->num_rows_ != tmp.num_rows_ || this->num_cols_ != tmp.num_cols_) {
        if (tmp.num_rows_ == 0) return;  // do nothing in this case.
        else KALDI_ERR << "Matrix::Read, size mismatch "
                       << this->num_rows_ <<  ", " << this->num_cols_
                       << " vs. " << tmp.num_rows_ << ", " << tmp.num_cols_;
      }
    }
    this->AddMat(1.0, tmp);
    return;
  }

  // now assume add == false.
  MatrixIndexT pos_at_start = is.tellg();
  std::ostringstream specific_error;

  if (binary) {  // Read in binary mode.
    int peekval = Peek(is, binary);
    if (peekval == 'C') {
      // This code enable us to read CompressedMatrix as a regular matrix.
      CompressedMatrix compressed_mat;
      compressed_mat.Read(is, binary); // at this point, add == false.
      this->Resize(compressed_mat.NumRows(), compressed_mat.NumCols());
      compressed_mat.CopyToMat(this);
      return;
    }
    const char *my_token =  (sizeof(Real) == 4 ? "FM" : "DM");
    char other_token_start = (sizeof(Real) == 4 ? 'D' : 'F');
    if (peekval == other_token_start) {  // need to instantiate the other type to read it.
      typedef typename OtherReal<Real>::Real OtherType;  // if Real == float, OtherType == double, and vice versa.
      Matrix<OtherType> other(this->num_rows_, this->num_cols_);
      other.Read(is, binary, false);  // add is false at this point anyway.
      this->Resize(other.NumRows(), other.NumCols());
      this->CopyFromMat(other);
      return;
    }
    std::string token;
    ReadToken(is, binary, &token);
    if (token != my_token) {
      if (token.length() > 20) token = token.substr(0, 17) + "...";
      specific_error << ": Expected token " << my_token << ", got " << token;
      goto bad;
    }
    int32 rows, cols;
    ReadBasicType(is, binary, &rows);  // throws on error.
    ReadBasicType(is, binary, &cols);  // throws on error.
    if ((MatrixIndexT)rows != this->num_rows_ || (MatrixIndexT)cols != this->num_cols_) {
      this->Resize(rows, cols);
    }
    if (this->Stride() == this->NumCols() && rows*cols!=0) {
      is.read(reinterpret_cast<char*>(this->Data()),
              sizeof(Real)*rows*cols);
      if (is.fail()) goto bad;
    } else {
      for (MatrixIndexT i = 0; i < (MatrixIndexT)rows; i++) {
        is.read(reinterpret_cast<char*>(this->RowData(i)), sizeof(Real)*cols);
        if (is.fail()) goto bad;
      }
    }
    if (is.eof()) return;
    if (is.fail()) goto bad;
    return;
  } else {  // Text mode.
    std::string str;
    is >> str; // get a token
    if (is.fail()) { specific_error << ": Expected \"[\", got EOF"; goto bad; }
    // if ((str.compare("DM") == 0) || (str.compare("FM") == 0)) {  // Back compatibility.
    // is >> str;  // get #rows
    //  is >> str;  // get #cols
    //  is >> str;  // get "["
    // }
    if (str == "[]") { Resize(0, 0); return; } // Be tolerant of variants.
    else if (str != "[") {
      if (str.length() > 20) str = str.substr(0, 17) + "...";
      specific_error << ": Expected \"[\", got \"" << str << '"';
      goto bad;
    }
    // At this point, we have read "[".
    std::vector<std::vector<Real>* > data;
    std::vector<Real> *cur_row = new std::vector<Real>;
    while (1) {
      int i = is.peek();
      if (i == -1) { specific_error << "Got EOF while reading matrix data"; goto cleanup; }
      else if (static_cast<char>(i) == ']') {  // Finished reading matrix.
        is.get();  // eat the "]".
        i = is.peek();
        if (static_cast<char>(i) == '\r') {
          is.get();
          is.get();  // get \r\n (must eat what we wrote)
        } else if (static_cast<char>(i) == '\n') { is.get(); } // get \n (must eat what we wrote)
        if (is.fail()) {
          KALDI_WARN << "After end of matrix data, read error.";
          // we got the data we needed, so just warn for this error.
        }
        // Now process the data.
        if (!cur_row->empty()) data.push_back(cur_row);
        else delete(cur_row);
        cur_row = NULL;
        if (data.empty()) { this->Resize(0, 0); return; }
        else {
          int32 num_rows = data.size(), num_cols = data[0]->size();
          this->Resize(num_rows, num_cols);
          for (int32 i = 0; i < num_rows; i++) {
            if (static_cast<int32>(data[i]->size()) != num_cols) {
              specific_error << "Matrix has inconsistent #cols: " << num_cols
                             << " vs." << data[i]->size() << " (processing row"
                             << i << ")";
              goto cleanup;
            }
            for (int32 j = 0; j < num_cols; j++)
              (*this)(i, j) = (*(data[i]))[j];
            delete data[i];
            data[i] = NULL;
          }
        }
        return;
      } else if (static_cast<char>(i) == '\n' || static_cast<char>(i) == ';') {
        // End of matrix row.
        is.get();
        if (cur_row->size() != 0) {
          data.push_back(cur_row);
          cur_row = new std::vector<Real>;
          cur_row->reserve(data.back()->size());
        }
      } else if ( (i >= '0' && i <= '9') || i == '-' ) {  // A number...
        Real r;
        is >> r;
        if (is.fail()) {
          specific_error << "Stream failure/EOF while reading matrix data.";
          goto cleanup;
        }
        cur_row->push_back(r);
      } else if (isspace(i)) {
        is.get();  // eat the space and do nothing.
      } else {  // NaN or inf or error.
        std::string str;
        is >> str;
        if (!KALDI_STRCASECMP(str.c_str(), "inf") ||
            !KALDI_STRCASECMP(str.c_str(), "infinity")) {
          cur_row->push_back(std::numeric_limits<Real>::infinity());
          KALDI_WARN << "Reading infinite value into matrix.";
        } else if (!KALDI_STRCASECMP(str.c_str(), "nan")) {
          cur_row->push_back(std::numeric_limits<Real>::quiet_NaN());
          KALDI_WARN << "Reading NaN value into matrix.";
        } else {
          if (str.length() > 20) str = str.substr(0, 17) + "...";
          specific_error << "Expecting numeric matrix data, got " << str;
          goto cleanup;
        }
      }
    }
    // Note, we never leave the while () loop before this
    // line (we return from it.)
 cleanup: // We only reach here in case of error in the while loop above.
    if(cur_row != NULL)
      delete cur_row;
    for (size_t i = 0; i < data.size(); i++)
      if(data[i] != NULL)
        delete data[i];
    // and then go on to "bad" below, where we print error.
  }
bad:
  KALDI_ERR << "Failed to read matrix from stream.  " << specific_error.str()
            << " File position at start is "
            << pos_at_start << ", currently " << is.tellg();
}


// Constructor... note that this is not const-safe as it would
// be quite complicated to implement a "const SubMatrix" class that
// would not allow its contents to be changed.
template<typename Real>
SubMatrix<Real>::SubMatrix(const MatrixBase<Real> &M,
                           const MatrixIndexT ro,
                           const MatrixIndexT r,
                           const MatrixIndexT co,
                           const MatrixIndexT c) {
  if (r == 0 || c == 0) {
    // we support the empty sub-matrix as a special case.
    KALDI_ASSERT(c == 0 && r == 0);
    this->data_ = NULL;
    this->num_cols_ = 0;
    this->num_rows_ = 0;
    this->stride_ = 0;
    return;
  }
  KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(ro) <
               static_cast<UnsignedMatrixIndexT>(M.num_rows_) &&
               static_cast<UnsignedMatrixIndexT>(co) <
               static_cast<UnsignedMatrixIndexT>(M.num_cols_) &&
               static_cast<UnsignedMatrixIndexT>(r) <=
               static_cast<UnsignedMatrixIndexT>(M.num_rows_ - ro) &&
               static_cast<UnsignedMatrixIndexT>(c) <=
               static_cast<UnsignedMatrixIndexT>(M.num_cols_ - co));
  // point to the begining of window
  MatrixBase<Real>::num_rows_ = r;
  MatrixBase<Real>::num_cols_ = c;
  MatrixBase<Real>::stride_ = M.Stride();
  MatrixBase<Real>::data_ = M.Data_workaround() +
      static_cast<size_t>(co) +
      static_cast<size_t>(ro) * static_cast<size_t>(M.Stride());
}


template<typename Real>
SubMatrix<Real>::SubMatrix(Real *data,
                           MatrixIndexT num_rows,
                           MatrixIndexT num_cols,
                           MatrixIndexT stride):
    MatrixBase<Real>(data, num_cols, num_rows, stride) { // caution: reversed order!
  if (data == NULL) {
    KALDI_ASSERT(num_rows * num_cols == 0);
    this->num_rows_ = 0;
    this->num_cols_ = 0;
    this->stride_ = 0;
  } else {
    KALDI_ASSERT(this->stride_ >= this->num_cols_);
  }
}


template<typename Real>
void MatrixBase<Real>::Add(const Real alpha) {
  Real *data = data_;
  MatrixIndexT stride = stride_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      data[c + stride*r] += alpha;
}

template<typename Real>
void MatrixBase<Real>::AddToDiag(const Real alpha) {
  Real *data = data_;
  MatrixIndexT this_stride = stride_ + 1,
      num_to_add = std::min(num_rows_, num_cols_);
  for (MatrixIndexT r = 0; r < num_to_add; r++)
    data[r * this_stride] += alpha;
}


template<typename Real>
Real MatrixBase<Real>::Cond() const {
  KALDI_ASSERT(num_rows_ > 0&&num_cols_ > 0);
  Vector<Real> singular_values(std::min(num_rows_, num_cols_));
  Svd(&singular_values);  // Get singular values...
  Real min = singular_values(0), max = singular_values(0);  // both absolute values...
  for (MatrixIndexT i = 1;i < singular_values.Dim();i++) {
    min = std::min((Real)std::abs(singular_values(i)), min); max = std::max((Real)std::abs(singular_values(i)), max);
  }
  if (min > 0) return max/min;
  else return std::numeric_limits<Real>::infinity();
}

template<typename Real>
Real MatrixBase<Real>::Trace(bool check_square) const  {
  KALDI_ASSERT(!check_square || num_rows_ == num_cols_);
  Real ans = 0.0;
  for (MatrixIndexT r = 0;r < std::min(num_rows_, num_cols_);r++) ans += data_ [r + stride_*r];
  return ans;
}

template<typename Real>
Real MatrixBase<Real>::Max() const {
  KALDI_ASSERT(num_rows_ > 0 && num_cols_ > 0);
  Real ans= *data_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      if (data_[c + stride_*r] > ans)
        ans = data_[c + stride_*r];
  return ans;
}

template<typename Real>
Real MatrixBase<Real>::Min() const {
  KALDI_ASSERT(num_rows_ > 0 && num_cols_ > 0);
  Real ans= *data_;
  for (MatrixIndexT r = 0; r < num_rows_; r++)
    for (MatrixIndexT c = 0; c < num_cols_; c++)
      if (data_[c + stride_*r] < ans)
        ans = data_[c + stride_*r];
  return ans;
}



template <typename Real>
void MatrixBase<Real>::AddMatMatMat(Real alpha,
                                    const MatrixBase<Real> &A, MatrixTransposeType transA,
                                    const MatrixBase<Real> &B, MatrixTransposeType transB,
                                    const MatrixBase<Real> &C, MatrixTransposeType transC,
                                    Real beta) {
  // Note on time taken with different orders of computation.  Assume not transposed in this /
  // discussion. Firstly, normalize expressions using A.NumCols == B.NumRows and B.NumCols == C.NumRows, prefer
  // rows where there is a choice.
  // time taken for (AB) is:  A.NumRows*B.NumRows*C.Rows
  // time taken for (AB)C is A.NumRows*C.NumRows*C.Cols
  // so this order is A.NumRows*B.NumRows*C.NumRows + A.NumRows*C.NumRows*C.NumCols.

  // time taken for (BC) is: B.NumRows*C.NumRows*C.Cols
  // time taken for A(BC) is: A.NumRows*B.NumRows*C.Cols
  // so this order is B.NumRows*C.NumRows*C.NumCols + A.NumRows*B.NumRows*C.Cols

  MatrixIndexT ARows = A.num_rows_, ACols = A.num_cols_, BRows = B.num_rows_, BCols = B.num_cols_,
      CRows = C.num_rows_, CCols = C.num_cols_;
  if (transA == kTrans) std::swap(ARows, ACols);
  if (transB == kTrans) std::swap(BRows, BCols);
  if (transC == kTrans) std::swap(CRows, CCols);

  MatrixIndexT AB_C_time = ARows*BRows*CRows + ARows*CRows*CCols;
  MatrixIndexT A_BC_time = BRows*CRows*CCols + ARows*BRows*CCols;

  if (AB_C_time < A_BC_time) {
    Matrix<Real> AB(ARows, BCols);
    AB.AddMatMat(1.0, A, transA, B, transB, 0.0);  // AB = A * B.
    (*this).AddMatMat(alpha, AB, kNoTrans, C, transC, beta);
  } else {
    Matrix<Real> BC(BRows, CCols);
    BC.AddMatMat(1.0, B, transB, C, transC, 0.0);  // BC = B * C.
    (*this).AddMatMat(alpha, A, transA, BC, kNoTrans, beta);
  }
}




template<typename Real>
void MatrixBase<Real>::DestructiveSvd(VectorBase<Real> *s, MatrixBase<Real> *U, MatrixBase<Real> *Vt) {
  // Svd, *this = U*diag(s)*Vt.
  // With (*this).num_rows_ == m, (*this).num_cols_ == n,
  // Support only skinny Svd with m>=n (NumRows>=NumCols), and zero sizes for U and Vt mean
  // we do not want that output.  We expect that s.Dim() == m,
  // U is either 0 by 0 or m by n, and rv is either 0 by 0 or n by n.
  // Throws exception on error.

  KALDI_ASSERT(num_rows_>=num_cols_ && "Svd requires that #rows by >= #cols.");  // For compatibility with JAMA code.
  KALDI_ASSERT(s->Dim() == num_cols_);  // s should be the smaller dim.
  KALDI_ASSERT(U == NULL || (U->num_rows_ == num_rows_&&U->num_cols_ == num_cols_));
  KALDI_ASSERT(Vt == NULL || (Vt->num_rows_ == num_cols_&&Vt->num_cols_ == num_cols_));

  Real prescale = 1.0;
  if ( std::abs((*this)(0, 0) ) < 1.0e-30) {  // Very tiny value... can cause problems in Svd.
    Real max_elem = LargestAbsElem();
    if (max_elem != 0) {
      prescale = 1.0 / max_elem;
      if (std::abs(prescale) == std::numeric_limits<Real>::infinity()) { prescale = 1.0e+40; }
      (*this).Scale(prescale);
    }
  }

#if !defined(HAVE_ATLAS) && !defined(USE_KALDI_SVD)
  // "S" == skinny Svd (only one we support because of compatibility with Jama one which is only skinny),
  // "N"== no eigenvectors wanted.
  LapackGesvd(s, U, Vt);
#else
  /*  if (num_rows_ > 1 && num_cols_ > 1 && (*this)(0, 0) == (*this)(1, 1)
      && Max() == Min() && (*this)(0, 0) != 0.0) { // special case that JamaSvd sometimes crashes on.
      KALDI_WARN << "Jama SVD crashes on this type of matrix, perturbing it to prevent crash.";
      for(int32 i = 0; i < NumRows(); i++)
      (*this)(i, i)  *= 1.00001;
      }*/
  bool ans = JamaSvd(s, U, Vt);
  if (Vt != NULL) Vt->Transpose();  // possibly to do: change this and also the transpose inside the JamaSvd routine.  note, Vt is square.
  if (!ans) {
    KALDI_ERR << "Error doing Svd";  // This one will be caught.
  }
#endif
  if (prescale != 1.0) s->Scale(1.0/prescale);
}

template<typename Real>
void MatrixBase<Real>::Svd(VectorBase<Real> *s, MatrixBase<Real> *U, MatrixBase<Real> *Vt) const {
  try {
    if (num_rows_ >= num_cols_) {
      Matrix<Real> tmp(*this);
      tmp.DestructiveSvd(s, U, Vt);
    } else {
      Matrix<Real> tmp(*this, kTrans);  // transpose of *this.
      // rVt will have different dim so cannot transpose in-place --> use a temp matrix.
      Matrix<Real> Vt_Trans(Vt ? Vt->num_cols_ : 0, Vt ? Vt->num_rows_ : 0);
      // U will be transpose
      tmp.DestructiveSvd(s, Vt ? &Vt_Trans : NULL, U);
      if (U) U->Transpose();
      if (Vt) Vt->CopyFromMat(Vt_Trans, kTrans);  // copy with transpose.
    }
  } catch (...) {
    KALDI_ERR << "Error doing Svd (did not converge), first part of matrix is\n"
              << SubMatrix<Real>(*this, 0, std::min((MatrixIndexT)10, num_rows_),
                                 0, std::min((MatrixIndexT)10, num_cols_))
              << ", min and max are: " << Min() << ", " << Max();
  }
}

template<typename Real>
bool MatrixBase<Real>::IsSymmetric(Real cutoff) const {
  MatrixIndexT R = num_rows_, C = num_cols_;
  if (R != C) return false;
  Real bad_sum = 0.0, good_sum = 0.0;
  for (MatrixIndexT i = 0;i < R;i++) {
    for (MatrixIndexT j = 0;j < i;j++) {
      Real a = (*this)(i, j), b = (*this)(j, i), avg = 0.5*(a+b), diff = 0.5*(a-b);
      good_sum += std::abs(avg); bad_sum += std::abs(diff);
    }
    good_sum += std::abs((*this)(i, i));
  }
  if (bad_sum > cutoff*good_sum) return false;
  return true;
}

template<typename Real>
bool MatrixBase<Real>::IsDiagonal(Real cutoff) const{
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real bad_sum = 0.0, good_sum = 0.0;
  for (MatrixIndexT i = 0;i < R;i++) {
    for (MatrixIndexT j = 0;j < C;j++) {
      if (i == j) good_sum += std::abs((*this)(i, j));
      else bad_sum += std::abs((*this)(i, j));
    }
  }
  return (!(bad_sum > good_sum * cutoff));
}

// This does nothing, it's designed to trigger Valgrind errors
// if any memory is uninitialized.
template<typename Real>
void MatrixBase<Real>::TestUninitialized() const {
  MatrixIndexT R = num_rows_, C = num_cols_, positive = 0;
  for (MatrixIndexT i = 0; i < R; i++)
    for (MatrixIndexT j = 0; j < C; j++)
      if ((*this)(i, j) > 0.0) positive++;
  if (positive > R * C)
    KALDI_ERR << "Error....";
}


template<typename Real>
bool MatrixBase<Real>::IsUnit(Real cutoff) const {
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real bad_max = 0.0;
  for (MatrixIndexT i = 0; i < R;i++)
    for (MatrixIndexT j = 0; j < C;j++)
      bad_max = std::max(bad_max, static_cast<Real>(std::abs( (*this)(i, j) - (i == j?1.0:0.0))));
  return (bad_max <= cutoff);
}

template<typename Real>
bool MatrixBase<Real>::IsZero(Real cutoff)const {
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real bad_max = 0.0;
  for (MatrixIndexT i = 0;i < R;i++)
    for (MatrixIndexT j = 0;j < C;j++)
      bad_max = std::max(bad_max, static_cast<Real>(std::abs( (*this)(i, j) )));
  return (bad_max <= cutoff);
}

template<typename Real>
Real MatrixBase<Real>::FrobeniusNorm() const{
  return std::sqrt(TraceMatMat(*this, *this, kTrans));
}

template<typename Real>
bool MatrixBase<Real>::ApproxEqual(const MatrixBase<Real> &other, float tol) const {
  if (num_rows_ != other.num_rows_ || num_cols_ != other.num_cols_)
    KALDI_ERR << "ApproxEqual: size mismatch.";
  Matrix<Real> tmp(*this);
  tmp.AddMat(-1.0, other);
  return (tmp.FrobeniusNorm() <= static_cast<Real>(tol) *
          this->FrobeniusNorm());
}

template<typename Real>
bool MatrixBase<Real>::Equal(const MatrixBase<Real> &other) const {
  if (num_rows_ != other.num_rows_ || num_cols_ != other.num_cols_)
    KALDI_ERR << "Equal: size mismatch.";
  for (MatrixIndexT i = 0; i < num_rows_; i++)
    for (MatrixIndexT j = 0; j < num_cols_; j++)
      if ( (*this)(i, j) != other(i, j))
        return false;
  return true;
}


template<typename Real>
Real MatrixBase<Real>::LargestAbsElem() const{
  MatrixIndexT R = num_rows_, C = num_cols_;
  Real largest = 0.0;
  for (MatrixIndexT i = 0;i < R;i++)
    for (MatrixIndexT j = 0;j < C;j++)
      largest = std::max(largest, (Real)std::abs((*this)(i, j)));
  return largest;
}


template<typename Real>
void MatrixBase<Real>::OrthogonalizeRows() {
  KALDI_ASSERT(NumRows() <= NumCols());
  MatrixIndexT num_rows = num_rows_;
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    int32 counter = 0;
    while (1) {
      Real start_prod = VecVec(this->Row(i), this->Row(i));
      if (start_prod - start_prod != 0.0 || start_prod == 0.0) {
        KALDI_WARN << "Self-product of row " << i << " of matrix is "
                   << start_prod << ", randomizing.";
        this->Row(i).SetRandn();
        counter++;
        continue;  // loop again.
      }
      for (MatrixIndexT j = 0; j < i; j++) {
        Real prod = VecVec(this->Row(i), this->Row(j));
        this->Row(i).AddVec(-prod, this->Row(j));
      }
      Real end_prod = VecVec(this->Row(i), this->Row(i));
      if (end_prod <= 0.01 * start_prod) { // We removed
        // almost all of the vector during orthogonalization,
        // so we have reason to doubt (for roundoff reasons)
        // that it's still orthogonal to the other vectors.
        // We need to orthogonalize again.
        if (end_prod == 0.0) { // Row is exactly zero:
          // generate random direction.
          this->Row(i).SetRandn();
        }
        counter++;
        if (counter > 100)
          KALDI_ERR << "Loop detected while orthogalizing matrix.";
      } else {
        this->Row(i).Scale(1.0 / std::sqrt(end_prod));
        break;
      }
    }
  }
}


// Uses Svd to compute the eigenvalue decomposition of a symmetric positive semidefinite
//   matrix:
// (*this) = rU * diag(rs) * rU^T, with rU an orthogonal matrix so rU^{-1} = rU^T.
// Does this by computing svd (*this) = U diag(rs) V^T ... answer is just U diag(rs) U^T.
// Throws exception if this failed to within supplied precision (typically because *this was not
// symmetric positive definite).

template<typename Real>
void MatrixBase<Real>::SymPosSemiDefEig(VectorBase<Real> *rs, MatrixBase<Real> *rU, Real check_thresh) // e.g. check_thresh = 0.001
{
  const MatrixIndexT D = num_rows_;

  KALDI_ASSERT(num_rows_ == num_cols_);
  KALDI_ASSERT(IsSymmetric() && "SymPosSemiDefEig: expecting input to be symmetrical.");
  KALDI_ASSERT(rU->num_rows_ == D && rU->num_cols_ == D && rs->Dim() == D);

  Matrix<Real>  Vt(D, D);
  Svd(rs, rU, &Vt);

  // First just zero any singular values if the column of U and V do not have +ve dot product--
  // this may mean we have small negative eigenvalues, and if we zero them the result will be closer to correct.
  for (MatrixIndexT i = 0;i < D;i++) {
    Real sum = 0.0;
    for (MatrixIndexT j = 0;j < D;j++) sum += (*rU)(j, i) * Vt(i, j);
    if (sum < 0.0) (*rs)(i) = 0.0;
  }

  {
    Matrix<Real> tmpU(*rU); Vector<Real> tmps(*rs); tmps.ApplyPow(0.5);
    tmpU.MulColsVec(tmps);
    SpMatrix<Real> tmpThis(D);
    tmpThis.AddMat2(1.0, tmpU, kNoTrans, 0.0);
    Matrix<Real> tmpThisFull(tmpThis);
    float new_norm = tmpThisFull.FrobeniusNorm();
    float old_norm = (*this).FrobeniusNorm();
    tmpThisFull.AddMat(-1.0, (*this));

    if (!(old_norm == 0 && new_norm == 0)) {
      float diff_norm = tmpThisFull.FrobeniusNorm();
      if (std::abs(new_norm-old_norm) > old_norm*check_thresh || diff_norm > old_norm*check_thresh) {
        KALDI_WARN << "SymPosSemiDefEig seems to have failed " << diff_norm << " !<< "
                   << check_thresh << "*" << old_norm << ", maybe matrix was not "
                   << "positive semi definite.  Continuing anyway.";
      }
    }
  }
}


template<typename Real>
Real MatrixBase<Real>::LogDet(Real *det_sign) const {
  Real log_det;
  Matrix<Real> tmp(*this);
  tmp.Invert(&log_det, det_sign, false);  // false== output not needed (saves some computation).
  return log_det;
}

template<typename Real>
void MatrixBase<Real>::InvertDouble(Real *log_det, Real *det_sign,
                                    bool inverse_needed) {
  double log_det_tmp, det_sign_tmp;
  Matrix<double> dmat(*this);
  dmat.Invert(&log_det_tmp, &det_sign_tmp, inverse_needed);
  if (inverse_needed) (*this).CopyFromMat(dmat);
  if (log_det) *log_det = log_det_tmp;
  if (det_sign) *det_sign = det_sign_tmp;
}

template<class Real>
void MatrixBase<Real>::CopyFromMat(const CompressedMatrix &mat) {
  mat.CopyToMat(this);
}

template<class Real>
Matrix<Real>::Matrix(const CompressedMatrix &M): MatrixBase<Real>() {
  Resize(M.NumRows(), M.NumCols(), kUndefined);
  M.CopyToMat(this);
}



template<typename Real>
void MatrixBase<Real>::InvertElements() {
  for (MatrixIndexT r = 0; r < num_rows_; r++) {
    for (MatrixIndexT c = 0; c < num_cols_; c++) {
      (*this)(r, c) = static_cast<Real>(1.0 / (*this)(r, c));
    }
  }
}

template<typename Real>
void MatrixBase<Real>::Transpose() {
  KALDI_ASSERT(num_rows_ == num_cols_);
  MatrixIndexT M = num_rows_;
  for (MatrixIndexT i = 0;i < M;i++)
    for (MatrixIndexT j = 0;j < i;j++) {
      Real &a = (*this)(i, j), &b = (*this)(j, i);
      std::swap(a, b);
    }
}


template<typename Real>
void Matrix<Real>::Transpose() {
  if (this->num_rows_ != this->num_cols_) {
    Matrix<Real> tmp(*this, kTrans);
    Resize(this->num_cols_, this->num_rows_);
    this->CopyFromMat(tmp);
  } else {
    (static_cast<MatrixBase<Real>&>(*this)).Transpose();
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyFloor(Real floor_val) {
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    Real *data = this->RowData(i);
    for (MatrixIndexT j = 0; j < num_cols; j++)
      data[j] = (data[j] < floor_val ? floor_val : data[j]);
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyCeiling(Real ceiling_val) {
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    Real *data = this->RowData(i);
    for (MatrixIndexT j = 0; j < num_cols; j++)
      data[j] = (data[j] > ceiling_val ? ceiling_val : data[j]);
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyLog() {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    Row(i).ApplyLog();
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyExp() {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    Row(i).ApplyExp();
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyExpSpecial() {
  int32 num_rows = num_rows_, num_cols = num_cols_,
      stride = stride_;
  Real *data = data_;
  for (MatrixIndexT i = 0; i < num_rows; ++i) {
    for (MatrixIndexT j = 0; j < num_cols; ++j) {
      Real &x = *(data + j + stride * i);
      x = x < Real(0) ? Exp(x) : x + Real(1);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyPow(Real power) {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    Row(i).ApplyPow(power);
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyPowAbs(Real power, bool include_sign) {
  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    Row(i).ApplyPowAbs(power, include_sign);
  }
}

template<typename Real>
void MatrixBase<Real>::ApplyHeaviside() {
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    Real *data = this->RowData(i);
    for (MatrixIndexT j = 0; j < num_cols; j++)
      data[j] = (data[j] > 0 ? 1.0 : 0.0);
  }
}

template<typename Real>
void MatrixBase<Real>::Heaviside(const MatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_;
  Real *row_data = data_;
  const Real *src_row_data = src.Data();
  for (MatrixIndexT row = 0; row < num_rows;
       row++,row_data += stride_, src_row_data += src.stride_) {
    for (MatrixIndexT col = 0; col < num_cols; col++)
      row_data[col] = (src_row_data[col] > 0 ? 1.0 : 0.0);
  }
}


template<typename Real>
bool MatrixBase<Real>::Power(Real power) {
  KALDI_ASSERT(num_rows_ > 0 && num_rows_ == num_cols_);
  MatrixIndexT n = num_rows_;
  Matrix<Real> P(n, n);
  Vector<Real> re(n), im(n);
  this->Eig(&P, &re, &im);
  // Now attempt to take the complex eigenvalues to this power.
  for (MatrixIndexT i = 0; i < n; i++)
    if (!AttemptComplexPower(&(re(i)), &(im(i)), power))
      return false;  // e.g. real and negative, or zero, eigenvalues.

  Matrix<Real> D(n, n);  // D to the power.
  CreateEigenvalueMatrix(re, im, &D);

  Matrix<Real> tmp(n, n);  // P times D
  tmp.AddMatMat(1.0, P, kNoTrans, D, kNoTrans, 0.0);  // tmp := P*D
  P.Invert();
  // next line is: *this = tmp * P^{-1} = P * D * P^{-1}
  (*this).AddMatMat(1.0, tmp, kNoTrans, P, kNoTrans, 0.0);
  return true;
}

template<typename Real>
void Matrix<Real>::Swap(Matrix<Real> *other) {
  std::swap(this->data_, other->data_);
  std::swap(this->num_cols_, other->num_cols_);
  std::swap(this->num_rows_, other->num_rows_);
  std::swap(this->stride_, other->stride_);
}

// Repeating this comment that appeared in the header:
// Eigenvalue Decomposition of a square NxN matrix into the form (*this) = P D
// P^{-1}.  Be careful: the relationship of D to the eigenvalues we output is
// slightly complicated, due to the need for P to be real.  In the symmetric
// case D is diagonal and real, but in
// the non-symmetric case there may be complex-conjugate pairs of eigenvalues.
// In this case, for the equation (*this) = P D P^{-1} to hold, D must actually
// be block diagonal, with 2x2 blocks corresponding to any such pairs.  If a
// pair is lambda +- i*mu, D will have a corresponding 2x2 block
// [lambda, mu; -mu, lambda].
// Note that if the input matrix (*this) is non-invertible, P may not be invertible
// so in this case instead of the equation (*this) = P D P^{-1} holding, we have
// instead (*this) P = P D.
//
// By making the pointer arguments non-NULL or NULL, the user can choose to take
// not to take the eigenvalues directly, and/or the matrix D which is block-diagonal
// with 2x2 blocks.
template<typename Real>
void MatrixBase<Real>::Eig(MatrixBase<Real> *P,
                           VectorBase<Real> *r,
                           VectorBase<Real> *i) const {
  EigenvalueDecomposition<Real>  eig(*this);
  if (P) eig.GetV(P);
  if (r) eig.GetRealEigenvalues(r);
  if (i) eig.GetImagEigenvalues(i);
}


// Begin non-member function definitions.

//  /**
//   * @brief Extension of the HTK header
//  */
// struct HtkHeaderExt
//  {
// INT_32 mHeaderSize;
// INT_32 mVersion;
// INT_32 mSampSize;
// };

template<typename Real>
bool ReadHtk(std::istream &is, Matrix<Real> *M_ptr, HtkHeader *header_ptr)
{
  // check instantiated with double or float.
  KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  Matrix<Real> &M = *M_ptr;
  HtkHeader htk_hdr;

  // TODO(arnab): this fails if the HTK file has CRC cheksum or is compressed.
  is.read((char*)&htk_hdr, sizeof(htk_hdr));  // we're being really POSIX here!
  if (is.fail()) {
    KALDI_WARN << "Could not read header from HTK feature file ";
    return false;
  }

  KALDI_SWAP4(htk_hdr.mNSamples);
  KALDI_SWAP4(htk_hdr.mSamplePeriod);
  KALDI_SWAP2(htk_hdr.mSampleSize);
  KALDI_SWAP2(htk_hdr.mSampleKind);

  bool has_checksum = false;
  {
    // See HParm.h in HTK code for sources of these things.
    enum BaseParmKind{
      Waveform, Lpc, Lprefc, Lpcepstra, Lpdelcep,
      Irefc, Mfcc, Fbank, Melspec, User, Discrete, Plp, Anon };

    const int32 IsCompressed = 02000, HasChecksum = 010000, HasVq = 040000,
        Problem = IsCompressed | HasVq;
    int32 base_parm = htk_hdr.mSampleKind & (077);
    has_checksum = (base_parm & HasChecksum) != 0;
    htk_hdr.mSampleKind &= ~HasChecksum; // We don't support writing with
                                         // checksum so turn it off.
    if (htk_hdr.mSampleKind & Problem)
      KALDI_ERR << "Code to read HTK features does not support compressed "
          "features, or features with VQ.";
    if (base_parm == Waveform || base_parm == Irefc || base_parm == Discrete)
      KALDI_ERR << "Attempting to read HTK features from unsupported type "
          "(e.g. waveform or discrete features.";
  }

  KALDI_VLOG(3) << "HTK header: Num Samples: " << htk_hdr.mNSamples
                << "; Sample period: " << htk_hdr.mSamplePeriod
                << "; Sample size: " << htk_hdr.mSampleSize
                << "; Sample kind: " << htk_hdr.mSampleKind;

  M.Resize(htk_hdr.mNSamples, htk_hdr.mSampleSize / sizeof(float));

  MatrixIndexT i;
  MatrixIndexT j;
  if (sizeof(Real) == sizeof(float)) {
    for (i = 0; i< M.NumRows(); i++) {
      is.read((char*)M.RowData(i), sizeof(float)*M.NumCols());
      if (is.fail()) {
        KALDI_WARN << "Could not read data from HTK feature file ";
        return false;
      }
      if (MachineIsLittleEndian()) {
        MatrixIndexT C = M.NumCols();
        for (j = 0; j < C; j++) {
          KALDI_SWAP4((M(i, j)));  // The HTK standard is big-endian!
        }
      }
    }
  } else {
    float *pmem = new float[M.NumCols()];
    for (i = 0; i < M.NumRows(); i++) {
      is.read((char*)pmem, sizeof(float)*M.NumCols());
      if (is.fail()) {
        KALDI_WARN << "Could not read data from HTK feature file ";
        delete [] pmem;
        return false;
      }
      MatrixIndexT C = M.NumCols();
      for (j = 0; j < C; j++) {
        if (MachineIsLittleEndian())  // HTK standard is big-endian!
          KALDI_SWAP4(pmem[j]);
        M(i, j) = static_cast<Real>(pmem[j]);
      }
    }
    delete [] pmem;
  }
  if (header_ptr) *header_ptr = htk_hdr;
  if (has_checksum) {
    int16 checksum;
    is.read((char*)&checksum, sizeof(checksum));
    if (is.fail())
      KALDI_WARN << "Could not read checksum from HTK feature file ";
    // We ignore the checksum.
  }
  return true;
}


template
bool ReadHtk(std::istream &is, Matrix<float> *M, HtkHeader *header_ptr);

template
bool ReadHtk(std::istream &is, Matrix<double> *M, HtkHeader *header_ptr);

template<typename Real>
bool WriteHtk(std::ostream &os, const MatrixBase<Real> &M, HtkHeader htk_hdr) // header may be derived from a previous call to ReadHtk.  Must be in binary mode.
{
  KALDI_ASSERT(M.NumRows() == static_cast<MatrixIndexT>(htk_hdr.mNSamples));
  KALDI_ASSERT(M.NumCols() == static_cast<MatrixIndexT>(htk_hdr.mSampleSize) /
               static_cast<MatrixIndexT>(sizeof(float)));

  KALDI_SWAP4(htk_hdr.mNSamples);
  KALDI_SWAP4(htk_hdr.mSamplePeriod);
  KALDI_SWAP2(htk_hdr.mSampleSize);
  KALDI_SWAP2(htk_hdr.mSampleKind);

  os.write((char*)&htk_hdr, sizeof(htk_hdr));
  if (os.fail())  goto bad;

  MatrixIndexT i;
  MatrixIndexT j;
  if (sizeof(Real) == sizeof(float) && !MachineIsLittleEndian()) {
    for (i = 0; i< M.NumRows(); i++) {  // Unlikely to reach here ever!
      os.write((char*)M.RowData(i), sizeof(float)*M.NumCols());
      if (os.fail()) goto bad;
    }
  } else {
    float *pmem = new float[M.NumCols()];

    for (i = 0; i < M.NumRows(); i++) {
      const Real *rowData = M.RowData(i);
      for (j = 0;j < M.NumCols();j++)
        pmem[j] =  static_cast<float> ( rowData[j] );
      if (MachineIsLittleEndian())
        for (j = 0;j < M.NumCols();j++)
          KALDI_SWAP4(pmem[j]);
      os.write((char*)pmem, sizeof(float)*M.NumCols());
      if (os.fail()) {
        delete [] pmem;
        goto bad;
      }
    }
    delete [] pmem;
  }
  return true;
bad:
  KALDI_WARN << "Could not write to HTK feature file ";
  return false;
}

template
bool WriteHtk(std::ostream &os, const MatrixBase<float> &M, HtkHeader htk_hdr);

template
bool WriteHtk(std::ostream &os, const MatrixBase<double> &M, HtkHeader htk_hdr);

template<class Real>
bool WriteSphinx(std::ostream &os, const MatrixBase<Real> &M)
{
  // CMUSphinx mfc file header contains count of the floats, followed
  // by the data in float little endian format.

  int size = M.NumRows() * M.NumCols();
  os.write((char*)&size, sizeof(int));
  if (os.fail())  goto bad;

  MatrixIndexT i;
  MatrixIndexT j;
  if (sizeof(Real) == sizeof(float) && MachineIsLittleEndian()) {
    for (i = 0; i< M.NumRows(); i++) {  // Unlikely to reach here ever!
      os.write((char*)M.RowData(i), sizeof(float)*M.NumCols());
      if (os.fail()) goto bad;
    }
  } else {
    float *pmem = new float[M.NumCols()];

    for (i = 0; i < M.NumRows(); i++) {
      const Real *rowData = M.RowData(i);
      for (j = 0;j < M.NumCols();j++)
        pmem[j] =  static_cast<float> ( rowData[j] );
      if (!MachineIsLittleEndian())
        for (j = 0;j < M.NumCols();j++)
          KALDI_SWAP4(pmem[j]);
      os.write((char*)pmem, sizeof(float)*M.NumCols());
      if (os.fail()) {
        delete [] pmem;
        goto bad;
      }
    }
    delete [] pmem;
  }
  return true;
bad:
  KALDI_WARN << "Could not write to Sphinx feature file";
  return false;
}

template
bool WriteSphinx(std::ostream &os, const MatrixBase<float> &M);

template
bool WriteSphinx(std::ostream &os, const MatrixBase<double> &M);

template <typename Real>
Real TraceMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                    const MatrixBase<Real> &B, MatrixTransposeType transB,
                    const MatrixBase<Real> &C, MatrixTransposeType transC) {
  MatrixIndexT ARows = A.NumRows(), ACols = A.NumCols(), BRows = B.NumRows(), BCols = B.NumCols(),
      CRows = C.NumRows(), CCols = C.NumCols();
  if (transA == kTrans) std::swap(ARows, ACols);
  if (transB == kTrans) std::swap(BRows, BCols);
  if (transC == kTrans) std::swap(CRows, CCols);
  KALDI_ASSERT( CCols == ARows && ACols == BRows && BCols == CRows && "TraceMatMatMat: args have mismatched dimensions.");
  if (ARows*BCols < std::min(BRows*CCols, CRows*ACols)) {
    Matrix<Real> AB(ARows, BCols);
    AB.AddMatMat(1.0, A, transA, B, transB, 0.0);  // AB = A * B.
    return TraceMatMat(AB, C, transC);
  } else if ( BRows*CCols < CRows*ACols) {
    Matrix<Real> BC(BRows, CCols);
    BC.AddMatMat(1.0, B, transB, C, transC, 0.0);  // BC = B * C.
    return TraceMatMat(BC, A, transA);
  } else {
    Matrix<Real> CA(CRows, ACols);
    CA.AddMatMat(1.0, C, transC, A, transA, 0.0);  // CA = C * A
    return TraceMatMat(CA, B, transB);
  }
}

template
float TraceMatMatMat(const MatrixBase<float> &A, MatrixTransposeType transA,
                     const MatrixBase<float> &B, MatrixTransposeType transB,
                     const MatrixBase<float> &C, MatrixTransposeType transC);

template
double TraceMatMatMat(const MatrixBase<double> &A, MatrixTransposeType transA,
                      const MatrixBase<double> &B, MatrixTransposeType transB,
                      const MatrixBase<double> &C, MatrixTransposeType transC);


template <typename Real>
Real TraceMatMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                       const MatrixBase<Real> &B, MatrixTransposeType transB,
                       const MatrixBase<Real> &C, MatrixTransposeType transC,
                       const MatrixBase<Real> &D, MatrixTransposeType transD) {
  MatrixIndexT ARows = A.NumRows(), ACols = A.NumCols(), BRows = B.NumRows(), BCols = B.NumCols(),
      CRows = C.NumRows(), CCols = C.NumCols(), DRows = D.NumRows(), DCols = D.NumCols();
  if (transA == kTrans) std::swap(ARows, ACols);
  if (transB == kTrans) std::swap(BRows, BCols);
  if (transC == kTrans) std::swap(CRows, CCols);
  if (transD == kTrans) std::swap(DRows, DCols);
  KALDI_ASSERT( DCols == ARows && ACols == BRows && BCols == CRows && CCols == DRows && "TraceMatMatMat: args have mismatched dimensions.");
  if (ARows*BCols < std::min(BRows*CCols, std::min(CRows*DCols, DRows*ACols))) {
    Matrix<Real> AB(ARows, BCols);
    AB.AddMatMat(1.0, A, transA, B, transB, 0.0);  // AB = A * B.
    return TraceMatMatMat(AB, kNoTrans, C, transC, D, transD);
  } else if ((BRows*CCols) < std::min(CRows*DCols, DRows*ACols)) {
    Matrix<Real> BC(BRows, CCols);
    BC.AddMatMat(1.0, B, transB, C, transC, 0.0);  // BC = B * C.
    return TraceMatMatMat(BC, kNoTrans, D, transD, A, transA);
  } else if (CRows*DCols < DRows*ACols) {
    Matrix<Real> CD(CRows, DCols);
    CD.AddMatMat(1.0, C, transC, D, transD, 0.0);  // CD = C * D
    return TraceMatMatMat(CD, kNoTrans, A, transA, B, transB);
  } else {
    Matrix<Real> DA(DRows, ACols);
    DA.AddMatMat(1.0, D, transD, A, transA, 0.0);  // DA = D * A
    return TraceMatMatMat(DA, kNoTrans, B, transB, C, transC);
  }
}

template
float TraceMatMatMatMat(const MatrixBase<float> &A, MatrixTransposeType transA,
                        const MatrixBase<float> &B, MatrixTransposeType transB,
                        const MatrixBase<float> &C, MatrixTransposeType transC,
                        const MatrixBase<float> &D, MatrixTransposeType transD);

template
double TraceMatMatMatMat(const MatrixBase<double> &A, MatrixTransposeType transA,
                         const MatrixBase<double> &B, MatrixTransposeType transB,
                         const MatrixBase<double> &C, MatrixTransposeType transC,
                         const MatrixBase<double> &D, MatrixTransposeType transD);

template<typename Real> void  SortSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                                   MatrixBase<Real> *Vt, bool sort_on_absolute_value) {
  /// Makes sure the Svd is sorted (from greatest to least absolute value).
  MatrixIndexT num_singval = s->Dim();
  KALDI_ASSERT(U == NULL || U->NumCols() == num_singval);
  KALDI_ASSERT(Vt == NULL || Vt->NumRows() == num_singval);

  std::vector<std::pair<Real, MatrixIndexT> > vec(num_singval);
  // negative because we want revese order.
  for (MatrixIndexT d = 0; d < num_singval; d++) {
    Real val = (*s)(d),
        sort_val = -(sort_on_absolute_value ? std::abs(val) : val);
    vec[d] = std::pair<Real, MatrixIndexT>(sort_val, d);
  }
  std::sort(vec.begin(), vec.end());
  Vector<Real> s_copy(*s);
  for (MatrixIndexT d = 0; d < num_singval; d++)
    (*s)(d) = s_copy(vec[d].second);
  if (U != NULL) {
    Matrix<Real> Utmp(*U);
    MatrixIndexT dim = Utmp.NumRows();
    for (MatrixIndexT d = 0; d < num_singval; d++) {
      MatrixIndexT oldidx = vec[d].second;
      for (MatrixIndexT e = 0; e < dim; e++)
        (*U)(e, d) = Utmp(e, oldidx);
    }
  }
  if (Vt != NULL) {
    Matrix<Real> Vttmp(*Vt);
    for (MatrixIndexT d = 0; d < num_singval; d++)
      (*Vt).Row(d).CopyFromVec(Vttmp.Row(vec[d].second));
  }
}

template
void SortSvd(VectorBase<float> *s, MatrixBase<float> *U,
             MatrixBase<float> *Vt, bool);

template
void SortSvd(VectorBase<double> *s, MatrixBase<double> *U,
             MatrixBase<double> *Vt, bool);

template<typename Real>
void CreateEigenvalueMatrix(const VectorBase<Real> &re, const VectorBase<Real> &im,
                            MatrixBase<Real> *D) {
  MatrixIndexT n = re.Dim();
  KALDI_ASSERT(im.Dim() == n && D->NumRows() == n && D->NumCols() == n);

  MatrixIndexT j = 0;
  D->SetZero();
  while (j < n) {
    if (im(j) == 0) {  // Real eigenvalue
      (*D)(j, j) = re(j);
      j++;
    } else {  // First of a complex pair
      KALDI_ASSERT(j+1 < n && ApproxEqual(im(j+1), -im(j))
                   && ApproxEqual(re(j+1), re(j)));
      /// if (im(j) < 0.0) KALDI_WARN << "Negative first im part of pair";  // TEMP
      Real lambda = re(j), mu = im(j);
      // create 2x2 block [lambda, mu; -mu, lambda]
      (*D)(j, j) = lambda;
      (*D)(j, j+1) = mu;
      (*D)(j+1, j) = -mu;
      (*D)(j+1, j+1) = lambda;
      j += 2;
    }
  }
}

template
void CreateEigenvalueMatrix(const VectorBase<float> &re, const VectorBase<float> &im,
                            MatrixBase<float> *D);
template
void CreateEigenvalueMatrix(const VectorBase<double> &re, const VectorBase<double> &im,
                            MatrixBase<double> *D);



template<typename Real>
bool AttemptComplexPower(Real *x_re, Real *x_im, Real power) {
  // Used in Matrix<Real>::Power().
  // Attempts to take the complex value x to the power "power",
  // assuming that power is fractional (i.e. we don't treat integers as a
  // special case).  Returns false if this is not possible, either
  // because x is negative and real (hence there is no obvious answer
  // that is "closest to 1", and anyway this case does not make sense
  // in the Matrix<Real>::Power() routine);
  // or because power is negative, and x is zero.

  // First solve for r and theta in
  // x_re = r*cos(theta), x_im = r*sin(theta)
  if (*x_re < 0.0 && *x_im == 0.0) return false;  // can't do
  // it for negative real values.
  Real r = std::sqrt((*x_re * *x_re) + (*x_im * *x_im));  // r == radius.
  if (power < 0.0 && r == 0.0) return false;
  Real theta = std::atan2(*x_im, *x_re);
  // Take the power.
  r = std::pow(r, power);
  theta *= power;
  *x_re = r * std::cos(theta);
  *x_im = r * std::sin(theta);
  return true;
}

template
bool AttemptComplexPower(float *x_re, float *x_im, float power);
template
bool AttemptComplexPower(double *x_re, double *x_im, double power);



template <typename Real>
Real TraceMatMat(const MatrixBase<Real> &A,
                  const MatrixBase<Real> &B,
                  MatrixTransposeType trans) {  // tr(A B), equivalent to sum of each element of A times same element in B'
  MatrixIndexT aStride = A.stride_, bStride = B.stride_;
  if (trans == kNoTrans) {
    KALDI_ASSERT(A.NumRows() == B.NumCols() && A.NumCols() == B.NumRows());
    Real ans = 0.0;
    Real *adata = A.data_, *bdata = B.data_;
    MatrixIndexT arows = A.NumRows(), acols = A.NumCols();
    for (MatrixIndexT row = 0;row < arows;row++, adata+=aStride, bdata++)
      ans += cblas_Xdot(acols, adata, 1, bdata, bStride);
    return ans;
  } else {
    KALDI_ASSERT(A.NumRows() == B.NumRows() && A.NumCols() == B.NumCols());
    Real ans = 0.0;
    Real *adata = A.data_, *bdata = B.data_;
    MatrixIndexT arows = A.NumRows(), acols = A.NumCols();
    for (MatrixIndexT row = 0;row < arows;row++, adata+=aStride, bdata+=bStride)
      ans += cblas_Xdot(acols, adata, 1, bdata, 1);
    return ans;
  }
}


// Instantiate the template above for float and double.
template
float TraceMatMat(const MatrixBase<float> &A,
                  const MatrixBase<float> &B,
                  MatrixTransposeType trans);
template
double TraceMatMat(const MatrixBase<double> &A,
                  const MatrixBase<double> &B,
                  MatrixTransposeType trans);


template<typename Real>
Real MatrixBase<Real>::LogSumExp(Real prune) const {
  Real sum;
  if (sizeof(sum) == 8) sum = kLogZeroDouble;
  else sum = kLogZeroFloat;
  Real max_elem = Max(), cutoff;
  if (sizeof(Real) == 4) cutoff = max_elem + kMinLogDiffFloat;
  else cutoff = max_elem + kMinLogDiffDouble;
  if (prune > 0.0 && max_elem - prune > cutoff) // explicit pruning...
    cutoff = max_elem - prune;

  double sum_relto_max_elem = 0.0;

  for (MatrixIndexT i = 0; i < num_rows_; i++) {
    for (MatrixIndexT j = 0; j < num_cols_; j++) {
      BaseFloat f = (*this)(i, j);
      if (f >= cutoff)
        sum_relto_max_elem += Exp(f - max_elem);
    }
  }
  return max_elem + Log(sum_relto_max_elem);
}

template<typename Real>
Real MatrixBase<Real>::ApplySoftMax() {
  Real max = this->Max(), sum = 0.0;
  // the 'max' helps to get in good numeric range.
  for (MatrixIndexT i = 0; i < num_rows_; i++)
    for (MatrixIndexT j = 0; j < num_cols_; j++)
      sum += ((*this)(i, j) = Exp((*this)(i, j) - max));
  this->Scale(1.0 / sum);
  return max + Log(sum);
}

template<typename Real>
void MatrixBase<Real>::Tanh(const MatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));

  if (num_cols_ == stride_ && src.num_cols_ == src.stride_) {
    SubVector<Real> src_vec(src.data_, num_rows_ * num_cols_),
        dst_vec(this->data_, num_rows_ * num_cols_);
    dst_vec.Tanh(src_vec);
  } else {
    for (MatrixIndexT r = 0; r < num_rows_; r++) {
      SubVector<Real> src_vec(src, r), dest_vec(*this, r);
      dest_vec.Tanh(src_vec);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::SoftHinge(const MatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));
  int32 num_rows = num_rows_, num_cols = num_cols_;
  for (MatrixIndexT r = 0; r < num_rows; r++) {
    Real *row_data = this->RowData(r);
    const Real *src_row_data = src.RowData(r);
    for (MatrixIndexT c = 0; c < num_cols; c++) {
      Real x = src_row_data[c], y;
      if (x > 10.0) y = x; // avoid exponentiating large numbers; function
      // approaches y=x.
      else y = Log1p(Exp(x)); // these defined in kaldi-math.h
      row_data[c] = y;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::GroupPnorm(const MatrixBase<Real> &src, Real power) {
  KALDI_ASSERT(src.NumCols() % this->NumCols() == 0 &&
               src.NumRows() == this->NumRows());
  int group_size = src.NumCols() / this->NumCols(),
    num_rows = this->NumRows(), num_cols = this->NumCols();
  for (MatrixIndexT i = 0; i < num_rows; i++)
    for (MatrixIndexT j = 0; j < num_cols; j++)
      (*this)(i, j) = src.Row(i).Range(j * group_size,  group_size).Norm(power);
}

template<typename Real>
void MatrixBase<Real>::GroupMax(const MatrixBase<Real> &src) {
  KALDI_ASSERT(src.NumCols() % this->NumCols() == 0 &&
               src.NumRows() == this->NumRows());
  int group_size = src.NumCols() / this->NumCols(),
      num_rows = this->NumRows(), num_cols = this->NumCols();
  for (MatrixIndexT i = 0; i < num_rows; i++) {
    const Real *src_row_data = src.RowData(i);
    for (MatrixIndexT j = 0; j < num_cols; j++) {
      Real max_val = -1e20;
      for (MatrixIndexT k = 0; k < group_size; k++) {
        Real src_data = src_row_data[j * group_size + k];
        if (src_data > max_val)
          max_val = src_data;
      }
      (*this)(i, j) = max_val;
    }
  }
}

template<typename Real>
void MatrixBase<Real>::CopyCols(const MatrixBase<Real> &src,
                                const MatrixIndexT *indices) {
  KALDI_ASSERT(NumRows() == src.NumRows());
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      this_stride = stride_, src_stride = src.stride_;
  Real *this_data = this->data_;
  const Real *src_data = src.data_;
#ifdef KALDI_PARANOID
  MatrixIndexT src_cols = src.NumCols();
  for (MatrixIndexT i = 0; i < num_cols; i++)
    KALDI_ASSERT(indices[i] >= -1 && indices[i] < src_cols);
#endif

  // For the sake of memory locality we do this row by row, rather
  // than doing it column-wise using cublas_Xcopy
  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride, src_data += src_stride) {
    const MatrixIndexT *index_ptr = &(indices[0]);
    for (MatrixIndexT c = 0; c < num_cols; c++, index_ptr++) {
      if (*index_ptr < 0) this_data[c] = 0;
      else this_data[c] = src_data[*index_ptr];
    }
  }
}


template<typename Real>
void MatrixBase<Real>::AddCols(const MatrixBase<Real> &src,
                               const MatrixIndexT *indices) {
  KALDI_ASSERT(NumRows() == src.NumRows());
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      this_stride = stride_, src_stride = src.stride_;
  Real *this_data = this->data_;
  const Real *src_data = src.data_;
#ifdef KALDI_PARANOID
  MatrixIndexT src_cols = src.NumCols();
  for (MatrixIndexT i = 0; i < num_cols; i++)
    KALDI_ASSERT(indices[i] >= -1 && indices[i] < src_cols);
#endif

  // For the sake of memory locality we do this row by row, rather
  // than doing it column-wise using cublas_Xcopy
  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride, src_data += src_stride) {
    const MatrixIndexT *index_ptr = &(indices[0]);
    for (MatrixIndexT c = 0; c < num_cols; c++, index_ptr++) {
      if (*index_ptr >= 0)
        this_data[c] += src_data[*index_ptr];
    }
  }
}

template<typename Real>
void MatrixBase<Real>::CopyRows(const MatrixBase<Real> &src,
                                const MatrixIndexT *indices) {
  KALDI_ASSERT(NumCols() == src.NumCols());
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      this_stride = stride_;
  Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    MatrixIndexT index = indices[r];
    if (index < 0) memset(this_data, 0, sizeof(Real) * num_cols_);
    else cblas_Xcopy(num_cols, src.RowData(index), 1, this_data, 1);
  }
}

template<typename Real>
void MatrixBase<Real>::CopyRows(const Real *const *src) {
  MatrixIndexT num_rows = num_rows_,
      num_cols = num_cols_, this_stride = stride_;
  Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    const Real *const src_data = src[r];
    if (src_data == NULL) memset(this_data, 0, sizeof(Real) * num_cols);
    else cblas_Xcopy(num_cols, src_data, 1, this_data, 1);
  }
}

template<typename Real>
void MatrixBase<Real>::CopyToRows(Real *const *dst) const {
  MatrixIndexT num_rows = num_rows_,
      num_cols = num_cols_, this_stride = stride_;
  const Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    Real *const dst_data = dst[r];
    if (dst_data != NULL)
      cblas_Xcopy(num_cols, this_data, 1, dst_data, 1);
  }
}

template<typename Real>
void MatrixBase<Real>::AddRows(Real alpha,
                               const MatrixBase<Real> &src,
                               const MatrixIndexT *indexes) {
  KALDI_ASSERT(NumCols() == src.NumCols());
  MatrixIndexT num_rows = num_rows_,
      num_cols = num_cols_, this_stride = stride_;
  Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    MatrixIndexT index = indexes[r];
    KALDI_ASSERT(index >= -1 && index < src.NumRows());
    if (index != -1)
      cblas_Xaxpy(num_cols, alpha, src.RowData(index), 1, this_data, 1);
  }
}

template<typename Real>
void MatrixBase<Real>::AddRows(Real alpha, const Real *const *src) {
  MatrixIndexT num_rows = num_rows_,
      num_cols = num_cols_, this_stride = stride_;
  Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    const Real *const src_data = src[r];
    if (src_data != NULL)
      cblas_Xaxpy(num_cols, alpha, src_data, 1, this_data, 1);
  }
}

template<typename Real>
void MatrixBase<Real>::AddToRows(Real alpha,
                                 const MatrixIndexT *indexes,
                                 MatrixBase<Real> *dst) const {
  KALDI_ASSERT(NumCols() == dst->NumCols());
  MatrixIndexT num_rows = num_rows_,
      num_cols = num_cols_, this_stride = stride_;
  Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    MatrixIndexT index = indexes[r];
    KALDI_ASSERT(index >= -1 && index < dst->NumRows());
    if (index != -1)
      cblas_Xaxpy(num_cols, alpha, this_data, 1, dst->RowData(index), 1);
  }
}

template<typename Real>
void MatrixBase<Real>::AddToRows(Real alpha, Real *const *dst) const {
  MatrixIndexT num_rows = num_rows_,
      num_cols = num_cols_, this_stride = stride_;
  const Real *this_data = this->data_;

  for (MatrixIndexT r = 0; r < num_rows; r++, this_data += this_stride) {
    Real *const dst_data = dst[r];
    if (dst_data != NULL)
      cblas_Xaxpy(num_cols, alpha, this_data, 1, dst_data, 1);
  }
}

template<typename Real>
void MatrixBase<Real>::Sigmoid(const MatrixBase<Real> &src) {
  KALDI_ASSERT(SameDim(*this, src));

  if (num_cols_ == stride_ && src.num_cols_ == src.stride_) {
    SubVector<Real> src_vec(src.data_, num_rows_ * num_cols_),
        dst_vec(this->data_, num_rows_ * num_cols_);
    dst_vec.Sigmoid(src_vec);
  } else {
    for (MatrixIndexT r = 0; r < num_rows_; r++) {
      SubVector<Real> src_vec(src, r), dest_vec(*this, r);
      dest_vec.Sigmoid(src_vec);
    }
  }
}

template<typename Real>
void MatrixBase<Real>::DiffSigmoid(const MatrixBase<Real> &value,
                                   const MatrixBase<Real> &diff) {
  KALDI_ASSERT(SameDim(*this, value) && SameDim(*this, diff));
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      stride = stride_, value_stride = value.stride_, diff_stride = diff.stride_;
  Real *data = data_;
  const Real *value_data = value.data_, *diff_data = diff.data_;
  for (MatrixIndexT r = 0; r < num_rows; r++) {
    for (MatrixIndexT c = 0; c < num_cols; c++)
      data[c] = diff_data[c] * value_data[c] * (1.0 - value_data[c]);
    data += stride;
    value_data += value_stride;
    diff_data += diff_stride;
  }
}

template<typename Real>
void MatrixBase<Real>::DiffTanh(const MatrixBase<Real> &value,
                                   const MatrixBase<Real> &diff) {
  KALDI_ASSERT(SameDim(*this, value) && SameDim(*this, diff));
  MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      stride = stride_, value_stride = value.stride_, diff_stride = diff.stride_;
  Real *data = data_;
  const Real *value_data = value.data_, *diff_data = diff.data_;
  for (MatrixIndexT r = 0; r < num_rows; r++) {
    for (MatrixIndexT c = 0; c < num_cols; c++)
      data[c] = diff_data[c] * (1.0 - (value_data[c] * value_data[c]));
    data += stride;
    value_data += value_stride;
    diff_data += diff_stride;
  }
}


template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::AddVecToRows(const Real alpha, const VectorBase<OtherReal> &v) {
  const MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      stride = stride_;
  KALDI_ASSERT(v.Dim() == num_cols);
  if(num_cols <= 64) {
    Real *data = data_;
    const OtherReal *vdata = v.Data();
    for (MatrixIndexT i = 0; i < num_rows; i++, data += stride) {
      for (MatrixIndexT j = 0; j < num_cols; j++)
        data[j] += alpha * vdata[j];
    }

  } else {
    Vector<OtherReal> ones(num_rows);
    ones.Set(1.0);
    this->AddVecVec(alpha, ones, v);
   }
}

template void MatrixBase<float>::AddVecToRows(const float alpha,
                                              const VectorBase<float> &v);
template void MatrixBase<float>::AddVecToRows(const float alpha,
                                              const VectorBase<double> &v);
template void MatrixBase<double>::AddVecToRows(const double alpha,
                                               const VectorBase<float> &v);
template void MatrixBase<double>::AddVecToRows(const double alpha,
                                               const VectorBase<double> &v);


template<typename Real>
template<typename OtherReal>
void MatrixBase<Real>::AddVecToCols(const Real alpha, const VectorBase<OtherReal> &v) {
  const MatrixIndexT num_rows = num_rows_, num_cols = num_cols_,
      stride = stride_;
  KALDI_ASSERT(v.Dim() == num_rows);

  if (num_rows <= 64) {
    Real *data = data_;
    const OtherReal *vdata = v.Data();
    for (MatrixIndexT i = 0; i < num_rows; i++, data += stride) {
      Real to_add = alpha * vdata[i];
      for (MatrixIndexT j = 0; j < num_cols; j++)
        data[j] += to_add;
    }

  } else {
    Vector<OtherReal> ones(num_cols);
    ones.Set(1.0);
    this->AddVecVec(alpha, v, ones);
  }
}

template void MatrixBase<float>::AddVecToCols(const float alpha,
                                              const VectorBase<float> &v);
template void MatrixBase<float>::AddVecToCols(const float alpha,
                                              const VectorBase<double> &v);
template void MatrixBase<double>::AddVecToCols(const double alpha,
                                               const VectorBase<float> &v);
template void MatrixBase<double>::AddVecToCols(const double alpha,
                                               const VectorBase<double> &v);

//Explicit instantiation of the classes
//Apparently, it seems to be necessary that the instantiation
//happens at the end of the file. Otherwise, not all the member
//functions will get instantiated.

template class Matrix<float>;
template class Matrix<double>;
template class MatrixBase<float>;
template class MatrixBase<double>;
template class SubMatrix<float>;
template class SubMatrix<double>;

} // namespace kaldi
