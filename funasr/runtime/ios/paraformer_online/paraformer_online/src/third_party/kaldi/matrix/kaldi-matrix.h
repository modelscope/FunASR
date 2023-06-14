// matrix/kaldi-matrix.h

// Copyright 2009-2011  Ondrej Glembek;  Microsoft Corporation;  Lukas Burget;
//                      Saarland University;  Petr Schwarz;  Yanmin Qian;
//                      Karel Vesely;  Go Vivace Inc.;  Haihua Xu
//           2017       Shiyin Kang

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

#ifndef KALDI_MATRIX_KALDI_MATRIX_H_
#define KALDI_MATRIX_KALDI_MATRIX_H_ 1

#include "matrix/matrix-common.h"

namespace kaldi {

/// @{ \addtogroup matrix_funcs_scalar

/// We need to declare this here as it will be a friend function.
/// tr(A B), or tr(A B^T).
template<typename Real>
Real TraceMatMat(const MatrixBase<Real> &A, const MatrixBase<Real> &B,
                 MatrixTransposeType trans = kNoTrans);
/// @}

/// \addtogroup matrix_group
/// @{

/// Base class which provides matrix operations not involving resizing
/// or allocation.   Classes Matrix and SubMatrix inherit from it and take care
/// of allocation and resizing.
template<typename Real>
class MatrixBase {
 public:
  // so this child can access protected members of other instances.
  friend class Matrix<Real>;
  // friend declarations for CUDA matrices (see ../cudamatrix/)
  friend class CuMatrixBase<Real>;
  friend class CuMatrix<Real>;
  friend class CuSubMatrix<Real>;
  friend class CuPackedMatrix<Real>;
  friend class PackedMatrix<Real>;
  friend class SparseMatrix<Real>;
  friend class SparseMatrix<float>;
  friend class SparseMatrix<double>;

  /// Returns number of rows (or zero for emtpy matrix).
  inline MatrixIndexT  NumRows() const { return num_rows_; }

  /// Returns number of columns (or zero for emtpy matrix).
  inline MatrixIndexT NumCols() const { return num_cols_; }

  /// Stride (distance in memory between each row).  Will be >= NumCols.
  inline MatrixIndexT Stride() const {  return stride_; }

  /// Returns size in bytes of the data held by the matrix.
  size_t  SizeInBytes() const {
    return static_cast<size_t>(num_rows_) * static_cast<size_t>(stride_) *
        sizeof(Real);
  }

  /// Gives pointer to raw data (const).
  inline const Real* Data() const {
    return data_;
  }

  /// Gives pointer to raw data (non-const).
  inline Real* Data() { return data_; }

  /// Returns pointer to data for one row (non-const)
  inline  Real* RowData(MatrixIndexT i) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return data_ + i * stride_;
  }

  /// Returns pointer to data for one row (const)
  inline const Real* RowData(MatrixIndexT i) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return data_ + i * stride_;
  }

  /// Indexing operator, non-const
  /// (only checks sizes if compiled with -DKALDI_PARANOID)
  inline Real&  operator() (MatrixIndexT r, MatrixIndexT c) {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                          static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                          static_cast<UnsignedMatrixIndexT>(c) <
                          static_cast<UnsignedMatrixIndexT>(num_cols_));
    return *(data_ + r * stride_ + c);
  }
  /// Indexing operator, provided for ease of debugging (gdb doesn't work
  /// with parenthesis operator).
  Real &Index (MatrixIndexT r, MatrixIndexT c) {  return (*this)(r, c); }

  /// Indexing operator, const
  /// (only checks sizes if compiled with -DKALDI_PARANOID)
  inline const Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    KALDI_PARANOID_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                          static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                          static_cast<UnsignedMatrixIndexT>(c) <
                          static_cast<UnsignedMatrixIndexT>(num_cols_));
    return *(data_ + r * stride_ + c);
  }

  /*   Basic setting-to-special values functions. */

  /// Sets matrix to zero.
  void SetZero();
  /// Sets all elements to a specific value.
  void Set(Real);
  /// Sets to zero, except ones along diagonal [for non-square matrices too]
  void SetUnit();
  /// Sets to random values of a normal distribution
  void SetRandn();
  /// Sets to numbers uniformly distributed on (0, 1)
  void SetRandUniform();

  /*  Copying functions.  These do not resize the matrix! */


  /// Copy given matrix. (no resize is done).
  template<typename OtherReal>
  void CopyFromMat(const MatrixBase<OtherReal> & M,
                   MatrixTransposeType trans = kNoTrans);

  /// Copy from compressed matrix.
  void CopyFromMat(const CompressedMatrix &M);

  /// Copy given spmatrix. (no resize is done).
  template<typename OtherReal>
  void CopyFromSp(const SpMatrix<OtherReal> &M);

  /// Copy given tpmatrix. (no resize is done).
  template<typename OtherReal>
  void CopyFromTp(const TpMatrix<OtherReal> &M,
                  MatrixTransposeType trans = kNoTrans);

  /// Copy from CUDA matrix.  Implemented in ../cudamatrix/cu-matrix.h
  template<typename OtherReal>
  void CopyFromMat(const CuMatrixBase<OtherReal> &M,
                   MatrixTransposeType trans = kNoTrans);

  /// This function has two modes of operation.  If v.Dim() == NumRows() *
  /// NumCols(), then treats the vector as a row-by-row concatenation of a
  /// matrix and copies to *this.
  /// if v.Dim() == NumCols(), it sets each row of *this to a copy of v.
  void CopyRowsFromVec(const VectorBase<Real> &v);

  /// This version of CopyRowsFromVec is implemented in ../cudamatrix/cu-vector.cc
  void CopyRowsFromVec(const CuVectorBase<Real> &v);

  template<typename OtherReal>
  void CopyRowsFromVec(const VectorBase<OtherReal> &v);

  /// Copies vector into matrix, column-by-column.
  /// Note that rv.Dim() must either equal NumRows()*NumCols() or NumRows();
  /// this has two modes of operation.
  void CopyColsFromVec(const VectorBase<Real> &v);

  /// Copy vector into specific column of matrix.
  void CopyColFromVec(const VectorBase<Real> &v, const MatrixIndexT col);
  /// Copy vector into specific row of matrix.
  void CopyRowFromVec(const VectorBase<Real> &v, const MatrixIndexT row);
  /// Copy vector into diagonal of matrix.
  void CopyDiagFromVec(const VectorBase<Real> &v);

  /* Accessing of sub-parts of the matrix. */

  /// Return specific row of matrix [const].
  inline const SubVector<Real> Row(MatrixIndexT i) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return SubVector<Real>(data_ + (i * stride_), NumCols());
  }

  /// Return specific row of matrix.
  inline SubVector<Real> Row(MatrixIndexT i) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(i) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_));
    return SubVector<Real>(data_ + (i * stride_), NumCols());
  }

  /// Return a sub-part of matrix.
  inline SubMatrix<Real> Range(const MatrixIndexT row_offset,
                               const MatrixIndexT num_rows,
                               const MatrixIndexT col_offset,
                               const MatrixIndexT num_cols) const {
    return SubMatrix<Real>(*this, row_offset, num_rows,
                           col_offset, num_cols);
  }
  inline SubMatrix<Real> RowRange(const MatrixIndexT row_offset,
                                  const MatrixIndexT num_rows) const {
    return SubMatrix<Real>(*this, row_offset, num_rows, 0, num_cols_);
  }
  inline SubMatrix<Real> ColRange(const MatrixIndexT col_offset,
                                  const MatrixIndexT num_cols) const {
    return SubMatrix<Real>(*this, 0, num_rows_, col_offset, num_cols);
  }

  /* Various special functions. */
  /// Returns sum of all elements in matrix.
  Real Sum() const;
  /// Returns trace of matrix.
  Real Trace(bool check_square = true) const;
  // If check_square = true, will crash if matrix is not square.

  /// Returns maximum element of matrix.
  Real Max() const;
  /// Returns minimum element of matrix.
  Real Min() const;

  /// Element by element multiplication with a given matrix.
  void MulElements(const MatrixBase<Real> &A);

  /// Divide each element by the corresponding element of a given matrix.
  void DivElements(const MatrixBase<Real> &A);

  /// Multiply each element with a scalar value.
  void Scale(Real alpha);

  /// Set, element-by-element, *this = max(*this, A)
  void Max(const MatrixBase<Real> &A);
  /// Set, element-by-element, *this = min(*this, A)
  void Min(const MatrixBase<Real> &A);

  /// Equivalent to (*this) = (*this) * diag(scale).  Scaling
  /// each column by a scalar taken from that dimension of the vector.
  void MulColsVec(const VectorBase<Real> &scale);

  /// Equivalent to (*this) = diag(scale) * (*this).  Scaling
  /// each row by a scalar taken from that dimension of the vector.
  void MulRowsVec(const VectorBase<Real> &scale);

  /// Divide each row into src.NumCols() equal groups, and then scale i'th row's
  /// j'th group of elements by src(i, j).  Requires src.NumRows() ==
  /// this->NumRows() and this->NumCols() % src.NumCols() == 0.
  void MulRowsGroupMat(const MatrixBase<Real> &src);

  /// Returns logdet of matrix.
  Real LogDet(Real *det_sign = NULL) const;

  /// matrix inverse.
  /// if inverse_needed = false, will fill matrix with garbage.
  /// (only useful if logdet wanted).
  void Invert(Real *log_det = NULL, Real *det_sign = NULL,
              bool inverse_needed = true);
  /// matrix inverse [double].
  /// if inverse_needed = false, will fill matrix with garbage
  /// (only useful if logdet wanted).
  /// Does inversion in double precision even if matrix was not double.
  void InvertDouble(Real *LogDet = NULL, Real *det_sign = NULL,
                      bool inverse_needed = true);

  /// Inverts all the elements of the matrix
  void InvertElements();

  /// Transpose the matrix.  This one is only
  /// applicable to square matrices (the one in the
  /// Matrix child class works also for non-square.
  void Transpose();

  /// Copies column r from column indices[r] of src.
  /// As a special case, if indexes[i] == -1, sets column i to zero.
  /// all elements of "indices" must be in [-1, src.NumCols()-1],
  /// and src.NumRows() must equal this.NumRows()
  void CopyCols(const MatrixBase<Real> &src,
                const MatrixIndexT *indices);

  /// Copies row r from row indices[r] of src (does nothing
  /// As a special case, if indexes[i] == -1, sets row i to zero.
  /// all elements of "indices" must be in [-1, src.NumRows()-1],
  /// and src.NumCols() must equal this.NumCols()
  void CopyRows(const MatrixBase<Real> &src,
                const MatrixIndexT *indices);

  /// Add column indices[r] of src to column r.
  /// As a special case, if indexes[i] == -1, skip column i
  /// indices.size() must equal this->NumCols(),
  /// all elements of "reorder" must be in [-1, src.NumCols()-1],
  /// and src.NumRows() must equal this.NumRows()
  void AddCols(const MatrixBase<Real> &src,
               const MatrixIndexT *indices);

  /// Copies row r of this matrix from an array of floats at the location given
  /// by src[r]. If any src[r] is NULL then this.Row(r) will be set to zero.
  /// Note: we are using "pointer to const pointer to const object" for "src",
  ///       because we may create "src" by calling Data() of const CuArray
  void CopyRows(const Real *const *src);

  /// Copies row r of this matrix to the array of floats at the location given
  /// by dst[r]. If dst[r] is NULL, does not copy anywhere.  Requires that none
  /// of the memory regions pointed to by the pointers in "dst" overlap (e.g.
  /// none of the pointers should be the same).
  void CopyToRows(Real *const *dst) const;

  /// Does for each row r, this.Row(r) += alpha * src.row(indexes[r]).
  /// If indexes[r] < 0, does not add anything. all elements of "indexes" must
  /// be in [-1, src.NumRows()-1], and src.NumCols() must equal this.NumCols().
  void AddRows(Real alpha,
               const MatrixBase<Real> &src,
               const MatrixIndexT *indexes);

  /// Does for each row r, this.Row(r) += alpha * src[r], treating src[r] as the
  /// beginning of a region of memory representing a vector of floats, of the
  /// same length as this.NumCols(). If src[r] is NULL, does not add anything.
  void AddRows(Real alpha, const Real *const *src);

  /// For each row r of this matrix, adds it (times alpha) to the array of
  /// floats at the location given by dst[r]. If dst[r] is NULL, does not do
  /// anything for that row. Requires that none of the memory regions pointed
  /// to by the pointers in "dst" overlap (e.g. none of the pointers should be
  /// the same).
  void AddToRows(Real alpha, Real *const *dst) const;

  /// For each row i of *this, adds this->Row(i) to
  /// dst->Row(indexes(i)) if indexes(i) >= 0, else do nothing.
  /// Requires that all the indexes[i] that are >= 0
  /// be distinct, otherwise the behavior is undefined.
  void AddToRows(Real alpha,
                 const MatrixIndexT *indexes,
                 MatrixBase<Real> *dst) const;

  /// Applies floor to all matrix elements
  void ApplyFloor(Real floor_val);

  /// Applies floor to all matrix elements
  void ApplyCeiling(Real ceiling_val);

  /// Calculates log of all the matrix elemnts
  void ApplyLog();

  /// Exponentiate each of the elements.
  void ApplyExp();

  /// For each element x of the matrix, set it to
  /// (x < 0 ? exp(x) : x + 1).  This function is used
  /// in our RNNLM training.
  void ApplyExpSpecial();

  /// Applies power to all matrix elements
  void ApplyPow(Real power);

  /// Apply power to the absolute value of each element.
  /// Include the sign of the input element if include_sign == true.
  /// If the power is negative and the input to the power is zero,
  /// The output will be set zero.
  void ApplyPowAbs(Real power, bool include_sign=false);

  /// Applies the Heaviside step function (x > 0 ? 1 : 0) to all matrix elements
  /// Note: in general you can make different choices for x = 0, but for now
  /// please leave it as it (i.e. returning zero) because it affects the
  /// RectifiedLinearComponent in the neural net code.
  void ApplyHeaviside();

  /// Eigenvalue Decomposition of a square NxN matrix into the form (*this) = P D
  /// P^{-1}.  Be careful: the relationship of D to the eigenvalues we output is
  /// slightly complicated, due to the need for P to be real.  In the symmetric
  /// case D is diagonal and real, but in
  /// the non-symmetric case there may be complex-conjugate pairs of eigenvalues.
  /// In this case, for the equation (*this) = P D P^{-1} to hold, D must actually
  /// be block diagonal, with 2x2 blocks corresponding to any such pairs.  If a
  /// pair is lambda +- i*mu, D will have a corresponding 2x2 block
  /// [lambda, mu; -mu, lambda].
  /// Note that if the input matrix (*this) is non-invertible, P may not be invertible
  /// so in this case instead of the equation (*this) = P D P^{-1} holding, we have
  /// instead (*this) P = P D.
  ///
  /// The non-member function CreateEigenvalueMatrix creates D from eigs_real and eigs_imag.
  void Eig(MatrixBase<Real> *P,
           VectorBase<Real> *eigs_real,
           VectorBase<Real> *eigs_imag) const;

  /// The Power method attempts to take the matrix to a power using a method that
  /// works in general for fractional and negative powers.  The input matrix must
  /// be invertible and have reasonable condition (or we don't guarantee the
  /// results.  The method is based on the eigenvalue decomposition.  It will
  /// return false and leave the matrix unchanged, if at entry the matrix had
  /// real negative eigenvalues (or if it had zero eigenvalues and the power was
  /// negative).
  bool Power(Real pow);

  /** Singular value decomposition
     Major limitations:
     For nonsquare matrices, we assume m>=n (NumRows >= NumCols), and we return
     the "skinny" Svd, i.e. the matrix in the middle is diagonal, and the
     one on the left is rectangular.

     In Svd, *this = U*diag(S)*Vt.
     Null pointers for U and/or Vt at input mean we do not want that output.  We
     expect that S.Dim() == m, U is either NULL or m by n,
     and v is either NULL or n by n.
     The singular values are not sorted (use SortSvd for that).  */
  void DestructiveSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                      MatrixBase<Real> *Vt);  // Destroys calling matrix.

  /// Compute SVD (*this) = U diag(s) Vt.   Note that the V in the call is already
  /// transposed; the normal formulation is U diag(s) V^T.
  /// Null pointers for U or V mean we don't want that output (this saves
  /// compute).  The singular values are not sorted (use SortSvd for that).
  void Svd(VectorBase<Real> *s, MatrixBase<Real> *U,
           MatrixBase<Real> *Vt) const;
  /// Compute SVD but only retain the singular values.
  void Svd(VectorBase<Real> *s) const { Svd(s, NULL, NULL); }


  /// Returns smallest singular value.
  Real MinSingularValue() const {
    Vector<Real> tmp(std::min(NumRows(), NumCols()));
    Svd(&tmp);
    return tmp.Min();
  }

  void TestUninitialized() const; // This function is designed so that if any element
  // if the matrix is uninitialized memory, valgrind will complain.

  /// Returns condition number by computing Svd.  Works even if cols > rows.
  /// Returns infinity if all singular values are zero.
  Real Cond() const;

  /// Returns true if matrix is Symmetric.
  bool IsSymmetric(Real cutoff = 1.0e-05) const;  // replace magic number

  /// Returns true if matrix is Diagonal.
  bool IsDiagonal(Real cutoff = 1.0e-05) const;  // replace magic number

  /// Returns true if the matrix is all zeros, except for ones on diagonal.  (it
  /// does not have to be square).  More specifically, this function returns
  /// false if for any i, j, (*this)(i, j) differs by more than cutoff from the
  /// expression (i == j ? 1 : 0).
  bool IsUnit(Real cutoff = 1.0e-05) const;     // replace magic number

  /// Returns true if matrix is all zeros.
  bool IsZero(Real cutoff = 1.0e-05) const;     // replace magic number

  /// Frobenius norm, which is the sqrt of sum of square elements.  Same as Schatten 2-norm,
  /// or just "2-norm".
  Real FrobeniusNorm() const;

  /// Returns true if ((*this)-other).FrobeniusNorm()
  /// <= tol * (*this).FrobeniusNorm().
  bool ApproxEqual(const MatrixBase<Real> &other, float tol = 0.01) const;

  /// Tests for exact equality.  It's usually preferable to use ApproxEqual.
  bool Equal(const MatrixBase<Real> &other) const;

  /// largest absolute value.
  Real LargestAbsElem() const;  // largest absolute value.

  /// Returns log(sum(exp())) without exp overflow
  /// If prune > 0.0, it uses a pruning beam, discarding
  /// terms less than (max - prune).  Note: in future
  /// we may change this so that if prune = 0.0, it takes
  /// the max, so use -1 if you don't want to prune.
  Real LogSumExp(Real prune = -1.0) const;

  /// Apply soft-max to the collection of all elements of the
  /// matrix and return normalizer (log sum of exponentials).
  Real ApplySoftMax();

  /// Set each element to the sigmoid of the corresponding element of "src".
  void Sigmoid(const MatrixBase<Real> &src);

  /// Sets each element to the Heaviside step function (x > 0 ? 1 : 0) of the
  /// corresponding element in "src".  Note: in general you can make different
  /// choices for x = 0, but for now please leave it as it (i.e. returning zero)
  /// because it affects the RectifiedLinearComponent in the neural net code.
  void Heaviside(const MatrixBase<Real> &src);

  /// Set each element to y = log(1 + exp(x))
  void SoftHinge(const MatrixBase<Real> &src);

  /// Apply the function y(i) = (sum_{j = i*G}^{(i+1)*G-1} x_j^(power))^(1 / p).
  /// Requires src.NumRows() == this->NumRows() and  src.NumCols() % this->NumCols() == 0.
  void GroupPnorm(const MatrixBase<Real> &src, Real power);

  /// Calculate derivatives for the GroupPnorm function above...
  /// if "input" is the input to the GroupPnorm function above (i.e. the "src" variable),
  /// and "output" is the result of the computation (i.e. the "this" of that function
  /// call), and *this has the same dimension as "input", then it sets each element
  /// of *this to the derivative d(output-elem)/d(input-elem) for each element of "input", where
  /// "output-elem" is whichever element of output depends on that input element.
  void GroupPnormDeriv(const MatrixBase<Real> &input, const MatrixBase<Real> &output,
                       Real power);

  /// Apply the function y(i) = (max_{j = i*G}^{(i+1)*G-1} x_j
  /// Requires src.NumRows() == this->NumRows() and  src.NumCols() % this->NumCols() == 0.
  void GroupMax(const MatrixBase<Real> &src);

  /// Calculate derivatives for the GroupMax function above, where
  /// "input" is the input to the GroupMax function above (i.e. the "src" variable),
  /// and "output" is the result of the computation (i.e. the "this" of that function
  /// call), and *this must have the same dimension as "input". Each element
  /// of *this will be set to 1 if the corresponding input equals the output of
  /// the group, and 0 otherwise. The equals the function derivative where it is
  /// defined (it's not defined where multiple inputs in the group are equal to the output).
  void GroupMaxDeriv(const MatrixBase<Real> &input, const MatrixBase<Real> &output);

  /// Set each element to the tanh of the corresponding element of "src".
  void Tanh(const MatrixBase<Real> &src);

  // Function used in backpropagating derivatives of the sigmoid function:
  // element-by-element, set *this = diff * value * (1.0 - value).
  void DiffSigmoid(const MatrixBase<Real> &value,
                   const MatrixBase<Real> &diff);

  // Function used in backpropagating derivatives of the tanh function:
  // element-by-element, set *this = diff * (1.0 - value^2).
  void DiffTanh(const MatrixBase<Real> &value,
                const MatrixBase<Real> &diff);

  /** Uses Svd to compute the eigenvalue decomposition of a symmetric positive
   * semi-definite matrix: (*this) = rP * diag(rS) * rP^T, with rP an
   * orthogonal matrix so rP^{-1} = rP^T.   Throws exception if input was not
   * positive semi-definite (check_thresh controls how stringent the check is;
   * set it to 2 to ensure it won't ever complain, but it will zero out negative
   * dimensions in your matrix.
  */
  void SymPosSemiDefEig(VectorBase<Real> *s, MatrixBase<Real> *P,
                        Real check_thresh = 0.001);

  friend Real kaldi::TraceMatMat<Real>(const MatrixBase<Real> &A,
      const MatrixBase<Real> &B, MatrixTransposeType trans);  // tr (A B)

  // so it can get around const restrictions on the pointer to data_.
  friend class SubMatrix<Real>;

  /// Add a scalar to each element
  void Add(const Real alpha);

  /// Add a scalar to each diagonal element.
  void AddToDiag(const Real alpha);

  /// *this += alpha * a * b^T
  template<typename OtherReal>
  void AddVecVec(const Real alpha, const VectorBase<OtherReal> &a,
                 const VectorBase<OtherReal> &b);

  /// [each row of *this] += alpha * v
  template<typename OtherReal>
  void AddVecToRows(const Real alpha, const VectorBase<OtherReal> &v);

  /// [each col of *this] += alpha * v
  template<typename OtherReal>
  void AddVecToCols(const Real alpha, const VectorBase<OtherReal> &v);

  /// *this += alpha * M [or M^T]
  void AddMat(const Real alpha, const MatrixBase<Real> &M,
              MatrixTransposeType transA = kNoTrans);

  /// *this += alpha * A [or A^T].
  void AddSmat(Real alpha, const SparseMatrix<Real> &A,
               MatrixTransposeType trans = kNoTrans);

  /// (*this) = alpha * op(A) * B + beta * (*this), where A is sparse.
  /// Multiplication of sparse with dense matrix.  See also AddMatSmat.
  void AddSmatMat(Real alpha, const SparseMatrix<Real> &A,
                  MatrixTransposeType transA, const MatrixBase<Real> &B,
                  Real beta);

  /// (*this) = alpha * A * op(B) + beta * (*this), where B is sparse
  /// and op(B) is either B or trans(B) depending on the 'transB' argument.
  /// This is multiplication of a dense by a sparse matrix.  See also
  /// AddSmatMat.
  void AddMatSmat(Real alpha, const MatrixBase<Real> &A,
                  const SparseMatrix<Real> &B, MatrixTransposeType transB,
                  Real beta);

  /// *this = beta * *this + alpha * M M^T, for symmetric matrices.  It only
  /// updates the lower triangle of *this.  It will leave the matrix asymmetric;
  /// if you need it symmetric as a regular matrix, do CopyLowerToUpper().
  void SymAddMat2(const Real alpha, const MatrixBase<Real> &M,
                  MatrixTransposeType transA, Real beta);

  /// *this = beta * *this + alpha * diag(v) * M [or M^T].
  /// The same as adding M but scaling each row M_i by v(i).
  void AddDiagVecMat(const Real alpha, const VectorBase<Real> &v,
                     const MatrixBase<Real> &M, MatrixTransposeType transM,
                     Real beta = 1.0);

  /// *this = beta * *this + alpha * M [or M^T] * diag(v)
  /// The same as adding M but scaling each column M_j by v(j).
  void AddMatDiagVec(const Real alpha,
                     const MatrixBase<Real> &M, MatrixTransposeType transM,
                     VectorBase<Real> &v,
                     Real beta = 1.0);

  /// *this = beta * *this + alpha * A .* B (.* element by element multiplication)
  void AddMatMatElements(const Real alpha,
                         const MatrixBase<Real>& A,
                         const MatrixBase<Real>& B,
                         const Real beta);

  /// *this += alpha * S
  template<typename OtherReal>
  void AddSp(const Real alpha, const SpMatrix<OtherReal> &S);

  void AddMatMat(const Real alpha,
                 const MatrixBase<Real>& A, MatrixTransposeType transA,
                 const MatrixBase<Real>& B, MatrixTransposeType transB,
                 const Real beta);

  /// *this = a * b / c (by element; when c = 0, *this = a)
  void SetMatMatDivMat(const MatrixBase<Real>& A,
                       const MatrixBase<Real>& B,
                       const MatrixBase<Real>& C);

  /// A version of AddMatMat specialized for when the second argument
  /// contains a lot of zeroes.
  void AddMatSmat(const Real alpha,
                  const MatrixBase<Real>& A, MatrixTransposeType transA,
                  const MatrixBase<Real>& B, MatrixTransposeType transB,
                  const Real beta);

  /// A version of AddMatMat specialized for when the first argument
  /// contains a lot of zeroes.
  void AddSmatMat(const Real alpha,
                  const MatrixBase<Real>& A, MatrixTransposeType transA,
                  const MatrixBase<Real>& B, MatrixTransposeType transB,
                  const Real beta);

  /// this <-- beta*this + alpha*A*B*C.
  void AddMatMatMat(const Real alpha,
                    const MatrixBase<Real>& A, MatrixTransposeType transA,
                    const MatrixBase<Real>& B, MatrixTransposeType transB,
                    const MatrixBase<Real>& C, MatrixTransposeType transC,
                    const Real beta);

  /// this <-- beta*this + alpha*SpA*B.
  // This and the routines below are really
  // stubs that need to be made more efficient.
  void AddSpMat(const Real alpha,
                const SpMatrix<Real>& A,
                const MatrixBase<Real>& B, MatrixTransposeType transB,
                const Real beta) {
    Matrix<Real> M(A);
    return AddMatMat(alpha, M, kNoTrans, B, transB, beta);
  }
  /// this <-- beta*this + alpha*A*B.
  void AddTpMat(const Real alpha,
                const TpMatrix<Real>& A, MatrixTransposeType transA,
                const MatrixBase<Real>& B, MatrixTransposeType transB,
                const Real beta) {
    Matrix<Real> M(A);
    return AddMatMat(alpha, M, transA, B, transB, beta);
  }
  /// this <-- beta*this + alpha*A*B.
  void AddMatSp(const Real alpha,
                const MatrixBase<Real>& A, MatrixTransposeType transA,
                const SpMatrix<Real>& B,
                const Real beta) {
    Matrix<Real> M(B);
    return AddMatMat(alpha, A, transA, M, kNoTrans, beta);
  }
  /// this <-- beta*this + alpha*A*B*C.
  void AddSpMatSp(const Real alpha,
                  const SpMatrix<Real> &A,
                  const MatrixBase<Real>& B, MatrixTransposeType transB,
                  const SpMatrix<Real>& C,
                const Real beta) {
    Matrix<Real> M(A), N(C);
    return AddMatMatMat(alpha, M, kNoTrans, B, transB, N, kNoTrans, beta);
  }
  /// this <-- beta*this + alpha*A*B.
  void AddMatTp(const Real alpha,
                const MatrixBase<Real>& A, MatrixTransposeType transA,
                const TpMatrix<Real>& B, MatrixTransposeType transB,
                const Real beta) {
    Matrix<Real> M(B);
    return AddMatMat(alpha, A, transA, M, transB, beta);
  }

  /// this <-- beta*this + alpha*A*B.
  void AddTpTp(const Real alpha,
               const TpMatrix<Real>& A, MatrixTransposeType transA,
               const TpMatrix<Real>& B, MatrixTransposeType transB,
               const Real beta) {
    Matrix<Real> M(A), N(B);
    return AddMatMat(alpha, M, transA, N, transB, beta);
  }

  /// this <-- beta*this + alpha*A*B.
  // This one is more efficient, not like the others above.
  void AddSpSp(const Real alpha,
               const SpMatrix<Real>& A, const SpMatrix<Real>& B,
               const Real beta);

  /// Copy lower triangle to upper triangle (symmetrize)
  void CopyLowerToUpper();

  /// Copy upper triangle to lower triangle (symmetrize)
  void CopyUpperToLower();

  /// This function orthogonalizes the rows of a matrix using the Gram-Schmidt
  /// process.  It is only applicable if NumRows() <= NumCols().  It will use
  /// random number generation to fill in rows with something nonzero, in cases
  /// where the original matrix was of deficient row rank.
  void OrthogonalizeRows();

  /// stream read.
  /// Use instead of stream<<*this, if you want to add to existing contents.
  // Will throw exception on failure.
  void Read(std::istream & in, bool binary, bool add = false);
  /// write to stream.
  void Write(std::ostream & out, bool binary) const;

  // Below is internal methods for Svd, user does not have to know about this.
#if !defined(HAVE_ATLAS) && !defined(USE_KALDI_SVD)
  // protected:
  // Should be protected but used directly in testing routine.
  // destroys *this!
  void LapackGesvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                     MatrixBase<Real> *Vt);
#else
 protected:
  // destroys *this!
  bool JamaSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
               MatrixBase<Real> *V);

#endif
 protected:

  ///  Initializer, callable only from child.
  explicit MatrixBase(Real *data, MatrixIndexT cols, MatrixIndexT rows, MatrixIndexT stride) :
    data_(data), num_cols_(cols), num_rows_(rows), stride_(stride) {
    KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  }

  ///  Initializer, callable only from child.
  /// Empty initializer, for un-initialized matrix.
  explicit MatrixBase(): data_(NULL) {
    KALDI_ASSERT_IS_FLOATING_TYPE(Real);
  }

  // Make sure pointers to MatrixBase cannot be deleted.
  ~MatrixBase() { }

  /// A workaround that allows SubMatrix to get a pointer to non-const data
  /// for const Matrix. Unfortunately C++ does not allow us to declare a
  /// "public const" inheritance or anything like that, so it would require
  /// a lot of work to make the SubMatrix class totally const-correct--
  /// we would have to override many of the Matrix functions.
  inline Real*  Data_workaround() const {
    return data_;
  }

  /// data memory area
  Real*   data_;

  /// these atributes store the real matrix size as it is stored in memory
  /// including memalignment
  MatrixIndexT    num_cols_;   /// < Number of columns
  MatrixIndexT    num_rows_;   /// < Number of rows
  /** True number of columns for the internal matrix. This number may differ
   * from num_cols_ as memory alignment might be used. */
  MatrixIndexT    stride_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(MatrixBase);
};

/// A class for storing matrices.
template<typename Real>
class Matrix : public MatrixBase<Real> {
 public:

  /// Empty constructor.
  Matrix();

  /// Basic constructor.
  Matrix(const MatrixIndexT r, const MatrixIndexT c,
         MatrixResizeType resize_type = kSetZero,
         MatrixStrideType stride_type = kDefaultStride):
      MatrixBase<Real>() { Resize(r, c, resize_type, stride_type); }

  /// Copy constructor from CUDA matrix
  /// This is defined in ../cudamatrix/cu-matrix.h
  template<typename OtherReal>
  explicit Matrix(const CuMatrixBase<OtherReal> &cu,
                  MatrixTransposeType trans = kNoTrans);


  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(Matrix<Real> *other);

  /// Defined in ../cudamatrix/cu-matrix.cc
  void Swap(CuMatrix<Real> *mat);

  /// Constructor from any MatrixBase. Can also copy with transpose.
  /// Allocates new memory.
  explicit Matrix(const MatrixBase<Real> & M,
                  MatrixTransposeType trans = kNoTrans);

  /// Same as above, but need to avoid default copy constructor.
  Matrix(const Matrix<Real> & M);  //  (cannot make explicit)

  /// Copy constructor: as above, but from another type.
  template<typename OtherReal>
  explicit Matrix(const MatrixBase<OtherReal> & M,
                    MatrixTransposeType trans = kNoTrans);

  /// Copy constructor taking SpMatrix...
  /// It is symmetric, so no option for transpose, and NumRows == Cols
  template<typename OtherReal>
  explicit Matrix(const SpMatrix<OtherReal> & M) : MatrixBase<Real>() {
    Resize(M.NumRows(), M.NumRows(), kUndefined);
    this->CopyFromSp(M);
  }

  /// Constructor from CompressedMatrix
  explicit Matrix(const CompressedMatrix &C);

  /// Copy constructor taking TpMatrix...
  template <typename OtherReal>
  explicit Matrix(const TpMatrix<OtherReal> & M,
                  MatrixTransposeType trans = kNoTrans) : MatrixBase<Real>() {
    if (trans == kNoTrans) {
      Resize(M.NumRows(), M.NumCols(), kUndefined);
      this->CopyFromTp(M);
    } else {
      Resize(M.NumCols(), M.NumRows(), kUndefined);
      this->CopyFromTp(M, kTrans);
    }
  }

  /// read from stream.
  // Unlike one in base, allows resizing.
  void Read(std::istream & in, bool binary, bool add = false);

  /// Remove a specified row.
  void RemoveRow(MatrixIndexT i);

  /// Transpose the matrix.  Works for non-square
  /// matrices as well as square ones.
  void Transpose();

  /// Distructor to free matrices.
  ~Matrix() { Destroy(); }

  /// Sets matrix to a specified size (zero is OK as long as both r and c are
  /// zero).  The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  ///
  /// You can set stride_type to kStrideEqualNumCols to force the stride
  /// to equal the number of columns; by default it is set so that the stride
  /// in bytes is a multiple of 16.
  ///
  /// This function takes time proportional to the number of data elements.
  void Resize(const MatrixIndexT r,
              const MatrixIndexT c,
              MatrixResizeType resize_type = kSetZero,
              MatrixStrideType stride_type = kDefaultStride);

  /// Assignment operator that takes MatrixBase.
  Matrix<Real> &operator = (const MatrixBase<Real> &other) {
    if (MatrixBase<Real>::NumRows() != other.NumRows() ||
        MatrixBase<Real>::NumCols() != other.NumCols())
      Resize(other.NumRows(), other.NumCols(), kUndefined);
    MatrixBase<Real>::CopyFromMat(other);
    return *this;
  }

  /// Assignment operator. Needed for inclusion in std::vector.
  Matrix<Real> &operator = (const Matrix<Real> &other) {
    if (MatrixBase<Real>::NumRows() != other.NumRows() ||
        MatrixBase<Real>::NumCols() != other.NumCols())
      Resize(other.NumRows(), other.NumCols(), kUndefined);
    MatrixBase<Real>::CopyFromMat(other);
    return *this;
  }


 private:
  /// Deallocates memory and sets to empty matrix (dimension 0, 0).
  void Destroy();

  /// Init assumes the current class contents are invalid (i.e. junk or have
  /// already been freed), and it sets the matrix to newly allocated memory with
  /// the specified number of rows and columns.  r == c == 0 is acceptable.  The data
  /// memory contents will be undefined.
  void Init(const MatrixIndexT r,
            const MatrixIndexT c,
            const MatrixStrideType stride_type);

};
/// @} end "addtogroup matrix_group"

/// \addtogroup matrix_funcs_io
/// @{

/// A structure containing the HTK header.
/// [TODO: change the style of the variables to Kaldi-compliant]
struct HtkHeader {
  /// Number of samples.
  int32    mNSamples;
  /// Sample period.
  int32    mSamplePeriod;
  /// Sample size
  int16    mSampleSize;
  /// Sample kind.
  uint16   mSampleKind;
};

// Read HTK formatted features from file into matrix.
template<typename Real>
bool ReadHtk(std::istream &is, Matrix<Real> *M, HtkHeader *header_ptr);

// Write (HTK format) features to file from matrix.
template<typename Real>
bool WriteHtk(std::ostream &os, const MatrixBase<Real> &M, HtkHeader htk_hdr);

// Write (CMUSphinx format) features to file from matrix.
template<typename Real>
bool WriteSphinx(std::ostream &os, const MatrixBase<Real> &M);

/// @} end of "addtogroup matrix_funcs_io"

/**
  Sub-matrix representation.
  Can work with sub-parts of a matrix using this class.
  Note that SubMatrix is not very const-correct-- it allows you to
  change the contents of a const Matrix.  Be careful!
*/

template<typename Real>
class SubMatrix : public MatrixBase<Real> {
 public:
  // Initialize a SubMatrix from part of a matrix; this is
  // a bit like A(b:c, d:e) in Matlab.
  // This initializer is against the proper semantics of "const", since
  // SubMatrix can change its contents.  It would be hard to implement
  // a "const-safe" version of this class.
  SubMatrix(const MatrixBase<Real>& T,
            const MatrixIndexT ro,  // row offset, 0 < ro < NumRows()
            const MatrixIndexT r,   // number of rows, r > 0
            const MatrixIndexT co,  // column offset, 0 < co < NumCols()
            const MatrixIndexT c);   // number of columns, c > 0

  // This initializer is mostly intended for use in CuMatrix and related
  // classes.  Be careful!
  SubMatrix(Real *data,
            MatrixIndexT num_rows,
            MatrixIndexT num_cols,
            MatrixIndexT stride);

  ~SubMatrix<Real>() {}

  /// This type of constructor is needed for Range() to work [in Matrix base
  /// class]. Cannot make it explicit.
  SubMatrix<Real> (const SubMatrix &other):
  MatrixBase<Real> (other.data_, other.num_cols_, other.num_rows_,
                    other.stride_) {}

 private:
  /// Disallow assignment.
  SubMatrix<Real> &operator = (const SubMatrix<Real> &other);
};
/// @} End of "addtogroup matrix_funcs_io".

/// \addtogroup matrix_funcs_scalar
/// @{

// Some declarations.  These are traces of products.


template<typename Real>
bool ApproxEqual(const MatrixBase<Real> &A,
                 const MatrixBase<Real> &B, Real tol = 0.01) {
  return A.ApproxEqual(B, tol);
}

template<typename Real>
inline void AssertEqual(const MatrixBase<Real> &A, const MatrixBase<Real> &B,
                        float tol = 0.01) {
  KALDI_ASSERT(A.ApproxEqual(B, tol));
}

/// Returns trace of matrix.
template <typename Real>
double TraceMat(const MatrixBase<Real> &A) { return A.Trace(); }


/// Returns tr(A B C)
template <typename Real>
Real TraceMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                      const MatrixBase<Real> &B, MatrixTransposeType transB,
                      const MatrixBase<Real> &C, MatrixTransposeType transC);

/// Returns tr(A B C D)
template <typename Real>
Real TraceMatMatMatMat(const MatrixBase<Real> &A, MatrixTransposeType transA,
                         const MatrixBase<Real> &B, MatrixTransposeType transB,
                         const MatrixBase<Real> &C, MatrixTransposeType transC,
                         const MatrixBase<Real> &D, MatrixTransposeType transD);

/// @} end "addtogroup matrix_funcs_scalar"


/// \addtogroup matrix_funcs_misc
/// @{


/// Function to ensure that SVD is sorted.  This function is made as generic as
/// possible, to be applicable to other types of problems.  s->Dim() should be
/// the same as U->NumCols(), and we sort s from greatest to least absolute
/// value (if sort_on_absolute_value == true) or greatest to least value
/// otherwise, moving the columns of U, if it exists, and the rows of Vt, if it
/// exists, around in the same way.  Note: the "absolute value" part won't matter
/// if this is an actual SVD, since singular values are non-negative.
template<typename Real> void SortSvd(VectorBase<Real> *s, MatrixBase<Real> *U,
                                     MatrixBase<Real>* Vt = NULL,
                                     bool sort_on_absolute_value = true);

/// Creates the eigenvalue matrix D that is part of the decomposition used Matrix::Eig.
/// D will be block-diagonal with blocks of size 1 (for real eigenvalues) or 2x2
/// for complex pairs.  If a complex pair is lambda +- i*mu, D will have a corresponding
/// 2x2 block [lambda, mu; -mu, lambda].
/// This function will throw if any complex eigenvalues are not in complex conjugate
/// pairs (or the members of such pairs are not consecutively numbered).
template<typename Real>
void CreateEigenvalueMatrix(const VectorBase<Real> &real, const VectorBase<Real> &imag,
                            MatrixBase<Real> *D);

/// The following function is used in Matrix::Power, and separately tested, so we
/// declare it here mainly for the testing code to see.  It takes a complex value to
/// a power using a method that will work for noninteger powers (but will fail if the
/// complex value is real and negative).
template<typename Real>
bool AttemptComplexPower(Real *x_re, Real *x_im, Real power);



/// @} end of addtogroup matrix_funcs_misc

/// \addtogroup matrix_funcs_io
/// @{
template<typename Real>
std::ostream & operator << (std::ostream & Out, const MatrixBase<Real> & M);

template<typename Real>
std::istream & operator >> (std::istream & In, MatrixBase<Real> & M);

// The Matrix read allows resizing, so we override the MatrixBase one.
template<typename Real>
std::istream & operator >> (std::istream & In, Matrix<Real> & M);


template<typename Real>
bool SameDim(const MatrixBase<Real> &M, const MatrixBase<Real> &N) {
  return (M.NumRows() == N.NumRows() && M.NumCols() == N.NumCols());
}

/// @} end of \addtogroup matrix_funcs_io


}  // namespace kaldi



// we need to include the implementation and some
// template specializations.
#include "matrix/kaldi-matrix-inl.h"


#endif  // KALDI_MATRIX_KALDI_MATRIX_H_
