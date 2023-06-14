// matrix/sparse-matrix.h

// Copyright  2015  Johns Hopkins University (author: Daniel Povey)
//            2015  Guoguo Chen
//            2017  Shiyin Kang


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

#ifndef KALDI_MATRIX_SPARSE_MATRIX_H_
#define KALDI_MATRIX_SPARSE_MATRIX_H_ 1

#include <utility>
#include <vector>

#include "matrix/matrix-common.h"
#include "matrix/kaldi-matrix.h"
#include "matrix/kaldi-vector.h"
#include "matrix/compressed-matrix.h"

namespace kaldi {


/// \addtogroup matrix_group
/// @{

template <typename Real>
class SparseVector {
 public:
  MatrixIndexT Dim() const { return dim_; }

  Real Sum() const;

  template <class OtherReal>
  void CopyElementsToVec(VectorBase<OtherReal> *vec) const;

  // *vec += alpha * *this.
  template <class OtherReal>
  void AddToVec(Real alpha,
                VectorBase<OtherReal> *vec) const;

  template <class OtherReal>
  void CopyFromSvec(const SparseVector<OtherReal> &other);

  SparseVector<Real> &operator = (const SparseVector<Real> &other);

  SparseVector(const SparseVector<Real> &other) { *this = other; }

  void Swap(SparseVector<Real> *other);

  // Returns the maximum value in this row and outputs the index associated with
  // it.  This is not the index into the Data() pointer, it is the index into
  // the vector it represents, i.e. the .first value in the pair.
  // If this vector's Dim() is zero it is an error to call this function.
  // If all the elements stored were negative and there underlying vector had
  // zero indexes not listed in the elements, or if no elements are stored, it
  // will return the first un-listed index, whose value (implicitly) is zero.
  Real Max(int32 *index) const;

  /// Returns the number of nonzero elements.
  MatrixIndexT NumElements() const { return pairs_.size(); }

  /// get an indexed element (0 <= i < NumElements()).
  const std::pair<MatrixIndexT, Real> &GetElement(MatrixIndexT i) const {
    return pairs_[i];
  }

  // returns pointer to element data, or NULL if empty (use with NumElements()).
  std::pair<MatrixIndexT, Real> *Data();

  // returns pointer to element data, or NULL if empty (use with NumElements());
  // const version
  const std::pair<MatrixIndexT, Real> *Data() const;

  /// Sets elements to zero with probability zero_prob, else normally
  /// distributed.  Useful in testing.
  void SetRandn(BaseFloat zero_prob);

  SparseVector(): dim_(0) { }

  explicit SparseVector(MatrixIndexT dim): dim_(dim) { KALDI_ASSERT(dim >= 0); }

  // constructor from pairs; does not assume input pairs are sorted and uniq
  SparseVector(MatrixIndexT dim,
               const std::vector<std::pair<MatrixIndexT, Real> > &pairs);

  // constructor from a VectorBase that keeps only the nonzero elements of 'vec'.
  explicit SparseVector(const VectorBase<Real> &vec);

  /// Resizes to this dimension.  resize_type == kUndefined
  /// behaves the same as kSetZero.
  void Resize(MatrixIndexT dim, MatrixResizeType resize_type = kSetZero);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &os, bool binary);

  /// Scale all elements of sparse vector.
  void Scale(Real alpha);

 private:
  MatrixIndexT dim_;
  // pairs of (row-index, value).  Stored in sorted order with no duplicates.
  // For now we use std::vector, but we could change this.
  std::vector<std::pair<MatrixIndexT, Real> > pairs_;
};


template <typename Real>
Real VecSvec(const VectorBase<Real> &vec,
             const SparseVector<Real> &svec);



template <typename Real>
class SparseMatrix {
 public:
  MatrixIndexT NumRows() const;

  MatrixIndexT NumCols() const;

  MatrixIndexT NumElements() const;

  Real Sum() const;

  Real FrobeniusNorm() const;


  /// This constructor creates a SparseMatrix that just contains the nonzero
  /// elements of 'mat'.
  explicit SparseMatrix(const MatrixBase<Real> &mat);

  /// Copy to matrix.  It must already have the correct size.
  template <class OtherReal>
  void CopyToMat(MatrixBase<OtherReal> *other,
                 MatrixTransposeType t = kNoTrans) const;

  /// Copies the values of all the elements in SparseMatrix into a VectorBase
  /// object.
  void CopyElementsToVec(VectorBase<Real> *other) const;

  /// Copies data from another sparse matrix.
  template<class OtherReal>
  void CopyFromSmat(const SparseMatrix<OtherReal> &other,
                    MatrixTransposeType trans = kNoTrans);

  /// Does *other = *other + alpha * *this.
  void AddToMat(BaseFloat alpha, MatrixBase<Real> *other,
                MatrixTransposeType t = kNoTrans) const;

  SparseMatrix<Real> &operator = (const SparseMatrix<Real> &other);

  SparseMatrix(const SparseMatrix<Real> &other, MatrixTransposeType trans =
                   kNoTrans) {
    this->CopyFromSmat(other, trans);
  }

  void Swap(SparseMatrix<Real> *other);

  // returns pointer to element data, or NULL if empty (use with NumElements()).
  SparseVector<Real> *Data();

  // returns pointer to element data, or NULL if empty (use with NumElements());
  // const version
  const SparseVector<Real> *Data() const;

  // initializer from the type that elsewhere in Kaldi is referred to as type
  // Posterior. indexed first by row-index; the pairs are (column-index, value),
  // and the constructor does not require them to be sorted and uniq.
  SparseMatrix(
      int32 dim,
      const std::vector<std::vector<std::pair<MatrixIndexT, Real> > > &pairs);

  /// Sets up to a pseudo-randomly initialized matrix, with each element zero
  /// with probability zero_prob and else normally distributed- mostly for
  /// purposes of testing.
  void SetRandn(BaseFloat zero_prob);

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &os, bool binary);

  const SparseVector<Real> &Row(MatrixIndexT r) const;

  /// Sets row r to "vec"; makes sure it has the correct dimension.
  void SetRow(int32 r, const SparseVector<Real> &vec);

  /// Select a subset of the rows of a SparseMatrix.
  /// Sets *this to only the rows of 'smat_other' that are listed
  /// in 'row_indexes'.
  /// 'row_indexes' must satisfy 0 <= row_indexes[i] < smat_other.NumRows().
  void SelectRows(const std::vector<int32> &row_indexes,
                  const SparseMatrix<Real> &smat_other);


  /// Sets *this to all the rows of *inputs appended together; this
  /// function is destructive of the inputs.  Requires, obviously,
  /// that the inputs all have the same dimension (although some may be
  /// empty).
  void AppendSparseMatrixRows(std::vector<SparseMatrix<Real> > *inputs);

  SparseMatrix() { }

  SparseMatrix(int32 num_rows, int32 num_cols) { Resize(num_rows, num_cols); }

  /// Constructor from an array of indexes.
  /// If trans == kNoTrans, construct a sparse matrix
  /// with num-rows == indexes.Dim() and num-cols = 'dim'.
  /// 'indexes' is expected to contain elements in the
  /// range [0, dim - 1].  Each row 'i' of *this after
  /// calling the constructor will contain  a single
  /// element at column-index indexes[i] with value 1.0.
  ///
  /// If trans == kTrans, the result will be the transpose
  /// of the sparse matrix described above.
  SparseMatrix(const std::vector<int32> &indexes, int32 dim,
               MatrixTransposeType trans = kNoTrans);

  /// Constructor from an array of indexes and an array of
  /// weights; requires indexes.Dim() == weights.Dim().
  /// If trans == kNoTrans, construct a sparse matrix
  /// with num-rows == indexes.Dim() and num-cols = 'dim'.
  /// 'indexes' is expected to contain elements in the
  /// range [0, dim - 1].  Each row 'i' of *this after
  /// calling the constructor will contain a single
  /// element at column-index indexes[i] with value weights[i].
  /// If trans == kTrans, the result will be the transpose
  /// of the sparse matrix described above.
  SparseMatrix(const std::vector<int32> &indexes,
               const VectorBase<Real> &weights, int32 dim,
               MatrixTransposeType trans = kNoTrans);

  /// Resizes the matrix; analogous to Matrix::Resize().  resize_type ==
  /// kUndefined behaves the same as kSetZero.
  void Resize(MatrixIndexT rows, MatrixIndexT cols,
              MatrixResizeType resize_type = kSetZero);

  /// Scale all elements in sparse matrix.
  void Scale(Real alpha);

  // Use the Matrix::CopyFromSmat() function to copy from this to Matrix.  Also
  // see Matrix::AddSmat().  There is not very extensive functionality for
  // SparseMat just yet (e.g. no matrix multiply); we will add things as needed
  // and as it seems necessary.
 private:
  // vector of SparseVectors, all of same dime (use an stl vector for now; this
  // could change).
  std::vector<SparseVector<Real> > rows_;
};


template<typename Real>
Real TraceMatSmat(const MatrixBase<Real> &A,
                  const SparseMatrix<Real> &B,
                  MatrixTransposeType trans = kNoTrans);


enum GeneralMatrixType {
  kFullMatrix,
  kCompressedMatrix,
  kSparseMatrix
};

/// This class is a wrapper that enables you to store a matrix
/// in one of three forms: either as a Matrix<BaseFloat>, or a CompressedMatrix,
/// or a SparseMatrix<BaseFloat>.  It handles the I/O for you, i.e. you read
/// and write a single object type.  It is useful for neural-net training
/// targets which might be sparse or not, and might be compressed or not.
class GeneralMatrix {
 public:
  /// Returns the type of the matrix: kSparseMatrix, kCompressedMatrix or
  /// kFullMatrix.  If this matrix is empty, returns kFullMatrix.
  GeneralMatrixType Type() const;

  void Compress();  // If it was a full matrix, compresses, changing Type() to
                    // kCompressedMatrix; otherwise does nothing.

  void Uncompress();  // If it was a compressed matrix, uncompresses, changing
                      // Type() to kFullMatrix; otherwise does nothing.

  void Write(std::ostream &os, bool binary) const;


  /// Note: if you write a compressed matrix in text form, it will be read as
  /// a regular full matrix.
  void Read(std::istream &is, bool binary);

  /// Returns the contents as a SparseMatrix.  This will only work if
  /// Type() returns kSparseMatrix, or NumRows() == 0; otherwise it will crash.
  const SparseMatrix<BaseFloat> &GetSparseMatrix() const;

  /// Swaps the with the given SparseMatrix.  This will only work if
  /// Type() returns kSparseMatrix, or NumRows() == 0.
  void SwapSparseMatrix(SparseMatrix<BaseFloat> *smat);

  /// Returns the contents as a compressed matrix.  This will only work if
  /// Type() returns kCompressedMatrix, or NumRows() == 0; otherwise it will
  /// crash.
  const CompressedMatrix &GetCompressedMatrix() const;

  /// Swaps the with the given CompressedMatrix.  This will only work if
  /// Type() returns kCompressedMatrix, or NumRows() == 0.
  void SwapCompressedMatrix(CompressedMatrix *cmat);

  /// Returns the contents as a Matrix<BaseFloat>.  This will only work if
  /// Type() returns kFullMatrix, or NumRows() == 0; otherwise it will crash.
  const Matrix<BaseFloat>& GetFullMatrix() const;

  /// Outputs the contents as a matrix.  This will work regardless of
  /// Type().  Sizes its output, unlike CopyToMat().
  void GetMatrix(Matrix<BaseFloat> *mat) const;

  /// Swaps the with the given Matrix.  This will only work if
  /// Type() returns kFullMatrix, or NumRows() == 0.
  void SwapFullMatrix(Matrix<BaseFloat> *mat);

  /// Copies contents, regardless of type, to "mat", which must be correctly
  /// sized.  See also GetMatrix(), which will size its output for you.
  void CopyToMat(MatrixBase<BaseFloat> *mat,
                 MatrixTransposeType trans = kNoTrans) const;

  /// Copies contents, regardless of type, to "cu_mat", which must be
  /// correctly sized.  Implemented in ../cudamatrix/cu-sparse-matrix.cc
  void CopyToMat(CuMatrixBase<BaseFloat> *cu_mat,
                 MatrixTransposeType trans = kNoTrans) const;

  /// Adds alpha times *this to mat.
  void AddToMat(BaseFloat alpha, MatrixBase<BaseFloat> *mat,
                MatrixTransposeType trans = kNoTrans) const;

  /// Adds alpha times *this to cu_mat.
  /// Implemented in ../cudamatrix/cu-sparse-matrix.cc
  void AddToMat(BaseFloat alpha, CuMatrixBase<BaseFloat> *cu_mat,
                MatrixTransposeType trans = kNoTrans) const;

  /// Scale each element of matrix by alpha.
  void Scale(BaseFloat alpha);

  /// Assignment from regular matrix.
  GeneralMatrix &operator= (const MatrixBase<BaseFloat> &mat);

  /// Assignment from compressed matrix.
  GeneralMatrix &operator= (const CompressedMatrix &mat);

  /// Assignment from SparseMatrix<BaseFloat>
  GeneralMatrix &operator= (const SparseMatrix<BaseFloat> &smat);

  MatrixIndexT NumRows() const;

  MatrixIndexT NumCols() const;

  explicit GeneralMatrix(const MatrixBase<BaseFloat> &mat) { *this = mat; }

  explicit GeneralMatrix(const CompressedMatrix &cmat) { *this = cmat; }

  explicit GeneralMatrix(const SparseMatrix<BaseFloat> &smat) { *this = smat; }

  GeneralMatrix() { }
  // Assignment operator.
  GeneralMatrix &operator =(const GeneralMatrix &other);
  // Copy constructor
  GeneralMatrix(const GeneralMatrix &other) { *this = other; }
  // Sets to the empty matrix.
  void Clear();
  // shallow swap
  void Swap(GeneralMatrix *other);
 private:
  // We don't explicitly store the type of the matrix.  Rather, we make
  // sure that only one of the matrices is ever nonempty, and the Type()
  // returns that one, or kFullMatrix if all are empty.
  Matrix<BaseFloat> mat_;
  CompressedMatrix cmat_;
  SparseMatrix<BaseFloat> smat_;
};


/// Appends all the matrix rows of a list of GeneralMatrixes, to get a single
/// GeneralMatrix.  Preserves sparsity if all inputs were sparse (or empty).
/// Does not preserve compression, if inputs were compressed; you have to
/// re-compress manually, if that's what you need.
void AppendGeneralMatrixRows(const std::vector<const GeneralMatrix *> &src,
                             GeneralMatrix *mat);


/// Outputs a SparseMatrix<Real> containing only the rows r of "in" such that
/// keep_rows[r] == true.  keep_rows.size() must equal in.NumRows(), and rows
/// must contain at least one "true" element.
template <typename Real>
void FilterSparseMatrixRows(const SparseMatrix<Real> &in,
                            const std::vector<bool> &keep_rows,
                            SparseMatrix<Real> *out);

/// Outputs a Matrix<Real> containing only the rows r of "in" such that
/// keep_keep_rows[r] == true.  keep_rows.size() must equal in.NumRows(), and
/// keep_rows must contain at least one "true" element.
template <typename Real>
void FilterMatrixRows(const Matrix<Real> &in,
                      const std::vector<bool> &keep_rows,
                      Matrix<Real> *out);

/// Outputs a Matrix<Real> containing only the rows r of "in" such that
/// keep_rows[r] == true.  keep_rows.size() must equal in.NumRows(), and rows
/// must contain at least one "true" element.
void FilterCompressedMatrixRows(const CompressedMatrix &in,
                                const std::vector<bool> &keep_rows,
                                Matrix<BaseFloat> *out);


/// Outputs a GeneralMatrix containing only the rows r of "in" such that
/// keep_rows[r] == true.  keep_rows.size() must equal in.NumRows(), and
/// keep_rows must contain at least one "true" element.  If in.Type() is
/// kCompressedMatrix, the result will not be compressed; otherwise, the type
/// is preserved.
void FilterGeneralMatrixRows(const GeneralMatrix &in,
                             const std::vector<bool> &keep_rows,
                             GeneralMatrix *out);

/// This function extracts a row-range of a GeneralMatrix and writes
/// as a GeneralMatrix containing the same type of underlying
/// matrix.  If the row-range is partly outside the row-range of 'in'
/// (i.e. if row_offset < 0 or row_offset + num_rows > in.NumRows())
/// then it will pad with copies of the first and last row as
/// needed.
/// This is more efficient than un-compressing and
/// re-compressing the underlying CompressedMatrix, and causes
/// less accuracy loss due to re-compression (no loss in most cases).
void ExtractRowRangeWithPadding(
    const GeneralMatrix &in,
    int32 row_offset,
    int32 num_rows,
    GeneralMatrix *out);


/// @} end of \addtogroup matrix_group


}  // namespace kaldi

#endif  // KALDI_MATRIX_SPARSE_MATRIX_H_
