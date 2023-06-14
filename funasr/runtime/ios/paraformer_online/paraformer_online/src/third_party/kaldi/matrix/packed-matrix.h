// matrix/packed-matrix.h

// Copyright 2009-2013  Ondrej Glembek;  Lukas Burget;  Microsoft Corporation;
//                      Saarland University;  Yanmin Qian;
//                      Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_MATRIX_PACKED_MATRIX_H_
#define KALDI_MATRIX_PACKED_MATRIX_H_

#include "matrix/matrix-common.h"
#include <algorithm>

namespace kaldi {

/// \addtogroup matrix_funcs_io
// we need to declare the friend << operator here
template<typename Real>
std::ostream & operator <<(std::ostream & out, const PackedMatrix<Real>& M);


/// \addtogroup matrix_group
/// @{

/// @brief Packed matrix: base class for triangular and symmetric matrices.
template<typename Real> class PackedMatrix {
  friend class CuPackedMatrix<Real>;
 public:
  //friend class CuPackedMatrix<Real>;

  PackedMatrix() : data_(NULL), num_rows_(0) {}

  explicit PackedMatrix(MatrixIndexT r, MatrixResizeType resize_type = kSetZero):
      data_(NULL) {  Resize(r, resize_type);  }

  explicit PackedMatrix(const PackedMatrix<Real> &orig) : data_(NULL) {
    Resize(orig.num_rows_, kUndefined);
    CopyFromPacked(orig);
  }

  template<typename OtherReal>
  explicit PackedMatrix(const PackedMatrix<OtherReal> &orig) : data_(NULL) {
    Resize(orig.NumRows(), kUndefined);
    CopyFromPacked(orig);
  }
  
  void SetZero();  /// < Set to zero
  void SetUnit();  /// < Set to unit matrix.
  void SetRandn(); /// < Set to random values of a normal distribution

  Real Trace() const;

  // Needed for inclusion in std::vector
  PackedMatrix<Real> & operator =(const PackedMatrix<Real> &other) {
    Resize(other.NumRows());
    CopyFromPacked(other);
    return *this;
  }

  ~PackedMatrix() {
    Destroy();
  }

  /// Set packed matrix to a specified size (can be zero).
  /// The value of the new data depends on resize_type:
  ///   -if kSetZero, the new data will be zero
  ///   -if kUndefined, the new data will be undefined
  ///   -if kCopyData, the new data will be the same as the old data in any
  ///      shared positions, and zero elsewhere.
  /// This function takes time proportional to the number of data elements.
  void Resize(MatrixIndexT nRows, MatrixResizeType resize_type = kSetZero);

  void AddToDiag(const Real r); // Adds r to diaginal

  void ScaleDiag(const Real alpha);  // Scales diagonal by alpha.

  void SetDiag(const Real alpha);  // Sets diagonal to this value.

  template<typename OtherReal>
  void CopyFromPacked(const PackedMatrix<OtherReal> &orig);
  
  /// CopyFromVec just interprets the vector as having the same layout
  /// as the packed matrix.  Must have the same dimension, i.e.
  /// orig.Dim() == (NumRows()*(NumRows()+1)) / 2;
  template<typename OtherReal>
  void CopyFromVec(const SubVector<OtherReal> &orig);
  
  Real* Data() { return data_; }
  const Real* Data() const { return data_; }
  inline MatrixIndexT NumRows() const { return num_rows_; }
  inline MatrixIndexT NumCols() const { return num_rows_; }
  size_t SizeInBytes() const {
    size_t nr = static_cast<size_t>(num_rows_);
    return ((nr * (nr+1)) / 2) * sizeof(Real);
  }

  //MatrixIndexT Stride() const { return stride_; }

  // This code is duplicated in child classes to avoid extra levels of calls.
  Real operator() (MatrixIndexT r, MatrixIndexT c) const {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(c) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_)
                 && c <= r);
    return *(data_ + (r * (r + 1)) / 2 + c);
  }

  // This code is duplicated in child classes to avoid extra levels of calls.
  Real &operator() (MatrixIndexT r, MatrixIndexT c) {
    KALDI_ASSERT(static_cast<UnsignedMatrixIndexT>(r) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_) &&
                 static_cast<UnsignedMatrixIndexT>(c) <
                 static_cast<UnsignedMatrixIndexT>(num_rows_)
                 && c <= r);
    return *(data_ + (r * (r + 1)) / 2 + c);
  }

  Real Max() const {
    KALDI_ASSERT(num_rows_ > 0);
    return * (std::max_element(data_, data_ + ((num_rows_*(num_rows_+1))/2) ));
  }

  Real Min() const {
    KALDI_ASSERT(num_rows_ > 0);
    return * (std::min_element(data_, data_ + ((num_rows_*(num_rows_+1))/2) ));
  }

  void Scale(Real c);

  friend std::ostream & operator << <> (std::ostream & out,
                                     const PackedMatrix<Real> &m);
  // Use instead of stream<<*this, if you want to add to existing contents.
  // Will throw exception on failure.
  void Read(std::istream &in, bool binary, bool add = false);

  void Write(std::ostream &out, bool binary) const;
  
  void Destroy();

  /// Swaps the contents of *this and *other.  Shallow swap.
  void Swap(PackedMatrix<Real> *other);
  void Swap(Matrix<Real> *other);


 protected:
  // Will only be called from this class or derived classes.
  void AddPacked(const Real alpha, const PackedMatrix<Real>& M);
  Real *data_;
  MatrixIndexT num_rows_;
  //MatrixIndexT stride_;
 private:
  /// Init assumes the current contents of the class are is invalid (i.e. junk or
  /// has already been freed), and it sets the matrixd to newly allocated memory
  /// with the specified dimension.  dim == 0 is acceptable.  The memory contents
  /// pointed to by data_ will be undefined.
  void Init(MatrixIndexT dim);

};
/// @} end "addtogroup matrix_group"


/// \addtogroup matrix_funcs_io
/// @{

template<typename Real>
std::ostream & operator << (std::ostream & os, const PackedMatrix<Real>& M) {
  M.Write(os, false);
  return os;
}

template<typename Real>
std::istream & operator >> (std::istream &is, PackedMatrix<Real> &M) {
  M.Read(is, false);
  return is;
}

/// @}

}  // namespace kaldi

#endif

