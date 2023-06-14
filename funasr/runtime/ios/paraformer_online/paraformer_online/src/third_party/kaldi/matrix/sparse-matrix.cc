// matrix/sparse-matrix.cc

// Copyright 2015     Johns Hopkins University (author: Daniel Povey)
//           2015     Guoguo Chen
//           2017     Shiyin Kang

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

#include <algorithm>
#include <limits>
#include <string>

#include "matrix/sparse-matrix.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

template <typename Real>
std::pair<MatrixIndexT, Real>* SparseVector<Real>::Data() {
  if (pairs_.empty())
    return NULL;
  else
    return &(pairs_[0]);
}

template <typename Real>
const std::pair<MatrixIndexT, Real>* SparseVector<Real>::Data() const {
  if (pairs_.empty())
    return NULL;
  else
    return &(pairs_[0]);
}

template <typename Real>
Real SparseVector<Real>::Sum() const {
  Real sum = 0;
  for (int32 i = 0; i < pairs_.size(); ++i) {
    sum += pairs_[i].second;
  }
  return sum;
}

template<typename Real>
void SparseVector<Real>::Scale(Real alpha) {
  for (int32 i = 0; i < pairs_.size(); ++i)
    pairs_[i].second *= alpha;
}

template <typename Real>
template <typename OtherReal>
void SparseVector<Real>::CopyElementsToVec(VectorBase<OtherReal> *vec) const {
  KALDI_ASSERT(vec->Dim() == this->dim_);
  vec->SetZero();
  OtherReal *other_data = vec->Data();
  typename std::vector<std::pair<MatrixIndexT, Real> >::const_iterator
      iter = pairs_.begin(), end = pairs_.end();
  for (; iter != end; ++iter)
    other_data[iter->first] = iter->second;
}
template
void SparseVector<float>::CopyElementsToVec(VectorBase<float> *vec) const;
template
void SparseVector<float>::CopyElementsToVec(VectorBase<double> *vec) const;
template
void SparseVector<double>::CopyElementsToVec(VectorBase<float> *vec) const;
template
void SparseVector<double>::CopyElementsToVec(VectorBase<double> *vec) const;

template <typename Real>
template <typename OtherReal>
void SparseVector<Real>::AddToVec(Real alpha,
                                  VectorBase<OtherReal> *vec) const {
  KALDI_ASSERT(vec->Dim() == dim_);
  OtherReal *other_data = vec->Data();
  typename std::vector<std::pair<MatrixIndexT, Real> >::const_iterator
      iter = pairs_.begin(), end = pairs_.end();
  if (alpha == 1.0) {  // treat alpha==1.0 case specially.
    for (; iter != end; ++iter)
      other_data[iter->first] += iter->second;
  } else {
    for (; iter != end; ++iter)
      other_data[iter->first] += alpha * iter->second;
  }
}

template
void SparseVector<float>::AddToVec(float alpha, VectorBase<float> *vec) const;
template
void SparseVector<float>::AddToVec(float alpha, VectorBase<double> *vec) const;
template
void SparseVector<double>::AddToVec(double alpha, VectorBase<float> *vec) const;
template
void SparseVector<double>::AddToVec(double alpha,
                                    VectorBase<double> *vec) const;

template <typename Real>
template <typename OtherReal>
void SparseVector<Real>::CopyFromSvec(const SparseVector<OtherReal> &other) {
  dim_ = other.Dim();
  pairs_.clear();
  if (dim_ == 0) return;
  for (int32 i = 0; i < other.NumElements(); ++i) {
    pairs_.push_back(std::make_pair(
        other.GetElement(i).first,
        static_cast<Real>(other.GetElement(i).second)));
  }
}
template
void SparseVector<float>::CopyFromSvec(const SparseVector<float> &svec);
template
void SparseVector<float>::CopyFromSvec(const SparseVector<double> &svec);
template
void SparseVector<double>::CopyFromSvec(const SparseVector<float> &svec);
template
void SparseVector<double>::CopyFromSvec(const SparseVector<double> &svec);


template <typename Real>
SparseVector<Real>& SparseVector<Real>::operator = (
    const SparseVector<Real> &other) {
  this->CopyFromSvec(other);
  dim_ = other.dim_;
  pairs_ = other.pairs_;
  return *this;
}

template <typename Real>
void SparseVector<Real>::Swap(SparseVector<Real> *other) {
  pairs_.swap(other->pairs_);
  std::swap(dim_, other->dim_);
}

template <typename Real>
void SparseVector<Real>::Write(std::ostream &os, bool binary) const {
  if (binary) {
    WriteToken(os, binary, "SV");
    WriteBasicType(os, binary, dim_);
    MatrixIndexT num_elems = pairs_.size();
    WriteBasicType(os, binary, num_elems);
    typename std::vector<std::pair<MatrixIndexT, Real> >::const_iterator
        iter = pairs_.begin(), end = pairs_.end();
    for (; iter != end; ++iter) {
      WriteBasicType(os, binary, iter->first);
      WriteBasicType(os, binary, iter->second);
    }
  } else {
    // In text-mode, use a human-friendly, script-friendly format;
    // format is "dim=5 [ 0 0.2 3 0.9 ] "
    os << "dim=" << dim_ << " [ ";
    typename std::vector<std::pair<MatrixIndexT, Real> >::const_iterator
        iter = pairs_.begin(), end = pairs_.end();
    for (; iter != end; ++iter)
      os << iter->first << ' ' << iter->second << ' ';
    os << "] ";
  }
}


template <typename Real>
void SparseVector<Real>::Read(std::istream &is, bool binary) {
  if (binary) {
    ExpectToken(is, binary, "SV");
    ReadBasicType(is, binary, &dim_);
    KALDI_ASSERT(dim_ >= 0);
    int32 num_elems;
    ReadBasicType(is, binary, &num_elems);
    KALDI_ASSERT(num_elems >= 0 && num_elems <= dim_);
    pairs_.resize(num_elems);
    typename std::vector<std::pair<MatrixIndexT, Real> >::iterator
        iter = pairs_.begin(), end = pairs_.end();
    for (; iter != end; ++iter) {
      ReadBasicType(is, binary, &(iter->first));
      ReadBasicType(is, binary, &(iter->second));
    }
  } else {
    // In text-mode, format is "dim=5 [ 0 0.2 3 0.9 ]
    std::string str;
    is >> str;
    if (str.substr(0, 4) != "dim=")
      KALDI_ERR << "Reading sparse vector, expected 'dim=xxx', got " << str;
    std::string dim_str = str.substr(4, std::string::npos);
    std::istringstream dim_istr(dim_str);
    int32 dim = -1;
    dim_istr >> dim;
    if (dim < 0 || dim_istr.fail()) {
      KALDI_ERR << "Reading sparse vector, expected 'dim=[int]', got " << str;
    }
    dim_ = dim;
    is >> std::ws;
    is >> str;
    if (str != "[")
      KALDI_ERR << "Reading sparse vector, expected '[', got " << str;
    pairs_.clear();
    while (1) {
      is >> std::ws;
      if (is.peek() == ']') {
        is.get();
        break;
      }
      MatrixIndexT i;
      BaseFloat p;
      is >> i >> p;
      if (is.fail())
        KALDI_ERR << "Error reading sparse vector, expecting numbers.";
      KALDI_ASSERT(i >= 0 && i < dim
                   && (pairs_.empty() || i > pairs_.back().first));
      pairs_.push_back(std::pair<MatrixIndexT, BaseFloat>(i, p));
    }
  }
}


namespace sparse_vector_utils {
template <typename Real>
struct CompareFirst {
  inline bool operator() (const std::pair<MatrixIndexT, Real> &p1,
                           const std::pair<MatrixIndexT, Real> &p2) const {
    return p1.first < p2.first;
  }
};
}

template <typename Real>
SparseVector<Real>::SparseVector(
    MatrixIndexT dim, const std::vector<std::pair<MatrixIndexT, Real> > &pairs):
    dim_(dim),
    pairs_(pairs) {
  std::sort(pairs_.begin(), pairs_.end(),
            sparse_vector_utils::CompareFirst<Real>());
  typename std::vector<std::pair<MatrixIndexT, Real> >::iterator
      out = pairs_.begin(), in = out,  end = pairs_.end();
  // special case: while there is nothing to be changed, skip over
  // initial input (avoids unnecessary copying).
  while (in + 1 < end && in[0].first != in[1].first && in[0].second != 0.0) {
    in++;
    out++;
  }
  while (in < end) {
    // We reach this point only at the first element of
    // each stretch of identical .first elements.
    *out = *in;
    ++in;
    while (in < end && in->first == out->first) {
      out->second += in->second;  // this is the merge operation.
      ++in;
    }
    if (out->second != Real(0.0))  // Don't keep zero elements.
      out++;
  }
  pairs_.erase(out, end);
  if (!pairs_.empty()) {
    // range check.
    KALDI_ASSERT(pairs_.front().first >= 0 && pairs_.back().first < dim_);
  }
}

template <typename Real>
void SparseVector<Real>::SetRandn(BaseFloat zero_prob) {
  pairs_.clear();
  KALDI_ASSERT(zero_prob >= 0 && zero_prob <= 1.0);
  for (MatrixIndexT i = 0; i < dim_; i++)
    if (WithProb(1.0 - zero_prob))
      pairs_.push_back(std::pair<MatrixIndexT, Real>(i, RandGauss()));
}

template <typename Real>
void SparseVector<Real>::Resize(MatrixIndexT dim,
                                MatrixResizeType resize_type) {
  if (resize_type != kCopyData || dim == 0)
    pairs_.clear();
  KALDI_ASSERT(dim >= 0);
  if (dim < dim_ && resize_type == kCopyData)
    while (!pairs_.empty() && pairs_.back().first >= dim)
      pairs_.pop_back();
  dim_ = dim;
}

template <typename Real>
MatrixIndexT SparseMatrix<Real>::NumRows() const {
  return rows_.size();
}

template <typename Real>
MatrixIndexT SparseMatrix<Real>::NumCols() const {
  if (rows_.empty())
    return 0.0;
  else
    return rows_[0].Dim();
}

template <typename Real>
MatrixIndexT SparseMatrix<Real>::NumElements() const {
  int32 num_elements = 0;
  for (int32 i = 0; i < rows_.size(); ++i) {
    num_elements += rows_[i].NumElements();
  }
  return num_elements;
}

template <typename Real>
SparseVector<Real>* SparseMatrix<Real>::Data() {
  if (rows_.empty())
    return NULL;
  else
    return rows_.data();
}

template <typename Real>
const SparseVector<Real>* SparseMatrix<Real>::Data() const {
  if (rows_.empty())
    return NULL;
  else
    return rows_.data();
}

template <typename Real>
Real SparseMatrix<Real>::Sum() const {
  Real sum = 0;
  for (int32 i = 0; i < rows_.size(); ++i) {
    sum += rows_[i].Sum();
  }
  return sum;
}

template<typename Real>
Real SparseMatrix<Real>::FrobeniusNorm() const {
  Real squared_sum = 0;
  for (int32 i = 0; i < rows_.size(); ++i) {
    const std::pair<MatrixIndexT, Real> *row_data = rows_[i].Data();
    for (int32 j = 0; j < rows_[i].NumElements(); ++j) {
      squared_sum += row_data[j].second * row_data[j].second;
    }
  }
  return std::sqrt(squared_sum);
}

template <typename Real>
template <typename OtherReal>
void SparseMatrix<Real>::CopyToMat(MatrixBase<OtherReal> *other,
                                   MatrixTransposeType trans) const {
  if (trans == kNoTrans) {
    MatrixIndexT num_rows = rows_.size();
    KALDI_ASSERT(other->NumRows() == num_rows);
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      SubVector<OtherReal> vec(*other, i);
      rows_[i].CopyElementsToVec(&vec);
    }
  } else {
    OtherReal *other_col_data = other->Data();
    MatrixIndexT other_stride = other->Stride(),
        num_rows = NumRows(), num_cols = NumCols();
    KALDI_ASSERT(num_rows == other->NumCols() && num_cols == other->NumRows());
    other->SetZero();
    for (MatrixIndexT row = 0; row < num_rows; row++, other_col_data++) {
      const SparseVector<Real> &svec = rows_[row];
      MatrixIndexT num_elems = svec.NumElements();
      const std::pair<MatrixIndexT, Real> *sdata = svec.Data();
      for (MatrixIndexT e = 0; e < num_elems; e++)
        other_col_data[sdata[e].first * other_stride] = sdata[e].second;
    }
  }
}

template
void SparseMatrix<float>::CopyToMat(MatrixBase<float> *other,
                                    MatrixTransposeType trans) const;
template
void SparseMatrix<float>::CopyToMat(MatrixBase<double> *other,
                                    MatrixTransposeType trans) const;
template
void SparseMatrix<double>::CopyToMat(MatrixBase<float> *other,
                                    MatrixTransposeType trans) const;
template
void SparseMatrix<double>::CopyToMat(MatrixBase<double> *other,
                                    MatrixTransposeType trans) const;

template <typename Real>
void SparseMatrix<Real>::CopyElementsToVec(VectorBase<Real> *other) const {
  KALDI_ASSERT(other->Dim() == NumElements());
  Real *dst_data = other->Data();
  int32 dst_index = 0;
  for (int32 i = 0; i < rows_.size(); ++i) {
    for (int32 j = 0; j < rows_[i].NumElements(); ++j) {
      dst_data[dst_index] =
          static_cast<Real>(rows_[i].GetElement(j).second);
      dst_index++;
    }
  }
}

template<typename Real>
template<typename OtherReal>
void SparseMatrix<Real>::CopyFromSmat(const SparseMatrix<OtherReal> &other,
                                      MatrixTransposeType trans) {
  if (trans == kNoTrans) {
    rows_.resize(other.NumRows());
    if (rows_.size() == 0)
      return;
    for (int32 r = 0; r < rows_.size(); ++r) {
      rows_[r].CopyFromSvec(other.Row(r));
    }
  } else {
    std::vector<std::vector<std::pair<MatrixIndexT, Real> > > pairs(
        other.NumCols());
    for (MatrixIndexT i = 0; i < other.NumRows(); ++i) {
      for (int id = 0; id < other.Row(i).NumElements(); ++id) {
        MatrixIndexT j = other.Row(i).GetElement(id).first;
        Real v = static_cast<Real>(other.Row(i).GetElement(id).second);
        pairs[j].push_back( { i, v });
      }
    }
    SparseMatrix<Real> temp(other.NumRows(), pairs);
    Swap(&temp);
  }
}
template
void SparseMatrix<float>::CopyFromSmat(const SparseMatrix<float> &other,
                                       MatrixTransposeType trans);
template
void SparseMatrix<float>::CopyFromSmat(const SparseMatrix<double> &other,
                                       MatrixTransposeType trans);
template
void SparseMatrix<double>::CopyFromSmat(const SparseMatrix<float> &other,
                                        MatrixTransposeType trans);
template
void SparseMatrix<double>::CopyFromSmat(const SparseMatrix<double> &other,
                                        MatrixTransposeType trans);

template <typename Real>
void SparseMatrix<Real>::Write(std::ostream &os, bool binary) const {
  if (binary) {
    // Note: we can use the same marker for float and double SparseMatrix,
    // because internally we use WriteBasicType and ReadBasicType to read the
    // floats and doubles, and this will automatically take care of type
    // conversion.
    WriteToken(os, binary, "SM");
    int32 num_rows = rows_.size();
    WriteBasicType(os, binary, num_rows);
    for (int32 row = 0; row < num_rows; row++)
      rows_[row].Write(os, binary);
  } else {
    // The format is "rows=10 dim=20 [ 1 0.4  9 1.2 ] dim=20 [ 3 1.7 19 0.6 ] ..
    // not 100% efficient, but easy to work with, and we can re-use the
    // read/write code from SparseVector.
    int32 num_rows = rows_.size();
    os << "rows=" << num_rows << " ";
    for (int32 row = 0; row < num_rows; row++)
      rows_[row].Write(os, binary);
    os << "\n";  // Might make it a little more readable.
  }
}

template <typename Real>
void SparseMatrix<Real>::Read(std::istream &is, bool binary) {
  if (binary) {
    ExpectToken(is, binary, "SM");
    int32 num_rows;
    ReadBasicType(is, binary, &num_rows);
    KALDI_ASSERT(num_rows >= 0 && num_rows < 10000000);
    rows_.resize(num_rows);
    for (int32 row = 0; row < num_rows; row++)
      rows_[row].Read(is, binary);
  } else {
    std::string str;
    is >> str;
    if (str.substr(0, 5) != "rows=")
      KALDI_ERR << "Reading sparse matrix, expected 'rows=xxx', got " << str;
    std::string rows_str = str.substr(5, std::string::npos);
    std::istringstream rows_istr(rows_str);
    int32 num_rows = -1;
    rows_istr >> num_rows;
    if (num_rows < 0 || rows_istr.fail()) {
      KALDI_ERR << "Reading sparse vector, expected 'rows=[int]', got " << str;
    }
    rows_.resize(num_rows);
    for (int32 row = 0; row < num_rows; row++)
      rows_[row].Read(is, binary);
  }
}


template <typename Real>
void SparseMatrix<Real>::AddToMat(BaseFloat alpha,
                                  MatrixBase<Real> *other,
                                  MatrixTransposeType trans) const {
  if (trans == kNoTrans) {
    MatrixIndexT num_rows = rows_.size();
    KALDI_ASSERT(other->NumRows() == num_rows);
    for (MatrixIndexT i = 0; i < num_rows; i++) {
      SubVector<Real> vec(*other, i);
      rows_[i].AddToVec(alpha, &vec);
    }
  } else {
    Real *other_col_data = other->Data();
    MatrixIndexT other_stride = other->Stride(),
        num_rows = NumRows(), num_cols = NumCols();
    KALDI_ASSERT(num_rows == other->NumCols() && num_cols == other->NumRows());
    for (MatrixIndexT row = 0; row < num_rows; row++, other_col_data++) {
      const SparseVector<Real> &svec = rows_[row];
      MatrixIndexT num_elems = svec.NumElements();
      const std::pair<MatrixIndexT, Real> *sdata = svec.Data();
      for (MatrixIndexT e = 0; e < num_elems; e++)
        other_col_data[sdata[e].first * other_stride] +=
            alpha * sdata[e].second;
    }
  }
}

template <typename Real>
Real VecSvec(const VectorBase<Real> &vec,
             const SparseVector<Real> &svec) {
  KALDI_ASSERT(vec.Dim() == svec.Dim());
  MatrixIndexT n = svec.NumElements();
  const std::pair<MatrixIndexT, Real> *sdata = svec.Data();
  const Real *data = vec.Data();
  Real ans = 0.0;
  for (MatrixIndexT i = 0; i < n; i++)
    ans += data[sdata[i].first] * sdata[i].second;
  return ans;
}

template
float VecSvec(const VectorBase<float> &vec,
              const SparseVector<float> &svec);
template
double VecSvec(const VectorBase<double> &vec,
              const SparseVector<double> &svec);

template <typename Real>
const SparseVector<Real> &SparseMatrix<Real>::Row(MatrixIndexT r) const {
  KALDI_ASSERT(static_cast<size_t>(r) < rows_.size());
  return rows_[r];
}

template <typename Real>
void SparseMatrix<Real>::SetRow(int32 r, const SparseVector<Real> &vec) {
  KALDI_ASSERT(static_cast<size_t>(r) < rows_.size() &&
               vec.Dim() == rows_[0].Dim());
  rows_[r] = vec;
}


template<typename Real>
void SparseMatrix<Real>::SelectRows(const std::vector<int32> &row_indexes,
                                    const SparseMatrix<Real> &smat_other) {
  Resize(row_indexes.size(), smat_other.NumCols());
  for (int i = 0; i < row_indexes.size(); ++i) {
    SetRow(i, smat_other.Row(row_indexes[i]));
  }
}

template<typename Real>
SparseMatrix<Real>::SparseMatrix(const std::vector<int32> &indexes, int32 dim,
                                 MatrixTransposeType trans) {
  const std::vector<int32>& idx = indexes;
  std::vector<std::vector<std::pair<MatrixIndexT, Real> > > pair(idx.size());
  for (int i = 0; i < idx.size(); ++i) {
    if (idx[i] >= 0) {
      pair[i].push_back( { idx[i], Real(1) });
    }
  }
  SparseMatrix<Real> smat_cpu(dim, pair);
  if (trans == kNoTrans) {
    this->Swap(&smat_cpu);
  } else {
    SparseMatrix<Real> tmp(smat_cpu, kTrans);
    this->Swap(&tmp);
  }
}

template<typename Real>
SparseMatrix<Real>::SparseMatrix(const std::vector<int32> &indexes,
                                 const VectorBase<Real> &weights, int32 dim,
                                 MatrixTransposeType trans) {
  const std::vector<int32>& idx = indexes;
  const VectorBase<Real>& w = weights;
  std::vector<std::vector<std::pair<MatrixIndexT, Real> > > pair(idx.size());
  for (int i = 0; i < idx.size(); ++i) {
    if (idx[i] >= 0) {
      pair[i].push_back( { idx[i], w(i) });
    }
  }
  SparseMatrix<Real> smat_cpu(dim, pair);
  if (trans == kNoTrans) {
    this->Swap(&smat_cpu);
  } else {
    SparseMatrix<Real> tmp(smat_cpu, kTrans);
    this->Swap(&tmp);
  }
}

template <typename Real>
SparseMatrix<Real>& SparseMatrix<Real>::operator = (
    const SparseMatrix<Real> &other) {
  rows_ = other.rows_;
  return *this;
}

template <typename Real>
void SparseMatrix<Real>::Swap(SparseMatrix<Real> *other) {
  rows_.swap(other->rows_);
}

template<typename Real>
SparseMatrix<Real>::SparseMatrix(
    MatrixIndexT dim,
    const std::vector<std::vector<std::pair<MatrixIndexT, Real> > > &pairs):
    rows_(pairs.size()) {
  MatrixIndexT num_rows = pairs.size();
  for (MatrixIndexT row = 0; row < num_rows; row++) {
    SparseVector<Real> svec(dim, pairs[row]);
    rows_[row].Swap(&svec);
  }
}

template <typename Real>
void SparseMatrix<Real>::SetRandn(BaseFloat zero_prob) {
  MatrixIndexT num_rows = rows_.size();
  for (MatrixIndexT row = 0; row < num_rows; row++)
    rows_[row].SetRandn(zero_prob);
}

template <typename Real>
void SparseMatrix<Real>::Resize(MatrixIndexT num_rows,
                                MatrixIndexT num_cols,
                                MatrixResizeType resize_type) {
  KALDI_ASSERT(num_rows >= 0 && num_cols >= 0);
  if (resize_type == kSetZero || resize_type == kUndefined) {
    rows_.clear();
    Resize(num_rows, num_cols, kCopyData);
  } else {
    // Assume resize_type == kCopyData from here.
    int32 old_num_rows = rows_.size(), old_num_cols = NumCols();
    SparseVector<Real> initializer(num_cols);
    rows_.resize(num_rows, initializer);
    if (num_cols != old_num_cols)
      for (int32 row = 0; row < old_num_rows; row++)
        rows_[row].Resize(num_cols, kCopyData);
  }
}

template <typename Real>
void SparseMatrix<Real>::AppendSparseMatrixRows(
    std::vector<SparseMatrix<Real> > *inputs) {
  rows_.clear();
  size_t num_rows = 0;
  typename std::vector<SparseMatrix<Real> >::iterator
      input_iter = inputs->begin(),
      input_end = inputs->end();
  for (; input_iter != input_end; ++input_iter)
    num_rows += input_iter->rows_.size();
  rows_.resize(num_rows);
  typename std::vector<SparseVector<Real> >::iterator
      row_iter = rows_.begin(),
      row_end = rows_.end();
  for (input_iter = inputs->begin(); input_iter != input_end; ++input_iter) {
    typename std::vector<SparseVector<Real> >::iterator
        input_row_iter = input_iter->rows_.begin(),
        input_row_end = input_iter->rows_.end();
    for (; input_row_iter != input_row_end; ++input_row_iter, ++row_iter)
      row_iter->Swap(&(*input_row_iter));
  }
  KALDI_ASSERT(row_iter == row_end);
  int32 num_cols = NumCols();
  for (row_iter = rows_.begin(); row_iter != row_end; ++row_iter) {
    if (row_iter->Dim() != num_cols)
      KALDI_ERR << "Appending rows with inconsistent dimensions, "
                << row_iter->Dim() << " vs. " << num_cols;
  }
  inputs->clear();
}

template<typename Real>
void SparseMatrix<Real>::Scale(Real alpha) {
  MatrixIndexT num_rows = rows_.size();
  for (MatrixIndexT row = 0; row < num_rows; row++)
    rows_[row].Scale(alpha);
}

template<typename Real>
SparseMatrix<Real>::SparseMatrix(const MatrixBase<Real> &mat) {
  MatrixIndexT num_rows = mat.NumRows();
  rows_.resize(num_rows);
  for (int32 row = 0; row < num_rows; row++) {
    SparseVector<Real> this_row(mat.Row(row));
    rows_[row].Swap(&this_row);
  }
}

template<typename Real>
Real TraceMatSmat(const MatrixBase<Real> &A,
                  const SparseMatrix<Real> &B,
                  MatrixTransposeType trans) {
  Real sum = 0.0;
  if (trans == kTrans) {
    MatrixIndexT num_rows = A.NumRows();
    KALDI_ASSERT(B.NumRows() == num_rows);
    for (MatrixIndexT r = 0; r < num_rows; r++)
      sum += VecSvec(A.Row(r), B.Row(r));
  } else {
    const Real *A_col_data = A.Data();
    MatrixIndexT Astride = A.Stride(), Acols = A.NumCols(), Arows = A.NumRows();
    KALDI_ASSERT(Arows == B.NumCols() && Acols == B.NumRows());
    sum = 0.0;
    for (MatrixIndexT i = 0; i < Acols; i++, A_col_data++) {
      Real col_sum = 0.0;
      const SparseVector<Real> &svec = B.Row(i);
      MatrixIndexT num_elems = svec.NumElements();
      const std::pair<MatrixIndexT, Real> *sdata = svec.Data();
      for (MatrixIndexT e = 0; e < num_elems; e++)
        col_sum += A_col_data[Astride * sdata[e].first] * sdata[e].second;
      sum += col_sum;
    }
  }
  return sum;
}

template
float TraceMatSmat(const MatrixBase<float> &A,
                   const SparseMatrix<float> &B,
                   MatrixTransposeType trans);
template
double TraceMatSmat(const MatrixBase<double> &A,
                   const SparseMatrix<double> &B,
                   MatrixTransposeType trans);

void GeneralMatrix::Clear() {
  mat_.Resize(0, 0);
  cmat_.Clear();
  smat_.Resize(0, 0);
}

GeneralMatrix& GeneralMatrix::operator= (const MatrixBase<BaseFloat> &mat) {
  Clear();
  mat_ = mat;
  return *this;
}

GeneralMatrix& GeneralMatrix::operator= (const CompressedMatrix &cmat) {
  Clear();
  cmat_ = cmat;
  return *this;
}

GeneralMatrix& GeneralMatrix::operator= (const SparseMatrix<BaseFloat> &smat) {
  Clear();
  smat_ = smat;
  return *this;
}

GeneralMatrix& GeneralMatrix::operator= (const GeneralMatrix &gmat) {
  mat_ = gmat.mat_;
  smat_ = gmat.smat_;
  cmat_ = gmat.cmat_;
  return *this;
}


GeneralMatrixType GeneralMatrix::Type() const {
  if (smat_.NumRows() != 0)
    return kSparseMatrix;
  else if (cmat_.NumRows() != 0)
    return kCompressedMatrix;
  else
    return kFullMatrix;
}

MatrixIndexT GeneralMatrix::NumRows() const {
  MatrixIndexT r = smat_.NumRows();
  if (r != 0)
    return r;
  r = cmat_.NumRows();
  if (r != 0)
    return r;
  return mat_.NumRows();
}

MatrixIndexT GeneralMatrix::NumCols() const {
  MatrixIndexT r = smat_.NumCols();
  if (r != 0)
    return r;
  r = cmat_.NumCols();
  if (r != 0)
    return r;
  return mat_.NumCols();
}


void GeneralMatrix::Compress() {
  if (mat_.NumRows() != 0) {
    cmat_.CopyFromMat(mat_);
    mat_.Resize(0, 0);
  }
}

void GeneralMatrix::Uncompress() {
  if (cmat_.NumRows() != 0) {
    cmat_.CopyToMat(&mat_);
    cmat_.Clear();
  }
}

void GeneralMatrix::GetMatrix(Matrix<BaseFloat> *mat) const {
  if (mat_.NumRows() !=0) {
    *mat = mat_;
  } else if (cmat_.NumRows() != 0) {
    mat->Resize(cmat_.NumRows(), cmat_.NumCols(), kUndefined);
    cmat_.CopyToMat(mat);
  } else if (smat_.NumRows() != 0) {
    mat->Resize(smat_.NumRows(), smat_.NumCols(), kUndefined);
    smat_.CopyToMat(mat);
  } else {
    mat->Resize(0, 0);
  }
}

void GeneralMatrix::CopyToMat(MatrixBase<BaseFloat> *mat,
                              MatrixTransposeType trans) const {
  if (mat_.NumRows() !=0) {
    mat->CopyFromMat(mat_, trans);
  } else if (cmat_.NumRows() != 0) {
    cmat_.CopyToMat(mat, trans);
  } else if (smat_.NumRows() != 0) {
    smat_.CopyToMat(mat, trans);
  } else {
    KALDI_ASSERT(mat->NumRows() == 0);
  }
}

void GeneralMatrix::Scale(BaseFloat alpha) {
  if (mat_.NumRows() != 0) {
    mat_.Scale(alpha);
  } else if (cmat_.NumRows() != 0) {
    cmat_.Scale(alpha);
  } else if (smat_.NumRows() != 0) {
    smat_.Scale(alpha);
  }

}
const SparseMatrix<BaseFloat>& GeneralMatrix::GetSparseMatrix() const {
  if (mat_.NumRows() != 0 || cmat_.NumRows() != 0)
    KALDI_ERR << "GetSparseMatrix called on GeneralMatrix of wrong type.";
  return smat_;
}

void GeneralMatrix::SwapSparseMatrix(SparseMatrix<BaseFloat> *smat) {
  if (mat_.NumRows() != 0 || cmat_.NumRows() != 0)
    KALDI_ERR << "GetSparseMatrix called on GeneralMatrix of wrong type.";
  smat->Swap(&smat_);
}

void GeneralMatrix::SwapCompressedMatrix(CompressedMatrix *cmat) {
  if (mat_.NumRows() != 0 || smat_.NumRows() != 0)
    KALDI_ERR << "GetSparseMatrix called on GeneralMatrix of wrong type.";
  cmat->Swap(&cmat_);
}

const CompressedMatrix &GeneralMatrix::GetCompressedMatrix() const {
  if (mat_.NumRows() != 0 || smat_.NumRows() != 0)
    KALDI_ERR << "GetCompressedMatrix called on GeneralMatrix of wrong type.";
  return cmat_;
}

const Matrix<BaseFloat> &GeneralMatrix::GetFullMatrix() const {
  if (smat_.NumRows() != 0 || cmat_.NumRows() != 0)
    KALDI_ERR << "GetFullMatrix called on GeneralMatrix of wrong type.";
  return mat_;
}


void GeneralMatrix::SwapFullMatrix(Matrix<BaseFloat> *mat) {
  if (cmat_.NumRows() != 0 || smat_.NumRows() != 0)
    KALDI_ERR << "SwapMatrix called on GeneralMatrix of wrong type.";
  mat->Swap(&mat_);
}

void GeneralMatrix::Write(std::ostream &os, bool binary) const {
  if (smat_.NumRows() != 0) {
    smat_.Write(os, binary);
  } else if (cmat_.NumRows() != 0) {
    cmat_.Write(os, binary);
  } else {
    mat_.Write(os, binary);
  }
}

void GeneralMatrix::Read(std::istream &is, bool binary) {
  Clear();
  if (binary) {
    int peekval = is.peek();
    if (peekval == 'C') {
      // Token CM for compressed matrix
      cmat_.Read(is, binary);
    } else if (peekval == 'S') {
      // Token SM for sparse matrix
      smat_.Read(is, binary);
    } else {
      mat_.Read(is, binary);
    }
  } else {
    // note: in text mode we will only ever read regular
    // or sparse matrices, because the compressed-matrix format just
    // gets written as a regular matrix in text mode.
    is >> std::ws;  // Eat up white space.
    int peekval = is.peek();
    if (peekval == 'r') {  // sparse format starts rows=[int].
      smat_.Read(is, binary);
    } else {
      mat_.Read(is, binary);
    }
  }
}


void AppendGeneralMatrixRows(const std::vector<const GeneralMatrix *> &src,
                             GeneralMatrix *mat) {
  mat->Clear();
  int32 size = src.size();
  if (size == 0)
    return;
  bool all_sparse = true;
  for (int32 i = 0; i < size; i++) {
    if (src[i]->Type() != kSparseMatrix && src[i]->NumRows() != 0) {
      all_sparse = false;
      break;
    }
  }
  if (all_sparse) {
    std::vector<SparseMatrix<BaseFloat> > sparse_mats(size);
    for (int32 i = 0; i < size; i++)
      sparse_mats[i] = src[i]->GetSparseMatrix();
    SparseMatrix<BaseFloat> appended_mat;
    appended_mat.AppendSparseMatrixRows(&sparse_mats);
    mat->SwapSparseMatrix(&appended_mat);
  } else {
    int32 tot_rows = 0, num_cols = -1;
    for (int32 i = 0; i < size; i++) {
      const GeneralMatrix &src_mat = *(src[i]);
      int32 src_rows = src_mat.NumRows(), src_cols = src_mat.NumCols();
      if (src_rows != 0) {
        tot_rows += src_rows;
        if (num_cols == -1) num_cols = src_cols;
        else if (num_cols != src_cols)
          KALDI_ERR << "Appending rows of matrices with inconsistent num-cols: "
                    << num_cols << " vs. " << src_cols;
      }
    }
    Matrix<BaseFloat> appended_mat(tot_rows, num_cols, kUndefined);
    int32 row_offset = 0;
    for (int32 i = 0; i < size; i++) {
      const GeneralMatrix &src_mat = *(src[i]);
      int32 src_rows = src_mat.NumRows();
      if (src_rows != 0) {
        SubMatrix<BaseFloat> dest_submat(appended_mat, row_offset, src_rows,
                                         0, num_cols);
        src_mat.CopyToMat(&dest_submat);
        row_offset += src_rows;
      }
    }
    KALDI_ASSERT(row_offset == tot_rows);
    mat->SwapFullMatrix(&appended_mat);
  }
}

void FilterCompressedMatrixRows(const CompressedMatrix &in,
                                const std::vector<bool> &keep_rows,
                                Matrix<BaseFloat> *out) {
  KALDI_ASSERT(keep_rows.size() == static_cast<size_t>(in.NumRows()));
  int32 num_kept_rows = 0;
  std::vector<bool>::const_iterator iter = keep_rows.begin(),
                                     end = keep_rows.end();
  for (; iter != end; ++iter)
    if (*iter)
      num_kept_rows++;
  if (num_kept_rows == 0)
    KALDI_ERR << "No kept rows";
  if (num_kept_rows == static_cast<int32>(keep_rows.size())) {
    out->Resize(in.NumRows(), in.NumCols(), kUndefined);
    in.CopyToMat(out);
    return;
  }
  const BaseFloat heuristic = 0.33;
  // should be > 0 and < 1.0.  represents the performance hit we get from
  // iterating row-wise versus column-wise in compressed-matrix uncompression.

  if (num_kept_rows > heuristic * in.NumRows()) {
    // if quite a few of the the rows are kept, it may be more efficient
    // to uncompress the entire compressed matrix, since per-column operation
    // is more efficient.
    Matrix<BaseFloat> full_mat(in);
    FilterMatrixRows(full_mat, keep_rows, out);
  } else {
    out->Resize(num_kept_rows, in.NumCols(), kUndefined);

    iter = keep_rows.begin();
    int32 out_row = 0;
    for (int32 in_row = 0; iter != end; ++iter, ++in_row) {
      if (*iter) {
        SubVector<BaseFloat> dest(*out, out_row);
        in.CopyRowToVec(in_row, &dest);
        out_row++;
      }
    }
    KALDI_ASSERT(out_row == num_kept_rows);
  }
}

template <typename Real>
void FilterMatrixRows(const Matrix<Real> &in,
                      const std::vector<bool> &keep_rows,
                      Matrix<Real> *out) {
  KALDI_ASSERT(keep_rows.size() == static_cast<size_t>(in.NumRows()));
  int32 num_kept_rows = 0;
  std::vector<bool>::const_iterator iter = keep_rows.begin(),
                                     end = keep_rows.end();
  for (; iter != end; ++iter)
    if (*iter)
      num_kept_rows++;
  if (num_kept_rows == 0)
    KALDI_ERR << "No kept rows";
  if (num_kept_rows == static_cast<int32>(keep_rows.size())) {
    *out = in;
    return;
  }
  out->Resize(num_kept_rows, in.NumCols(), kUndefined);
  iter = keep_rows.begin();
  int32 out_row = 0;
  for (int32 in_row = 0; iter != end; ++iter, ++in_row) {
    if (*iter) {
      SubVector<Real> src(in, in_row);
      SubVector<Real> dest(*out, out_row);
      dest.CopyFromVec(src);
      out_row++;
    }
  }
  KALDI_ASSERT(out_row == num_kept_rows);
}

template
void FilterMatrixRows(const Matrix<float> &in,
                      const std::vector<bool> &keep_rows,
                      Matrix<float> *out);
template
void FilterMatrixRows(const Matrix<double> &in,
                      const std::vector<bool> &keep_rows,
                      Matrix<double> *out);

template <typename Real>
void FilterSparseMatrixRows(const SparseMatrix<Real> &in,
                            const std::vector<bool> &keep_rows,
                            SparseMatrix<Real> *out) {
  KALDI_ASSERT(keep_rows.size() == static_cast<size_t>(in.NumRows()));
  int32 num_kept_rows = 0;
  std::vector<bool>::const_iterator iter = keep_rows.begin(),
                                     end = keep_rows.end();
  for (; iter != end; ++iter)
    if (*iter)
      num_kept_rows++;
  if (num_kept_rows == 0)
    KALDI_ERR << "No kept rows";
  if (num_kept_rows == static_cast<int32>(keep_rows.size())) {
    *out = in;
    return;
  }
  out->Resize(num_kept_rows, in.NumCols(), kUndefined);
  iter = keep_rows.begin();
  int32 out_row = 0;
  for (int32 in_row = 0; iter != end; ++iter, ++in_row) {
    if (*iter) {
      out->SetRow(out_row, in.Row(in_row));
      out_row++;
    }
  }
  KALDI_ASSERT(out_row == num_kept_rows);
}

template
void FilterSparseMatrixRows(const SparseMatrix<float> &in,
                            const std::vector<bool> &keep_rows,
                            SparseMatrix<float> *out);
template
void FilterSparseMatrixRows(const SparseMatrix<double> &in,
                            const std::vector<bool> &keep_rows,
                            SparseMatrix<double> *out);


void FilterGeneralMatrixRows(const GeneralMatrix &in,
                             const std::vector<bool> &keep_rows,
                             GeneralMatrix *out) {
  out->Clear();
  KALDI_ASSERT(keep_rows.size() == static_cast<size_t>(in.NumRows()));
  int32 num_kept_rows = 0;
  std::vector<bool>::const_iterator iter = keep_rows.begin(),
                                     end = keep_rows.end();
  for (; iter != end; ++iter)
    if (*iter)
      num_kept_rows++;
  if (num_kept_rows == 0)
    KALDI_ERR << "No kept rows";
  if (num_kept_rows == static_cast<int32>(keep_rows.size())) {
    *out = in;
    return;
  }
  switch (in.Type()) {
    case kCompressedMatrix: {
      const CompressedMatrix &cmat = in.GetCompressedMatrix();
      Matrix<BaseFloat> full_mat;
      FilterCompressedMatrixRows(cmat, keep_rows, &full_mat);
      out->SwapFullMatrix(&full_mat);
      return;
    }
    case kSparseMatrix: {
      const SparseMatrix<BaseFloat> &smat = in.GetSparseMatrix();
      SparseMatrix<BaseFloat> smat_out;
      FilterSparseMatrixRows(smat, keep_rows, &smat_out);
      out->SwapSparseMatrix(&smat_out);
      return;
    }
    case kFullMatrix: {
      const Matrix<BaseFloat> &full_mat = in.GetFullMatrix();
      Matrix<BaseFloat> full_mat_out;
      FilterMatrixRows(full_mat, keep_rows, &full_mat_out);
      out->SwapFullMatrix(&full_mat_out);
      return;
    }
    default:
      KALDI_ERR << "Invalid general-matrix type.";
  }
}

void GeneralMatrix::AddToMat(BaseFloat alpha, MatrixBase<BaseFloat> *mat,
                             MatrixTransposeType trans) const {
  switch (this->Type()) {
    case kFullMatrix: {
      mat->AddMat(alpha, mat_, trans);
      break;
    }
    case kSparseMatrix: {
      smat_.AddToMat(alpha, mat, trans);
      break;
    }
    case kCompressedMatrix: {
      Matrix<BaseFloat> temp_mat(cmat_);
      mat->AddMat(alpha, temp_mat, trans);
      break;
    }
    default:
      KALDI_ERR << "Invalid general-matrix type.";
  }
}

template <class Real>
Real SparseVector<Real>::Max(int32 *index_out) const {
  KALDI_ASSERT(dim_ > 0 && pairs_.size() <= static_cast<size_t>(dim_));
  Real ans = -std::numeric_limits<Real>::infinity();
  int32 index = 0;
  typename std::vector<std::pair<MatrixIndexT, Real> >::const_iterator
      iter = pairs_.begin(), end = pairs_.end();
  for (; iter != end; ++iter) {
    if (iter->second > ans) {
      ans = iter->second;
      index = iter->first;
    }
  }
  if (ans >= 0 || pairs_.size() == dim_) {
    // ans >= 0 will be the normal case.
    // if pairs_.size() == dim_ then we need to return
    // even a negative answer as there are no spaces (hence no unlisted zeros).
    *index_out = index;
    return ans;
  }
  // all the stored elements are < 0, but there are unlisted
  // elements -> pick the first unlisted element.
  // Note that this class requires that the indexes are sorted
  // and unique.
  index = 0;  // "index" will always be the next index, that
              // we haven't seen listed yet.
  iter = pairs_.begin();
  for (; iter != end; ++iter) {
    if (iter->first > index) {  // index "index" is not listed.
      *index_out = index;
      return 0.0;
    } else {
      // index is the next potential gap in the indexes.
      index = iter->first + 1;
    }
  }
  // we can reach here if either pairs_.empty(), or
  // pairs_ is nonempty but contains a sequence (0, 1, 2,...).
  if (!pairs_.empty())
    index = pairs_.back().first + 1;
  // else leave index at zero
  KALDI_ASSERT(index < dim_);
  *index_out = index;
  return 0.0;
}

template <typename Real>
SparseVector<Real>::SparseVector(const VectorBase<Real> &vec) {
  MatrixIndexT dim = vec.Dim();
  dim_ = dim;
  if (dim == 0)
    return;
  const Real *ptr = vec.Data();
  for (MatrixIndexT i = 0; i < dim; i++) {
    Real val = ptr[i];
    if (val != 0.0)
      pairs_.push_back(std::pair<MatrixIndexT,Real>(i,val));
  }
}

void GeneralMatrix::Swap(GeneralMatrix *other) {
  mat_.Swap(&(other->mat_));
  cmat_.Swap(&(other->cmat_));
  smat_.Swap(&(other->smat_));
}


void ExtractRowRangeWithPadding(
    const GeneralMatrix &in,
    int32 row_offset,
    int32 num_rows,
    GeneralMatrix *out) {
  // make sure 'out' is empty to start with.
  Matrix<BaseFloat> empty_mat;
  *out = empty_mat;
  if (num_rows == 0) return;
  switch (in.Type()) {
    case kFullMatrix: {
      const Matrix<BaseFloat> &mat_in = in.GetFullMatrix();
      int32 num_rows_in = mat_in.NumRows(), num_cols = mat_in.NumCols();
      KALDI_ASSERT(num_rows_in > 0);  // we can't extract >0 rows from an empty
                                      // matrix.
      Matrix<BaseFloat> mat_out(num_rows, num_cols, kUndefined);
      for (int32 row = 0; row < num_rows; row++) {
        int32 row_in = row + row_offset;
        if (row_in < 0) row_in = 0;
        else if (row_in >= num_rows_in) row_in = num_rows_in - 1;
        SubVector<BaseFloat> vec_in(mat_in, row_in),
            vec_out(mat_out, row);
        vec_out.CopyFromVec(vec_in);
      }
      out->SwapFullMatrix(&mat_out);
      break;
    }
    case kSparseMatrix: {
      const SparseMatrix<BaseFloat> &smat_in = in.GetSparseMatrix();
      int32 num_rows_in = smat_in.NumRows(),
          num_cols = smat_in.NumCols();
      KALDI_ASSERT(num_rows_in > 0);  // we can't extract >0 rows from an empty
                                      // matrix.
      SparseMatrix<BaseFloat> smat_out(num_rows, num_cols);
      for (int32 row = 0; row < num_rows; row++) {
        int32 row_in = row + row_offset;
        if (row_in < 0) row_in = 0;
        else if (row_in >= num_rows_in) row_in = num_rows_in - 1;
        smat_out.SetRow(row, smat_in.Row(row_in));
      }
      out->SwapSparseMatrix(&smat_out);
      break;
    }
    case kCompressedMatrix: {
      const CompressedMatrix &cmat_in = in.GetCompressedMatrix();
      bool allow_padding = true;
      CompressedMatrix cmat_out(cmat_in, row_offset, num_rows,
                                0, cmat_in.NumCols(), allow_padding);
      out->SwapCompressedMatrix(&cmat_out);
      break;
    }
    default:
      KALDI_ERR << "Bad matrix type.";
  }
}



template class SparseVector<float>;
template class SparseVector<double>;
template class SparseMatrix<float>;
template class SparseMatrix<double>;

}  // namespace kaldi
