// matrix/compressed-matrix.cc

// Copyright 2012    Johns Hopkins University (author: Daniel Povey)
//                   Frantisek Skala, Wei Shi
//           2015    Tom Ko

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

#include "matrix/compressed-matrix.h"
#include <algorithm>

namespace kaldi {

//static
MatrixIndexT CompressedMatrix::DataSize(const GlobalHeader &header) {
  // Returns size in bytes of the data.
  DataFormat format = static_cast<DataFormat>(header.format);
  if (format == kOneByteWithColHeaders) {
    return sizeof(GlobalHeader) +
        header.num_cols * (sizeof(PerColHeader) + header.num_rows);
  } else if (format == kTwoByte) {
    return sizeof(GlobalHeader) +
        2 * header.num_rows * header.num_cols;
  } else {
    KALDI_ASSERT(format == kOneByte);
    return sizeof(GlobalHeader) +
        header.num_rows * header.num_cols;
  }
}

// scale all element of matrix by scaling floats
// in GlobalHeader with alpha.
void CompressedMatrix::Scale(float alpha) {
  if (data_ != NULL) {
    GlobalHeader *h = reinterpret_cast<GlobalHeader*>(data_);
    // scale the floating point values in each PerColHolder
    // and leave all integers the same.
    h->min_value *= alpha;
    h->range *= alpha;
  }
}

template<typename Real>  // static inline
void CompressedMatrix::ComputeGlobalHeader(
    const MatrixBase<Real> &mat, CompressionMethod method,
    GlobalHeader *header) {
  if (method == kAutomaticMethod) {
    if (mat.NumRows() > 8) method = kSpeechFeature;
    else method = kTwoByteAuto;
  }

  switch (method) {
    case kSpeechFeature:
      header->format = static_cast<int32>(kOneByteWithColHeaders);  // 1.
      break;
    case kTwoByteAuto: case kTwoByteSignedInteger:
      header->format = static_cast<int32>(kTwoByte);  // 2.
      break;
    case kOneByteAuto: case kOneByteUnsignedInteger: case kOneByteZeroOne:
      header->format = static_cast<int32>(kOneByte);  // 3.
      break;
    default:
      KALDI_ERR << "Invalid compression type: "
                << static_cast<int32>(method);
  }

  header->num_rows = mat.NumRows();
  header->num_cols = mat.NumCols();

  // Now compute 'min_value' and 'range'.
  switch (method) {
    case kSpeechFeature: case kTwoByteAuto: case kOneByteAuto: {
      float min_value = mat.Min(), max_value = mat.Max();
      // ensure that max_value is strictly greater than min_value, even if matrix is
      // constant; this avoids crashes in ComputeColHeader when compressing speech
      // featupres.
      if (max_value == min_value)
        max_value = min_value + (1.0 + fabs(min_value));
      KALDI_ASSERT(min_value - min_value == 0 &&
                   max_value - max_value == 0 &&
                   "Cannot compress a matrix with Nan's or Inf's");

      header->min_value = min_value;
      header->range = max_value - min_value;

      // we previously checked that max_value != min_value, so their
      // difference should be nonzero.
      KALDI_ASSERT(header->range > 0.0);
      break;
    }
    case kTwoByteSignedInteger: {
      header->min_value = -32768.0;
      header->range = 65535.0;
      break;
    }
    case kOneByteUnsignedInteger: {
      header->min_value = 0.0;
      header->range = 255.0;
      break;
    }
    case kOneByteZeroOne: {
      header->min_value = 0.0;
      header->range = 1.0;
      break;
    }
    default:
      KALDI_ERR << "Unknown compression method = "
                << static_cast<int32>(method);
  }
  KALDI_COMPILE_TIME_ASSERT(sizeof(*header) == 20);  // otherwise
  // something weird is happening and our code probably won't work or
  // won't be robust across platforms.
}

template<typename Real>
void CompressedMatrix::CopyFromMat(
    const MatrixBase<Real> &mat, CompressionMethod method) {
  if (data_ != NULL) {
    delete [] static_cast<float*>(data_);  // call delete [] because was allocated with new float[]
    data_ = NULL;
  }
  if (mat.NumRows() == 0) { return; }  // Zero-size matrix stored as zero pointer.


  GlobalHeader global_header;
  ComputeGlobalHeader(mat, method, &global_header);

  int32 data_size = DataSize(global_header);

  data_ = AllocateData(data_size);

  *(reinterpret_cast<GlobalHeader*>(data_)) = global_header;

  DataFormat format = static_cast<DataFormat>(global_header.format);
  if (format == kOneByteWithColHeaders) {
    PerColHeader *header_data =
        reinterpret_cast<PerColHeader*>(static_cast<char*>(data_) +
                                        sizeof(GlobalHeader));
    uint8 *byte_data =
        reinterpret_cast<uint8*>(header_data + global_header.num_cols);

    const Real *matrix_data = mat.Data();

    for (int32 col = 0; col < global_header.num_cols; col++) {
      CompressColumn(global_header,
                     matrix_data + col, mat.Stride(),
                     global_header.num_rows,
                     header_data, byte_data);
      header_data++;
      byte_data += global_header.num_rows;
    }
  } else if (format == kTwoByte) {
    uint16 *data = reinterpret_cast<uint16*>(static_cast<char*>(data_) +
                                             sizeof(GlobalHeader));
    int32 num_rows = mat.NumRows(), num_cols = mat.NumCols();
    for (int32 r = 0; r < num_rows; r++) {
      const Real *row_data = mat.RowData(r);
      for (int32 c = 0; c < num_cols; c++)
        data[c] = FloatToUint16(global_header, row_data[c]);
      data += num_cols;
    }
  } else {
    KALDI_ASSERT(format == kOneByte);
    uint8 *data = reinterpret_cast<uint8*>(static_cast<char*>(data_) +
                                           sizeof(GlobalHeader));
    int32 num_rows = mat.NumRows(), num_cols = mat.NumCols();
    for (int32 r = 0; r < num_rows; r++) {
      const Real *row_data = mat.RowData(r);
      for (int32 c = 0; c < num_cols; c++)
        data[c] = FloatToUint8(global_header, row_data[c]);
      data += num_cols;
    }
  }
}

// Instantiate the template for float and double.
template
void CompressedMatrix::CopyFromMat(const MatrixBase<float> &mat,
                                   CompressionMethod method);

template
void CompressedMatrix::CopyFromMat(const MatrixBase<double> &mat,
                                   CompressionMethod method);


CompressedMatrix::CompressedMatrix(
    const CompressedMatrix &cmat,
    const MatrixIndexT row_offset,
    const MatrixIndexT num_rows,
    const MatrixIndexT col_offset,
    const MatrixIndexT num_cols,
    bool allow_padding): data_(NULL) {
  int32 old_num_rows = cmat.NumRows(), old_num_cols = cmat.NumCols();

  if (old_num_rows == 0) {
    KALDI_ASSERT(num_rows == 0 && num_cols == 0);
    // The empty matrix is stored as a zero pointer.
    return;
  }

  KALDI_ASSERT(row_offset < old_num_rows);
  KALDI_ASSERT(col_offset < old_num_cols);
  KALDI_ASSERT(row_offset >= 0 || allow_padding);
  KALDI_ASSERT(col_offset >= 0);
  KALDI_ASSERT(row_offset + num_rows <= old_num_rows || allow_padding);
  KALDI_ASSERT(col_offset + num_cols <= old_num_cols);

  if (num_rows == 0 || num_cols == 0) { return; }

  bool padding_is_used = (row_offset < 0 ||
                          row_offset + num_rows > old_num_rows);

  GlobalHeader new_global_header;
  KALDI_COMPILE_TIME_ASSERT(sizeof(new_global_header) == 20);

  GlobalHeader *old_global_header = reinterpret_cast<GlobalHeader*>(cmat.Data());

  new_global_header = *old_global_header;
  new_global_header.num_cols = num_cols;
  new_global_header.num_rows = num_rows;

  // We don't switch format from 1 -> 2 (in case of size reduction) yet; if this
  // is needed, we will do this below by creating a temporary Matrix.
  new_global_header.format = old_global_header->format;

  data_ = AllocateData(DataSize(new_global_header));  // allocate memory
  *(reinterpret_cast<GlobalHeader*>(data_)) = new_global_header;


  DataFormat format = static_cast<DataFormat>(old_global_header->format);
  if (format == kOneByteWithColHeaders) {
    PerColHeader *old_per_col_header =
        reinterpret_cast<PerColHeader*>(old_global_header + 1);
    uint8 *old_byte_data =
        reinterpret_cast<uint8*>(old_per_col_header +
                                 old_global_header->num_cols);
    PerColHeader *new_per_col_header =
        reinterpret_cast<PerColHeader*>(
            reinterpret_cast<GlobalHeader*>(data_) + 1);

    memcpy(new_per_col_header, old_per_col_header + col_offset,
           sizeof(PerColHeader) * num_cols);

    uint8 *new_byte_data =
        reinterpret_cast<uint8*>(new_per_col_header + num_cols);
    if (!padding_is_used) {
      uint8 *old_start_of_subcol =
          old_byte_data + row_offset + (col_offset * old_num_rows),
          *new_start_of_col = new_byte_data;
      for (int32 i = 0; i < num_cols; i++) {
        memcpy(new_start_of_col, old_start_of_subcol, num_rows);
        new_start_of_col += num_rows;
        old_start_of_subcol += old_num_rows;
      }
    } else {
      uint8 *old_start_of_col =
          old_byte_data + (col_offset * old_num_rows),
          *new_start_of_col = new_byte_data;
      for (int32 i = 0; i < num_cols; i++) {

        for (int32 j = 0; j < num_rows; j++) {
          int32 old_j = j + row_offset;
          if (old_j < 0) old_j = 0;
          else if (old_j >= old_num_rows) old_j = old_num_rows - 1;
          new_start_of_col[j] = old_start_of_col[old_j];
        }
        new_start_of_col += num_rows;
        old_start_of_col += old_num_rows;
      }
    }
  } else if (format == kTwoByte) {
    const uint16 *old_data =
        reinterpret_cast<const uint16*>(old_global_header + 1);
    uint16 *new_row_data =
        reinterpret_cast<uint16*>(reinterpret_cast<GlobalHeader*>(data_) + 1);

    for (int32 row = 0; row < num_rows; row++) {
      int32 old_row = row + row_offset;
      // The next two lines are only relevant if padding_is_used.
      if (old_row < 0) old_row = 0;
      else if (old_row >= old_num_rows) old_row = old_num_rows - 1;
      const uint16 *old_row_data =
          old_data + col_offset + (old_num_cols * old_row);
      memcpy(new_row_data, old_row_data, sizeof(uint16) * num_cols);
      new_row_data += num_cols;
    }
  } else {
    KALDI_ASSERT(format == kOneByte);
    const uint8 *old_data =
        reinterpret_cast<const uint8*>(old_global_header + 1);
    uint8 *new_row_data =
        reinterpret_cast<uint8*>(reinterpret_cast<GlobalHeader*>(data_) + 1);

    for (int32 row = 0; row < num_rows; row++) {
      int32 old_row = row + row_offset;
      // The next two lines are only relevant if padding_is_used.
      if (old_row < 0) old_row = 0;
      else if (old_row >= old_num_rows) old_row = old_num_rows - 1;
      const uint8 *old_row_data =
          old_data + col_offset + (old_num_cols * old_row);
      memcpy(new_row_data, old_row_data, sizeof(uint8) * num_cols);
      new_row_data += num_cols;
    }
  }

  if (num_rows < 8 && format == kOneByteWithColHeaders) {
    // format was 1 but we want it to be 2 -> create a temporary
    // Matrix (uncompress), re-compress, and swap.
    // This gives us almost exact reconstruction while saving
    // memory (the elements take more space but there will be
    // no per-column headers).
    Matrix<float> temp(this->NumRows(), this->NumCols(),
                       kUndefined);
    this->CopyToMat(&temp);
    CompressedMatrix temp_cmat(temp, kTwoByteAuto);
    this->Swap(&temp_cmat);
  }
}


template<typename Real>
CompressedMatrix &CompressedMatrix::operator =(const MatrixBase<Real> &mat) {
  this->CopyFromMat(mat);
  return *this;
}

// Instantiate the template for float and double.
template
CompressedMatrix& CompressedMatrix::operator =(const MatrixBase<float> &mat);

template
CompressedMatrix& CompressedMatrix::operator =(const MatrixBase<double> &mat);

inline uint16 CompressedMatrix::FloatToUint16(
    const GlobalHeader &global_header,
    float value) {
  float f = (value - global_header.min_value) /
      global_header.range;
  if (f > 1.0) f = 1.0;  // Note: this should not happen.
  if (f < 0.0) f = 0.0;  // Note: this should not happen.
  return static_cast<int>(f * 65535 + 0.499);  // + 0.499 is to
  // round to closest int; avoids bias.
}


inline uint8 CompressedMatrix::FloatToUint8(
    const GlobalHeader &global_header,
    float value) {
  float f = (value - global_header.min_value) /
      global_header.range;
  if (f > 1.0) f = 1.0;  // Note: this should not happen.
  if (f < 0.0) f = 0.0;  // Note: this should not happen.
  return static_cast<int>(f * 255 + 0.499);  // + 0.499 is to
  // round to closest int; avoids bias.
}


inline float CompressedMatrix::Uint16ToFloat(
    const GlobalHeader &global_header,
    uint16 value) {
  // the constant 1.52590218966964e-05 is 1/65535.
  return global_header.min_value
      + global_header.range * 1.52590218966964e-05F * value;
}

template<typename Real>  // static
void CompressedMatrix::ComputeColHeader(
    const GlobalHeader &global_header,
    const Real *data, MatrixIndexT stride,
    int32 num_rows, CompressedMatrix::PerColHeader *header) {
  KALDI_ASSERT(num_rows > 0);
  std::vector<Real> sdata(num_rows); // the sorted data.
  for (size_t i = 0, size = sdata.size(); i < size; i++)
    sdata[i] = data[i*stride];

  if (num_rows >= 5) {
    int quarter_nr = num_rows/4;
    // std::sort(sdata.begin(), sdata.end());
    // The elements at positions 0, quarter_nr,
    // 3*quarter_nr, and num_rows-1 need to be in sorted order.
    std::nth_element(sdata.begin(), sdata.begin() + quarter_nr, sdata.end());
    // Now, sdata.begin() + quarter_nr contains the element that would appear
    // in sorted order, in that position.
    std::nth_element(sdata.begin(), sdata.begin(), sdata.begin() + quarter_nr);
    // Now, sdata.begin() and sdata.begin() + quarter_nr contain the elements
    // that would appear at those positions in sorted order.
    std::nth_element(sdata.begin() + quarter_nr + 1,
                     sdata.begin() + (3*quarter_nr), sdata.end());
    // Now, sdata.begin(), sdata.begin() + quarter_nr, and sdata.begin() +
    // 3*quarter_nr, contain the elements that would appear at those positions
    // in sorted order.
    std::nth_element(sdata.begin() + (3*quarter_nr) + 1, sdata.end() - 1,
                     sdata.end());
    // Now, sdata.begin(), sdata.begin() + quarter_nr, and sdata.begin() +
    // 3*quarter_nr, and sdata.end() - 1, contain the elements that would appear
    // at those positions in sorted order.

    header->percentile_0 =
        std::min<uint16>(FloatToUint16(global_header, sdata[0]), 65532);
    header->percentile_25 =
        std::min<uint16>(
            std::max<uint16>(
                FloatToUint16(global_header, sdata[quarter_nr]),
                header->percentile_0 + static_cast<uint16>(1)), 65533);
    header->percentile_75 =
        std::min<uint16>(
            std::max<uint16>(
                FloatToUint16(global_header, sdata[3*quarter_nr]),
                header->percentile_25 + static_cast<uint16>(1)), 65534);
    header->percentile_100 = std::max<uint16>(
        FloatToUint16(global_header, sdata[num_rows-1]),
        header->percentile_75 + static_cast<uint16>(1));

  } else {  // handle this pathological case.
    std::sort(sdata.begin(), sdata.end());
    // Note: we know num_rows is at least 1.
    header->percentile_0 =
        std::min<uint16>(FloatToUint16(global_header, sdata[0]),
                         65532);
    if (num_rows > 1)
      header->percentile_25 =
          std::min<uint16>(
              std::max<uint16>(FloatToUint16(global_header, sdata[1]),
                               header->percentile_0 + 1), 65533);
    else
      header->percentile_25 = header->percentile_0 + 1;
    if (num_rows > 2)
      header->percentile_75 =
          std::min<uint16>(
              std::max<uint16>(FloatToUint16(global_header, sdata[2]),
                               header->percentile_25 + 1), 65534);
    else
      header->percentile_75 = header->percentile_25 + 1;
    if (num_rows > 3)
      header->percentile_100 =
          std::max<uint16>(FloatToUint16(global_header, sdata[3]),
                           header->percentile_75 + 1);
    else
      header->percentile_100 = header->percentile_75 + 1;
  }
}

// static
inline uint8 CompressedMatrix::FloatToChar(
    float p0, float p25, float p75, float p100,
    float value) {
  int ans;
  if (value < p25) {  // range [ p0, p25 ) covered by
    // characters 0 .. 64.  We round to the closest int.
    float f = (value - p0) / (p25 - p0);
    ans = static_cast<int>(f * 64 + 0.5);
    // Note: the checks on the next two lines
    // are necessary in pathological cases when all the elements in a row
    // are the same and the percentile_* values are separated by one.
    if (ans < 0) ans = 0;
    if (ans > 64) ans = 64;
  } else if (value < p75) {  // range [ p25, p75 )covered
    // by characters 64 .. 192.  We round to the closest int.
    float f = (value - p25) / (p75 - p25);
    ans = 64 + static_cast<int>(f * 128 + 0.5);
    if (ans < 64) ans = 64;
    if (ans > 192) ans = 192;
  } else {  // range [ p75, p100 ] covered by
    // characters 192 .. 255.  Note: this last range
    // has fewer characters than the left range, because
    // we go up to 255, not 256.
    float f = (value - p75) / (p100 - p75);
    ans = 192 + static_cast<int>(f * 63 + 0.5);
    if (ans < 192) ans = 192;
    if (ans > 255) ans = 255;
  }
  return static_cast<uint8>(ans);
}


// static
inline float CompressedMatrix::CharToFloat(
    float p0, float p25, float p75, float p100,
    uint8 value) {
  if (value <= 64) {
    return p0 + (p25 - p0) * value * (1/64.0);
  } else if (value <= 192) {
    return p25 + (p75 - p25) * (value - 64) * (1/128.0);
  } else {
    return p75 + (p100 - p75) * (value - 192) * (1/63.0);
  }
}


template<typename Real>  // static
void CompressedMatrix::CompressColumn(
    const GlobalHeader &global_header,
    const Real *data, MatrixIndexT stride,
    int32 num_rows, CompressedMatrix::PerColHeader *header,
    uint8 *byte_data) {
  ComputeColHeader(global_header, data, stride,
                   num_rows, header);

  float p0 = Uint16ToFloat(global_header, header->percentile_0),
      p25 = Uint16ToFloat(global_header, header->percentile_25),
      p75 = Uint16ToFloat(global_header, header->percentile_75),
      p100 = Uint16ToFloat(global_header, header->percentile_100);

  for (int32 i = 0; i < num_rows; i++) {
    Real this_data = data[i * stride];
    byte_data[i] = FloatToChar(p0, p25, p75, p100, this_data);
  }
}

// static
void* CompressedMatrix::AllocateData(int32 num_bytes) {
  KALDI_ASSERT(num_bytes > 0);
  KALDI_COMPILE_TIME_ASSERT(sizeof(float) == 4);
  // round size up to nearest number of floats.
  return reinterpret_cast<void*>(new float[(num_bytes/3) + 4]);
}

void CompressedMatrix::Write(std::ostream &os, bool binary) const {
  if (binary) {  // Binary-mode write:
    if (data_ != NULL) {
      GlobalHeader &h = *reinterpret_cast<GlobalHeader*>(data_);
      DataFormat format = static_cast<DataFormat>(h.format);
      if (format == kOneByteWithColHeaders) {
        WriteToken(os, binary, "CM");
      } else if (format == kTwoByte) {
        WriteToken(os, binary, "CM2");
      } else if (format == kOneByte) {
        WriteToken(os, binary, "CM3");
      }
      MatrixIndexT size = DataSize(h);  // total size of data in data_
      // We don't write out the "int32 format", hence the + 4, - 4.
      os.write(reinterpret_cast<const char*>(data_) + 4, size - 4);
    } else {  // special case: where data_ == NULL, we treat it as an empty
      // matrix.
      WriteToken(os, binary, "CM");
      GlobalHeader h;
      h.range = h.min_value = 0.0;
      h.num_rows = h.num_cols = 0;
      os.write(reinterpret_cast<const char*>(&h), sizeof(h));
    }
  } else {
    // In text mode, just use the same format as a regular matrix.
    // This is not compressed.
    Matrix<BaseFloat> temp_mat(this->NumRows(), this->NumCols(),
                               kUndefined);
    this->CopyToMat(&temp_mat);
    temp_mat.Write(os, binary);
  }
  if (os.fail())
    KALDI_ERR << "Error writing compressed matrix to stream.";
}

void CompressedMatrix::Read(std::istream &is, bool binary) {
  if (data_ != NULL) {
    delete [] (static_cast<float*>(data_));
    data_ = NULL;
  }
  if (binary) {
    int peekval = Peek(is, binary);
    if (peekval == 'C') {
      std::string tok; // Should be CM (format 1) or CM2 (format 2)
      ReadToken(is, binary, &tok);
      GlobalHeader h;
      if (tok == "CM") { h.format = 1; } //  kOneByteWithColHeaders
      else if (tok == "CM2") { h.format = 2; }  // kTwoByte
      else if (tok == "CM3") { h.format = 3; }  // kOneByte
      else {
        KALDI_ERR << "Unexpected token " << tok << ", expecting CM, CM2 or CM3";
      }
      // don't read the "format" -> hence + 4, - 4.
      is.read(reinterpret_cast<char*>(&h) + 4, sizeof(h) - 4);
      if (is.fail())
        KALDI_ERR << "Failed to read header";
      if (h.num_cols == 0) // empty matrix.
        return;
      int32 size = DataSize(h), remaining_size = size - sizeof(GlobalHeader);
      data_ = AllocateData(size);
      *(reinterpret_cast<GlobalHeader*>(data_)) = h;
      is.read(reinterpret_cast<char*>(data_) + sizeof(GlobalHeader),
              remaining_size);
    } else {
      // Assume that what we're reading is a regular Matrix.  This might be the
      // case if you changed your code, making a Matrix into a CompressedMatrix,
      // and you want back-compatibility for reading.
      Matrix<BaseFloat> M;
      M.Read(is, binary); // This will crash if it was not a Matrix.
      this->CopyFromMat(M);
    }
  } else {  // Text-mode read.  In this case you don't get to
    // choose the compression type.  Anyway this branch would only
    // be taken when debugging.
    Matrix<BaseFloat> temp;
    temp.Read(is, binary);
    this->CopyFromMat(temp);
  }
  if (is.fail())
    KALDI_ERR << "Failed to read data.";
}

template<typename Real>
void CompressedMatrix::CopyToMat(MatrixBase<Real> *mat,
                                 MatrixTransposeType trans) const {
  if (trans == kTrans) {
    Matrix<Real> temp(this->NumCols(), this->NumRows());
    CopyToMat(&temp, kNoTrans);
    mat->CopyFromMat(temp, kTrans);
    return;
  }

  if (data_ == NULL) {
    KALDI_ASSERT(mat->NumRows() == 0);
    KALDI_ASSERT(mat->NumCols() == 0);
    return;
  }
  GlobalHeader *h = reinterpret_cast<GlobalHeader*>(data_);
  int32 num_cols = h->num_cols, num_rows = h->num_rows;
  KALDI_ASSERT(mat->NumRows() == num_rows);
  KALDI_ASSERT(mat->NumCols() == num_cols);

  DataFormat format = static_cast<DataFormat>(h->format);
  if (format == kOneByteWithColHeaders) {
    PerColHeader *per_col_header = reinterpret_cast<PerColHeader*>(h+1);
    uint8 *byte_data = reinterpret_cast<uint8*>(per_col_header +
                                                h->num_cols);
    for (int32 i = 0; i < num_cols; i++, per_col_header++) {
      float p0 = Uint16ToFloat(*h, per_col_header->percentile_0),
          p25 = Uint16ToFloat(*h, per_col_header->percentile_25),
          p75 = Uint16ToFloat(*h, per_col_header->percentile_75),
          p100 = Uint16ToFloat(*h, per_col_header->percentile_100);
      for (int32 j = 0; j < num_rows; j++, byte_data++) {
        float f = CharToFloat(p0, p25, p75, p100, *byte_data);
        (*mat)(j, i) = f;
      }
    }
  } else if (format == kTwoByte) {
    const uint16 *data = reinterpret_cast<const uint16*>(h + 1);
    float min_value = h->min_value,
        increment = h->range * (1.0 / 65535.0);
    for (int32 i = 0; i < num_rows; i++) {
      Real *row_data = mat->RowData(i);
      for (int32 j = 0; j < num_cols; j++)
        row_data[j] = min_value + data[j] * increment;
      data += num_cols;
    }
  } else {
    KALDI_ASSERT(format == kOneByte);
    float min_value = h->min_value, increment = h->range * (1.0 / 255.0);

    const uint8 *data = reinterpret_cast<const uint8*>(h + 1);
    for (int32 i = 0; i < num_rows; i++) {
      Real *row_data = mat->RowData(i);
      for (int32 j = 0; j < num_cols; j++)
        row_data[j] = min_value + data[j] * increment;
      data += num_cols;
    }
  }
}

// Instantiate the template for float and double.
template
void CompressedMatrix::CopyToMat(MatrixBase<float> *mat,
                                 MatrixTransposeType trans) const;
template
void CompressedMatrix::CopyToMat(MatrixBase<double> *mat,
                                 MatrixTransposeType trans) const;

template<typename Real>
void CompressedMatrix::CopyRowToVec(MatrixIndexT row,
                                    VectorBase<Real> *v) const {
  KALDI_ASSERT(row < this->NumRows());
  KALDI_ASSERT(row >= 0);
  KALDI_ASSERT(v->Dim() == this->NumCols());

  GlobalHeader *h = reinterpret_cast<GlobalHeader*>(data_);
  DataFormat format = static_cast<DataFormat>(h->format);
  if (format == kOneByteWithColHeaders) {
    PerColHeader *per_col_header = reinterpret_cast<PerColHeader*>(h+1);
    uint8 *byte_data = reinterpret_cast<uint8*>(per_col_header +
                                                h->num_cols);
    byte_data += row;  // point to first value we are interested in
    for (int32 i = 0; i < h->num_cols;
         i++, per_col_header++, byte_data += h->num_rows) {
      float p0 = Uint16ToFloat(*h, per_col_header->percentile_0),
          p25 = Uint16ToFloat(*h, per_col_header->percentile_25),
          p75 = Uint16ToFloat(*h, per_col_header->percentile_75),
          p100 = Uint16ToFloat(*h, per_col_header->percentile_100);
      float f = CharToFloat(p0, p25, p75, p100, *byte_data);
      (*v)(i) = f;
    }
  } else if (format == kTwoByte) {
    int32 num_cols = h->num_cols;
    float min_value = h->min_value,
        increment = h->range * (1.0 / 65535.0);
    const uint16 *row_data = reinterpret_cast<uint16*>(h + 1) + (num_cols * row);
    Real *v_data = v->Data();
    for (int32 c = 0; c < num_cols; c++)
      v_data[c] = min_value + row_data[c] * increment;
  } else {
    KALDI_ASSERT(format == kOneByte);
    int32 num_cols = h->num_cols;
    float min_value = h->min_value,
        increment = h->range * (1.0 / 255.0);
    const uint8 *row_data = reinterpret_cast<uint8*>(h + 1) + (num_cols * row);
    Real *v_data = v->Data();
    for (int32 c = 0; c < num_cols; c++)
      v_data[c] = min_value + row_data[c] * increment;
  }
}

template<typename Real>
void CompressedMatrix::CopyColToVec(MatrixIndexT col,
                                    VectorBase<Real> *v) const {
  KALDI_ASSERT(col < this->NumCols());
  KALDI_ASSERT(col >= 0);
  KALDI_ASSERT(v->Dim() == this->NumRows());

  GlobalHeader *h = reinterpret_cast<GlobalHeader*>(data_);

  DataFormat format = static_cast<DataFormat>(h->format);
  if (format == kOneByteWithColHeaders) {
    PerColHeader *per_col_header = reinterpret_cast<PerColHeader*>(h+1);
    uint8 *byte_data = reinterpret_cast<uint8*>(per_col_header +
                                                h->num_cols);
    byte_data += col*h->num_rows;  // point to first value in the column we want
    per_col_header += col;
    float p0 = Uint16ToFloat(*h, per_col_header->percentile_0),
        p25 = Uint16ToFloat(*h, per_col_header->percentile_25),
        p75 = Uint16ToFloat(*h, per_col_header->percentile_75),
        p100 = Uint16ToFloat(*h, per_col_header->percentile_100);
    for (int32 i = 0; i < h->num_rows; i++, byte_data++) {
      float f = CharToFloat(p0, p25, p75, p100, *byte_data);
      (*v)(i) = f;
    }
  } else if (format == kTwoByte) {
    int32 num_rows = h->num_rows, num_cols = h->num_cols;
    float min_value = h->min_value,
        increment = h->range * (1.0 / 65535.0);
    const uint16 *col_data = reinterpret_cast<uint16*>(h + 1) + col;
    Real *v_data = v->Data();
    for (int32 r = 0; r < num_rows; r++)
      v_data[r] = min_value + increment * col_data[r * num_cols];
  } else {
    KALDI_ASSERT(format == kOneByte);
    int32 num_rows = h->num_rows, num_cols = h->num_cols;
    float min_value = h->min_value,
        increment = h->range * (1.0 / 255.0);
    const uint8 *col_data = reinterpret_cast<uint8*>(h + 1) + col;
    Real *v_data = v->Data();
    for (int32 r = 0; r < num_rows; r++)
      v_data[r] = min_value + increment * col_data[r * num_cols];
  }
}

// instantiate the templates.
template void
CompressedMatrix::CopyColToVec(MatrixIndexT, VectorBase<double> *) const;
template void
CompressedMatrix::CopyColToVec(MatrixIndexT, VectorBase<float> *) const;
template void
CompressedMatrix::CopyRowToVec(MatrixIndexT, VectorBase<double> *) const;
template void
CompressedMatrix::CopyRowToVec(MatrixIndexT, VectorBase<float> *) const;

template<typename Real>
void CompressedMatrix::CopyToMat(int32 row_offset,
                                 int32 col_offset,
                                 MatrixBase<Real> *dest) const {
  KALDI_PARANOID_ASSERT(row_offset < this->NumRows());
  KALDI_PARANOID_ASSERT(col_offset < this->NumCols());
  KALDI_PARANOID_ASSERT(row_offset >= 0);
  KALDI_PARANOID_ASSERT(col_offset >= 0);
  KALDI_ASSERT(row_offset+dest->NumRows() <= this->NumRows());
  KALDI_ASSERT(col_offset+dest->NumCols() <= this->NumCols());
  // everything is OK
  GlobalHeader *h = reinterpret_cast<GlobalHeader*>(data_);
  int32 num_rows = h->num_rows, num_cols = h->num_cols,
      tgt_cols = dest->NumCols(), tgt_rows = dest->NumRows();

  DataFormat format = static_cast<DataFormat>(h->format);
  if (format == kOneByteWithColHeaders) {
    PerColHeader *per_col_header = reinterpret_cast<PerColHeader*>(h+1);
    uint8 *byte_data = reinterpret_cast<uint8*>(per_col_header +
                                                h->num_cols);

    uint8 *start_of_subcol = byte_data+row_offset;  // skip appropriate
    // number of columns
    start_of_subcol += col_offset*num_rows;  // skip appropriate number of rows

    per_col_header += col_offset;  // skip the appropriate number of headers

    for (int32 i = 0;
         i < tgt_cols;
         i++, per_col_header++, start_of_subcol+=num_rows) {
      byte_data = start_of_subcol;
      float p0 = Uint16ToFloat(*h, per_col_header->percentile_0),
          p25 = Uint16ToFloat(*h, per_col_header->percentile_25),
          p75 = Uint16ToFloat(*h, per_col_header->percentile_75),
          p100 = Uint16ToFloat(*h, per_col_header->percentile_100);
      for (int32 j = 0; j < tgt_rows; j++, byte_data++) {
        float f = CharToFloat(p0, p25, p75, p100, *byte_data);
        (*dest)(j, i) = f;
      }
    }
  } else if (format == kTwoByte) {
    const uint16 *data = reinterpret_cast<const uint16*>(h+1) + col_offset +
        (num_cols * row_offset);
    float min_value = h->min_value,
        increment = h->range * (1.0 / 65535.0);

    for (int32 row = 0; row < tgt_rows; row++) {
      Real *dest_row = dest->RowData(row);
      for (int32 col = 0; col < tgt_cols; col++)
        dest_row[col] = min_value + increment * data[col];
      data += num_cols;
    }
  } else {
    KALDI_ASSERT(format == kOneByte);
    const uint8 *data = reinterpret_cast<const uint8*>(h+1) + col_offset +
        (num_cols * row_offset);
    float min_value = h->min_value,
        increment = h->range * (1.0 / 255.0);
    for (int32 row = 0; row < tgt_rows; row++) {
      Real *dest_row = dest->RowData(row);
      for (int32 col = 0; col < tgt_cols; col++)
        dest_row[col] = min_value + increment * data[col];
      data += num_cols;
    }
  }
}

// instantiate the templates.
template void CompressedMatrix::CopyToMat(int32,
                                          int32,
                                          MatrixBase<float> *dest) const;
template void CompressedMatrix::CopyToMat(int32,
                                          int32,
                                          MatrixBase<double> *dest) const;

void CompressedMatrix::Clear() {
  if (data_ != NULL) {
    delete [] static_cast<float*>(data_);
    data_ = NULL;
  }
}

CompressedMatrix::CompressedMatrix(const CompressedMatrix &mat): data_(NULL) {
  *this = mat; // use assignment operator.
}

CompressedMatrix &CompressedMatrix::operator = (const CompressedMatrix &mat) {
  Clear(); // now this->data_ == NULL.
  if (mat.data_ != NULL) {
    MatrixIndexT data_size = DataSize(*static_cast<GlobalHeader*>(mat.data_));
    data_ = AllocateData(data_size);
    memcpy(static_cast<void*>(data_),
           static_cast<void*>(mat.data_),
           data_size);
  }
  return *this;
}


}  // namespace kaldi
