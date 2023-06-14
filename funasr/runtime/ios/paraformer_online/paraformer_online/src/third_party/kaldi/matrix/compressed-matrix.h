// matrix/compressed-matrix.h

// Copyright 2012  Johns Hopkins University (author: Daniel Povey)
//                 Frantisek Skala, Wei Shi

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

#ifndef KALDI_MATRIX_COMPRESSED_MATRIX_H_
#define KALDI_MATRIX_COMPRESSED_MATRIX_H_ 1

#include "matrix/kaldi-matrix.h"

namespace kaldi {

/// \addtogroup matrix_group
/// @{



/*
  The enum CompressionMethod is used when creating a CompressedMatrix (a lossily
  compressed matrix) from a regular Matrix.  It dictates how we choose the
  compressed format and how we choose the ranges of floats that are represented
  by particular integers.

    kAutomaticMethod = 1 This is the default when you don't specify the
                        compression method.  It is a shorthand for using
                        kSpeechFeature if the num-rows is more than 8, and
                        kTwoByteAuto otherwise.
    kSpeechFeature = 2  This is the most complicated of the compression methods,
                        and was designed for speech features which have a roughly
                        Gaussian distribution with different ranges for each
                        dimension.  Each element is stored in one byte, but there
                        is an 8-byte header per column; the spacing of the
                        integer values is not uniform but is in 3 ranges.
    kTwoByteAuto = 3    Each element is stored in two bytes as a uint16, with
                        the representable range of values chosen automatically
                        with the minimum and maximum elements of the matrix as
                        its edges.
    kTwoByteSignedInteger = 4
                        Each element is stored in two bytes as a uint16, with
                        the representable range of value chosen to coincide with
                        what you'd get if you stored signed integers, i.e.
                        [-32768.0, 32767.0].  Suitable for waveform data that
                        was previously stored as 16-bit PCM.
    kOneByteAuto = 5    Each element is stored in one byte as a uint8, with the
                        representable range of values chosen automatically with
                        the minimum and maximum elements of the matrix as its
                        edges.
    kOneByteUnsignedInteger = 6 Each element is stored in
                        one byte as a uint8, with the representable range of
                        values equal to [0.0, 255.0].
    kOneByteZeroOne = 7 Each element is stored in
                        one byte as a uint8, with the representable range of
                        values equal to [0.0, 1.0].  Suitable for image data
                        that has previously been compressed as int8.

    // We can add new methods here as needed: if they just imply different ways
    // of selecting the min_value and range, and a num-bytes = 1 or 2, they will
    // be trivial to implement.
*/
enum CompressionMethod {
  kAutomaticMethod = 1,
  kSpeechFeature = 2,
  kTwoByteAuto = 3,
  kTwoByteSignedInteger = 4,
  kOneByteAuto = 5,
  kOneByteUnsignedInteger = 6,
  kOneByteZeroOne = 7
};


/*
  This class does lossy compression of a matrix.  It supports various compression
  methods, see enum CompressionMethod.
*/

class CompressedMatrix {
 public:
  CompressedMatrix(): data_(NULL) { }

  ~CompressedMatrix() { Clear(); }

  template<typename Real>
  explicit CompressedMatrix(const MatrixBase<Real> &mat,
                            CompressionMethod method = kAutomaticMethod):
      data_(NULL) { CopyFromMat(mat, method); }

  /// Initializer that can be used to select part of an existing
  /// CompressedMatrix without un-compressing and re-compressing (note: unlike
  /// similar initializers for class Matrix, it doesn't point to the same memory
  /// location).
  ///
  /// This creates a CompressedMatrix with the size (num_rows, num_cols)
  /// starting at (row_offset, col_offset).
  ///
  /// If you specify allow_padding = true,
  /// it is permitted to have row_offset < 0 and
  /// row_offset + num_rows > mat.NumRows(), and the result will contain
  /// repeats of the first and last rows of 'mat' as necessary.
  CompressedMatrix(const CompressedMatrix &mat,
                   const MatrixIndexT row_offset,
                   const MatrixIndexT num_rows,
                   const MatrixIndexT col_offset,
                   const MatrixIndexT num_cols,
                   bool allow_padding = false);

  void *Data() const { return this->data_; }

  /// This will resize *this and copy the contents of mat to *this.
  template<typename Real>
  void CopyFromMat(const MatrixBase<Real> &mat,
                   CompressionMethod method = kAutomaticMethod);

  CompressedMatrix(const CompressedMatrix &mat);

  CompressedMatrix &operator = (const CompressedMatrix &mat); // assignment operator.

  template<typename Real>
  CompressedMatrix &operator = (const MatrixBase<Real> &mat); // assignment operator.

  /// Copies contents to matrix.  Note: mat must have the correct size.
  /// The kTrans case uses a temporary.
  template<typename Real>
  void CopyToMat(MatrixBase<Real> *mat,
                 MatrixTransposeType trans = kNoTrans) const;

  void Write(std::ostream &os, bool binary) const;

  void Read(std::istream &is, bool binary);

  /// Returns number of rows (or zero for emtpy matrix).
  inline MatrixIndexT NumRows() const { return (data_ == NULL) ? 0 :
      (*reinterpret_cast<GlobalHeader*>(data_)).num_rows; }

  /// Returns number of columns (or zero for emtpy matrix).
  inline MatrixIndexT NumCols() const { return (data_ == NULL) ? 0 :
      (*reinterpret_cast<GlobalHeader*>(data_)).num_cols; }

  /// Copies row #row of the matrix into vector v.
  /// Note: v must have same size as #cols.
  template<typename Real>
  void CopyRowToVec(MatrixIndexT row, VectorBase<Real> *v) const;

  /// Copies column #col of the matrix into vector v.
  /// Note: v must have same size as #rows.
  template<typename Real>
  void CopyColToVec(MatrixIndexT col, VectorBase<Real> *v) const;

  /// Copies submatrix of compressed matrix into matrix dest.
  /// Submatrix starts at row row_offset and column column_offset and its size
  /// is defined by size of provided matrix dest
  template<typename Real>
  void CopyToMat(int32 row_offset,
                 int32 column_offset,
                 MatrixBase<Real> *dest) const;

  void Swap(CompressedMatrix *other) { std::swap(data_, other->data_); }

  void Clear();

  /// scales all elements of matrix by alpha.
  /// It scales the floating point values in GlobalHeader by alpha.
  void Scale(float alpha);

  friend class Matrix<float>;
  friend class Matrix<double>;
 private:

  // This enum describes the different compressed-data formats: these are
  // distinct from the compression methods although all of the methods apart
  // from kAutomaticMethod dictate a particular compressed-data format.
  //
  //  kOneByteWithColHeaders means there is a GlobalHeader and each
  //    column has a PerColHeader; the actual data is stored in
  //    one byte per element, in column-major order (the mapping
  //    from integers to floats is a little complicated).
  //  kTwoByte means there is a global header but no PerColHeader;
  //    the actual data is stored in two bytes per element in
  //    row-major order; it's decompressed as:
  //       uint16 i;  GlobalHeader g;
  //       float f = g.min_value + i * (g.range / 65535.0)
  //  kOneByte means there is a global header but not PerColHeader;
  //    the data is stored in one byte per element in row-major
  //    order and is decompressed as:
  //       uint8 i;  GlobalHeader g;
  //       float f = g.min_value + i * (g.range / 255.0)
  enum DataFormat {
    kOneByteWithColHeaders = 1,
    kTwoByte = 2,
    kOneByte = 3
  };


  // allocates data using new [], ensures byte alignment
  // sufficient for float.
  static void *AllocateData(int32 num_bytes);

  struct GlobalHeader {
    int32 format;     // Represents the enum DataFormat.
    float min_value;  // min_value and range represent the ranges of the integer
                      // data in the kTwoByte and kOneByte formats, and the
                      // range of the PerColHeader uint16's in the
                      // kOneByteWithColheaders format.
    float range;
    int32 num_rows;
    int32 num_cols;
  };

  // This function computes the global header for compressing this data.
  template<typename Real>
  static inline void ComputeGlobalHeader(const MatrixBase<Real> &mat,
                                         CompressionMethod method,
                                         GlobalHeader *header);


  // The number of bytes we need to request when allocating 'data_'.
  static MatrixIndexT DataSize(const GlobalHeader &header);

  // This struct is only used in format kOneByteWithColHeaders.
  struct PerColHeader {
    uint16 percentile_0;
    uint16 percentile_25;
    uint16 percentile_75;
    uint16 percentile_100;
  };

  template<typename Real>
  static void CompressColumn(const GlobalHeader &global_header,
                             const Real *data, MatrixIndexT stride,
                             int32 num_rows, PerColHeader *header,
                             uint8 *byte_data);
  template<typename Real>
  static void ComputeColHeader(const GlobalHeader &global_header,
                               const Real *data, MatrixIndexT stride,
                               int32 num_rows, PerColHeader *header);

  static inline uint16 FloatToUint16(const GlobalHeader &global_header,
                                     float value);

  // this is used only in the kOneByte compression format.
  static inline uint8 FloatToUint8(const GlobalHeader &global_header,
                                   float value);

  static inline float Uint16ToFloat(const GlobalHeader &global_header,
                                    uint16 value);

  // this is used only in the kOneByteWithColHeaders compression format.
  static inline uint8 FloatToChar(float p0, float p25,
                                          float p75, float p100,
                                          float value);

  // this is used only in the kOneByteWithColHeaders compression format.
  static inline float CharToFloat(float p0, float p25,
                                  float p75, float p100,
                                  uint8 value);

  void *data_; // first GlobalHeader, then PerColHeader (repeated), then
  // the byte data for each column (repeated).  Note: don't intersperse
  // the byte data with the PerColHeaders, because of alignment issues.

};

/// @} end of \addtogroup matrix_group


}  // namespace kaldi


#endif  // KALDI_MATRIX_COMPRESSED_MATRIX_H_
