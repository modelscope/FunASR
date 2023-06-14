// util/kaldi-holder.h

// Copyright 2009-2011     Microsoft Corporation
//                2016     Johns Hopkins University (author: Daniel Povey)
//                2016     Xiaohui Zhang

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


#ifndef KALDI_UTIL_KALDI_HOLDER_H_
#define KALDI_UTIL_KALDI_HOLDER_H_

#include <algorithm>
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "matrix/kaldi-vector.h"
#include "matrix/sparse-matrix.h"

namespace kaldi {


// The Table class uses a Holder class to wrap objects, and make them behave
// in a "normalized" way w.r.t. reading and writing, so the Table class can
// be template-ized without too much trouble. Look below this
// comment (search for GenericHolder) to see what it looks like.
//
//  Requirements of the holder class:
//
// They can only contain objects that can be read/written without external
// information; other objects cannot be stored in this type of archive.
//
// In terms of what functions it should have, see GenericHolder below.
// It is just for documentation.
//
// (1) Requirements of the Read and Write functions
//
// The Read and Write functions should have the property that in a longer
// file, if the Read function is started from where the Write function started
// writing, it should go to where the Write function stopped writing, in either
// text or binary mode (but it's OK if it doesn't eat up trailing space).
//
//     [Desirable property: when writing in text mode the output should contain
//      exactly one newline, at the end of the output; this makes it easier to
//      manipulate]
//
//     [Desirable property for classes: the output should just be a binary-mode
//      header (if in binary mode and it's a Kaldi object, or no header
//      othewise), and then the output of Object.Write().  This means that when
//      written to individual files with the scp: type of wspecifier, we can
//      read the individual files in the "normal" Kaldi way by reading the
//      binary header and then the object.]
//
//
// The Write function takes a 'binary' argument.  In general, each object will
// have two formats: text and binary.  However, it's permitted to throw() if
// asked to read in the text format if there is none.  The file will be open, if
// the file system has binary/text modes, in the corresponding mode.  However,
// the object should have a file-mode in which it can read either text or binary
// output.  It announces this via the static IsReadInBinary() function.  This
// will generally be the binary mode and it means that where necessary, in text
// formats, we must ignore \r characters.
//
// Memory requirements: if it allocates memory, the destructor should
// free that memory.  Copying and assignment of Holder objects may be
// disallowed as the Table code never does this.


/// GenericHolder serves to document the requirements of the Holder interface;
/// it's not intended to be used.
template<class SomeType> class GenericHolder {
 public:
  typedef SomeType T;

  /// Must have a constructor that takes no arguments.
  GenericHolder() { }

  /// Write() writes this object of type T.  Possibly also writes a binary-mode
  /// header so that the Read function knows which mode to read in (since the
  /// Read function does not get this information).  It's a static member so we
  /// can write those not inside this class (can use this function with Value()
  /// to write from this class).  The Write method may throw if it cannot write
  /// the object in the given (binary/non-binary) mode.  The holder object can
  /// assume the stream has been opened in the given mode (where relevant).  The
  /// object can write the data how it likes.
  static bool Write(std::ostream &os, bool binary, const T &t);

  /// Reads into the holder.  Must work out from the stream (which will be
  /// opened on Windows in binary mode if the IsReadInBinary() function of this
  /// class returns true, and text mode otherwise) whether the actual data is
  /// binary or not (usually via reading the Kaldi binary-mode header).
  /// We put the responsibility for reading the Kaldi binary-mode header in the
  /// Read function (rather than making the binary mode an argument to this
  /// function), so that for non-Kaldi binary files we don't have to write the
  /// header, which would prevent the file being read by non-Kaldi programs
  /// (e.g. if we write to individual files using an scp).
  /// Read must deallocate any existing data we have here, if applicable (must
  /// not assume the object was newly constructed).
  /// Returns true on success.
  /// If Read() returns false, the contents of this object and hence the value
  /// returned by Value() may be undefined.
  bool Read(std::istream &is);

  /// IsReadInBinary() will return true if the object wants the file to be
  /// opened in binary for reading (if the file system has binary/text modes),
  /// and false otherwise.  Static function.  Kaldi objects always return true
  /// as they always read in binary mode.  Note that we must be able to read, in
  /// this mode, objects written in both text and binary mode by Write (which
  /// may mean ignoring "\r" characters).  I doubt we will ever want this
  /// function to return false.
  static bool IsReadInBinary() { return true; }

  /// Returns the value of the object held here.  Will only
  /// ever be called if Read() has been previously called and it returned
  /// true (so OK to throw exception if no object was read).
  T &Value() { return t_; }  // if t is a pointer, would return *t_;

  /// The Clear() function doesn't have to do anything.  Its purpose is to
  /// allow the object to free resources if they're no longer needed.
  void Clear() { }

  /// This swaps the objects held by *this and *other (preferably a shallow
  /// swap).  Note, this is just an example.  The swap is with the *same type*
  /// of holder, not with some nonexistent base-class (remember, GenericHolder is
  /// an example for documentation, not a base-class).
  void Swap(GenericHolder<T> *other) { std::swap(t_, other->t_); }

  /// At the time of writing this will only do something meaningful
  /// KaldiObjectHolder holding matrix objects, in order to extract a holder
  /// holding a sub-matrix specified by 'range', e.g. [0:3,2:10], like in Matlab
  /// but with zero-based indexing. It returns true with successful extraction
  /// of the range, false if the range was invalid or outside the bounds of the
  /// matrix.  For other types of holder it just throws an error.
  bool ExtractRange(const GenericHolder<T> &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  /// If the object held pointers, the destructor would free them.
  ~GenericHolder() { }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(GenericHolder);
  T t_;  // t_ may alternatively be of type T*.
};


// See kaldi-holder-inl.h for examples of some actual Holder
// classes and templates.


// The following two typedefs should probably be in their own file, but they're
// here until there are enough of them to warrant their own header.


/// \addtogroup holders
/// @{

/// KaldiObjectHolder works for Kaldi objects that have the "standard" Read
/// and Write functions, and a copy constructor.
template<class KaldiType> class KaldiObjectHolder;

/// BasicHolder is valid for float, double, bool, and integer
/// types.  There will be a compile time error otherwise, because
/// we make sure that the {Write, Read}BasicType functions do not
/// get instantiated for other types.
template<class BasicType> class BasicHolder;


// A Holder for a vector of basic types, e.g.
// std::vector<int32>, std::vector<float>, and so on.
// Note: a basic type is defined as a type for which ReadBasicType
// and WriteBasicType are implemented, i.e. integer and floating
// types, and bool.
template<class BasicType> class BasicVectorHolder;


// A holder for vectors of vectors of basic types, e.g.
// std::vector<std::vector<int32> >, and so on.
// Note: a basic type is defined as a type for which ReadBasicType
// and WriteBasicType are implemented, i.e. integer and floating
// types, and bool.
template<class BasicType> class BasicVectorVectorHolder;

// A holder for vectors of pairsof basic types, e.g.
// std::vector<std::pair<int32, int32> >, and so on.
// Note: a basic type is defined as a type for which ReadBasicType
// and WriteBasicType are implemented, i.e. integer and floating
// types, and bool.  Text format is (e.g. for integers),
// "1 12 ; 43 61 ; 17 8 \n"
template<class BasicType> class BasicPairVectorHolder;

/// We define a Token (not a typedef, just a word) as a nonempty, printable,
/// whitespace-free std::string.  The binary and text formats here are the same
/// (newline-terminated) and as such we don't bother with the binary-mode
/// headers.
class TokenHolder;

/// Class TokenVectorHolder is a Holder class for vectors of Tokens
/// (T == std::string).
class TokenVectorHolder;

/// A class for reading/writing HTK-format matrices.
/// T == std::pair<Matrix<BaseFloat>, HtkHeader>
class HtkMatrixHolder;

/// A class for reading/writing Sphinx format matrices.
template<int kFeatDim = 13> class SphinxMatrixHolder;

/// This templated function exists so that we can write .scp files with
/// 'object ranges' specified: the canonical example is a [first:last] range
/// of rows of a matrix, or [first-row:last-row,first-column,last-column]
/// of a matrix.  We can also support [begin-time:end-time] of a wave
/// file.  The string 'range' is whatever is in the square brackets; it is
/// parsed inside this function.
/// This function returns true if the partial object was successfully extracted,
/// and false if there was an error such as an invalid range.
/// The generic version of this function just fails; we overload the template
/// whenever we need it for a specific class.
template <class T>
bool ExtractObjectRange(const T &input, const std::string &range, T *output) {
  KALDI_ERR << "Ranges not supported for objects of this type.";
  return false;
}

/// The template is specialized with a version that actually does something,
/// for types Matrix<float> and Matrix<double>.  We can later add versions of
/// this template for other types, such as Vector, which can meaningfully
/// have ranges extracted.
template <class Real>
bool ExtractObjectRange(const Matrix<Real> &input, const std::string &range,
                        Matrix<Real> *output);

/// The template is specialized types Vector<float> and Vector<double>.
template <class Real>
bool ExtractObjectRange(const Vector<Real> &input, const std::string &range,
                        Vector<Real> *output);

/// GeneralMatrix is always of type BaseFloat
bool ExtractObjectRange(const GeneralMatrix &input, const std::string &range,
                        GeneralMatrix *output);

/// CompressedMatrix is always of the type BaseFloat but it is more
/// efficient to provide template as it uses CompressedMatrix's own
/// conversion to Matrix<Real>
template <class Real>
bool ExtractObjectRange(const CompressedMatrix &input, const std::string &range,
                        Matrix<Real> *output);

// In SequentialTableReaderScriptImpl and RandomAccessTableReaderScriptImpl, for
// cases where the scp contained 'range specifiers' (things in square brackets
// identifying parts of objects like matrices), use this function to separate
// the input string 'rxfilename_with_range' (e.g "1.ark:100[1:2,2:10]") into the data_rxfilename
// (e.g. "1.ark:100") and the optional range specifier which will be everything
// inside the square brackets.  It returns true if everything seems OK, and
// false if for example the string contained more than one '['.  This function
// should only be called if 'line' ends in ']', otherwise it is an error.
bool ExtractRangeSpecifier(const std::string &rxfilename_with_range,
                           std::string *data_rxfilename,
                           std::string *range);


/// @} end "addtogroup holders"


}  // end namespace kaldi

#include "util/kaldi-holder-inl.h"

#endif  // KALDI_UTIL_KALDI_HOLDER_H_
