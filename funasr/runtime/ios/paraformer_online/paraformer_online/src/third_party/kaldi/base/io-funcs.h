// base/io-funcs.h

// Copyright 2009-2011  Microsoft Corporation;  Saarland University;
//                      Jan Silovsky;   Yanmin Qian
//                2016  Xiaohui Zhang

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

#ifndef KALDI_BASE_IO_FUNCS_H_
#define KALDI_BASE_IO_FUNCS_H_

// This header only contains some relatively low-level I/O functions.
// The full Kaldi I/O declarations are in ../util/kaldi-io.h
// and ../util/kaldi-table.h
// They were put in util/ in order to avoid making the Matrix library
// dependent on them.

#include <cctype>
#include <vector>
#include <string>
#include "base/kaldi-common.h"

namespace kaldi {



/*
  This comment describes the Kaldi approach to I/O.  All objects can be written
  and read in two modes: binary and text.  In addition we want to make the I/O
  work if we redefine the typedef "BaseFloat" between floats and doubles.
  We also want to have control over whitespace in text mode without affecting
  the meaning of the file, for pretty-printing purposes.

  Errors are handled by throwing an exception (std::runtime_error).

  For integer and floating-point types (and boolean values):

   WriteBasicType(std::ostream &, bool binary, const T&);
   ReadBasicType(std::istream &, bool binary, T*);

  and we expect these functions to be defined in such a way that they work when
  the type T changes between float and double, so you can read float into double
  and vice versa].  Note that for efficiency and space-saving reasons, the Vector
  and Matrix classes do not use these functions [but they preserve the type
  interchangeability in their own way]

  For a class (or struct) C:
  class C {
  ..
    Write(std::ostream &, bool binary, [possibly extra optional args for specific classes]) const;
    Read(std::istream &, bool binary, [possibly extra optional args for specific classes]);
  ..
  }
  NOTE: The only actual optional args we used are the "add" arguments in
  Vector/Matrix classes, which specify whether we should sum the data already
  in the class with the data being read.

  For types which are typedef's involving stl classes, I/O is as follows:
  typedef std::vector<std::pair<A, B> > MyTypedefName;

  The user should define something like:

   WriteMyTypedefName(std::ostream &, bool binary, const MyTypedefName &t);
   ReadMyTypedefName(std::ostream &, bool binary, MyTypedefName *t);

  The user would have to write these functions.

  For a type std::vector<T>:

   void WriteIntegerVector(std::ostream &os, bool binary, const std::vector<T> &v);
   void ReadIntegerVector(std::istream &is, bool binary, std::vector<T> *v);

  For other types, e.g. vectors of pairs, the user should create a routine of the
  type WriteMyTypedefName.  This is to avoid introducing confusing templated functions;
  we could easily create templated functions to handle most of these cases but they
  would have to share the same name.

  It also often happens that the user needs to write/read special tokens as part
  of a file.  These might be class headers, or separators/identifiers in the class.
  We provide special functions for manipulating these.  These special tokens must
  be nonempty and must not contain any whitespace.

    void WriteToken(std::ostream &os, bool binary, const char*);
    void WriteToken(std::ostream &os, bool binary, const std::string & token);
    int Peek(std::istream &is, bool binary);
    void ReadToken(std::istream &is, bool binary, std::string *str);
    void PeekToken(std::istream &is, bool binary, std::string *str);

  WriteToken writes the token and one space (whether in binary or text mode).

  Peek returns the first character of the next token, by consuming whitespace
  (in text mode) and then returning the peek() character.  It returns -1 at EOF;
  it doesn't throw.  It's useful if a class can have various forms based on
  typedefs and virtual classes, and wants to know which version to read.

  ReadToken allow the caller to obtain the next token.  PeekToken works just
  like ReadToken, but seeks back to the beginning of the token.  A subsequent
  call to ReadToken will read the same token again.  This is useful when
  different object types are written to the same file; using PeekToken one can
  decide which of the objects to read.

  There is currently no special functionality for writing/reading strings (where the strings
  contain data rather than "special tokens" that are whitespace-free and nonempty).  This is
  because Kaldi is structured in such a way that strings don't appear, except as OpenFst symbol
  table entries (and these have their own format).


  NOTE: you should not call ReadIntegerType and WriteIntegerType with types,
  such as int and size_t, that are machine-independent -- at least not
  if you want your file formats to port between machines.  Use int32 and
  int64 where necessary.  There is no way to detect this using compile-time
  assertions because C++ only keeps track of the internal representation of
  the type.
*/

/// \addtogroup io_funcs_basic
/// @{


/// WriteBasicType is the name of the write function for bool, integer types,
/// and floating-point types. They all throw on error.
template<class T> void WriteBasicType(std::ostream &os, bool binary, T t);

/// ReadBasicType is the name of the read function for bool, integer types,
/// and floating-point types. They all throw on error.
template<class T> void ReadBasicType(std::istream &is, bool binary, T *t);


// Declare specialization for bool.
template<>
void WriteBasicType<bool>(std::ostream &os, bool binary, bool b);

template <>
void ReadBasicType<bool>(std::istream &is, bool binary, bool *b);

// Declare specializations for float and double.
template<>
void WriteBasicType<float>(std::ostream &os, bool binary, float f);

template<>
void WriteBasicType<double>(std::ostream &os, bool binary, double f);

template<>
void ReadBasicType<float>(std::istream &is, bool binary, float *f);

template<>
void ReadBasicType<double>(std::istream &is, bool binary, double *f);

// Define ReadBasicType that accepts an "add" parameter to add to
// the destination.  Caution: if used in Read functions, be careful
// to initialize the parameters concerned to zero in the default
// constructor.
template<class T>
inline void ReadBasicType(std::istream &is, bool binary, T *t, bool add) {
  if (!add) {
    ReadBasicType(is, binary, t);
  } else {
    T tmp = T(0);
    ReadBasicType(is, binary, &tmp);
    *t += tmp;
  }
}

/// Function for writing STL vectors of integer types.
template<class T> inline void WriteIntegerVector(std::ostream &os, bool binary,
                                                 const std::vector<T> &v);

/// Function for reading STL vector of integer types.
template<class T> inline void ReadIntegerVector(std::istream &is, bool binary,
                                                std::vector<T> *v);

/// Function for writing STL vectors of pairs of integer types.
template<class T>
inline void WriteIntegerPairVector(std::ostream &os, bool binary,
                                   const std::vector<std::pair<T, T> > &v);

/// Function for reading STL vector of pairs of integer types.
template<class T>
inline void ReadIntegerPairVector(std::istream &is, bool binary,
                                  std::vector<std::pair<T, T> > *v);

/// The WriteToken functions are for writing nonempty sequences of non-space
/// characters. They are not for general strings.
void WriteToken(std::ostream &os, bool binary, const char *token);
void WriteToken(std::ostream &os, bool binary, const std::string & token);

/// Peek consumes whitespace (if binary == false) and then returns the peek()
/// value of the stream.
int Peek(std::istream &is, bool binary);

/// ReadToken gets the next token and puts it in str (exception on failure).
void ReadToken(std::istream &is, bool binary, std::string *token);

/// PeekToken will return the first character of the next token, or -1 if end of
/// file.  It's the same as Peek(), except if the first character is '<' it will
/// skip over it and will return the next character.  It will unget the '<' so
/// the stream is where it was before you did PeekToken().
int PeekToken(std::istream &is, bool binary);

/// ExpectToken tries to read in the given token, and throws an exception
/// on failure.
void ExpectToken(std::istream &is, bool binary, const char *token);
void ExpectToken(std::istream &is, bool binary, const std::string & token);

/// ExpectPretty attempts to read the text in "token", but only in non-binary
/// mode.  Throws exception on failure.  It expects an exact match except that
/// arbitrary whitespace matches arbitrary whitespace.
void ExpectPretty(std::istream &is, bool binary, const char *token);
void ExpectPretty(std::istream &is, bool binary, const std::string & token);

/// @} end "addtogroup io_funcs_basic"


/// InitKaldiOutputStream initializes an opened stream for writing by writing an
/// optional binary header and modifying the floating-point precision; it will
/// typically not be called by users directly.
inline void InitKaldiOutputStream(std::ostream &os, bool binary);

/// InitKaldiInputStream initializes an opened stream for reading by detecting
/// the binary header and setting the "binary" value appropriately;
/// It will typically not be called by users directly.
inline bool InitKaldiInputStream(std::istream &is, bool *binary);

}  // end namespace kaldi.

#include "base/io-funcs-inl.h"

#endif  // KALDI_BASE_IO_FUNCS_H_
