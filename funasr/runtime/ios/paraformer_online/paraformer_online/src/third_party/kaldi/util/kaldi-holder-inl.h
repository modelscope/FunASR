// util/kaldi-holder-inl.h

// Copyright 2009-2011     Microsoft Corporation
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


#ifndef KALDI_UTIL_KALDI_HOLDER_INL_H_
#define KALDI_UTIL_KALDI_HOLDER_INL_H_

#include <algorithm>
#include <vector>
#include <utility>
#include <string>

#include "base/kaldi-utils.h"
#include "util/kaldi-io.h"
#include "util/text-utils.h"
#include "matrix/kaldi-matrix.h"

namespace kaldi {

/// \addtogroup holders
/// @{


// KaldiObjectHolder is valid only for Kaldi objects with
// copy constructors, default constructors, and "normal"
// Kaldi Write and Read functions.  E.g. it works for
// Matrix and Vector.
template<class KaldiType> class KaldiObjectHolder {
 public:
  typedef KaldiType T;

  KaldiObjectHolder(): t_(NULL) { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      t.Write(os, binary);
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object. " << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() {
    if (t_) {
      delete t_;
      t_ = NULL;
    }
  }

  // Reads into the holder.
  bool Read(std::istream &is) {
    delete t_;
    t_ = new T;
    // Don't want any existing state to complicate the read function: get new
    // object.
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object, failed reading binary header\n";
      return false;
    }
    try {
      t_->Read(is, is_binary);
      return true;
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object. " << e.what();
      delete t_;
      t_ = NULL;
      return false;
    }
  }

  // Kaldi objects always have the stream open in binary mode for
  // reading.
  static bool IsReadInBinary() { return true; }

  T &Value() {
    // code error if !t_.
    if (!t_) KALDI_ERR << "KaldiObjectHolder::Value() called wrongly.";
    return *t_;
  }

  void Swap(KaldiObjectHolder<T> *other) {
    // the t_ values are pointers so this is a shallow swap.
    std::swap(t_, other->t_);
  }

  bool ExtractRange(const KaldiObjectHolder<T> &other,
                    const std::string &range) {
    KALDI_ASSERT(other.t_ != NULL);
    delete t_;
    t_ = new T;
    // this call will fail for most object types.
    return ExtractObjectRange(*(other.t_), range, t_);
  }

  ~KaldiObjectHolder() { delete t_; }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(KaldiObjectHolder);
  T *t_;
};


// BasicHolder is valid for float, double, bool, and integer
// types.  There will be a compile time error otherwise, because
// we make sure that the {Write, Read}BasicType functions do not
// get instantiated for other types.

template<class BasicType> class BasicHolder {
 public:
  typedef BasicType T;

  BasicHolder(): t_(static_cast<T>(-1)) { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      WriteBasicType(os, binary, t);
      if (!binary) os << '\n';  // Makes output format more readable and
      // easier to manipulate.
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object. " << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() { }

  // Reads into the holder.
  bool Read(std::istream &is) {
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary"
          " header\n";
      return false;
    }
    try {
      int c;
      if (!is_binary) {  // This is to catch errors, the class would work
        // without it..
        // Eat up any whitespace and make sure it's not newline.
        while (isspace((c = is.peek())) && c != static_cast<int>('\n')) {
          is.get();
        }
        if (is.peek() == '\n') {
          KALDI_WARN << "Found newline but expected basic type.";
          return false;  // This is just to catch a more-
          // likely-than average type of error (empty line before the token),
          // since ReadBasicType will eat it up.
        }
      }

      ReadBasicType(is, is_binary, &t_);

      if (!is_binary) {  // This is to catch errors, the class would work
        // without it..
        // make sure there is a newline.
        while (isspace((c = is.peek())) && c != static_cast<int>('\n')) {
          is.get();
        }
        if (is.peek() != '\n') {
          KALDI_WARN << "BasicHolder::Read, expected newline, got "
                     << CharToString(is.peek()) << ", position " << is.tellg();
          return false;
        }
        is.get();  // Consume the newline.
      }
      return true;
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught reading Table object. " << e.what();
      return false;
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  T &Value() {
    return t_;
  }

  void Swap(BasicHolder<T> *other) {
    std::swap(t_, other->t_);
  }

  bool ExtractRange(const BasicHolder<T> &other, const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~BasicHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicHolder);

  T t_;
};


/// A Holder for a vector of basic types, e.g.
/// std::vector<int32>, std::vector<float>, and so on.
/// Note: a basic type is defined as a type for which ReadBasicType
/// and WriteBasicType are implemented, i.e. integer and floating
/// types, and bool.
template<class BasicType> class BasicVectorHolder {
 public:
  typedef std::vector<BasicType> T;

  BasicVectorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {  // need to write the size, in binary mode.
        KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(t.size())) ==
                     t.size());
        // Or this Write routine cannot handle such a large vector.
        // use int32 because it's fixed size regardless of compilation.
        // change to int64 (plus in Read function) if this becomes a problem.
        WriteBasicType(os, binary, static_cast<int32>(t.size()));
        for (typename std::vector<BasicType>::const_iterator iter = t.begin();
             iter != t.end(); ++iter)
          WriteBasicType(os, binary, *iter);

      } else {
        for (typename std::vector<BasicType>::const_iterator iter = t.begin();
             iter != t.end(); ++iter)
          WriteBasicType(os, binary, *iter);
        os << '\n';  // Makes output format more readable and
        // easier to manipulate.  In text mode, this function writes something
        // like "1 2 3\n".
      }
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object (BasicVector). "
                 << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary"
          " header\n";
      return false;
    }
    if (!is_binary) {
      // In text mode, we terminate with newline.
      std::string line;
      getline(is, line);  // this will discard the \n, if present.
      if (is.fail()) {
        KALDI_WARN << "BasicVectorHolder::Read, error reading line " <<
            (is.eof() ? "[eof]" : "");
        return false;  // probably eof.  fail in any case.
      }
      std::istringstream line_is(line);
      try {
        while (1) {
          line_is >> std::ws;  // eat up whitespace.
          if (line_is.eof()) break;
          BasicType bt;
          ReadBasicType(line_is, false, &bt);
          t_.push_back(bt);
        }
        return true;
      } catch(const std::exception &e) {
        KALDI_WARN << "BasicVectorHolder::Read, could not interpret line: "
                   << "'" << line << "'" << "\n" << e.what();
        return false;
      }
    } else {  // binary mode.
      size_t filepos = is.tellg();
      try {
        int32 size;
        ReadBasicType(is, true, &size);
        t_.resize(size);
        for (typename std::vector<BasicType>::iterator iter = t_.begin();
             iter != t_.end();
             ++iter) {
          ReadBasicType(is, true, &(*iter));
        }
        return true;
      } catch(...) {
        KALDI_WARN << "BasicVectorHolder::Read, read error or unexpected data"
            " at archive entry beginning at file position " << filepos;
        return false;
      }
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  T &Value() { return t_; }

  void Swap(BasicVectorHolder<BasicType> *other) {
    t_.swap(other->t_);
  }

  bool ExtractRange(const BasicVectorHolder<BasicType> &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~BasicVectorHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicVectorHolder);
  T t_;
};


/// BasicVectorVectorHolder is a Holder for a vector of vector of
/// a basic type, e.g. std::vector<std::vector<int32> >.
/// Note: a basic type is defined as a type for which ReadBasicType
/// and WriteBasicType are implemented, i.e. integer and floating
/// types, and bool.
template<class BasicType> class BasicVectorVectorHolder {
 public:
  typedef std::vector<std::vector<BasicType> > T;

  BasicVectorVectorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {  // need to write the size, in binary mode.
        KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(t.size())) ==
                     t.size());
        // Or this Write routine cannot handle such a large vector.
        // use int32 because it's fixed size regardless of compilation.
        // change to int64 (plus in Read function) if this becomes a problem.
        WriteBasicType(os, binary, static_cast<int32>(t.size()));
        for (typename std::vector<std::vector<BasicType> >::const_iterator
                 iter = t.begin();
             iter != t.end(); ++iter) {
          KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(iter->size()))
                       == iter->size());
          WriteBasicType(os, binary, static_cast<int32>(iter->size()));
          for (typename std::vector<BasicType>::const_iterator
                   iter2 = iter->begin();
               iter2 != iter->end(); ++iter2) {
            WriteBasicType(os, binary, *iter2);
          }
        }
      } else {  // text mode...
        // In text mode, we write out something like (for integers):
        // "1 2 3 ; 4 5 ; 6 ; ; 7 8 9 ;\n"
        // where the semicolon is a terminator, not a separator
        // (a separator would cause ambiguity between an
        // empty list, and a list containing a single empty list).
        for (typename std::vector<std::vector<BasicType> >::const_iterator
                 iter = t.begin();
             iter != t.end();
             ++iter) {
          for (typename std::vector<BasicType>::const_iterator
                   iter2 = iter->begin();
               iter2 != iter->end(); ++iter2)
            WriteBasicType(os, binary, *iter2);
          os << "; ";
        }
        os << '\n';
      }
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object. " << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Failed reading binary header\n";
      return false;
    }
    if (!is_binary) {
      // In text mode, we terminate with newline.
      try {  // catching errors from ReadBasicType..
        std::vector<BasicType> v;  // temporary vector
        while (1) {
          int i = is.peek();
          if (i == -1) {
            KALDI_WARN << "Unexpected EOF";
            return false;
          } else if (static_cast<char>(i) == '\n') {
            if (!v.empty()) {
              KALDI_WARN << "No semicolon before newline (wrong format)";
              return false;
            } else {
              is.get();
              return true;
            }
          } else if (std::isspace(i)) {
            is.get();
          } else if (static_cast<char>(i) == ';') {
            t_.push_back(v);
            v.clear();
            is.get();
          } else {  // some object we want to read...
            BasicType b;
            ReadBasicType(is, false, &b);  // throws on error.
            v.push_back(b);
          }
        }
      } catch(const std::exception &e) {
        KALDI_WARN << "BasicVectorVectorHolder::Read, read error. " << e.what();
        return false;
      }
    } else {  // binary mode.
      size_t filepos = is.tellg();
      try {
        int32 size;
        ReadBasicType(is, true, &size);
        t_.resize(size);
        for (typename std::vector<std::vector<BasicType> >::iterator
                 iter = t_.begin();
             iter != t_.end();
             ++iter) {
          int32 size2;
          ReadBasicType(is, true, &size2);
          iter->resize(size2);
          for (typename std::vector<BasicType>::iterator iter2 = iter->begin();
               iter2 != iter->end();
               ++iter2)
            ReadBasicType(is, true, &(*iter2));
        }
        return true;
      } catch(...) {
        KALDI_WARN << "Read error or unexpected data at archive entry beginning"
            " at file position " << filepos;
        return false;
      }
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  T &Value() {  return t_; }

  void Swap(BasicVectorVectorHolder<BasicType> *other) {
    t_.swap(other->t_);
  }

  bool ExtractRange(BasicVectorVectorHolder<BasicType> &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~BasicVectorVectorHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicVectorVectorHolder);
  T t_;
};


/// BasicPairVectorHolder is a Holder for a vector of pairs of
/// a basic type, e.g. std::vector<std::pair<int32, int32> >.
/// Note: a basic type is defined as a type for which ReadBasicType
/// and WriteBasicType are implemented, i.e. integer and floating
/// types, and bool.
template<class BasicType> class BasicPairVectorHolder {
 public:
  typedef std::vector<std::pair<BasicType, BasicType> > T;

  BasicPairVectorHolder() { }

  static bool Write(std::ostream &os, bool binary, const T &t) {
    InitKaldiOutputStream(os, binary);  // Puts binary header if binary mode.
    try {
      if (binary) {  // need to write the size, in binary mode.
        KALDI_ASSERT(static_cast<size_t>(static_cast<int32>(t.size())) ==
                     t.size());
        // Or this Write routine cannot handle such a large vector.
        // use int32 because it's fixed size regardless of compilation.
        // change to int64 (plus in Read function) if this becomes a problem.
        WriteBasicType(os, binary, static_cast<int32>(t.size()));
        for (typename T::const_iterator iter = t.begin();
             iter != t.end(); ++iter) {
          WriteBasicType(os, binary, iter->first);
          WriteBasicType(os, binary, iter->second);
        }
      } else {  // text mode...
        // In text mode, we write out something like (for integers):
        // "1 2 ; 4 5 ; 6 7 ; 8 9 \n"
        // where the semicolon is a separator, not a terminator.
        for (typename T::const_iterator iter = t.begin();
             iter != t.end();) {
          WriteBasicType(os, binary, iter->first);
          WriteBasicType(os, binary, iter->second);
          ++iter;
          if (iter != t.end())
            os << "; ";
        }
        os << '\n';
      }
      return os.good();
    } catch(const std::exception &e) {
      KALDI_WARN << "Exception caught writing Table object. " << e.what();
      return false;  // Write failure.
    }
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();
    bool is_binary;
    if (!InitKaldiInputStream(is, &is_binary)) {
      KALDI_WARN << "Reading Table object [integer type], failed reading binary"
          " header\n";
      return false;
    }
    if (!is_binary) {
      // In text mode, we terminate with newline.
      try {  // catching errors from ReadBasicType..
        std::vector<BasicType> v;  // temporary vector
        while (1) {
          int i = is.peek();
          if (i == -1) {
            KALDI_WARN << "Unexpected EOF";
            return false;
          } else if (static_cast<char>(i) == '\n') {
            if (t_.empty() && v.empty()) {
              is.get();
              return true;
            } else if (v.size() == 2) {
              t_.push_back(std::make_pair(v[0], v[1]));
              is.get();
              return true;
            } else {
              KALDI_WARN << "Unexpected newline, reading vector<pair<?> >; got "
                         << v.size() << " elements, expected 2.";
              return false;
            }
          } else if (std::isspace(i)) {
            is.get();
          } else if (static_cast<char>(i) == ';') {
            if (v.size() != 2) {
              KALDI_WARN << "Wrong input format, reading vector<pair<?> >; got "
                         << v.size() << " elements, expected 2.";
              return false;
            }
            t_.push_back(std::make_pair(v[0], v[1]));
            v.clear();
            is.get();
          } else {  // some object we want to read...
            BasicType b;
            ReadBasicType(is, false, &b);  // throws on error.
            v.push_back(b);
          }
        }
      } catch(const std::exception &e) {
        KALDI_WARN << "BasicPairVectorHolder::Read, read error. " << e.what();
        return false;
      }
    } else {  // binary mode.
      size_t filepos = is.tellg();
      try {
        int32 size;
        ReadBasicType(is, true, &size);
        t_.resize(size);
        for (typename T::iterator iter = t_.begin();
             iter != t_.end();
             ++iter) {
          ReadBasicType(is, true, &(iter->first));
          ReadBasicType(is, true, &(iter->second));
        }
        return true;
      } catch(...) {
        KALDI_WARN << "BasicVectorHolder::Read, read error or unexpected data"
            " at archive entry beginning at file position " << filepos;
        return false;
      }
    }
  }

  // Objects read/written with the Kaldi I/O functions always have the stream
  // open in binary mode for reading.
  static bool IsReadInBinary() { return true; }

  T &Value() {  return t_; }

  void Swap(BasicPairVectorHolder<BasicType> *other) {
    t_.swap(other->t_);
  }

  bool ExtractRange(const BasicPairVectorHolder<BasicType> &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

  ~BasicPairVectorHolder() { }
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(BasicPairVectorHolder);
  T t_;
};




// We define a Token as a nonempty, printable, whitespace-free std::string.
// The binary and text formats here are the same (newline-terminated)
// and as such we don't bother with the binary-mode headers.
class TokenHolder {
 public:
  typedef std::string T;

  TokenHolder() {}

  static bool Write(std::ostream &os, bool, const T &t) {  // ignore binary-mode
    KALDI_ASSERT(IsToken(t));
    os << t << '\n';
    return os.good();
  }

  void Clear() { t_.clear(); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    is >> t_;
    if (is.fail()) return false;
    char c;
    while (isspace(c = is.peek()) && c!= '\n') is.get();
    if (is.peek() != '\n') {
      KALDI_WARN << "TokenHolder::Read, expected newline, got char "
        << CharToString(is.peek())
        << ", at stream pos " << is.tellg();
      return false;
    }
    is.get();  // get '\n'
    return true;
  }


  // Since this is fundamentally a text format, read in text mode (would work
  // fine either way, but doing it this way will exercise more of the code).
  static bool IsReadInBinary() { return false; }

  T &Value() { return t_; }

  ~TokenHolder() { }

  void Swap(TokenHolder *other) {
    t_.swap(other->t_);
  }

  bool ExtractRange(const TokenHolder &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(TokenHolder);
  T t_;
};

// A Token is a nonempty, whitespace-free std::string.
// Class TokenVectorHolder is a Holder class for vectors of these.
class TokenVectorHolder {
 public:
  typedef std::vector<std::string> T;

  TokenVectorHolder() { }

  static bool Write(std::ostream &os, bool, const T &t) {  // ignore binary-mode
    for (std::vector<std::string>::const_iterator iter = t.begin();
         iter != t.end();
         ++iter) {
      KALDI_ASSERT(IsToken(*iter));  // make sure it's whitespace-free,
      // printable and nonempty.
      os << *iter << ' ';
    }
    os << '\n';
    return os.good();
  }

  void Clear() { t_.clear(); }


  // Reads into the holder.
  bool Read(std::istream &is) {
    t_.clear();

    // there is no binary/non-binary mode.

    std::string line;
    getline(is, line);  // this will discard the \n, if present.
    if (is.fail()) {
      KALDI_WARN << "BasicVectorHolder::Read, error reading line " << (is.eof()
                                                                       ? "[eof]" : "");
      return false;  // probably eof.  fail in any case.
    }
    const char *white_chars = " \t\n\r\f\v";
    SplitStringToVector(line, white_chars, true, &t_);  // true== omit
    // empty strings e.g. between spaces.
    return true;
  }

  // Read in text format since it's basically a text-mode thing.. doesn't really
  // matter, it would work either way since we ignore the extra '\r'.
  static bool IsReadInBinary() { return false; }

  T &Value() { return t_; }

  void Swap(TokenVectorHolder *other) {
    t_.swap(other->t_);
  }

  bool ExtractRange(const TokenVectorHolder &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(TokenVectorHolder);
  T t_;
};


class HtkMatrixHolder {
 public:
  typedef std::pair<Matrix<BaseFloat>, HtkHeader> T;

  HtkMatrixHolder() {}

  static bool Write(std::ostream &os, bool binary, const T &t) {
    if (!binary)
      KALDI_ERR << "Non-binary HTK-format write not supported.";
    bool ans = WriteHtk(os, t.first, t.second);
    if (!ans)
      KALDI_WARN << "Error detected writing HTK-format matrix.";
    return ans;
  }

  void Clear() { t_.first.Resize(0, 0); }

  // Reads into the holder.
  bool Read(std::istream &is) {
    bool ans = ReadHtk(is, &t_.first, &t_.second);
    if (!ans) {
      KALDI_WARN << "Error detected reading HTK-format matrix.";
      return false;
    }
    return ans;
  }

  // HTK-format matrices only read in binary.
  static bool IsReadInBinary() { return true; }

  T &Value() { return t_; }

  void Swap(HtkMatrixHolder *other) {
    t_.first.Swap(&(other->t_.first));
    std::swap(t_.second, other->t_.second);
  }

  bool ExtractRange(const HtkMatrixHolder &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }
  // Default destructor.
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(HtkMatrixHolder);
  T t_;
};

// SphinxMatrixHolder can be used to read and write feature files in
// CMU Sphinx format. 13-dimensional big-endian features are assumed.
// The ultimate reference is SphinxBase's source code (for example see
// feat_s2mfc_read() in src/libsphinxbase/feat/feat.c).
// We can't fully automate the detection of machine/feature file endianess
// mismatch here, because for this Sphinx relies on comparing the feature
// file's size with the number recorded in its header. We are working with
// streams, however(what happens if this is a Kaldi archive?). This should
// be no problem, because the usage help of Sphinx' "wave2feat" for example
// says that Sphinx features are always big endian.
// Note: the kFeatDim defaults to 13, see forward declaration in kaldi-holder.h
template<int kFeatDim> class SphinxMatrixHolder {
 public:
  typedef Matrix<BaseFloat> T;

  SphinxMatrixHolder() {}

  void Clear() { feats_.Resize(0, 0); }

  // Writes Sphinx-format features
  static bool Write(std::ostream &os, bool binary, const T &m) {
    if (!binary) {
      KALDI_WARN << "SphinxMatrixHolder can't write Sphinx features in text ";
      return false;
    }

    int32 size = m.NumRows() * m.NumCols();
    if (MachineIsLittleEndian())
      KALDI_SWAP4(size);
    // write the header
    os.write(reinterpret_cast<char*> (&size), sizeof(size));

    for (MatrixIndexT i = 0; i < m.NumRows(); i++) {
      std::vector<float32> tmp(m.NumCols());
      for (MatrixIndexT j = 0; j < m.NumCols(); j++) {
        tmp[j] = static_cast<float32>(m(i, j));
        if (MachineIsLittleEndian())
          KALDI_SWAP4(tmp[j]);
      }
      os.write(reinterpret_cast<char*>(&(tmp[0])),
               tmp.size() * 4);
    }
    return true;
  }

  // Reads the features into a Kaldi Matrix
  bool Read(std::istream &is) {
    int32 nmfcc;

    is.read(reinterpret_cast<char*> (&nmfcc), sizeof(nmfcc));
    if (MachineIsLittleEndian())
      KALDI_SWAP4(nmfcc);
    KALDI_VLOG(2) << "#feats: " << nmfcc;
    int32 nfvec = nmfcc / kFeatDim;
    if ((nmfcc % kFeatDim) != 0) {
      KALDI_WARN << "Sphinx feature count is inconsistent with vector length ";
      return false;
    }

    feats_.Resize(nfvec, kFeatDim);
    for (MatrixIndexT i = 0; i < feats_.NumRows(); i++) {
      if (sizeof(BaseFloat) == sizeof(float32)) {
        is.read(reinterpret_cast<char*> (feats_.RowData(i)),
                kFeatDim * sizeof(float32));
        if (!is.good()) {
          KALDI_WARN << "Unexpected error/EOF while reading Sphinx features ";
          return false;
        }
        if (MachineIsLittleEndian()) {
          for (MatrixIndexT j = 0; j < kFeatDim; j++)
            KALDI_SWAP4(feats_(i, j));
        }
      } else {  // KALDI_DOUBLEPRECISION=1
        float32 tmp[kFeatDim];
        is.read(reinterpret_cast<char*> (tmp), sizeof(tmp));
        if (!is.good()) {
          KALDI_WARN << "Unexpected error/EOF while reading Sphinx features ";
          return false;
        }
        for (MatrixIndexT j = 0; j < kFeatDim; j++) {
          if (MachineIsLittleEndian())
            KALDI_SWAP4(tmp[j]);
          feats_(i, j) = static_cast<BaseFloat>(tmp[j]);
        }
      }
    }

    return true;
  }

  // Only read in binary
  static bool IsReadInBinary() { return true; }

  T &Value() { return feats_; }

  void Swap(SphinxMatrixHolder *other) {
    feats_.Swap(&(other->feats_));
  }

  bool ExtractRange(const SphinxMatrixHolder &other,
                    const std::string &range) {
    KALDI_ERR << "ExtractRange is not defined for this type of holder.";
    return false;
  }

 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(SphinxMatrixHolder);
  T feats_;
};


/// @} end "addtogroup holders"

}  // end namespace kaldi



#endif  // KALDI_UTIL_KALDI_HOLDER_INL_H_
