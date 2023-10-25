// util/kaldi-pipebuf.h

// Copyright 2009-2011  Ondrej Glembek

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


/** @file kaldi-pipebuf.h
 *  This is an Kaldi C++ Library header.
 */

#ifndef KALDI_UTIL_KALDI_PIPEBUF_H_
#define KALDI_UTIL_KALDI_PIPEBUF_H_

#include<string>
#if !defined(_LIBCPP_VERSION)  // libc++
#include <fstream>
#else
#include "util/basic-filebuf.h"
#endif

namespace kaldi {
// This class provides a way to initialize a filebuf with a FILE* pointer
// directly; it will not close the file pointer when it is deleted.
// The C++ standard does not allow implementations of C++ to provide
// this constructor within basic_filebuf, which makes it hard to deal
// with pipes using completely native C++.  This is a workaround

#ifdef _MSC_VER
#elif defined(_LIBCPP_VERSION)  // libc++
template<class CharType, class Traits = std::char_traits<CharType> >
class basic_pipebuf : public basic_filebuf<CharType, Traits> {
 public:
  typedef basic_pipebuf<CharType, Traits>   ThisType;

 public:
  basic_pipebuf(FILE *fptr, std::ios_base::openmode mode)
      : basic_filebuf<CharType, Traits>() {
    this->open(fptr, mode);
    if (!this->is_open()) {
      KALDI_WARN << "Error initializing pipebuf";  // probably indicates
      // code error, if the fptr was good.
      return;
    }
  }
};  // class basic_pipebuf
#else
template<class CharType, class Traits = std::char_traits<CharType> >
class basic_pipebuf : public std::basic_filebuf<CharType, Traits> {
 public:
  typedef basic_pipebuf<CharType, Traits>   ThisType;

 public:
  basic_pipebuf(FILE *fptr, std::ios_base::openmode mode)
      : std::basic_filebuf<CharType, Traits>() {
    this->_M_file.sys_open(fptr, mode);
    if (!this->is_open()) {
      KALDI_WARN << "Error initializing pipebuf";  // probably indicates
      // code error, if the fptr was good.
      return;
    }
    this->_M_mode = mode;
    this->_M_buf_size = BUFSIZ;
    this->_M_allocate_internal_buffer();
    this->_M_reading = false;
    this->_M_writing = false;
    this->_M_set_buffer(-1);
  }
};  // class basic_pipebuf
#endif  // _MSC_VER

}  // namespace kaldi

#endif  // KALDI_UTIL_KALDI_PIPEBUF_H_
