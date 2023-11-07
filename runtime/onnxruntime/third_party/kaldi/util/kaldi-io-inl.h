// util/kaldi-io-inl.h

// Copyright 2009-2011 Microsoft Corporation

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
#ifndef KALDI_UTIL_KALDI_IO_INL_H_
#define KALDI_UTIL_KALDI_IO_INL_H_

#include<string>

namespace kaldi {

bool Input::Open(const std::string &rxfilename, bool *binary) {
  return OpenInternal(rxfilename, true, binary);
}

bool Input::OpenTextMode(const std::string &rxfilename) {
  return OpenInternal(rxfilename, false, NULL);
}

bool Input::IsOpen() {
  return impl_ != NULL;
}

bool Output::IsOpen() {
  return impl_ != NULL;
}


}  // end namespace kaldi.


#endif  // KALDI_UTIL_KALDI_IO_INL_H_
