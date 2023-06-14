// util/const-integer-set-inl.h

// Copyright 2009-2011     Microsoft Corporation

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


#ifndef KALDI_UTIL_CONST_INTEGER_SET_INL_H_
#define KALDI_UTIL_CONST_INTEGER_SET_INL_H_

// Do not include this file directly.  It is included by const-integer-set.h


namespace kaldi {

template<class I>
void ConstIntegerSet<I>::InitInternal() {
  KALDI_ASSERT_IS_INTEGER_TYPE(I);
  quick_set_.clear();  // just in case we previously had data.
  if (slow_set_.size() == 0) {
    lowest_member_=(I) 1;
    highest_member_=(I) 0;
    contiguous_ = false;
    quick_ = false;
  } else {
    lowest_member_ = slow_set_.front();
    highest_member_ = slow_set_.back();
    size_t range = highest_member_ + 1 - lowest_member_;
    if (range == slow_set_.size()) {
      contiguous_ = true;
      quick_= false;
    } else {
      contiguous_ = false;
      // If it would be more compact to store as bool
      if (range < slow_set_.size() * 8 * sizeof(I)) {
        // (assuming 1 bit per element)...
        quick_set_.resize(range, false);
        for (size_t i = 0;i < slow_set_.size();i++)
          quick_set_[slow_set_[i] - lowest_member_] = true;
        quick_ = true;
      } else {
        quick_ = false;
      }
    }
  }
}

template<class I>
int ConstIntegerSet<I>::count(I i) const {
  if (i < lowest_member_ || i > highest_member_) {
    return 0;
  } else {
    if (contiguous_) return true;
    if (quick_) {
      return (quick_set_[i-lowest_member_] ? 1 : 0);
    } else {
      bool ans = std::binary_search(slow_set_.begin(), slow_set_.end(), i);
      return (ans ? 1 : 0);
    }
  }
}

template<class I>
void ConstIntegerSet<I>::Write(std::ostream &os, bool binary) const {
  WriteIntegerVector(os, binary, slow_set_);
}

template<class I>
void ConstIntegerSet<I>::Read(std::istream &is, bool binary) {
  ReadIntegerVector(is, binary, &slow_set_);
  InitInternal();
}



}  // end namespace kaldi

#endif  // KALDI_UTIL_CONST_INTEGER_SET_INL_H_
