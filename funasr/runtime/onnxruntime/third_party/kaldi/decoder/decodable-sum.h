// decoder/decodable-sum.h

// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Lukas Burget, Pawel Swietojanski

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

#ifndef KALDI_DECODER_DECODABLE_SUM_H_
#define KALDI_DECODER_DECODABLE_SUM_H_

#include <vector>
#include <utility>

#include "base/kaldi-common.h"
#include "itf/decodable-itf.h"

namespace kaldi {

// The DecodableSum object is a very simple object that just sums
// scores over a number of Decodable objects.  They must all have
// the same dimensions.

class DecodableSum: public DecodableInterface {
 public:
  // Does not take ownership of pointers!  They are just
  // pointers because they are non-const.
  DecodableSum(DecodableInterface *d1, BaseFloat w1,
               DecodableInterface *d2, BaseFloat w2) {
    decodables_.push_back(std::make_pair(d1, w1));
    decodables_.push_back(std::make_pair(d2, w2));
    CheckSizes();
  }

  // Does not take ownership of pointers!
  DecodableSum(
      const std::vector<std::pair<DecodableInterface*, BaseFloat> > &decodables) :
      decodables_(decodables) { CheckSizes(); }

  void CheckSizes() const {
    KALDI_ASSERT(decodables_.size() >= 1
                 && decodables_[0].first != NULL);
    for (size_t i = 1; i < decodables_.size(); i++)
      KALDI_ASSERT(decodables_[i].first != NULL &&
                   decodables_[i].first->NumIndices() ==
                   decodables_[0].first->NumIndices());
  }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    BaseFloat sum = 0.0;
    // int32 i=1;
    for (std::vector<std::pair<DecodableInterface*, BaseFloat> >::iterator iter = decodables_.begin();
         iter != decodables_.end();
         ++iter) {
      sum += iter->first->LogLikelihood(frame, state_index) * iter->second;
     }
    return sum;
  }

  virtual int32 NumIndices() const { return decodables_[0].first->NumIndices(); }

  virtual bool IsLastFrame(int32 frame) const {
    // We require all the decodables have the same #frames.  We don't check this though.
    return decodables_[0].first->IsLastFrame(frame);
  }

 private:
  std::vector<std::pair<DecodableInterface*, BaseFloat> > decodables_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableSum);
};

class DecodableSumScaled : public DecodableSum {
 public:
  DecodableSumScaled(DecodableInterface *d1, BaseFloat w1,
                     DecodableInterface *d2, BaseFloat w2,
                     BaseFloat scale)
    : DecodableSum(d1, w1, d2, w2), scale_(scale) {}

  DecodableSumScaled(const std::vector<std::pair<DecodableInterface*, BaseFloat> > &decodables,
                     BaseFloat scale)
    : DecodableSum(decodables), scale_(scale) {}

  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    return scale_ * DecodableSum::LogLikelihood(frame, state_index);
  }

 private:
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableSumScaled);
};

}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_SUM_H_

