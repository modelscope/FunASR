// decoder/decodable-mapped.h

// Copyright 2009-2011  Saarland University;  Microsoft Corporation;
//                      Lukas Burget

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

#ifndef KALDI_DECODER_DECODABLE_MAPPED_H_
#define KALDI_DECODER_DECODABLE_MAPPED_H_

#include <vector>

#include "base/kaldi-common.h"
#include "itf/decodable-itf.h"

namespace kaldi {

// The DecodableMapped object is initialized by a normal decodable object,
// and a vector that maps indices.  The "pdf index" into this decodable object
// is the index into the vector, and the value it finds there is used
// to index into the base decodable object.

class DecodableMapped: public DecodableInterface {
 public:
  DecodableMapped(const std::vector<int32> &index_map, DecodableInterface *d):
      index_map_(index_map), decodable_(d) { }

  // Note, frames are numbered from zero.  But state_index is numbered
  // from one (this routine is called by FSTs).
  virtual BaseFloat LogLikelihood(int32 frame, int32 state_index) {
    KALDI_ASSERT(static_cast<size_t>(state_index) < index_map_.size());
    return decodable_->LogLikelihood(frame, index_map_[state_index]);
  }

  // note: indices are assumed to be numbered from one, so
  // NumIndices() will be the same as the largest index.
  virtual int32 NumIndices() const { return static_cast<int32>(index_map_.size()) - 1; }

  virtual bool IsLastFrame(int32 frame) const {
    // We require all the decodables have the same #frames.  We don't check this though.
    return decodable_->IsLastFrame(frame);
  }

 private:
  std::vector<int32> index_map_;
  DecodableInterface *decodable_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableMapped);
};



}  // namespace kaldi

#endif  // KALDI_DECODER_DECODABLE_MAPPED_H_

