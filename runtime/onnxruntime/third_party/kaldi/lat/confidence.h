// lat/confidence.h

// Copyright 2013  Johns Hopkins University (Author: Daniel Povey)

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


#ifndef KALDI_LAT_CONFIDENCE_H_
#define KALDI_LAT_CONFIDENCE_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/// Caution: this function is not the only way to get confidences in Kaldi.
/// This only gives you sentence-level (utterance-level) confidence.  You can
/// get word-by-word confidence within a sentence, along with Minimum Bayes Risk
/// decoding, by looking at sausages.h.
/// Caution: confidences estimated using this type of method are not very
/// accurate.
/// This function will return the difference between the best path in clat and
/// the second-best path in clat (a positive number), or zero if clat was
/// equivalent to the empty FST (no successful paths), or infinity if there
/// was only one path in "clat".  It will output to "num_paths" (if non-NULL)
/// a number n = 0, 1 or 2 saying how many n-best paths (up to two) were found.
/// If n >= 1 it outputs to "best_sentence" (if non-NULL) the best word-sequence;
/// if n == 2 it outputs to "second_best_sentence" (if non-NULL) the second best
/// word-sequence (this may be useful for testing whether the two best word
/// sequences are somehow equivalent for the task at hand).  If you need more
/// information than this or want to look deeper inside the n-best list, then
/// look at the implementation of this function.
/// This function requires that distinct paths in "lat" have distinct word
/// sequences; this will automatically be the case if you generated "clat"
/// in any normal way, such as from a decoder, because a deterministic FST
/// has this property.
/// This function assumes that any acoustic scaling you want to apply,
/// has already been applied.
BaseFloat SentenceLevelConfidence(const CompactLattice &clat,
                                  int32 *num_paths,
                                  std::vector<int32> *best_sentence,
                                  std::vector<int32> *second_best_sentence);


/// This version of SentenceLevelConfidence takes as input a state-level lattice.
/// It needs to determinize it first, but it does so in a "smart" way that only generates
/// about as many output paths as it needs.
BaseFloat SentenceLevelConfidence(const Lattice &lat,
                                  int32 *num_paths,
                                  std::vector<int32> *best_sentence,
                                  std::vector<int32> *second_best_sentence);





}  // namespace kaldi

#endif  // KALDI_LAT_CONFIDENCE_H_
