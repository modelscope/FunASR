// util/edit-distance.h

// Copyright 2009-2011     Microsoft Corporation;  Haihua Xu

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


#ifndef KALDI_UTIL_EDIT_DISTANCE_H_
#define KALDI_UTIL_EDIT_DISTANCE_H_
#include <vector>
#include <set>
#include <algorithm>
#include <limits>
#include <cassert>
#include <utility>
#include "util/edit-distance-inl.h"
#include "base/kaldi-types.h"

namespace kaldi {

// Compute the edit-distance between two strings.
template<class T>
int32 LevenshteinEditDistance(const std::vector<T> &a,
                              const std::vector<T> &b);


// edit distance calculation with conventional method.
// note: noise word must be filtered out from the hypothesis and
// reference sequence
// before the following procedure conducted.
template<class T>
int32 LevenshteinEditDistance(const std::vector<T> &ref,
                              const std::vector<T> &hyp,
                              int32 *ins, int32 *del, int32 *sub);

// This version of the edit-distance computation outputs the alignment
// between the two.  This is a vector of pairs of (symbol a, symbol b).
// The epsilon symbol (eps_symbol) must not occur in sequences a or b.
// Where one aligned to no symbol in the other (insertion or deletion),
// epsilon will be the corresponding member of the pair.
// It returns the edit-distance between the two strings.

template<class T>
int32 LevenshteinAlignment(const std::vector<T> &a,
                           const std::vector<T> &b,
                           T eps_symbol,
                           std::vector<std::pair<T, T> > *output);

}  // end namespace kaldi

#endif  // KALDI_UTIL_EDIT_DISTANCE_H_
