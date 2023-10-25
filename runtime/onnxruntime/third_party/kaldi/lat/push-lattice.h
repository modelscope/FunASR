// lat/push-lattice.h

// Copyright 2013        Johns Hopkins University (Author: Daniel Povey)
//           2014        Guoguo Chen

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


#ifndef KALDI_LAT_PUSH_LATTICE_H_
#define KALDI_LAT_PUSH_LATTICE_H_

#include <vector>
#include <map>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"

namespace fst {


/// This function pushes the transition-ids as far towards the start as they
/// will go.  It can be useful prior to lattice-align-words (for non-linear
/// lattices).  We can't use the generic OpenFst "push" function because
/// it uses the sum as the divisor, which is not appropriate in this case
/// (a+b generally won't divide a or b in this semiring).
/// It returns true on success, false if it failed due to TopSort failing,
/// which should never happen, but we handle it gracefully by just leaving the
/// lattice the same.
/// This function used to be called just PushCompactLattice.
template<class Weight, class IntType>
bool PushCompactLatticeStrings(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat);

/// This function pushes the weights in the CompactLattice so that all states
/// except possibly the start state, have Weight components (of type
/// LatticeWeight) that "sum to one" in the LatticeWeight (i.e. interpreting the
/// weights as negated log-probs).  It returns true on success, false if it
/// failed due to TopSort failing, which should never happen, but we handle it
/// gracefully by just leaving the lattice the same.
template<class Weight, class IntType>
bool PushCompactLatticeWeights(
    MutableFst<ArcTpl<CompactLatticeWeightTpl<Weight, IntType> > > *clat);

}  // namespace fst

#endif  // KALDI_LAT_PUSH_LATTICE_H_
