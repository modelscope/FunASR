// fstext/epsilon-property.h

// Copyright 2014    Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_FSTEXT_EPSILON_PROPERTY_H_
#define KALDI_FSTEXT_EPSILON_PROPERTY_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

namespace fst {

enum {
  kStateHasEpsilonArcsEntering = 0x1,
  kStateHasNonEpsilonArcsEntering = 0x2,
  kStateHasEpsilonArcsLeaving = 0x4,
  kStateHasNonEpsilonArcsLeaving = 0x8
}; // use 'char' for this enum.

/// This function will set epsilon_info to have size equal to the
/// NumStates() of the FST, containing a logical-or of the enum
/// values kStateHasEpsilonArcsEntering, kStateHasNonEpsilonArcsEntering,
/// kStateHasEpsilonArcsLeaving, and kStateHasNonEpsilonArcsLeaving.
/// The meaning should be obvious.  Note: an epsilon arc is defined
/// as an arc where ilabel == olabel == 0.
template<class Arc>
void ComputeStateInfo(const VectorFst<Arc> &fst,
                      std::vector<char> *epsilon_info);

/// This function modifies the fst (while maintaining equivalence) in such a way
/// that, after the modification, all states of the FST which have epsilon-arcs
/// entering them, have no non-epsilon arcs entering them, and all states which
/// have epsilon-arcs leaving them, have no non-epsilon arcs leaving them.  It does
/// this by creating extra states and adding extra epsilon transitions.  An epsilon
/// arc is defined as an arc where both the ilabel and the olabel are epsilons.
/// This function may fail with KALDI_ASSERT for certain cyclic FSTs, but is safe
/// in the acyclic case.
template<class Arc>
void EnsureEpsilonProperty(VectorFst<Arc> *fst);

} // end namespace fst


#include "fstext/epsilon-property-inl.h"

#endif
