// fstext/prune-special.h

// Copyright 2014  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_FSTEXT_PRUNE_SPECIAL_H_
#define KALDI_FSTEXT_PRUNE_SPECIAL_H_

#include "fst/fstlib.h"
#include "fstext/lattice-weight.h"
#include "fstext/factor.h"

namespace fst {



/**
   The function PruneSpecial is like the standard OpenFst function "prune",
   except it does not expand the entire "ifst"- this is useful for cases where
   ifst is an on-demand FST such as a ComposeFst and we don't want to visit
   it all.  It supports pruning either to a specified beam (if beam is
   not One()), or to a specified max_states (if max_states is > 0).  One of the
   two must be specified.

   Requirements:
     - Costs must be non-negative (equivalently, weights must not be greater than One()).
     - There must be a Compare(a, b) function that compares two weights and returns (-1,0,1)
       if (a<b, a=b, a>b).  We define this in Kaldi, for TropicalWeight, LogWeight (I think),
       and LatticeWeight... also CompactLatticeWeight, but we doubt that will be used here;
       better to use PruneCompactLattice().
 */

template<class Arc>
void PruneSpecial(const Fst<Arc> &ifst,
                  VectorFst<Arc> *ofst,
                  typename Arc::Weight beam,
                  size_t max_states =  0);



} // end namespace fst

#include "fstext/prune-special-inl.h"

#endif  // KALDI_FSTEXT_PRUNE_SPECIAL_H_
