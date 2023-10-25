// fstext/determinize-star.h

// Copyright 2009-2011  Microsoft Corporation
//                2014  Guoguo Chen
//                2015  Hainan Xu

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

#ifndef KALDI_FSTEXT_DETERMINIZE_STAR_H_
#define KALDI_FSTEXT_DETERMINIZE_STAR_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <stdexcept> // this algorithm uses exceptions

namespace fst {

/// \addtogroup fst_extensions
///  @{


// For example of usage, see test-determinize-star.cc

/*
   DeterminizeStar implements determinization with epsilon removal, which we
   distinguish with a star.

   We define a determinized* FST as one in which no state has more than one
   transition with the same input-label.  Epsilon input labels are not allowed
   except starting from states that have exactly one arc exiting them (and are
   not final).  [In the normal definition of determinized, epsilon-input labels
   are not allowed at all, whereas in Mohri's definition, epsilons are treated
   as ordinary symbols].  The determinized* definition is intended to simulate
   the effect of allowing strings of output symbols at each state.

   The algorithm implemented here takes an Fst<Arc>, and a pointer to a
   MutableFst<Arc> where it puts its output.  The weight type is assumed to be a
   float-weight.  It does epsilon removal and determinization.
   This algorithm may fail if the input has epsilon cycles under
   certain circumstances (i.e. the semiring is non-idempotent, e.g. the log
   semiring, or there are negative cost epsilon cycles).

   This implementation is much less fancy than the one in fst/determinize.h, and
   does not have an "on-demand" version.

   The algorithm is a fairly normal determinization algorithm.  We keep in
   memory the subsets of states, together with their leftover strings and their
   weights.  The only difference is we detect input epsilon transitions and
   treat them "specially".
*/


// This algorithm will be slightly faster if you sort the input fst on input label.

/**
    This function implements the normal version of DeterminizeStar, in which the
    output strings are represented using sequences of arcs, where all but the
    first one has an epsilon on the input side.  The debug_ptr argument is an
    optional pointer to a bool that, if it becomes true while the algorithm is
    executing, the algorithm will print a traceback and terminate (used in
    fstdeterminizestar.cc debug non-terminating determinization).
    If max_states is positive, it will stop determinization and throw an
    exception as soon as the max-states is reached. This can be useful in test.
    If allow_partial is true, the algorithm will output partial results when the
    specified max_states is reached (when larger than zero), instead of throwing
    out an error.

    Caution, the return status is un-intuitive: this function will return false if
    determinization completed normally, and true if it was stopped early by
    reaching the 'max-states' limit, and a partial FST was generated.
*/
template<class F>
bool DeterminizeStar(F &ifst, MutableFst<typename F::Arc> *ofst,
                     float delta = kDelta,
                     bool *debug_ptr = NULL,
                     int max_states = -1,
                     bool allow_partial = false);



/*  This is a version of DeterminizeStar with a slightly more "natural" output format,
    where the output sequences are encoded using the GallicArc (i.e. the output symbols
    are strings.
    If max_states is positive, it will stop determinization and throw an
    exception as soon as the max-states is reached.  This can be useful in test.
    If allow_partial is true, the algorithm will output partial results when the
    specified max_states is reached (when larger than zero), instead of throwing
    out an error.

    Caution, the return status is un-intuitive: this function will return false if
    determinization completed normally, and true if it was stopped early by
    reaching the 'max-states' limit, and a partial FST was generated.
*/
template<class F>
bool DeterminizeStar(F &ifst, MutableFst<GallicArc<typename F::Arc> > *ofst,
                     float delta = kDelta, bool *debug_ptr = NULL,
                     int max_states = -1,
                     bool allow_partial = false);


/// @} end "addtogroup fst_extensions"

} // end namespace fst

#include "fstext/determinize-star-inl.h"

#endif
