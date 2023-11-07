// fstext/pre-determinize.h

// Copyright 2009-2011  Microsoft Corporation

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

#ifndef KALDI_FSTEXT_PRE_DETERMINIZE_H_
#define KALDI_FSTEXT_PRE_DETERMINIZE_H_
#include <algorithm>
#include <map>
#include <set>
#include <vector>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "base/kaldi-common.h"

namespace fst {

/* PreDeterminize inserts extra symbols on the input side of an FST as necessary to
   ensure that, after epsilon removal, it will be compactly determinizable by the
   determinize* algorithm.  By compactly determinizable we mean that
   no original FST state is represented in more than one determinized state).

   Caution: this code is now only used in testing.

   The new symbols start from the value "first_new_symbol", which should be
   higher than the largest-numbered symbol currently in the FST.  The new
   symbols added are put in the array syms_out, which should be empty at start.
*/

template<class Arc, class Int>
void PreDeterminize(MutableFst<Arc> *fst,
                    typename Arc::Label first_new_symbol,
                    std::vector<Int> *syms_out);


/* CreateNewSymbols is a helper function used inside PreDeterminize, and is also useful
   when you need to add a number of extra symbols to a different vocabulary from the one
   modified by PreDeterminize. */

template<class Label>
void CreateNewSymbols(SymbolTable *inputSymTable, int nSym,
                      std::string prefix, std::vector<Label> *syms_out);

/** AddSelfLoops is a function you will probably want to use alongside PreDeterminize,
    to add self-loops to any FSTs that you compose on the left hand side of the one
    modified by PreDeterminize.

    This function inserts loops with "special symbols" [e.g. \#0, \#1] into an FST.
    This is done at each final state and each state with non-epsilon output symbols on
    at least one arc out of it.  This is to ensure that these symbols, when inserted into
    the input side of an FST we will compose with on the right, can "pass through" this
    FST.

    At input, isyms and osyms must be vectors of the same size n, corresponding
    to symbols that currently do not exist in 'fst'.  For each state in n that has
    non-epsilon symbols on the output side of arcs leaving it, or which is a final state,
    this function inserts n self-loops with unit weight and one of the n pairs
    of symbols on its input and output.
*/
template<class Arc>
void AddSelfLoops(MutableFst<Arc> *fst, std::vector<typename Arc::Label> &isyms,
                     std::vector<typename Arc::Label> &osyms);


/* DeleteSymbols replaces any instances of symbols in the vector symsIn, appearing
   on the input side, with epsilon. */
/* It returns the number of instances of symbols deleted. */
template<class Arc>
int64 DeleteISymbols(MutableFst<Arc> *fst, std::vector<typename Arc::Label> symsIn);

/* CreateSuperFinal takes an FST, and creates an equivalent FST with a single final
   state with no transitions out and unit final weight, by inserting epsilon transitions
   as necessary. */
template<class Arc>
typename Arc::StateId CreateSuperFinal(MutableFst<Arc> *fst);


} // end namespace fst

#include "fstext/pre-determinize-inl.h"

#endif
