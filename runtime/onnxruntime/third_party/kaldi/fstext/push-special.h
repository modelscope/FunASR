// fstext/push-special.h

// Copyright 2012-2015  Johns Hopkins University (author: Daniel Povey)

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

#ifndef KALDI_FSTEXT_PUSH_SPECIAL_H_
#define KALDI_FSTEXT_PUSH_SPECIAL_H_

#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include "util/const-integer-set.h"

namespace fst {

/*
  This function does weight-pushing, in the log semiring,
  but in a special way, such that any "leftover weight" after pushing
  gets distributed evenly along the FST, and doesn't end up either
  at the start or at the end.  Basically it pushes the weights such
  that the total weight of each state (i.e. the sum of the arc
  probabilities plus the final-prob) is the same for all states.
*/
void PushSpecial(VectorFst<StdArc> *fst,
                 float delta = kDelta);

}

#endif
