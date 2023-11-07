// fstext/prune-special-test.cc

// Copyright 2014  Guoguo Chen

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


#include "fstext/prune-special.h"
#include "fstext/rand-fst.h"
#include "fstext/fstext-utils.h"

namespace fst {

static void TestPruneSpecial() {
  typedef StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  RandFstOptions opts;
  opts.acyclic = false;
  VectorFst<Arc> *ifst = RandFst<StdArc>(opts);

  float beam = 0.55;

  {
    FstPrinter<Arc> fstprinter(*ifst, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
    std::cout << std::endl;
  }

  // Do the special pruning.
  VectorFst<Arc> ofst1;
  PruneSpecial<StdArc>(*ifst, &ofst1, beam);
  {
    FstPrinter<Arc> fstprinter(ofst1, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
    std::cout << std::endl;
  }

  // Do the normal pruning.
  VectorFst<Arc> ofst2;
  Prune(*ifst, &ofst2, beam);
  {
    FstPrinter<Arc> fstprinter(ofst2, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
    std::cout << std::endl;
  }

  KALDI_ASSERT(RandEquivalent(ofst1, ofst2,
                              5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/,
                              100/*path length-- max?*/));

  delete ifst;
}


} // namespace fst

int main() {
  kaldi::g_kaldi_verbose_level = 4;
  using namespace fst;
  for (int i = 0; i < 25; i++) {
    TestPruneSpecial();
  }
}
