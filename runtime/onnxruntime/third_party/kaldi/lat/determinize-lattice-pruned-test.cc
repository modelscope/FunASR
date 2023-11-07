// lat/determinize-lattice-pruned-test.cc

// Copyright 2009-2012  Microsoft Corporation
//           2012-2013  Johns Hopkins University (Author: Daniel Povey)

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

#include "lat/determinize-lattice-pruned.h"
#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"

namespace fst {
// Caution: these tests are not as generic as you might think from all the
// templates in the code.  They are basically only valid for LatticeArc.
// This is partly due to the fact that certain templates need to be instantiated
// in other .cc files in this directory.

// test that determinization proceeds correctly on general
// FSTs (not guaranteed determinzable, but we use the
// max-states option to stop it getting out of control).
template<class Arc> void TestDeterminizeLatticePruned() {
  typedef kaldi::int32 Int;
  typedef typename Arc::Weight Weight;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;

  for(int i = 0; i < 100; i++) {
    RandFstOptions opts;
    opts.n_states = 4;
    opts.n_arcs = 10;
    opts.n_final = 2;
    opts.allow_empty = false;
    opts.weight_multiplier = 0.5; // impt for the randomly generated weights
    opts.acyclic = true;
    // to be exactly representable in float,
    // or this test fails because numerical differences can cause symmetry in
    // weights to be broken, which causes the wrong path to be chosen as far
    // as the string part is concerned.

    VectorFst<Arc> *fst = RandPairFst<Arc>(opts);

    bool sorted = TopSort(fst);
    KALDI_ASSERT(sorted);

    ILabelCompare<Arc> ilabel_comp;
    if (kaldi::Rand() % 2 == 0)
      ArcSort(fst, ilabel_comp);

    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> det_fst;
    try {
      DeterminizeLatticePrunedOptions lat_opts;
      lat_opts.max_mem = ((kaldi::Rand() % 2 == 0) ? 100 : 1000);
      lat_opts.max_states = ((kaldi::Rand() % 2 == 0) ? -1 : 20);
      lat_opts.max_arcs = ((kaldi::Rand() % 2 == 0) ? -1 : 30);
      bool ans = DeterminizeLatticePruned<Weight>(*fst, 10.0, &det_fst, lat_opts);

      std::cout << "FST after lattice-determinizing is:\n";
      {
        FstPrinter<Arc> fstprinter(det_fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
      KALDI_ASSERT(det_fst.Properties(kIDeterministic, true) & kIDeterministic);
      // OK, now determinize it a different way and check equivalence.
      // [note: it's not normal determinization, it's taking the best path
      // for any input-symbol sequence....


      VectorFst<Arc> pruned_fst(*fst);
      if (pruned_fst.NumStates() != 0)
        kaldi::PruneLattice(10.0, &pruned_fst);

      VectorFst<CompactArc> compact_pruned_fst, compact_pruned_det_fst;
      ConvertLattice<Weight, Int>(pruned_fst, &compact_pruned_fst, false);
      std::cout << "Compact pruned FST is:\n";
      {
        FstPrinter<CompactArc> fstprinter(compact_pruned_fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
      ConvertLattice<Weight, Int>(det_fst, &compact_pruned_det_fst, false);

      std::cout << "Compact version of determinized FST is:\n";
      {
        FstPrinter<CompactArc> fstprinter(compact_pruned_det_fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }

      if (ans)
        KALDI_ASSERT(RandEquivalent(compact_pruned_det_fst, compact_pruned_fst, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length, max*/));
    } catch (...) {
      std::cout << "Failed to lattice-determinize this FST (probably not determinizable)\n";
    }
    delete fst;
  }
}

// test that determinization proceeds without crash on acyclic FSTs
// (guaranteed determinizable in this sense).
template<class Arc> void TestDeterminizeLatticePruned2() {
  typedef typename Arc::Weight Weight;
  RandFstOptions opts;
  opts.acyclic = true;
  for(int i = 0; i < 100; i++) {
    VectorFst<Arc> *fst = RandPairFst<Arc>(opts);
    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> ofst;
    DeterminizeLatticePruned<Weight>(*fst, 10.0, &ofst);
    std::cout << "FST after lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(ofst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    delete fst;
  }
}


} // end namespace fst

int main() {
  using namespace fst;
  TestDeterminizeLatticePruned<kaldi::LatticeArc>();
  TestDeterminizeLatticePruned2<kaldi::LatticeArc>();
  std::cout << "Tests succeeded\n";
}
