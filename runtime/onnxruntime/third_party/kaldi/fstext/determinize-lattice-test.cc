// fstext/determinize-lattice-test.cc

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

#include "fstext/determinize-lattice.h"
#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"

namespace fst {
using std::vector;
using std::cout;

void TestLatticeStringRepository() {
  typedef int32 IntType;

  LatticeStringRepository<IntType> sr;
  typedef LatticeStringRepository<IntType>::Entry Entry;

  for(int i = 0; i < 100; i++) {
    int len = kaldi::Rand() % 5;
    vector<IntType> str(len), str2(kaldi::Rand() % 4);
    const Entry *e = NULL;
    for(int i = 0; i < len; i++) {
      str[i] = kaldi::Rand() % 5;
      e = sr.Successor(e, str[i]);
    }
    sr.ConvertToVector(e, &str2);
    assert(str == str2);

    int len2 = kaldi::Rand() % 5;
    str2.resize(len2);
    const Entry *f = sr.EmptyString(); // NULL
    for(int i = 0; i < len2; i++) {
      str2[i] = kaldi::Rand() % 5;
      f = sr.Successor(f, str2[i]);
    }
    vector<IntType> prefix, prefix2(kaldi::Rand() % 10),
        prefix3;
    for(int i = 0; i < len && i < len2; i++) {
      if (str[i] == str2[i]) prefix.push_back(str[i]);
      else break;
    }
    const Entry *g = sr.CommonPrefix(e, f);
    sr.ConvertToVector(g, &prefix2);
    sr.ConvertToVector(e, &prefix3);
    sr.ReduceToCommonPrefix(f, &prefix3);
    assert(prefix == prefix2);
    assert(prefix == prefix3);
    assert(sr.IsPrefixOf(g, e));
    assert(sr.IsPrefixOf(g, f));
    if (str.size() > prefix.size())
      assert(!sr.IsPrefixOf(e, g));
  }
}


// test that determinization proceeds correctly on general
// FSTs (not guaranteed determinzable, but we use the
// max-states option to stop it getting out of control).
template<class Arc> void TestDeterminizeLattice() {
  typedef typename Arc::Weight Weight;
  typedef int32 Int;
  typedef ArcTpl<CompactLatticeWeightTpl<Weight, Int> > CompactArc;

  for(int i = 0; i < 100; i++) {
    RandFstOptions opts;
    opts.n_states = 4;
    opts.n_arcs = 10;
    opts.n_final = 2;
    opts.allow_empty = false;
    opts.weight_multiplier = 0.5; // impt for the randomly generated weights
    // to be exactly representable in float,
    // or this test fails because numerical differences can cause symmetry in
    // weights to be broken, which causes the wrong path to be chosen as far
    // as the string part is concerned.

    VectorFst<Arc> *fst = RandFst<Arc>();
    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> det_fst;
    try {
      DeterminizeLatticeOptions lat_opts;
      lat_opts.max_mem = 100;

      if (!DeterminizeLattice<TropicalWeight, int32>(*fst, &det_fst, lat_opts, NULL))
        throw std::runtime_error("could not determinize");
      std::cout << "FST after lattice-determinizing is:\n";
      {
        FstPrinter<Arc> fstprinter(det_fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
      assert(det_fst.Properties(kIDeterministic, true) & kIDeterministic);
      // OK, now determinize it a different way and check equivalence.
      // [note: it's not normal determinization, it's taking the best path
      // for any input-symbol sequence....
      VectorFst<CompactArc> compact_fst, compact_det_fst;
      ConvertLattice<Weight, Int>(*fst, &compact_fst, false);
      std::cout << "Compact FST is:\n";
      {
        FstPrinter<CompactArc> fstprinter(compact_fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
      if (kaldi::Rand() % 2 == 1)
        ConvertLattice<Weight, Int>(det_fst, &compact_det_fst, false);
      else
        if (!DeterminizeLattice<TropicalWeight, int32>(*fst, &compact_det_fst, lat_opts, NULL))
          throw std::runtime_error("could not determinize");

      std::cout << "Compact version of determinized FST is:\n";
      {
        FstPrinter<CompactArc> fstprinter(compact_det_fst, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }

      assert(RandEquivalent(compact_det_fst, compact_fst, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length, max*/));
    } catch (...) {
      std::cout << "Failed to lattice-determinize this FST (probably not determinizable)\n";
    }
    delete fst;
  }
}

// test that determinization proceeds correctly on acyclic FSTs
// (guaranteed determinizable in this sense).
template<class Arc> void TestDeterminizeLattice2() {
  RandFstOptions opts;
  opts.acyclic = true;
  for(int i = 0; i < 100; i++) {
    VectorFst<Arc> *fst = RandFst<Arc>(opts);
    std::cout << "FST before lattice-determinizing is:\n";
    {
      FstPrinter<Arc> fstprinter(*fst, NULL, NULL, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }
    VectorFst<Arc> ofst;
    DeterminizeLattice<TropicalWeight, int32>(*fst, &ofst);
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
  TestLatticeStringRepository();
  TestDeterminizeLattice<StdArc>();
  TestDeterminizeLattice2<StdArc>();
  std::cout << "Tests succeeded\n";
}
