// fstext/trivial-factor-weight-test.cc

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

#include "base/kaldi-math.h"
#include "fstext/pre-determinize.h"
#include "fstext/determinize-star.h"
#include "fstext/trivial-factor-weight.h"
#include "fstext/fst-test-utils.h"
// Just check that it compiles, for now.

namespace fst
{
  using std::cout;
  using std::vector;

// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestFactor() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();
  int n_syms = 2 + kaldi::Rand() % 5, n_states = 3 + kaldi::Rand() % 10, n_arcs = 5 + kaldi::Rand() % 30, n_final = 1 + kaldi::Rand()%3;  // Up to 2 unique symbols.
  cout << "Testing pre-determinize with "<<n_syms<<" symbols, "<<n_states<<" states and "<<n_arcs<<" arcs and "<<n_final<<" final states.\n";
  SymbolTable *sptr = NULL;

  vector<Label> all_syms;  // including epsilon.
  // Put symbols in the symbol table from 1..n_syms-1.
  for (size_t i = 0;i < (size_t)n_syms;i++)
    all_syms.push_back(i);

  // Create states.
  vector<StateId> all_states;
  for (size_t i = 0;i < (size_t)n_states;i++) {
    StateId this_state = fst->AddState();
    if (i == 0) fst->SetStart(i);
    all_states.push_back(this_state);
  }
  // Set final states.
  for (size_t j = 0;j < (size_t)n_final;j++) {
    StateId id = all_states[kaldi::Rand() % n_states];
    Weight weight = (Weight)(0.33*(kaldi::Rand() % 5) );
    printf("calling SetFinal with %d and %f\n", id, weight.Value());
    fst->SetFinal(id, weight);
  }
  // Create arcs.
  for (size_t i = 0;i < (size_t)n_arcs;i++) {
    Arc a;
    a.nextstate = all_states[kaldi::Rand() % n_states];
    a.ilabel = all_syms[kaldi::Rand() % n_syms];
    a.olabel = all_syms[kaldi::Rand() % n_syms];  // same input+output vocab.
    a.weight = (Weight) (0.33*(kaldi::Rand() % 2));
    StateId start_state = all_states[kaldi::Rand() % n_states];
    fst->AddArc(start_state, a);
  }

  std::cout <<" printing before trimming\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }
  // Trim resulting FST.
  Connect(fst);

  std::cout <<" printing after trimming\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  vector<Label> extra_syms;
  if (fst->Start() != kNoStateId) {  // "Connect" did not make it empty....
    PreDeterminize(fst, 1000, &extra_syms);
  }

  std::cout <<" printing after predeterminization\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }


  {  // Remove epsilon.  All default args.
    bool connect = true;
    Weight weight_threshold = Weight::Zero();
    int64 nstate = -1;  // Relates to pruning.
    double delta = kDelta;  // I think a small weight value.  Relates to some kind of pruning,
    // I guess.  But with no epsilon cycles, probably doensn't matter.
    RmEpsilon(fst, connect,  weight_threshold, nstate, delta);
  }

  std::cout <<" printing after double-epsilon removal\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }
  VectorFst<Arc> ofst_star;

  {
    printf("Converting to Gallic semiring");
    VectorFst<GallicArc<Arc> > gallic_fst;
    VectorFst<GallicArc<Arc> > gallic_fst_noeps;
    VectorFst<GallicArc<Arc> > gallic_fst_det;


    {
      printf("Determinizing with DeterminizeStar, converting to Gallic\n");
      DeterminizeStar(*fst, &gallic_fst);
    }

    {
      std::cout <<" printing gallic FST\n";
      FstPrinter<GallicArc<Arc> >  fstprinter(gallic_fst, sptr, sptr, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }


    // Map(ofst_star, &gallic_fst, ToGallicMapper<Arc, STRING_LEFT>());

    printf("Converting gallic back to regular\n");
    TrivialFactorWeightFst< GallicArc<Arc, GALLIC_LEFT>, GallicFactor<typename Arc::Label,
        typename Arc::Weight, GALLIC_LEFT> > fwfst(gallic_fst);
    {
      std::cout <<" printing factor-weight FST\n";
      FstPrinter<GallicArc<Arc> >  fstprinter(fwfst, sptr, sptr, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }

    Map(fwfst, &ofst_star, FromGallicMapper<Arc, GALLIC_LEFT>());

    {
      std::cout <<" printing after converting back to regular FST\n";
      FstPrinter<Arc> fstprinter(ofst_star, sptr, sptr, NULL, false, true, "\t");
      fstprinter.Print(&std::cout, "standard output");
    }


    VectorFst<GallicArc<Arc> > new_gallic_fst;
    Map(ofst_star, &new_gallic_fst, ToGallicMapper<Arc, GALLIC_LEFT>());

    assert(RandEquivalent(gallic_fst, new_gallic_fst, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  }

  delete fst;
}


template<class Arc, class inttype> void TestStringRepository() {
  typedef typename Arc::Label Label;

  StringRepository<Label, inttype> sr;

  int N = 1000;
  if (sizeof(inttype) == 1) N = 64;
  vector<vector<Label> > strings(N);
  vector<inttype> ids(N);

  for (size_t i = 0;i < N;i++) {
    size_t len = kaldi::Rand() % 4;
    vector<Label> vec;
    for (size_t j = 0;j < len;j++) vec.push_back( (kaldi::Rand()%10) + 150*(kaldi::Rand()%2));  // make it have reasonable range.
    if (i < 500 && vec.size() == 0) ids[i] = sr.IdOfEmpty();
    else if (i < 500 && vec.size() == 1) ids[i] = sr.IdOfLabel(vec[0]);
    else ids[i] = sr.IdOfSeq(vec);

    strings[i] = vec;
  }

  for (size_t i = 0;i < N;i++) {
    vector<Label> tmpv;
    tmpv.push_back(10);  // just put in garbage.
    sr.SeqOfId(ids[i], &tmpv);
    assert(tmpv == strings[i]);
    assert(sr.IdOfSeq(strings[i]) == ids[i]);
    if (strings[i].size() == 0) assert(ids[i] == sr.IdOfEmpty());
    if (strings[i].size() == 1) assert(ids[i] == sr.IdOfLabel(strings[i][0]));

    if (sizeof(inttype) != 1) {
      size_t prefix_len = kaldi::Rand() % (strings[i].size() + 1);
      inttype s2 = sr.RemovePrefix(ids[i], prefix_len);
      vector<Label> vec2;
      sr.SeqOfId(s2, &vec2);
      for (size_t j = 0;j < strings[i].size()-prefix_len;j++) {
        assert(vec2[j] == strings[i][j+prefix_len]);
      }
    }

  }
}

} // end namespace fst

int main() {
  for (int i = 0;i < 25;i++) {
    fst::TestFactor<fst::StdArc>();
  }
}
