// fstext/pre-determinize-test.cc

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
#include "fstext/fst-test-utils.h"
#include "fstext/fstext-utils.h"

// Just check that it compiles, for now.

namespace fst
{
  using std::vector;
  using std::cout;

// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestPreDeterminize() {
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

  VectorFst<Arc> *fst_copy_orig = new VectorFst<Arc>(*fst);

  vector<Label> extra_syms;
  if (fst->Start() != kNoStateId) {  // "Connect" did not make it empty....
    typename Arc::Label highest_sym = HighestNumberedInputSymbol(*fst);
    PreDeterminize(fst, highest_sym+1, &extra_syms);
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

  std::cout <<" printing after epsilon removal\n";
  {
    FstPrinter<Arc> fstprinter(*fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }


  VectorFst<Arc> ofst;
  DeterminizeOptions<Arc> opts;  // Default options.
  Determinize(*fst, &ofst, opts);
  std::cout <<" printing after determinization\n";
  {
    FstPrinter<Arc> fstprinter(ofst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  int64 num_removed = DeleteISymbols(&ofst, extra_syms);
  std::cout <<" printing after removing "<<num_removed<<" instances of extra symbols\n";
  {
    FstPrinter<Arc> fstprinter(ofst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  std::cout <<" Checking equivalent to original FST.\n";
  // giving Rand() as a seed stops the random number generator from always being reset to
  // the same point each time, while maintaining determinism of the test.
  assert(RandEquivalent(ofst, *fst_copy_orig, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
  delete fst_copy_orig;
}

template<class Arc>  void TestAddSelfLoops() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();
  SymbolTable *ilabels = new SymbolTable("my-symbol-table");
  SymbolTable *olabels = new SymbolTable("my-symbol-table-2");
  Label i0 = ilabels->AddSymbol("<eps>");
  Label i1 = ilabels->AddSymbol("1");
  Label i2 = ilabels->AddSymbol("2");

  Label o0 = olabels->AddSymbol("<eps>");
  Label o1 = olabels->AddSymbol("1");

  assert(i0 == 0 && o0 == 0);
  StateId s0 = fst->AddState(), s1 = fst->AddState(), s2 = fst->AddState();
  fst->SetStart(s0);
  assert(s0 == 0);

  fst->SetFinal(s2, (Weight)2);  // state 2 is final.
  {
    Arc arc;
    arc.ilabel = i1;
    arc.olabel = o0;
    arc.nextstate = 1;
    arc.weight = (Weight)1;
    fst->AddArc(s0, arc);  // arc from 0 to 1 with epsilon out.
  }
  {
    Arc arc;
    arc.ilabel = i2;
    arc.olabel = o1;
    arc.nextstate = 2;
    arc.weight = (Weight)2;
    fst->AddArc(s1, arc);  // arc from 1 to 2 with "1" out.
  }
  std::cout <<" printing before adding self-loops\n";
  {
    FstPrinter<Arc> fstprinter(*fst, ilabels, olabels, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }


  // So states 1 and 2 should have self-loops on.
  size_t num_extra = kaldi::Rand() % 5;
  vector<Label> extra_ilabels, extra_olabels;
  CreateNewSymbols(ilabels,  num_extra, "in#", &extra_ilabels);
  CreateNewSymbols(olabels,  num_extra, "out#", &extra_olabels);

  AddSelfLoops(fst, extra_ilabels, extra_olabels);

  assert(fst->NumArcs(0) == 1);
  assert(fst->NumArcs(1) == 1 + num_extra);
  assert(fst->NumArcs(2) == num_extra);

  std::cout <<" printing after adding self-loops\n";
  {
    FstPrinter<Arc> fstprinter(*fst, ilabels, olabels, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  delete fst;
  delete ilabels;
  delete olabels;
}

} // end namespace fst.


int main() {
  for (int i = 0;i < 10;i++) {  // run it multiple times; it's a randomized testing algorithm.
    fst::TestPreDeterminize<fst::StdArc>();
  }
  for (int i = 0;i < 5;i++) {
    fst::TestAddSelfLoops<fst::StdArc>();
  }
}
