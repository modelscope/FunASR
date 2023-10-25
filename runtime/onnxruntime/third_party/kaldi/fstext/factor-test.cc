// fstext/factor-test.cc

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


#include "fstext/factor.h"
#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"


namespace fst
{
using std::vector;

// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc> static void TestFactor() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> fst;
  int n_syms = 2 + kaldi::Rand() % 5, n_arcs = 5 + kaldi::Rand() % 30, n_final = 1 + kaldi::Rand()%10;

  SymbolTable symtab("my-symbol-table"), *sptr = &symtab;

  vector<Label> all_syms;  // including epsilon.
  // Put symbols in the symbol table from 1..n_syms-1.
  for (size_t i = 0;i < (size_t)n_syms;i++) {
    std::stringstream ss;
    if (i == 0) ss << "<eps>";
    else ss<<i;
    Label cur_lab = sptr->AddSymbol(ss.str());
    assert(cur_lab == (Label)i);
    all_syms.push_back(cur_lab);
  }
  assert(all_syms[0] == 0);

  fst.AddState();
  int cur_num_states = 1;
  for (int i = 0; i < n_arcs; i++) {
    StateId src_state = kaldi::Rand() % cur_num_states;
    StateId dst_state;
    if (kaldi::RandUniform() < 0.1) dst_state = kaldi::Rand() % cur_num_states;
    else {
      dst_state = cur_num_states++; fst.AddState();
    }
    Arc arc;
    if (kaldi::RandUniform() < 0.5) arc.ilabel = all_syms[kaldi::Rand()%all_syms.size()];
    else arc.ilabel = 0;
    if (kaldi::RandUniform() < 0.5) arc.olabel = all_syms[kaldi::Rand()%all_syms.size()];
    else arc.olabel = 0;
    arc.weight = (Weight) (0 + 0.1*(kaldi::Rand() % 5));
    arc.nextstate = dst_state;
    fst.AddArc(src_state, arc);
  }
  for (int i = 0; i < n_final; i++) {
    fst.SetFinal(kaldi::Rand() % cur_num_states,  (Weight) (0 + 0.1*(kaldi::Rand() % 5)));
  }

  if (kaldi::RandUniform() < 0.8)   fst.SetStart(0);  // usually leads to nicer examples.
  else fst.SetStart(kaldi::Rand() % cur_num_states);

  std::cout <<" printing before trimming\n";
  {
    FstPrinter<Arc> fstprinter(fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }
  // Trim resulting FST.
  Connect(&fst);

  std::cout <<" printing after trimming\n";
  {
    FstPrinter<Arc> fstprinter(fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  if (fst.Start() == kNoStateId) return;  // "Connect" made it empty.

  VectorFst<Arc> fst_pushed;
  Push<Arc, REWEIGHT_TO_INITIAL>(fst, &fst_pushed, kPushLabels);

  VectorFst<Arc> fst_factored;
  vector<vector<typename Arc::Label> > symbols;

  Factor(fst, &fst_factored, &symbols);

  // Check no epsilons in "symbols".
  for (size_t i = 0; i < symbols.size(); i++)
    assert(symbols[i].size() == 0 || *(std::min(symbols[i].begin(), symbols[i].end())) > 0);

  VectorFst<Arc> fst_factored_pushed;
  vector<vector<typename Arc::Label> > symbols_pushed;
  Factor(fst_pushed, &fst_factored_pushed, &symbols_pushed);

  std::cout << "Unfactored has "<<fst.NumStates()<<" states, factored has "<<fst_factored.NumStates()<<", and pushed+factored has "<<fst_factored_pushed.NumStates()<<'\n';

  assert(fst_factored.NumStates() <= fst.NumStates());
  //  assert(fst_factored_pushed.NumStates() <= fst_factored.NumStates());  // pushing should only help. [ no, it doesn't]
  assert(fst_factored_pushed.NumStates() <= fst_pushed.NumStates());

  VectorFst<Arc> fst_factored_copy(fst_factored);

  VectorFst<Arc> fst_factored_unfactored(fst_factored);
  ExpandInputSequences(symbols, &fst_factored_unfactored);

  VectorFst<Arc> factor_fst;
  CreateFactorFst(symbols, &factor_fst);
  VectorFst<Arc> fst_factored_unfactored2;
  Compose(factor_fst, fst_factored, &fst_factored_unfactored2);

  ExpandInputSequences(symbols_pushed, &fst_factored_pushed);

  assert(RandEquivalent(fst, fst_factored_unfactored, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  assert(RandEquivalent(fst, fst_factored_unfactored2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  assert(RandEquivalent(fst, fst_factored_pushed, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  {  // Have tested for equivalence; now do another test: that FactorFst actually finds all
    // the factors.  Do this by inserting factors using ExpandInputSequences and making sure it gets
    // rid of them all.
    Label max_label = *(std::max_element(all_syms.begin(), all_syms.end()));
    vector<vector<Label> > new_labels(max_label+1);
    for (Label l = 1; l < static_cast<Label>(new_labels.size()); l++) {
      int n = kaldi::Rand() % 5;
      for (int i = 0; i < n; i++) new_labels[l].push_back(kaldi::Rand() % 100);
    }
    VectorFst<Arc> fst_expanded(fst);
    ExpandInputSequences(new_labels, &fst_expanded);

    vector<vector<Label> > factors;
    VectorFst<Arc> fst_reduced;
    Factor(fst_expanded, &fst_reduced, &factors);
    assert(fst_reduced.NumStates() <= fst.NumStates());  // Checking that it found all the factors.
  }

  {  // This block test MapInputSymbols [but relies on the correctness of Factor
    // and ExpandInputSequences to do so].

    std::map<Label, Label> symbols_reverse_map;  // from new->old.
    symbols_reverse_map[0] = 0;  // map eps to eps.
    for (Label i = 1; i < static_cast<Label>(symbols.size()); i++) {
      Label new_i;
      do {
        new_i = kaldi::Rand() % (symbols.size() + 20);
      } while (symbols_reverse_map.count(new_i) == 1);
      symbols_reverse_map[new_i] = i;
    }
    vector<vector<Label> > symbols_new;
    vector<Label> symbol_map(symbols.size());  // from old->new.
    typename std::map<Label, Label>::iterator iter = symbols_reverse_map.begin();
    for (; iter != symbols_reverse_map.end(); iter++) {
      Label new_label = iter->first, old_label = iter->second;
      if (new_label >= static_cast<Label>(symbols_new.size())) symbols_new.resize(new_label+1);
      symbols_new[new_label] = symbols[old_label];
      symbol_map[old_label] = new_label;
    }
    MapInputSymbols(symbol_map, &fst_factored_copy);
    ExpandInputSequences(symbols_new, &fst_factored_copy);
    assert(RandEquivalent(fst, fst_factored_copy,
                          5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/,
                          100/*path length-- max?*/));
  }

}


} // namespace fst

int main() {
  using namespace fst;
  for (int i = 0;i < 25;i++) {
    TestFactor<fst::StdArc>();
  }
}
