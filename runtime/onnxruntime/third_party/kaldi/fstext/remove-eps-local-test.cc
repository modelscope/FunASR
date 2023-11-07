// fstext/remove-eps-local-test.cc

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (author: Daniel Povey)

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


#include "fstext/remove-eps-local.h"
#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"
#include "base/kaldi-math.h"


namespace fst
{
using std::vector;
using std::cout;

// Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc> static void TestRemoveEpsLocal() {
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
    if (kaldi::RandUniform() < 0.3) arc.ilabel = all_syms[kaldi::Rand()%all_syms.size()];
    else arc.ilabel = 0;
    if (kaldi::RandUniform() < 0.3) arc.olabel = all_syms[kaldi::Rand()%all_syms.size()];
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

  Connect(&fst);
  if (fst.Start() == kNoStateId) return;  // "Connect" made it empty.

  std::cout <<" printing after trimming\n";
  {
    FstPrinter<Arc> fstprinter(fst, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  VectorFst<Arc> fst_copy1(fst);


  RemoveEpsLocal(&fst_copy1);



  {
    std::cout << "copy1 = \n";
    FstPrinter<Arc> fstprinter(fst_copy1, sptr, sptr, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }


  int num_states_0 = fst.NumStates();
  int num_states_1 = fst_copy1.NumStates();


  std::cout << "Number of states 0 = "<<num_states_0<<", 1 = "<<num_states_1<<'\n';

  assert(RandEquivalent(fst, fst_copy1, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
}


static void TestRemoveEpsLocalSpecial() {
  // test that RemoveEpsLocalSpecial preserves equivalence in tropical while
  // maintaining stochasticity in log.
  typedef VectorFst<LogArc> Fst;
  typedef LogArc::Weight Weight;
  typedef LogArc::StateId StateId;
  typedef LogArc Arc;
  VectorFst<LogArc> *logfst = RandFst<LogArc>();

  { // Make the FST stochastic.
    for (StateId s = 0; s < logfst->NumStates(); s++) {
      Weight w = logfst->Final(s);
      for (ArcIterator<Fst> aiter(*logfst, s); !aiter.Done(); aiter.Next()) {
        w = Plus(w, aiter.Value().weight);
      }
      if (w != Weight::Zero()) {
        logfst->SetFinal(s, Divide(logfst->Final(s), w, DIVIDE_ANY));
        for (MutableArcIterator<Fst> aiter(logfst, s); !aiter.Done(); aiter.Next()) {
          Arc a = aiter.Value();
          a.weight = Divide(a.weight, w, DIVIDE_ANY);
          aiter.SetValue(a);
        }
      }
    }
  }
#ifndef _MSC_VER
  assert(IsStochasticFst(*logfst, kDelta*10));
#endif
  {
    std::cout << "logfst = \n";
    FstPrinter<LogArc> fstprinter(*logfst, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }

  VectorFst<StdArc> fst;
  Cast(*logfst, &fst);
  VectorFst<StdArc> fst_copy(fst);
  RemoveEpsLocalSpecial(&fst);  // removes eps in std-arc but keep stochastic in log-arc
  // make sure equivalent.
  assert(RandEquivalent(fst, fst_copy, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
  VectorFst<LogArc> logfst2;
  Cast(fst, &logfst2);

  {
    std::cout << "logfst2 = \n";
    FstPrinter<LogArc> fstprinter(logfst2, NULL, NULL, NULL, false, true, "\t");
    fstprinter.Print(&std::cout, "standard output");
  }
  if (ApproxEqual(ShortestDistance(*logfst), ShortestDistance(logfst2))) {
    // make sure we preserved stochasticity in cases where doing so was
    // possible... if the log-semiring total weight changed, then it is
    // not possible so don't assert this.
    assert(IsStochasticFst(logfst2, kDelta*10));
  }
  delete logfst;
}

} // namespace fst

int main() {
  using namespace fst;
  for (int i = 0; i < 10; i++) {
    TestRemoveEpsLocal<fst::StdArc>();
    TestRemoveEpsLocalSpecial();
  }
}
