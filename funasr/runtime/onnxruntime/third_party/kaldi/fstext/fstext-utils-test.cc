// fstext/fstext-utils-test.cc

// Copyright 2009-2012  Microsoft Corporation  Daniel Povey

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

#include "base/kaldi-common.h" // for exceptions
#include "fstext/fstext-utils.h"
#include "fstext/fst-test-utils.h"
#include "util/stl-utils.h"
#include "base/kaldi-math.h"

namespace fst
{
using std::vector;
using std::cout;

template<class Arc, class I>
void TestMakeLinearAcceptor() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  int len = kaldi::Rand() % 10;
  vector<I> vec;
  vector<I> vec_nozeros;
  for (int i = 0; i < len; i++) {
    int j = kaldi::Rand() % len;
    vec.push_back(j);
    if (j != 0) vec_nozeros.push_back(j);
  }


  VectorFst<Arc> vfst;
  MakeLinearAcceptor(vec, &vfst);
  vector<I> vec2;
  vector<I> vec3;
  Weight w;
  GetLinearSymbolSequence(vfst, &vec2, &vec3, &w);
  assert(w == Weight::One());
  assert(vec_nozeros == vec2);
  assert(vec_nozeros == vec3);

  if (vec2.size() != 0 || vec3.size() != 0) { // This test might not work
    // for empty sequences...
    {
      vector<VectorFst<Arc> > fstvec;
      NbestAsFsts(vfst, 1, &fstvec);
      KALDI_ASSERT(fstvec.size() == 1);
      assert(RandEquivalent(vfst, fstvec[0], 2/*paths*/, 0.01/*delta*/,
                            kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
    }
  }
  bool include_eps = (kaldi::Rand() % 2 == 0);
  if (!include_eps) vec = vec_nozeros;
  kaldi::SortAndUniq(&vec);

  vector<I> vec4;
  GetInputSymbols(vfst, include_eps, &vec4);
  assert(vec4 == vec);
  vector<I> vec5;
  GetInputSymbols(vfst, include_eps, &vec5);
}


template<class Arc>  void TestDeterminizeStarInLog() {
  VectorFst<Arc> *fst = RandFst<Arc>();
  VectorFst<Arc> fst_copy(fst);
  typename Arc::Label next_sym = 1 + HighestNumberedInputSymbol(*fst);
  vector<typename Arc::Label> syms;
  PreDeterminize(fst, NULL, "#", next_sym, &syms);


}

  // Don't instantiate with log semiring, as RandEquivalent may fail.
template<class Arc>  void TestSafeDeterminizeWrapper() {  // also tests SafeDeterminizeMinimizeWrapper().
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = new VectorFst<Arc>();
  int n_syms = 2 + kaldi::Rand() % 5, n_states = 3 + kaldi::Rand() % 10, n_arcs = 5 + kaldi::Rand() % 30, n_final = 1 + kaldi::Rand()%3;  // Up to 2 unique symbols.
  cout << "Testing pre-determinize with "<<n_syms<<" symbols, "<<n_states<<" states and "<<n_arcs<<" arcs and "<<n_final<<" final states.\n";
  SymbolTable *sptr = new SymbolTable("my-symbol-table");
  sptr->AddSymbol("<eps>");
  delete sptr;
  sptr = new SymbolTable("my-symbol-table");

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

  VectorFst<Arc> *fst_det = new VectorFst<Arc>;

  vector<Label> extra_syms;
  if (fst->Start() != kNoStateId) {  // "Connect" did not make it empty....
    if (kaldi::Rand() % 2 == 0)
      SafeDeterminizeWrapper(fst_copy_orig, fst_det);
    else {
      if (kaldi::Rand() % 2 == 0)
        SafeDeterminizeMinimizeWrapper(fst_copy_orig, fst_det);
      else
        SafeDeterminizeMinimizeWrapperInLog(fst_copy_orig, fst_det);
    }

    // no because does shortest-dist on weights even if not pushing on them.
    // PushInLog<REWEIGHT_TO_INITIAL>(fst_det, kPushLabels);  // will always succeed.
    KALDI_LOG << "Num states [orig]: " << fst->NumStates() << "[det]" << fst_det->NumStates();
    assert(RandEquivalent(*fst, *fst_det, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
  }
  delete fst;
  delete fst_copy_orig;
  delete fst_det;
  delete sptr;
}


  // Don't instantiate with log semiring, as RandEquivalent may fail.
void TestPushInLog() {  // also tests SafeDeterminizeMinimizeWrapper().
  typedef StdArc Arc;
  typedef  Arc::Label Label;
  typedef  Arc::StateId StateId;
  typedef  Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<Arc>();
  VectorFst<Arc> fst2(*fst);
  PushInLog<REWEIGHT_TO_INITIAL>(&fst2, kPushLabels|kPushWeights, 0.01);  // speed it up using large delta.
  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}



template<class Arc>  void TestAcceptorMinimize() {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  VectorFst<Arc> *fst = RandFst<Arc>();

  Project(fst, PROJECT_INPUT);
  RemoveWeights(fst);

  VectorFst<Arc> fst2(*fst);
  internal::AcceptorMinimize(&fst2);

  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}


template<class Arc>  void TestMakeSymbolsSame() {

  VectorFst<Arc> *fst = RandFst<Arc>();
  bool foll = (kaldi::Rand() % 2 == 0);
  bool is_symbol = (kaldi::Rand() % 2 == 0);


  VectorFst<Arc> fst2(*fst);

  if (foll) {
    MakeFollowingInputSymbolsSame(is_symbol, &fst2);
    assert(FollowingInputSymbolsAreSame(is_symbol, fst2));
  } else {
    MakePrecedingInputSymbolsSame(is_symbol, &fst2);
    assert(PrecedingInputSymbolsAreSame(is_symbol, fst2));
  }


  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}


template<class Arc>
struct TestFunctor {
  typedef int32 Result;
  typedef typename Arc::Label Arg;
  Result operator () (Arg a) const {
    if (a == kNoLabel) return -1;
    else if (a == 0) return 0;
    else {
      return 1 + ((a-1) % 10);
    }
  }
};

template<class Arc>  void TestMakeSymbolsSameClass() {

  VectorFst<Arc> *fst = RandFst<Arc>();
  bool foll = (kaldi::Rand() % 2 == 0);
  bool is_symbol = (kaldi::Rand() % 2 == 0);


  VectorFst<Arc> fst2(*fst);

  TestFunctor<Arc> f;
  if (foll) {
    MakeFollowingInputSymbolsSameClass(is_symbol, &fst2, f);
    assert(FollowingInputSymbolsAreSameClass(is_symbol, fst2, f));
  } else {
    MakePrecedingInputSymbolsSameClass(is_symbol, &fst2, f);
    assert(PrecedingInputSymbolsAreSameClass(is_symbol, fst2, f));
  }

  assert(RandEquivalent(*fst, fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));

  delete fst;
}


// MakeLoopFstCompare is as MakeLoopFst but implmented differently [ less efficiently
// but more clearly], so we can check for equivalence.
template<class Arc>
VectorFst<Arc>* MakeLoopFstCompare(const vector<const ExpandedFst<Arc> *> &fsts) {
  VectorFst<Arc> *ans = new VectorFst<Arc>;
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  for (Label i = 0; i <  fsts.size(); i++) {
    if (fsts[i] != NULL) {
      VectorFst<Arc> i_fst;  // accepts symbol i on output.
      i_fst.AddState(); i_fst.AddState();
      i_fst.SetStart(0); i_fst.SetFinal(1, Weight::One());
      i_fst.AddArc(0, Arc(0, i, Weight::One(), 1));
      VectorFst<Arc> other_fst(*(fsts[i]));  // copy it.
      ClearSymbols(false, true, &other_fst);  // Clear output symbols so symbols
      // are on input side.
      Concat(&i_fst, other_fst);  // now i_fst is "i_fst [concat] other_fst".
      Union(ans, i_fst);
    }
  }
  Closure(ans, CLOSURE_STAR);
  return ans;
}


template<class Arc>  void TestMakeLoopFst() {

  int num_fsts = kaldi::Rand() % 10;
  vector<const ExpandedFst<Arc>* > fsts(num_fsts, (const ExpandedFst<Arc>*)NULL);
  for (int i = 0; i < num_fsts; i++) {
    if (kaldi::Rand() % 2 == 0) {  // put an fst there.
      VectorFst<Arc> *fst = RandFst<Arc>();
      Project(fst, PROJECT_INPUT);  // make input & output labels the same.
      fsts[i] = fst;
    } else { // this is to test that it works with the caching.
      fsts[i] = fsts[i/2];
    }
  }

  VectorFst<Arc> *fst1 = MakeLoopFst(fsts),
      *fst2 = MakeLoopFstCompare(fsts);

  assert(fst1->Properties(kOLabelSorted, kOLabelSorted) != 0);

  assert(RandEquivalent(*fst1, *fst2, 5/*paths*/, 0.01/*delta*/, kaldi::Rand()/*seed*/, 100/*path length-- max?*/));
  delete fst1;
  delete fst2;
  std::sort(fsts.begin(), fsts.end());
  fsts.erase(std::unique(fsts.begin(), fsts.end()), fsts.end());
  for (int i = 0; i < (int)fsts.size(); i++)
    delete fsts[i];
}



template<class Arc>
void TestEqualAlign() {
  for (size_t i = 0; i < 4; i++) {
    RandFstOptions opts;
    opts.allow_empty = false;
    VectorFst<Arc> *fst = RandFst<Arc>();
    int length = 10 + kaldi::Rand() % 20;

    VectorFst<Arc> fst_path;
    if (EqualAlign(*fst, length, kaldi::Rand(), &fst_path)) {
      std::cout << "EqualAlign succeeded\n";
      vector<int32> isymbol_seq, osymbol_seq;
      typename Arc::Weight weight;
      GetLinearSymbolSequence(fst_path, &isymbol_seq, &osymbol_seq, &weight);
      assert(isymbol_seq.size() == length);
      Invert(&fst_path);
      VectorFst<Arc> fst_composed;
      Compose(fst_path, *fst, &fst_composed);
      assert(fst_composed.Start() != kNoStateId);  // make sure nonempty.
    } else {
      std::cout << "EqualAlign did not generate alignment\n";
    }
    delete fst;
  }
}


template<class Arc> void Print(const Fst<Arc> &fst, std::string message) {
  std::cout << message << "\n";
  FstPrinter<Arc> fstprinter(fst, NULL, NULL, NULL, false, true, "\t");
  fstprinter.Print(&std::cout, "standard output");
}


template<class Arc>
void TestRemoveUselessArcs() {
  for (size_t i = 0; i < 4; i++) {
    RandFstOptions opts;
    opts.allow_empty = false;
    VectorFst<Arc> *fst = RandFst<Arc>();
    // Print(*fst, "[testremoveuselessarcs]:fst:");
    UniformArcSelector<Arc> selector;
    RandGenOptions<UniformArcSelector<Arc> > randgen_opts(selector);
    VectorFst<Arc> fst_path;
    RandGen(*fst, &fst_path, randgen_opts);
    Project(&fst_path, PROJECT_INPUT);
    // Print(fst_path, "[testremoveuselessarcs]:fstpath:");

    VectorFst<Arc> fst_nouseless(*fst);
    RemoveUselessArcs(&fst_nouseless);
    // Print(fst_nouseless, "[testremoveuselessarcs]:fst_nouseless:");

    VectorFst<Arc> orig_composed,
        nouseless_composed;
    Compose(fst_path, *fst, &orig_composed);
    Compose(fst_path, fst_nouseless, &nouseless_composed);

    // Print(orig_composed, "[testremoveuselessarcs]:orig_composed");
    // Print(nouseless_composed, "[testremoveuselessarcs]:nouseless_composed");

    VectorFst<Arc> orig_bestpath,
        nouseless_bestpath;
    ShortestPath(orig_composed, &orig_bestpath);
    ShortestPath(nouseless_composed, &nouseless_bestpath);
    // Print(orig_bestpath, "[testremoveuselessarcs]:orig_bestpath");
    // Print(nouseless_bestpath, "[testremoveuselessarcs]:nouseless_bestpath");

    typename Arc::Weight worig, wnouseless;
    GetLinearSymbolSequence<Arc, int>(orig_bestpath, NULL, NULL, &worig);
    GetLinearSymbolSequence<Arc, int>(nouseless_bestpath, NULL, NULL, &wnouseless);
    assert(ApproxEqual(worig, wnouseless, kDelta));

    // assert(RandEquivalent(orig_bestpath, nouseless_bestpath, 5/*paths*/, 0.01/*delta*/, Rand()/*seed*/, 100/*path length-- max?*/));
    delete fst;
  }
}



} // end namespace fst


int main() {
  for (int i = 0; i < 5; i++) {
    fst::TestMakeLinearAcceptor<fst::StdArc, int>();  // this also tests GetLinearSymbolSequence, GetInputSymbols and GetOutputSymbols.
    fst::TestMakeLinearAcceptor<fst::StdArc, int32>();
    fst::TestMakeLinearAcceptor<fst::StdArc, uint32>();
    fst::TestSafeDeterminizeWrapper<fst::StdArc>();
    fst::TestAcceptorMinimize<fst::StdArc>();
    fst::TestMakeSymbolsSame<fst::StdArc>();
    fst::TestMakeSymbolsSame<fst::LogArc>();
    fst::TestMakeSymbolsSameClass<fst::StdArc>();
    fst::TestMakeSymbolsSameClass<fst::LogArc>();
    fst::TestMakeLoopFst<fst::StdArc>();
    fst::TestMakeLoopFst<fst::LogArc>();
    fst::TestEqualAlign<fst::StdArc>();
    fst::TestEqualAlign<fst::LogArc>();
    fst::TestRemoveUselessArcs<fst::StdArc>();
  }
}
