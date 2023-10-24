// fstext/context-fst-test.cc

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

#include "fstext/context-fst.h"
#include "fstext/fst-test-utils.h"
#include "tree/context-dep.h"
#include "util/kaldi-io.h"
#include "base/kaldi-math.h"

namespace fst
{
using std::vector;
using std::cout;

// GenAcceptorFromSequence generates a linear acceptor (identical input+output symbols) that has this
// sequence of symbols, and
template<class Arc>
static VectorFst<Arc> *GenAcceptorFromSequence(const vector<typename Arc::Label> &symbols, float cost) {
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;

  vector<float> split_cost(symbols.size()+1, 0.0);  // for #-arcs + end-state.
  {  // compute split_cost.  it must sum to "cost".
    std::set<int32> indices;
    size_t num_indices = 1 + (kaldi::Rand() % split_cost.size());
    while (indices.size() < num_indices) indices.insert(kaldi::Rand() % split_cost.size());
    for (std::set<int32>::iterator iter = indices.begin(); iter != indices.end(); ++iter) {
      split_cost[*iter] = cost / num_indices;
    }
  }

  VectorFst<Arc> *fst = new VectorFst<Arc>();
  StateId cur_state = fst->AddState();
  fst->SetStart(cur_state);
  for (size_t i = 0; i < symbols.size(); i++) {
    StateId next_state = fst->AddState();
    Arc arc;
    arc.ilabel = symbols[i];
    arc.olabel = symbols[i];
    arc.nextstate = next_state;
    arc.weight = (Weight) split_cost[i];
    fst->AddArc(cur_state, arc);
    cur_state = next_state;

  }
  fst->SetFinal(cur_state, (Weight)split_cost[symbols.size()]);
  return fst;
}



// CheckPhones is used to test the correctness of an FST that is the result of
// composition with a ContextFst.
template<class Arc>
static float CheckPhones(const VectorFst<Arc> &linear_fst,
                          const vector<typename Arc::Label> &phone_ids,
                          const vector<typename Arc::Label> &disambig_ids,
                          const vector<typename Arc::Label> &phone_seq,
                          const vector<vector<typename Arc::Label> > &ilabel_info,
                          int N, int P) {
  typedef typename Arc::Label Label;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Weight Weight;

  assert(kaldi::IsSorted(phone_ids));  // so we can do binary_search.


  vector<int32> input_syms;
  vector<int32> output_syms;
  Weight tot_cost;
  bool ans = GetLinearSymbolSequence(linear_fst,  &input_syms,
                                     &output_syms, &tot_cost);
  assert(ans);  // should be linear.

  vector<int32> phone_seq_check;
  for (size_t i = 0; i < output_syms.size(); i++)
    if (std::binary_search(phone_ids.begin(), phone_ids.end(), output_syms[i]))
      phone_seq_check.push_back(output_syms[i]);

  assert(phone_seq_check  == phone_seq);

  vector<vector<int32> > input_syms_long;
  for (size_t i = 0; i < input_syms.size(); i++) {
    Label isym = input_syms[i];
    if (ilabel_info[isym].size() == 0) continue;  // epsilon.
    if ( (ilabel_info[isym].size() == 1 &&
         ilabel_info[isym][0] <= 0) ) continue;  // disambig.
    input_syms_long.push_back(ilabel_info[isym]);
  }

  for (size_t i = 0; i < input_syms_long.size(); i++) {
    vector<int32> phone_context_window(N);  // phone at pos i will be at pos P in this window.
    int pos = ((int)i) - P;  // pos of first phone in window [ may be out of range] .
    for (int j = 0; j < N; j++, pos++) {
      if (static_cast<size_t>(pos) < phone_seq.size()) phone_context_window[j] = phone_seq[pos];
      else phone_context_window[j] = 0;  // 0 is a special symbol that context-dep-itf expects to see
      // when no phone is present due to out-of-window.  context-fst knows about this too.
    }
    assert(input_syms_long[i] == phone_context_window);
  }
  return tot_cost.Value();
}




template<class Arc>
static VectorFst<Arc> *GenRandPhoneSeq(vector<typename Arc::Label> &phone_syms,
                                       vector<typename Arc::Label> &disambig_syms,
                                       typename Arc::Label subsequential_symbol,
                                       int num_subseq_syms,
                                       float seq_prob,
                                       vector<typename Arc::Label> *phoneseq_out) {
  KALDI_ASSERT(phoneseq_out != NULL);
  typedef typename Arc::Label Label;
  // Generate an FST that is a random phone sequence, ending
  // with "num_subseq_syms" subsequential symbols.  It will
  // have disambiguation symbols randomly interspersed throughout.
  // The number of phones is random (possibly zero).
  size_t len = (kaldi::Rand() % 4) * (kaldi::Rand() % 3);  // up to 3*2=6 phones.
  float disambig_prob = 0.33;
  phoneseq_out->clear();
  vector<Label> syms;  // the phones
  for (size_t i = 0; i < len; i++) {
    while (kaldi::RandUniform() < disambig_prob) syms.push_back(disambig_syms[kaldi::Rand() % disambig_syms.size()]);
    Label phone_id = phone_syms[kaldi::Rand() % phone_syms.size()];
    phoneseq_out->push_back(phone_id);  // record in output the underlying phone sequence.
    syms.push_back(phone_id);
  }
  for (size_t i = 0; static_cast<int32>(i) < num_subseq_syms; i++) {
    while (kaldi::RandUniform() < disambig_prob) syms.push_back(disambig_syms[kaldi::Rand() % disambig_syms.size()]);
    syms.push_back(subsequential_symbol);
  }
  while (kaldi::RandUniform() < disambig_prob) syms.push_back(disambig_syms[kaldi::Rand() % disambig_syms.size()]);

  // OK, now have the symbols of the FST as a vector.
  return GenAcceptorFromSequence<Arc>(syms, seq_prob);
}

// Don't instantiate with log semiring, as RandEquivalent may fail.
// TestContestFst also test ReadILabelInfo and WriteILabelInfo.
static void TestContextFst(bool verbose, bool use_matcher) {
  typedef StdArc Arc;
  typedef Arc::Label Label;
  typedef Arc::StateId StateId;
  typedef Arc::Weight Weight;

  // Generate a random set of phones.
  size_t num_phones = 1 + kaldi::Rand() % 10;
  std::set<int32> phones_set;
  while (phones_set.size() < num_phones) phones_set.insert(1 + kaldi::Rand() % (num_phones + 5));  // don't use 0 [== epsilon]
  vector<int32> phones;
  kaldi::CopySetToVector(phones_set, &phones);

  int N = 1 + kaldi::Rand() % 4;  // Context size, in range 1..4.
  int P = kaldi::Rand() % N;  // 1.. N-1.
  if (verbose) std::cout << "N = "<< N << ", P = "<<P<<'\n';

  Label subsequential_symbol = 1000;
  vector<int32> disambig_syms;
  for (size_t i =0; i < 5; i++) disambig_syms.push_back(500 + i);
  vector<int32> phone_syms;
  for (size_t i = 0; i < phones.size();i++) phone_syms.push_back(phones[i]);


  InverseContextFst inv_cfst(subsequential_symbol,
                             phones, disambig_syms,
                             N, P);


  /* Now create random phone-sequences and compose them with the context FST.
  */

  for (size_t p = 0; p < 10; p++) {
    vector<int32> phone_seq;
    int num_subseq = N - P - 1;  // zero if P == N-1, i.e. P is last element, i.e. left-context only.
    float tot_cost = 20.0 * kaldi::RandUniform();
    VectorFst<Arc> *f = GenRandPhoneSeq<Arc>(phone_syms, disambig_syms, subsequential_symbol, num_subseq, tot_cost, &phone_seq);
    if (verbose) {
      std::cout << "Sequence FST is:\n";
      {  // Try to print the fst.
        FstPrinter<Arc> fstprinter(*f, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
    }

    VectorFst<Arc> fst_composed;

    ComposeDeterministicOnDemandInverse(*f,  &inv_cfst, &fst_composed);


    // Testing WriteILabelInfo and ReadILabelInfo.
    {
      bool binary = (kaldi::Rand() % 2 == 0);
      WriteILabelInfo(kaldi::Output("tmpf", binary).Stream(),
                      binary, inv_cfst.IlabelInfo());

      bool binary_in;
      vector<vector<int32> > ilabel_info;
      kaldi::Input ki("tmpf", &binary_in);
      ReadILabelInfo(ki.Stream(),
                     binary_in, &ilabel_info);
      assert(ilabel_info == inv_cfst.IlabelInfo());
    }


    if (verbose) {
      std::cout << "Composed FST is:\n";
      {  // Try to print the fst.
        FstPrinter<Arc> fstprinter(fst_composed, NULL, NULL, NULL, false, true, "\t");
        fstprinter.Print(&std::cout, "standard output");
      }
    }

    // now check the composed FST.
    float tot_cost_check = CheckPhones<Arc>(fst_composed,
                                            phone_syms,
                                            disambig_syms,
                                            phone_seq,
                                            inv_cfst.IlabelInfo(),
                                            N, P);
    kaldi::AssertEqual(tot_cost, tot_cost_check);

    delete f;
  }

  unlink("tmpf");
}


} // namespace fst

int main() {

  for (int i = 0;i < 16;i++) {
    bool verbose = (i < 4);
    bool use_matcher = ( (i/4) % 2 == 0);
    fst::TestContextFst(verbose, use_matcher);
  }
}
