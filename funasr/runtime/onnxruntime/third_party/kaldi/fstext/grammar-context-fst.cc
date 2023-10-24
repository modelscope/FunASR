// fstext/grammar-context-fst.cc

// Copyright      2018  Johns Hopkins University (author: Daniel Povey)

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

#include "fstext/grammar-context-fst.h"
#include "base/kaldi-error.h"
#include "util/stl-utils.h"

namespace fst {
using std::vector;

InverseLeftBiphoneContextFst::InverseLeftBiphoneContextFst(
    Label nonterm_phones_offset,
    const vector<int32>& phones,
    const vector<int32>& disambig_syms):
    nonterm_phones_offset_(nonterm_phones_offset),
    phone_syms_(phones),
    disambig_syms_(disambig_syms) {

  { // This block does some checks.
    std::vector<int32> all_inputs(phones);
    all_inputs.insert(all_inputs.end(), disambig_syms.begin(),
                      disambig_syms.end());
    all_inputs.push_back(nonterm_phones_offset);
    size_t size = all_inputs.size();
    kaldi::SortAndUniq(&all_inputs);
    if (all_inputs.size() != size) {
      KALDI_ERR << "There was overlap between disambig symbols, phones, "
          "and/or --nonterm-phones-offset";
    }
    if (all_inputs.front() <= 0)
      KALDI_ERR << "Symbols <= 0 were passed in as phones, disambig-syms, "
          "or nonterm-phones-offset.";
    if (all_inputs.back() != nonterm_phones_offset) {
      // the value passed --nonterm-phones-offset is not higher numbered
      // than all the phones and disambig syms... do some more checking.
      for (int32 i = 1; i < 4; i++) {
        int32 symbol = nonterm_phones_offset + i;
        // None of the symbols --nonterm-phones-offset + {kNontermBos, kNontermBegin,
        //                  kNontermEnd, kNontermReenter, kNontermUserDefined}
        // (i.e. the special symbols plus the first user-defined symbol) may be
        // listed as phones or disambig symbols... this doesn't make sense.  We
        // do allow disambig symbols to be higher-numbered than the nonterminal
        // sybols, just in case that happens to be needed, but they can't overlap.
        if (std::binary_search(all_inputs.begin(), all_inputs.end(), symbol)) {
          KALDI_ERR << "The symbol " << symbol
                    << " = --nonterm-phones-offset + " << i
                    << " was listed as a phone or disambig symbol.";
        }
      }
    }
    if (phone_syms_.empty())
      KALDI_WARN << "Context FST created but there are no phone symbols: probably "
          "input FST was empty.";
  }

  // empty vector, will be the ilabel_info vector that corresponds to epsilon,
  // in case our FST needs to output epsilons.
  vector<int32> empty_vec;
  Label epsilon_label = FindLabel(empty_vec);
  // Make sure that a label is assigned for epsilon.
  KALDI_ASSERT(epsilon_label == 0);
}


InverseLeftBiphoneContextFst::Weight InverseLeftBiphoneContextFst::Final(StateId s) {
  if (s == 0 || phone_syms_.count(s) != 0 ||
      s == GetPhoneSymbolFor(kNontermEnd))
    return Weight::One();
  else
    return Weight::Zero();
}

bool InverseLeftBiphoneContextFst::GetArc(
    StateId s, Label ilabel, Arc *arc) {
  // it's a rule of the DeterministicOnDemandFst that the ilabel cannot be zero.q
  KALDI_ASSERT(ilabel != 0);

  arc->ilabel = ilabel;
  arc->weight = Weight::One();

  if (s == 0 || phone_syms_.count(s) != 0) {
    // This is an epsilon or phone state.
    if (phone_syms_.count(ilabel) != 0) {
      // The ilabel is a phone.
      std::vector<int32> context_window(2);
      context_window[0] = s;
      context_window[1] = ilabel;
      arc->olabel = FindLabel(context_window);
      arc->nextstate = ilabel;
      return true;
    } else if (disambig_syms_.count(ilabel) != 0) {
      // the ilabel is a disambiguation symbol.  Make a self-loop arc that
      // replicates the disambiguation symbol on the input.
      // The ilabel-info vector for disambig symbols is just a single element
      // consisting of the negative of the disambig symbols (for easier
      // identification from code).
      std::vector<int32> this_ilabel_info(1);
      this_ilabel_info[0] = -ilabel;
      arc->olabel = FindLabel(this_ilabel_info);
      arc->nextstate = s;
      return true;
    } else if (ilabel == GetPhoneSymbolFor(kNontermBegin) &&
               s == 0) {
      // We were at the start state and saw the symbol #nonterm_begin.
      // Output nothing, but transition to the special #nonterm_begin state.
      // when we're in that state, arcs for phones generate special
      // osymbols corresponding to pairs like (#nonterm_begin, p1).
      arc->olabel = 0;
      arc->nextstate = GetPhoneSymbolFor(kNontermBegin);
      return true;
    } else if (ilabel == GetPhoneSymbolFor(kNontermEnd)) {
      // we saw #nonterm_end.
      std::vector<int32> this_ilabel_info(2);
      this_ilabel_info[0] = -(GetPhoneSymbolFor(kNontermEnd));
      this_ilabel_info[1] = (s != 0 ? s : GetPhoneSymbolFor(kNontermBos));
      arc->olabel = FindLabel(this_ilabel_info);
      arc->nextstate = GetPhoneSymbolFor(kNontermEnd);
      return true;
    } else if (ilabel >= GetPhoneSymbolFor(kNontermUserDefined)) {
      // Assume this ilabel is a user-defined nonterminal.
      // Transition to the state kNontermUserDefined, with an olabel
      // (#nonterm:foo, p1) where 'p1' is the current left-context.
      std::vector<int32> this_ilabel_info(2);
      this_ilabel_info[0] = -ilabel;
      this_ilabel_info[1] = (s != 0 ? s : GetPhoneSymbolFor(kNontermBos));
      arc->olabel = FindLabel(this_ilabel_info);
      // the destination state is not specific to this user-defined symbol, it's
      // a generic destination state.
      arc->nextstate = GetPhoneSymbolFor(kNontermUserDefined);
      return true;
    } else {
      return false;
    }
  } else if (s == GetPhoneSymbolFor(kNontermBegin)) {
    if (phone_syms_.count(ilabel) != 0 || ilabel == GetPhoneSymbolFor(kNontermBos)) {
      std::vector<int32> this_ilabel_info(2);
      this_ilabel_info[0] = -GetPhoneSymbolFor(kNontermBegin);
      this_ilabel_info[1] = ilabel;
      arc->nextstate = (ilabel == GetPhoneSymbolFor(kNontermBos) ? 0 : ilabel);
      arc->olabel = FindLabel(this_ilabel_info);
      return true;
    } else {
      return false;
    }
  } else if (s == GetPhoneSymbolFor(kNontermEnd)) {
    return false;
  } else if (s == GetPhoneSymbolFor(kNontermUserDefined)) {
    if (phone_syms_.count(ilabel) != 0 || ilabel == GetPhoneSymbolFor(kNontermBos)) {
      std::vector<int32> this_ilabel_info(2);
      this_ilabel_info[0] = -GetPhoneSymbolFor(kNontermReenter);
      this_ilabel_info[1] = ilabel;
      arc->nextstate = (ilabel == GetPhoneSymbolFor(kNontermBos) ? 0 : ilabel);
      arc->olabel = FindLabel(this_ilabel_info);
      return true;
    } else {
      return false;
    }
  } else {
    // likely code error.
    KALDI_ERR << "Invalid state encountered";
    return false;  // won't get here.  suppress compiler error.
  }
}

StdArc::Label InverseLeftBiphoneContextFst::FindLabel(const vector<int32> &label_vec) {
  // Finds the ilabel corresponding to this vector (creates a new ilabel if
  // necessary).
  VectorToLabelMap::const_iterator iter = ilabel_map_.find(label_vec);
  if (iter == ilabel_map_.end()) {  // Not already in map.
    Label this_label = ilabel_info_.size();
    ilabel_info_.push_back(label_vec);
    ilabel_map_[label_vec] = this_label;
    return this_label;
  } else {
    return iter->second;
  }
}


void ComposeContextLeftBiphone(
    int32 nonterm_phones_offset,
    const vector<int32> &disambig_syms_in,
    const VectorFst<StdArc> &ifst,
    VectorFst<StdArc> *ofst,
    std::vector<std::vector<int32> > *ilabels) {

  vector<int32> disambig_syms(disambig_syms_in);
  std::sort(disambig_syms.begin(), disambig_syms.end());

  vector<int32> all_syms;
  GetInputSymbols(ifst, false/*no eps*/, &all_syms);
  std::sort(all_syms.begin(), all_syms.end());
  vector<int32> phones;
  for (size_t i = 0; i < all_syms.size(); i++)
    if (!std::binary_search(disambig_syms.begin(),
                            disambig_syms.end(), all_syms[i]) &&
        all_syms[i] < nonterm_phones_offset)
      phones.push_back(all_syms[i]);


  InverseLeftBiphoneContextFst inv_c(nonterm_phones_offset,
                                     phones, disambig_syms);

  // The following statement is equivalent to the following
  // (if FSTs had the '*' operator for composition):
  //   (*ofst) = inv(inv_c) * (*ifst)
  ComposeDeterministicOnDemandInverse(ifst, &inv_c, ofst);

  inv_c.SwapIlabelInfo(ilabels);
}

}  // end namespace fst
