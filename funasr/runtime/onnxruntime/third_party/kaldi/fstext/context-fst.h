// fstext/context-fst.h

// Copyright 2009-2011  Microsoft Corporation
//                2018  Johns Hopkins University (author: Daniel Povey)

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
//
// This file includes material from the OpenFST Library v1.2.7 available at
// http://www.openfst.org and released under the Apache License Version 2.0.
//
// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Copyright 2005-2010 Google, Inc.
// Author: riley@google.com (Michael Riley)



#ifndef KALDI_FSTEXT_CONTEXT_FST_H_
#define KALDI_FSTEXT_CONTEXT_FST_H_

/* This header defines a context FST "C" (the "C" in "HCLG") which transduces
   from symbols representing phone context windows (e.g. "a, b, c") to
   individual phones, e.g. "a".  Search for "hbka.pdf" ("Speech Recognition
   with Weighted Finite State Transducers") by M. Mohri, for more context.
*/

#include <unordered_map>
using std::unordered_map;

#include <algorithm>
#include <string>
#include <vector>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "util/const-integer-set.h"
#include "fstext/deterministic-fst.h"

namespace fst {




/// Utility function for writing ilabel-info vectors to disk.
void WriteILabelInfo(std::ostream &os, bool binary,
                     const std::vector<std::vector<int32> > &ilabel_info);

/// Utility function for reading ilabel-info vectors from disk.
void ReadILabelInfo(std::istream &is, bool binary,
                    std::vector<std::vector<int32> > *ilabel_info);


/// The following function is mainly of use for printing and debugging.
SymbolTable *CreateILabelInfoSymbolTable(const std::vector<std::vector<int32> > &ilabel_info,
                                         const SymbolTable &phones_symtab,
                                         std::string separator,
                                         std::string disambig_prefix);  // e.g. separator = "/", disambig_prefix = "#"



/**
   Used in the command-line tool fstcomposecontext.  It creates a context FST and
   composes it on the left with "ifst" to make "ofst".  It outputs the label
   information to ilabels_out.  "ifst" is mutable because we need to add the
   subsequential loop.

    @param [in] disambig_syms  List of disambiguation symbols, e.g. the integer
                 ids of #0, #1, #2 ... in the phones.txt.
    @param [in] context_width  Size of context window, e.g. 3 for triphone.
    @param [in] central_position  Central position in phonetic context window
                  (zero-based index), e.g. 1 for triphone.
    @param [in,out] ifst   The FST we are composing with C (e.g. LG.fst), mustable because
                  we need to add the subsequential loop to it.
    @param [out] ofst   Composed output FST (would be CLG.fst).
    @param [out] ilabels_out  Vector, indexed by ilabel of CLG.fst, providing information
                  about the meaning of that ilabel; see
                  "http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel".
    @param [in] project_ifst  This is intended only to be set to true
                  in the program 'fstmakecontextfst'... if true, it will
                  project on the input after adding the subsequential loop
                  to 'ifst', which allows us to reconstruct the context
                  fst C.fst.
 */
void ComposeContext(const std::vector<int32> &disambig_syms,
                    int32 context_width, int32 central_position,
                    VectorFst<StdArc> *ifst,
                    VectorFst<StdArc> *ofst,
                    std::vector<std::vector<int32> > *ilabels_out,
                    bool project_ifst = false);


/**
  Modifies an FST so that it transuces the same paths, but the input side of the
  paths can all have the subsequential symbol '$' appended to them any number of
  times (we could easily specify the number of times, but accepting any number of
  repetitions is just more convenient).  The actual way we do this is for each
  final state, we add a transition with weight equal to the final-weight of that
  state, with input-symbol '$' and output-symbols \<eps\>, and ending in a new
  super-final state that has unit final-probability and a unit-weight self-loop
  with '$' on its input and \<eps\> on its output.  The reason we don't just
  add a loop to each final-state has to do with preserving stochasticity
  (see \ref fst_algo_stochastic).  We keep the final-probability in all the
  original final-states rather than setting them to zero, so the resulting FST
  can accept zero '$' symbols at the end (in case we had no right context).
*/
void AddSubsequentialLoop(StdArc::Label subseq_symbol,
                          MutableFst<StdArc> *fst);


/*
   InverseContextFst represents the inverse of the context FST "C" (the "C" in
   "HCLG") which transduces from symbols representing phone context windows
   (e.g. "a, b, c") to individual phones, e.g. "a".  So InverseContextFst
   transduces from phones to symbols representing phone context windows.  The
   point is that the inverse is deterministic, so the DeterministicOnDemandFst
   interface is applicable, which turns out to be a convenient way to implement
   this.

   This doesn't implement the full Fst interface, it implements the
   DeterministicOnDemandFst interface which is much simpler and which is
   sufficient for what we need to do with this.

   Search for "hbka.pdf" ("Speech Recognition with Weighted Finite State
   Transducers") by M. Mohri, for more context.
*/

class InverseContextFst: public DeterministicOnDemandFst<StdArc> {
public:
  typedef StdArc Arc;
  typedef typename StdArc::StateId StateId;
  typedef typename StdArc::Weight Weight;
  typedef typename StdArc::Label Label;

  /**
     Constructor.
        @param [in] subsequential_symbol   The integer id of the 'subsequential symbol'
                          (usually represented as '$') that terminates sequences on the
                          output of C.fst (input of InverseContextFst).  Search for
                          "quential" in https://cs.nyu.edu/~mohri/pub/hbka.pdf.
                          This may just be the first unused integer id.  Must be nonzer.
        @param [in] phones      List of integer ids of phones, as you would see in phones.txt
        @param [in] disambig_syms   List of integer ids of disambiguation symbols,
                                   e.g. the ids of #0, #1, #2 in phones.txt
        @param [in] context_width  Size of context window, e.g. 3 for triphone.
        @param [in] central_position  Central position in context window (zero-based),
                                   e.g. 1 for triphone.
     See \ref graph_context for more details.
  */
  InverseContextFst(Label subsequential_symbol,
                    const std::vector<int32>& phones,
                    const std::vector<int32>& disambig_syms,
                    int32 context_width,
                    int32 central_position);


  virtual StateId Start() { return 0; }

  virtual Weight Final(StateId s);

  /// Note: ilabel must not be epsilon.
  virtual bool GetArc(StateId s, Label ilabel, Arc *arc);

  ~InverseContextFst() { }

  // Returns a reference to a vector<vector<int32> > with information about all
  // the input symbols of C (i.e. all the output symbols of this
  // InverseContextFst).  See
  // "http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel".
  const std::vector<std::vector<int32> > &IlabelInfo() const {
    return ilabel_info_;
  }

  // A way to destructively obtain the ilabel-info.  Only do this if you
  // are just about to destroy this object.
  void SwapIlabelInfo(std::vector<std::vector<int32> > *vec) { ilabel_info_.swap(*vec); }

private:

  /// Returns the state-id corresponding to this vector of phones; creates the
  /// state it if necessary.  Requires seq.size() == context_width_ - 1.
  StateId FindState(const std::vector<int32> &seq);

  /// Finds the label index corresponding to this context-window of phones
  /// (likely of width context_width_).  Inserts it into the
  /// ilabel_info_/ilabel_map_ tables if necessary.
  Label FindLabel(const std::vector<int32> &label_info);

  inline bool IsDisambigSymbol(Label lab) { return (disambig_syms_.count(lab) != 0); }

  inline bool IsPhoneSymbol(Label lab) { return (phone_syms_.count(lab) != 0); }

  /// Create disambiguation-symbol self-loop arc; where 'ilabel' must correspond to
  /// a disambiguation symbol.  Called from CreateArc().
  inline void CreateDisambigArc(StateId s, Label ilabel, Arc *arc);

  /// Creates an arc, this function is to be called only when 'ilabel'
  /// corresponds to a phone.  Called from CreateArc().  The olabel may end be
  /// epsilon, instead of a phone-in-context, if the system has right context
  /// and we are very near the beginning of the phone sequence.
  inline void CreatePhoneOrEpsArc(StateId src, StateId dst, Label ilabel,
                                  const std::vector<int32> &phone_seq, Arc *arc);


  /// If phone_seq is nonempty then this function it left by one and appends
  /// 'label' to it, otherwise it does nothing.  We expect (but do not check)
  /// that phone_seq->size() == context_width_ - 1.
  inline void ShiftSequenceLeft(Label label, std::vector<int32> *phone_seq);

  /// This utility function does something equivalent to the following 3 steps:
  ///   *full_phone_sequence =  seq;
  ///  full_phone_sequence->append(label)
  ///  Replace any values equal to 'subsequential_symbol_' in
  /// full_phone_sequence with zero (this is to avoid having to keep track of
  /// the value of 'subsequential_symbol_' outside of this program).
  /// This function assumes that seq.size() == context_width_ - 1, and also that
  /// 'subsequential_symbol_' does not appear in positions 0 through
  /// central_position_ of 'seq'.
  inline void GetFullPhoneSequence(const std::vector<int32> &seq, Label label,
                                   std::vector<int32> *full_phone_sequence);

  // Map type to map from vectors of int32 (representing phonetic contexts,
  // which will be of dimension context_width - 1) to StateId (corresponding to
  // the state index in this FST).
  typedef unordered_map<std::vector<int32>, StateId,
                        kaldi::VectorHasher<int32> > VectorToStateMap;

  // Map type to map from vectors of int32 (representing ilabel-info,
  // see http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel) to
  // Label (the output label in this FST).
  typedef unordered_map<std::vector<int32>, Label,
                        kaldi::VectorHasher<int32> > VectorToLabelMap;


  // Sometimes called N, context_width_ this is the width of the
  // phonetic context, e.g. 3 for triphone, 2 for biphone, one for monophone.
  // It is a user-specified value.
  int32 context_width_;

  // Sometimes called P, central_position_ is is the (zero-based) "central
  // position" in the context window, meaning the phone that is "in" a certain
  // context.  The most widely used values of (context-width, central-position)
  // are: (3,1) for triphone, (1,0) for monophone, and (2, 1) for left biphone.
  // This is also specified by the user.  As an example, in the left-biphone
  // [ 5, 6 ], we view it as "the phone numbered 6 with the phone numbered 5 as
  // its left-context".
  int32 central_position_;

  // The following three variables were also passed in by the caller:

  // 'phone_syms_' are a set of phone-ids, typically 1, 2, .. num_phones.
  kaldi::ConstIntegerSet<Label> phone_syms_;

  // disambig_syms_ is the set of integer ids of the disambiguation symbols,
  // usually represented in text form as #0, #1, #2, etc.  These are inserted
  // into the grammar (for #0) and the lexicon (for #1, #2, ...) in order to
  // make the composed FSTs determinizable.  They are treated "specially" by the
  // context FST in that they are not part of the context, they are just "passed
  // through" via self-loops.  See the Mohri chapter mrentioned above for more
  // information.
  kaldi::ConstIntegerSet<Label> disambig_syms_;

  // subsequential_symbol_, represented as "$" in the Mohri chapter mentioned
  // above, is something which terminates phonetic sequences to force out the
  // last phones-in-context.  In our implementation it's added to det(LG) as a
  // self-loop on final states before composing with C.
  // (c.f. AddSubsequentialLoop()).
  Label subsequential_symbol_;


  // pseudo_eps_symbol_, which in printed form we refer to as "#-1", is a symbol that
  // appears on the ilabels of the context transducer C, i.e. the olabels of this
  // FST which is C's inverse.  It is a symbol we introduce to solve a special problem
  // in systems with right-context (context_width_ > central_position_ + 1) that
  // use disambiguation symbols.  It exists to prevent CLG from being nondeterminizable.
  //
  // The issue is that, in this case, the disambiguation symbols are shifted
  // left w.r.t. the phones, and there becomes an ambiguity, if a disambiguation
  // symbol appears at the start of a sequence on the input of CLG, about
  // whether it was at the very start of the input of LG, or just after, say,
  // the first real phone.  This can lead to determinization failure under
  // certain circumstances.  What we do if we need pseudo_eps_symbol_ to be not
  // epsilon, we create a special symbol with symbol-id 1 and sequence
  // representation (ilabels entry) [ 0 ] .
  int32 pseudo_eps_symbol_;

  // maps from vector<int32>, representing phonetic contexts of length
  // context_width_ - 1, to StateId.  (The states of the "C" fst correspond to
  // phonetic contexts, but we only create them as and when they are needed).
  VectorToStateMap state_map_;

  // The inverse of 'state_map_': gives us the phonetic context corresponding to
  // each state-id.
  std::vector<std::vector<int32> > state_seqs_;

  // maps from vector<int32>, representing phonetic contexts of length
  // context_width_ - 1, to Label.  These are actually the output labels of this
  // InverseContextFst (because of the "Inverse" part), but for historical
  // reasons and because we've used the term ilabels" in the documentation, we
  // still call these "ilabels").
  VectorToLabelMap ilabel_map_;

  // ilabel_info_ is the reverse map of ilabel_map_.
  // Indexed by olabel (although we call this ilabel_info_ for historical
  // reasons and because is for the ilabels of C), ilabel_info_[i] gives
  // information about the meaning of each symbol on the input of C
  // aka the output of inv(C).
  // See "http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel".
  std::vector<std::vector<int32> > ilabel_info_;

};

}  // namespace fst


#endif  // KALDI_FSTEXT_CONTEXT_FST_H_
