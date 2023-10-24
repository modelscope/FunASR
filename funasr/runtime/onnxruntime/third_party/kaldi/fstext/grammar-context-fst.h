// fstext/grammar-context-fst.h

// Copyright   2018  Johns Hopkins University (author: Daniel Povey)

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


#ifndef KALDI_FSTEXT_GRAMMAR_CONTEXT_FST_H_
#define KALDI_FSTEXT_GRAMMAR_CONTEXT_FST_H_

/* This header defines a special form of the context FST "C" (the "C" in "HCLG")
   that integrates with our framework for building dynamic graphs for grammars
   that are too big to statically create, or graphs with on-the-fly pieces that
   you want to create at recognition time without building the whole graph.

   This framework is limited to only work with models with left-biphone context.
   (Fortunately this doesn't impact results, as our best models are all 'chain'
   models with left biphone context).

   The main code exported from here is the class InverseLeftBiphoneContextFst,
   which is similar to the InverseContextFst defined in context-fst.h, but
   is limited to left-biphone context and also supports certain special
   extensions we need to compile grammars.

   See \ref grammar (../doc/grammar.dox) for high-level
   documentation on how this framework works.
*/


#include <algorithm>
#include <string>
#include <vector>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "util/const-integer-set.h"
#include "fstext/deterministic-fst.h"
#include "fstext/context-fst.h"

namespace fst {


/**
   An anonymous enum to define some values for symbols used in our grammar-fst
   framework.  Please understand this with reference to the documentation in
   \ref grammar (../doc/grammar.dox).  This enum defines
   the values of nonterminal-related symbols in phones.txt.  They are not
   the actual values-- they will be shifted by adding the value
   nonterm_phones_offset which is passed in by the command-line flag
   --nonterm-phones-offset.

 */

enum NonterminalValues {
  kNontermBos = 0,  // #nonterm_bos
  kNontermBegin = 1,  // #nonterm_begin
  kNontermEnd = 2,  // #nonterm_end
  kNontermReenter = 3,  // #nonterm_reenter
  kNontermUserDefined = 4,   // the lowest-numbered user-defined nonterminal, e.g. #nonterm:foo
  // kNontermMediumNumber and kNontermBigNumber come into the encoding of
  // nonterminal-related symbols in HCLG.fst.  The only hard constraint on them
  // is that kNontermBigNumber must be bigger than the biggest transition-id in
  // your system, and kNontermMediumNumber must be >0.  These values were chosen
  // for ease of human inspection of numbers encoded with them.
  kNontermMediumNumber = 1000,
  kNontermBigNumber = 10000000
};



// Returns the smallest multiple of 1000 that is strictly greater than
// nonterm_phones_offset.  Used in the encoding of special symbol in HCLG;
// they are encoded as
//  special_symbol =
//     kNontermBigNumber + (nonterminal * encoding_multiple) + phone_index
inline int32 GetEncodingMultiple(int32 nonterm_phones_offset) {
  int32 medium_number = static_cast<int32>(kNontermMediumNumber);
  return medium_number *
      ((nonterm_phones_offset + medium_number) / medium_number);
}

/**
   This is a variant of the function ComposeContext() which is to be used
   with our "grammar FST" framework (see \ref graph_context, i.e.
   ../doc/grammar.dox, for more details).  This does not take
   the 'context_width' and 'central_position' arguments because they are
   assumed to be 2 and 1 respectively (meaning, left-biphone phonetic context).

   This function creates a context FST and composes it on the left with "ifst"
   to make "ofst".

    @param [in] nonterm_phones_offset  The integer id of the symbol
                  #nonterm_bos in the phones.txt file.  You can just set this
                  to a large value (like 1 million) if you are not actually using
                  nonterminals (e.g. for testing purposes).
    @param [in] disambig_syms  List of disambiguation symbols, e.g. the integer
                 ids of #0, #1, #2 ... in the phones.txt.
    @param [in,out] ifst   The FST we are composing with C (e.g. LG.fst).
    @param [out] ofst   Composed output FST (would be CLG.fst).
    @param [out] ilabels  Vector, indexed by ilabel of CLG.fst, providing information
                  about the meaning of that ilabel; see \ref tree_ilabel
                  (http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel)
                  and also \ref grammar_special_clg
                  (http://kaldi-asr.org/doc/grammar#grammar_special_clg).
  */
void ComposeContextLeftBiphone(
    int32 nonterm_phones_offset,
    const std::vector<int32> &disambig_syms,
    const VectorFst<StdArc> &ifst,
    VectorFst<StdArc> *ofst,
    std::vector<std::vector<int32> > *ilabels);



/*
   InverseLeftBiphoneContextFst represents the inverse of the context FST "C" (the "C" in
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

class InverseLeftBiphoneContextFst: public DeterministicOnDemandFst<StdArc> {
public:
  typedef StdArc Arc;
  typedef typename StdArc::StateId StateId;
  typedef typename StdArc::Weight Weight;
  typedef typename StdArc::Label Label;

  /**
     Constructor.  This does not take the arguments 'context_width' or
     'central_position' because they are assumed to be (2, 1) meaning a
     system with left-biphone context; and there is no subsequential
     symbol because it is not needed in systems without right context.

        @param [in] nonterm_phones_offset  The integer id of the symbol
                  #nonterm_bos in the phones.txt file. You can just set this to
                  a large value (like 1 million) if you are not actually using
                  nonterminals (e.g. for testing purposes).
        @param [in] phones      List of integer ids of phones, as you would see in phones.txt
        @param [in] disambig_syms   List of integer ids of disambiguation symbols,
                                   e.g. the ids of #0, #1, #2 in phones.txt

     See \ref graph_context for more details.
  */
  InverseLeftBiphoneContextFst(Label nonterm_phones_offset,
                               const std::vector<int32>& phones,
                               const std::vector<int32>& disambig_syms);

  /**
     Here is a note on the state space of InverseLeftBiphoneContextFst;
     see \ref grammar_special_c which has some documentation on this.

     The state space uses the same numbering as phones.txt.

       State 0 means the beginning-of-sequence state, where there is no left
       context.

       For each phone p in the list 'phones' passed to the constructor (i.e. in
       the set passed to the constructor), the state 'p' corresponds to a
       left-context of that phone.

       If p is equal to nonterm_phones_offset_ + kNontermBegin (i.e. the
       integer form of `\#nonterm_begin`), then this is the state we transition
       to when we see that symbol starting from left-context==0 (no context).  The
       transition to this special state will have epsilon on the output.  (talking
       here about inv(C), not C, so input/output are reversed).
       The state is nonfinal and when we see a regular phone p1 or #nonterm_bos, instead of
       outputting that phone in context, we output the pair (#nonterm_begin,p1) or
       (#nonterm_begin,#nonterm_bos).  This state is not final.

       If p is equal to nonterm_phones_offset_ + kNontermUserDefined, then this
       is the state we transition to when we see any user-defined nonterminal.
       Transitions to this special state have olabels of the form (#nonterm:foo,p1)
       where p1 is the preceding context (with #nonterm_begin if that context was
       0); transitions out of it have olabels of the form (#nonterm_reenter,p2), where
       p2 is the phone on the ilabel of that transition.  Again: talking about inv(C).
       This state is not final.

       If p is equal to nonterm_phones_offset_ + kNontermEnd, then this is
       the state we transition to when we see the ilabel #nonterm_end.  The olabels
       on the transitions to it (talking here about inv(C), so ilabels and olabels
       are reversed) are of the form (#nonterm_end, p1) where p1 corresponds to the
       context we were in.  This state is final.
   */


  virtual StateId Start() { return 0; }

  virtual Weight Final(StateId s);

  /// Note: ilabel must not be epsilon.
  virtual bool GetArc(StateId s, Label ilabel, Arc *arc);

  ~InverseLeftBiphoneContextFst() { }

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

  inline int32 GetPhoneSymbolFor(enum NonterminalValues n) {
    return nonterm_phones_offset_ + static_cast<int32>(n);
  }

  /// Finds the label index corresponding to this context-window of phones
  /// (likely of width context_width_).  Inserts it into the
  /// ilabel_info_/ilabel_map_ tables if necessary.
  Label FindLabel(const std::vector<int32> &label_info);


  // Map type to map from vectors of int32 (representing ilabel-info,
  // see http://kaldi-asr.org/doc/tree_externals.html#tree_ilabel) to
  // Label (the output label in this FST).
  typedef unordered_map<std::vector<int32>, Label,
                        kaldi::VectorHasher<int32> > VectorToLabelMap;


  // The following three variables were also passed in by the caller:
  int32 nonterm_phones_offset_;

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


#endif  // KALDI_FSTEXT_GRAMMAR_CONTEXT_FST_H_
