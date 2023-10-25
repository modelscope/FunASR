// Copyright 2020  Jiayu DU

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
#ifdef HAVE_KENLM
#ifndef KALDI_LM_KENLM_H
#define KALDI_LM_KENLM_H

#include <base/kaldi-common.h>
#include <fst/fstlib.h>
#include <fst/fst-decl.h>
#include <fstext/deterministic-fst.h>

#include "lm/model.hh"
#include "util/murmur_hash.hh"

namespace kaldi {

// KenLm class wraps kenlm model(supporting both "trie" or "probing" models):
//  1. provides interface for loading binary LM, and holds it with ownership
//  2. provides interface for ngram score query at runtime
//  3. handles the index mapping between kaldi's symbols & kenlm's words
// KenLm object is heavy, stateless and thread-safe, 
// can be shared by Fst wrapper class(i.e. KenLmDeterministicOnDemandFst)
class KenLm {
 public:
  typedef lm::WordIndex WordIndex;
  typedef lm::ngram::State State;

 public:
  KenLm() : 
    model_(nullptr), vocab_(nullptr),
    bos_sym_("<s>"), eos_sym_("</s>"), unk_sym_("<unk>"),
    bos_symid_(0), eos_symid_(0), unk_symid_(0)
  { }

  ~KenLm() {
    if (model_ != nullptr) {
      delete model_;
    }
    model_ = nullptr;
    vocab_ = nullptr;
    symid_to_wid_.clear();
  }

  // If you have big LM on SSD hard-drive,
  // you can set load_method to util::LoadMethod::LAZY,
  // which enables "on-demand" model reading(via POSIX mmap) at runtime.
  // Refer to tools/kenlm/util/mmap.hh for more load methods.
  int Load(std::string kenlm_filename, 
           std::string kaldi_symbol_table_filename,
           util::LoadMethod load_method = util::LoadMethod::POPULATE_OR_READ);

  inline WordIndex GetWordIndex(std::string word) const {
    return vocab_->Index(word.c_str());
  }

  inline WordIndex GetWordIndex(int32 symbol_id) const {
    return symid_to_wid_[symbol_id];
  }

  void SetStateToBeginOfSentence(State *s) const { model_->BeginSentenceWrite(s); }
  void SetStateToNull(State *s) const { model_->NullContextWrite(s); }

  int32 BosSymbolIndex() const { return bos_symid_; }
  int32 EosSymbolIndex() const { return eos_symid_; }
  int32 UnkSymbolIndex() const { return unk_symid_; }

  inline BaseFloat Score(const State *in_state,
                         WordIndex word,
                         State *out_state) const {
    return model_->BaseScore(in_state, word, out_state);
  }

  // This provides a fast state hashing, 
  // KenLmDeterministicOnDemandFst needs this for Fst states managing.
  struct StateHasher {
    inline size_t operator()(const State &s) const noexcept {
      return util::MurmurHashNative(s.words, sizeof(WordIndex) * s.Length());
    }
  };

 private:
  void ComputeSymbolToWordIndexMapping(std::string symbol_table);
  
 private:
  lm::base::Model *model_; // with ownership

  // without ownership, points to internal vocabulary of model_
  const lm::base::Vocabulary* vocab_;

  // There are two integerized indexing systems here:
  // 1. Kaldi's fst output *symbol index*(defined in words.txt),
  // 2. KenLm's *word index*(defined by word string hashing).
  // In order to rescore kaldi hypotheses with kenlm ngrams, 
  // we need to know the index mapping from symbol to word.
  // KenLm class precomputes (during model loading) and stores this mapping,
  // and apply the mapping at runtime.
  // This is slower, but at least we don't need
  // to modify/convert runtime resources.(e.g. HCLG/lattices or kenlm models)
  //
  // In the mapping, <eps> and #0 symbols are special:
  // They do not correspond to any word in KenLm,
  // so the mapping of these two symbols are logically undefined,
  // we just map them to KenLm's <unk> to avoid random invalid mapping.

  // symid_to_wid_[kaldi_symbol_index] -> kenlm word index
  std::vector<WordIndex> symid_to_wid_;

  // special lm symbols
  std::string bos_sym_;
  std::string eos_sym_;
  std::string unk_sym_;

  int32 bos_symid_;
  int32 eos_symid_;
  int32 unk_symid_;
}; // class KenLm


// DeterministicOnDemandFst wraps a KenLm object as a deteministic Fst.
// Internally, it manages dynamically expanded Fst states(so not thread-safe),
// different threads should create their own instances of this class.
// They are lightweight and can share the same KenLm object.
template<class Arc>
class KenLmDeterministicOnDemandFst : public fst::DeterministicOnDemandFst<Arc> {
 public:
  typedef typename Arc::Weight Weight;
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename KenLm::State State;
  typedef typename KenLm::WordIndex WordIndex;
 
  explicit KenLmDeterministicOnDemandFst(const KenLm *lm)
   : lm_(lm), num_states_(0), bos_state_id_(0)
  {
    // create bos to be FST start state
    MapElem e;
    lm->SetStateToBeginOfSentence(&e.first);
    e.second = bos_state_id_;
    std::pair<IterType, bool> r = state_map_.insert(e);
    KALDI_ASSERT(r.second == true); // bos successfully inserted into state map
    state_vec_.push_back(&r.first->first);
    num_states_++;

    eos_symbol_id_ = lm_->EosSymbolIndex();
  }
  virtual ~KenLmDeterministicOnDemandFst() { }

  virtual StateId Start() { 
    return bos_state_id_;
  }

  virtual bool GetArc(StateId s, Label label, Arc *oarc) {
    KALDI_ASSERT(s < static_cast<StateId>(state_vec_.size()));
    const State* istate = state_vec_[s];
    MapElem e;
    WordIndex word = lm_->GetWordIndex(label);
    BaseFloat log_10_prob = lm_->Score(istate, word, &e.first);
    e.second = num_states_;
    std::pair<IterType, bool> r = state_map_.insert(e);
    if (r.second == true) { // new state
      state_vec_.push_back(&(r.first->first));
      num_states_++;
    }

    oarc->ilabel = label;
    oarc->olabel = oarc->ilabel;
    oarc->nextstate = r.first->second;
    oarc->weight = Weight(-log_10_prob * M_LN10); // KenLm log10() -> Kaldi ln()

    return true;
  }

  virtual Weight Final(StateId s) {
    Arc oarc;
    GetArc(s, eos_symbol_id_, &oarc);
    return oarc.weight;
  }

 private:
  typedef std::pair<State, StateId> MapElem;
  typedef unordered_map<State, StateId, KenLm::StateHasher> MapType;
  typedef typename MapType::iterator IterType;

  const KenLm *lm_; // no ownership
  MapType state_map_;
  std::vector<const State*> state_vec_;
  StateId num_states_; // state vector index range, [0, num_states_)
  StateId bos_state_id_;  // fst start state id
  Label eos_symbol_id_;
}; // class KenLmDeterministicOnDemandFst
} // namespace kaldi
#endif
#endif
