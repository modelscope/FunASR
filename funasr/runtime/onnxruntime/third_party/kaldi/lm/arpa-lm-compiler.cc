// lm/arpa-lm-compiler.cc

// Copyright 2009-2011 Gilles Boulianne
// Copyright 2016 Smart Action LLC (kkm)
// Copyright 2017 Xiaohui Zhang

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

#include <algorithm>
#include <limits>
#include <sstream>
#include <utility>

#include "base/kaldi-math.h"
#include "lm/arpa-lm-compiler.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"
#include "fstext/remove-eps-local.h"

namespace kaldi {

class ArpaLmCompilerImplInterface {
 public:
  virtual ~ArpaLmCompilerImplInterface() { }
  virtual void ConsumeNGram(const NGram& ngram, bool is_highest) = 0;
};

namespace {

typedef int32 StateId;
typedef int32 Symbol;

// GeneralHistKey can represent state history in an arbitrarily large n
// n-gram model with symbol ids fitting int32.
class GeneralHistKey {
 public:
  // Construct key from being and end iterators.
  template<class InputIt>
  GeneralHistKey(InputIt begin, InputIt end) : vector_(begin, end) { }
  // Construct empty history key.
  GeneralHistKey() : vector_() { }
  // Return tails of the key as a GeneralHistKey. The tails of an n-gram
  // w[1..n] is the sequence w[2..n] (and the heads is w[1..n-1], but the
  // key class does not need this operartion).
  GeneralHistKey Tails() const {
    return GeneralHistKey(vector_.begin() + 1, vector_.end());
  }
  // Keys are equal if represent same state.
  friend bool operator==(const GeneralHistKey& a, const GeneralHistKey& b) {
    return a.vector_ == b.vector_;
  }
  // Public typename HashType for hashing.
  struct HashType {
    size_t operator()(const GeneralHistKey& key) const {
      return VectorHasher<Symbol>().operator()(key.vector_);
    }
  };

 private:
  std::vector<Symbol> vector_;
};

// OptimizedHistKey combines 3 21-bit symbol ID values into one 64-bit
// machine word. allowing significant memory reduction and some runtime
// benefit over GeneralHistKey. Since 3 symbols are enough to track history
// in a 4-gram model, this optimized key is used for smaller models with up
// to 4-gram and symbol values up to 2^21-1.
//
// See GeneralHistKey for interface requirements of a key class.
class OptimizedHistKey {
 public:
  enum {
    kShift = 21,  // 21 * 3 = 63 bits for data.
    kMaxData = (1 << kShift) - 1
  };
  template<class InputIt>
  OptimizedHistKey(InputIt begin, InputIt end) : data_(0) {
    for (uint32 shift = 0; begin != end; ++begin, shift += kShift) {
      data_ |= static_cast<uint64>(*begin) << shift;
    }
  }
  OptimizedHistKey() : data_(0) { }
  OptimizedHistKey Tails() const {
    return OptimizedHistKey(data_ >> kShift);
  }
  friend bool operator==(const OptimizedHistKey& a, const OptimizedHistKey& b) {
    return a.data_ == b.data_;
  }
  struct HashType {
    size_t operator()(const OptimizedHistKey& key) const { return key.data_; }
  };

 private:
  explicit OptimizedHistKey(uint64 data) : data_(data) { }
  uint64 data_;
};

}  // namespace

template <class HistKey>
class ArpaLmCompilerImpl : public ArpaLmCompilerImplInterface {
 public:
  ArpaLmCompilerImpl(ArpaLmCompiler* parent, fst::StdVectorFst* fst,
                     Symbol sub_eps);

  virtual void ConsumeNGram(const NGram &ngram, bool is_highest);

 private:
  StateId AddStateWithBackoff(HistKey key, float backoff);
  void CreateBackoff(HistKey key, StateId state, float weight);

  ArpaLmCompiler *parent_;  // Not owned.
  fst::StdVectorFst* fst_;  // Not owned.
  Symbol bos_symbol_;
  Symbol eos_symbol_;
  Symbol sub_eps_;

  StateId eos_state_;
  typedef unordered_map<HistKey, StateId,
                        typename HistKey::HashType> HistoryMap;
  HistoryMap history_;
};

template <class HistKey>
ArpaLmCompilerImpl<HistKey>::ArpaLmCompilerImpl(
    ArpaLmCompiler* parent, fst::StdVectorFst* fst, Symbol sub_eps)
    : parent_(parent), fst_(fst), bos_symbol_(parent->Options().bos_symbol),
      eos_symbol_(parent->Options().eos_symbol), sub_eps_(sub_eps) {
  // The algorithm maintains state per history. The 0-gram is a special state
  // for empty history. All unigrams (including BOS) backoff into this state.
  StateId zerogram = fst_->AddState();
  history_[HistKey()] = zerogram;

  // Also, if </s> is not treated as epsilon, create a common end state for
  // all transitions accepting the </s>, since they do not back off. This small
  // optimization saves about 2% states in an average grammar.
  if (sub_eps_ == 0) {
    eos_state_ = fst_->AddState();
    fst_->SetFinal(eos_state_, 0);
  }
}

template <class HistKey>
void ArpaLmCompilerImpl<HistKey>::ConsumeNGram(const NGram &ngram,
                                               bool is_highest) {
  // Generally, we do the following. Suppose we are adding an n-gram "A B
  // C". Then find the node for "A B", add a new node for "A B C", and connect
  // them with the arc accepting "C" with the specified weight. Also, add a
  // backoff arc from the new "A B C" node to its backoff state "B C".
  //
  // Two notable exceptions are the highest order n-grams, and final n-grams.
  //
  // When adding a highest order n-gram (e. g., our "A B C" is in a 3-gram LM),
  // the following optimization is performed. There is no point adding a node
  // for "A B C" with a "C" arc from "A B", since there will be no other
  // arcs ingoing to this node, and an epsilon backoff arc into the backoff
  // model "B C", with the weight of \bar{1}. To save a node, create an arc
  // accepting "C" directly from "A B" to "B C". This saves as many nodes
  // as there are the highest order n-grams, which is typically about half
  // the size of a large 3-gram model.
  //
  // Indeed, this does not apply to n-grams ending in EOS, since they do not
  // back off. These are special, as they do not have a back-off state, and
  // the node for "(..anything..) </s>" is always final. These are handled
  // in one of the two possible ways, If symbols <s> and </s> are being
  // replaced by epsilons, neither node nor arc is created, and the logprob
  // of the n-gram is applied to its source node as final weight. If <s> and
  // </s> are preserved, then a special final node for </s> is allocated and
  // used as the destination of the "</s>" acceptor arc.
  HistKey heads(ngram.words.begin(), ngram.words.end() - 1);
  typename HistoryMap::iterator source_it = history_.find(heads);
  if (source_it == history_.end()) {
    // There was no "A B", therefore the probability of "A B C" is zero.
    // Print a warning and discard current n-gram.
    if (parent_->ShouldWarn())
      KALDI_WARN << parent_->LineReference()
                 << " skipped: no parent (n-1)-gram exists";
    return;
  }

  StateId source = source_it->second;
  StateId dest;
  Symbol sym = ngram.words.back();
  float weight = -ngram.logprob;
  if (sym == sub_eps_ || sym == 0) {
    KALDI_ERR << " <eps> or disambiguation symbol " << sym << "found in the ARPA file. ";
  }
  if (sym == eos_symbol_) {
    if (sub_eps_ == 0) {
      // Keep </s> as a real symbol when not substituting.
      dest = eos_state_;
    } else {
      // Treat </s> as if it was epsilon: mark source final, with the weight
      // of the n-gram.
      fst_->SetFinal(source, weight);
      return;
    }
  } else {
    // For the highest order n-gram, this may find an existing state, for
    // non-highest, will create one (unless there are duplicate n-grams
    // in the grammar, which cannot be reliably detected if highest order,
    // so we better do not do that at all).
    dest = AddStateWithBackoff(
        HistKey(ngram.words.begin() + (is_highest ? 1 : 0),
                ngram.words.end()),
        -ngram.backoff);
  }

  if (sym == bos_symbol_) {
    weight = 0;  // Accepting <s> is always free.
    if (sub_eps_ == 0) {
      // <s> is as a real symbol, only accepted in the start state.
      source = fst_->AddState();
      fst_->SetStart(source);
    } else {
      // The new state for <s> unigram history *is* the start state.
      fst_->SetStart(dest);
      return;
    }
  }

  // Add arc from source to dest, whichever way it was found.
  fst_->AddArc(source, fst::StdArc(sym, sym, weight, dest));
  return;
}

// Find or create a new state for n-gram defined by key, and ensure it has a
// backoff transition.  The key is either the current n-gram for all but
// highest orders, or the tails of the n-gram for the highest order. The
// latter arises from the chain-collapsing optimization described above.
template <class HistKey>
StateId ArpaLmCompilerImpl<HistKey>::AddStateWithBackoff(HistKey key,
                                                         float backoff) {
  typename HistoryMap::iterator dest_it = history_.find(key);
  if (dest_it != history_.end()) {
    // Found an existing state in the history map. Invariant: if the state in
    // the map, then its backoff arc is in the FST. We are done.
    return dest_it->second;
  }
  // Otherwise create a new state and its backoff arc, and register in the map.
  StateId dest = fst_->AddState();
  history_[key] = dest;
  CreateBackoff(key.Tails(), dest, backoff);
  return dest;
}

// Create a backoff arc for a state. Key is a backoff destination that may or
// may not exist. When the destination is not found, naturally fall back to
// the lower order model, and all the way down until one is found (since the
// 0-gram model is always present, the search is guaranteed to terminate).
template <class HistKey>
inline void ArpaLmCompilerImpl<HistKey>::CreateBackoff(
    HistKey key, StateId state, float weight) {
  typename HistoryMap::iterator dest_it = history_.find(key);
  while (dest_it == history_.end()) {
    key = key.Tails();
    dest_it = history_.find(key);
  }

  // The arc should transduce either <eos> or #0 to <eps>, depending on the
  // epsilon substitution mode. This is the only case when input and output
  // label may differ.
  fst_->AddArc(state, fst::StdArc(sub_eps_, 0, weight, dest_it->second));
}

ArpaLmCompiler::~ArpaLmCompiler() {
  if (impl_ != NULL)
    delete impl_;
}

void ArpaLmCompiler::HeaderAvailable() {
  KALDI_ASSERT(impl_ == NULL);
  // Use optimized implementation if the grammar is 4-gram or less, and the
  // maximum attained symbol id will fit into the optimized range.
  int64 max_symbol = 0;
  if (Symbols() != NULL)
    max_symbol = Symbols()->AvailableKey() - 1;
  // If augmenting the symbol table, assume the worst case when all words in
  // the model being read are novel.
  if (Options().oov_handling == ArpaParseOptions::kAddToSymbols)
    max_symbol += NgramCounts()[0];

  if (NgramCounts().size() <= 4 && max_symbol < OptimizedHistKey::kMaxData) {
    impl_ = new ArpaLmCompilerImpl<OptimizedHistKey>(this, &fst_, sub_eps_);
  } else {
    impl_ = new ArpaLmCompilerImpl<GeneralHistKey>(this, &fst_, sub_eps_);
    KALDI_LOG << "Reverting to slower state tracking because model is large: "
              << NgramCounts().size() << "-gram with symbols up to "
              << max_symbol;
  }
}

void ArpaLmCompiler::ConsumeNGram(const NGram &ngram) {
  // <s> is invalid in tails, </s> in heads of an n-gram.
  for (int i = 0; i < ngram.words.size(); ++i) {
    if ((i > 0 && ngram.words[i] == Options().bos_symbol) ||
        (i + 1 < ngram.words.size()
         && ngram.words[i] == Options().eos_symbol)) {
      if (ShouldWarn())
        KALDI_WARN << LineReference()
                   << " skipped: n-gram has invalid BOS/EOS placement";
      return;
    }
  }

  bool is_highest = ngram.words.size() == NgramCounts().size();
  impl_->ConsumeNGram(ngram, is_highest);
}

void ArpaLmCompiler::RemoveRedundantStates() {
  fst::StdArc::Label backoff_symbol = sub_eps_;
  if (backoff_symbol == 0) {
    // The method of removing redundant states implemented in this function
    // leads to slow determinization of L o G when people use the older style of
    // usage of arpa2fst where the --disambig-symbol option was not specified.
    // The issue seems to be that it creates a non-deterministic FST, while G is
    // supposed to be deterministic.  By 'return'ing below, we just disable this
    // method if people were using an older script.  This method isn't really
    // that consequential anyway, and people will move to the newer-style
    // scripts (see current utils/format_lm.sh), so this isn't much of a
    // problem.
    return;
  }

  fst::StdArc::StateId num_states = fst_.NumStates();


  // replace the #0 symbols on the input of arcs out of redundant states (states
  // that are not final and have only a backoff arc leaving them), with <eps>.
  for (fst::StdArc::StateId state = 0; state < num_states; state++) {
    if (fst_.NumArcs(state) == 1 && fst_.Final(state) == fst::TropicalWeight::Zero()) {
      fst::MutableArcIterator<fst::StdVectorFst> iter(&fst_, state);
      fst::StdArc arc = iter.Value();
      if (arc.ilabel == backoff_symbol) {
        arc.ilabel = 0;
        iter.SetValue(arc);
      }
    }
  }

  // we could call fst::RemoveEps, and it would have the same effect in normal
  // cases, where backoff_symbol != 0 and there are no epsilons in unexpected
  // places, but RemoveEpsLocal is a bit safer in case something weird is going
  // on; it guarantees not to blow up the FST.
  fst::RemoveEpsLocal(&fst_);
  KALDI_LOG << "Reduced num-states from " << num_states << " to "
            << fst_.NumStates();
}

void ArpaLmCompiler::Check() const {
  if (fst_.Start() == fst::kNoStateId) {
    KALDI_ERR << "Arpa file did not contain the beginning-of-sentence symbol "
              << Symbols()->Find(Options().bos_symbol) << ".";
  }
}

void ArpaLmCompiler::ReadComplete() {
  fst_.SetInputSymbols(Symbols());
  fst_.SetOutputSymbols(Symbols());
  RemoveRedundantStates();
  Check();
}

}  // namespace kaldi
