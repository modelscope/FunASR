// lm/const-arpa-lm.h

// Copyright 2014  Guoguo Chen

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

#ifndef KALDI_LM_CONST_ARPA_LM_H_
#define KALDI_LM_CONST_ARPA_LM_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "fstext/deterministic-fst.h"
#include "lm/arpa-file-parser.h"
#include "util/common-utils.h"

namespace kaldi {

/**
    The following explains how the const arpa LM works. We will start from a toy
    example, and gradually get to the existing framework. Related classes are:
    LmState, ConstArpaLmBuilder and ConstArpaLm.

    First, let's explain how we can compute LM scores from an Arpa file. Suppose
    we want to get the N-gram prob for "A B C". We can code the lookup something
    very roughly like this:

    float GetNgramLogprob(hist, word) {  // hist = "A B", word = "C"
      backoff_logprob = 0.0;
      if ((state = GetLmState(hist)) != NULL) {
        // "A B" exists as a prefix in the LM
        if (state->HasWord(word)) {
          return state->Logprob(word);
        } else {
          // We'll need to backoff to "B C", but include the backoff penalty.
          backoff_logprob = state->BackoffLogprob();
        }
      }
      return backoff_logprob + GetNgramLogprob(hist_minus_first_word, word);
    }

    In terms of data-structures, in the most abstract form of it would be
    something like the following (note, we assume words in the lexicon can be
    represented as int32, and that these indexes are nonnegative):

    class LmState {  // e.g., LmState for "A B"
      // This is the actual LM-prob of this sequence, e.g. if this state is
      // "A B" then it would be the logprob of "A -> B".
      float logprob_;

      // Backoff probability for LM-state "A B" -> "X" backing off to "B" -> "X"
      // if "A B X" is not present in the language model.
      float backoff_logprob_;

      // e.g. "C" -> LmState of "A B C".
      std::unordered_map<int32, LmState*> children_;
    };

    The above design is very memory inefficient for two reasons:
    1. Suppose "A B" has no children, i.e. no C such that "A B C" is an n-gram.
       In this case the backoff_logprob will be zero and the 'children' vector
       will be empty. So all we need is the "float logprob_". Let's call "A B" a
       leaf in this case.
    2. The map std::unordered_map uses a lot of memory.

    A first iteration of making this efficient is to get rid of the map as
    follows:

    class LmState {
      float logprob_;
      float backoff_logprob_;
      std::vector<std::pair<int32, int32> > children_;
    };

    Here, the 'children_' vector contains pairs (child_word, child_info), sorted
    by 'child_word' so we can use binary search to locate the entry. We have to
    do some fancy bit-work to avoid having to allocate an LmState if a given
    N-gram is a leaf. We design the child_info in the children_ vector as
    follows:
    1. If it's an even number, then it represents a float (i.e. we
       reinterpret_cast to float), and the associated N-gram is a leaf. This
       requires losing the least significant bit of information in the float.
    2. If it's an odd number, then it will be used to represent a pointer to the
       LmState of the child. In order to use a 32-bit number to represent a
       possibly 64-bit pointer, we store the LmState structures in memory in
       a way that's sorted lexicographically by the vector of words, so that
       following "A B" will be the LmStates for "A B A", "A B B", "A B C" and so
       on (note, we actually deal with integers instead of letters). So if we
       make the pointers relative to the current LmState, most of them will be
       quite small (and all will be positive, due to the lexicographic sorting).
       As for the pointers that are too large, if any, we can have an "overflow
       buffer" indexed by a 30-bit index that stores, directly as pointers, the
       child LmStates. We use the first bit to distinguish the relative pointer
       case and the overflow pointer case, i.e.,
       a. If (child_info / 2) is positive, then (current_lmstate_pointer +
          child_info / 2) is the address of the child LmState.
       b. If (child_info / 2) is negative, then -1 * (child_info / 2) is the
          index into the overflow buffer which gives the address of the child
          LmState.

    Note that unigram LM-states are usually frequently accessed, so it makes
    sense to assign one LmState to each single word even if it would otherwise
    be "leaf" as defined above. We then can have an array of those unigram
    LM-states for efficient lookup.

    Also, we define the class LmState just to set up data structure for Arpa
    LM. In the end, we have a class like the following:

    class ConstArpaLm {
     public:
      // Some public functions.
     private:
      // Index of largest word-id, plus one; defines end of "unigram_states_"
      // array.
      int32 num_words_;

      // Loopup table for pointers of unigrams. The pointer could be NULL, for
      // example for those words that are in words.txt, but not in the language
      // model.
      int32 **unigram_states_;

      // Number of entries in the overflow buffer for pointers that couldn't be
      // represented as a 30-bit relative index
      int32 overflow_buffer_size_;

      // Technically a 32-bit number cannot represent a possibly 64-bit pointer.
      // We therefore use "relative" address instead of "absolute" address,
      // which will be a small number most of the time. This buffer is for the
      // case where the relative address has more than 30-bits.
      int32 **overflow_buffer_;

      // Size of the array lm_states_. This is required only for I/O.
      int64 lm_states_size_;

      // Data block for LmState.
      int32 *lm_states_;
    };

    Note, when we do I/O, we don't write out the arrays of pointers
    "overflow_buffer_" and "unigram_states_" directly. Instead we subtract
    "lm_states_" from each one before writing them out, so we are writing out
    indexes. Then, when we read them back in, after we allocate "lm_states_"
    we can convert them back to pointers. When we create these temporary arrays
    of indexes while reading and writing, we use int64, even if the pointer type
    of the machine is int32. This way the I/O is independent of the pointer size
    of the machine.

    Now it is time to put things together.

    ConstArpaLmBuilder takes charge of reading in the Arpa LM and building the
    ConstArpaLm.

    ConstArpaLM holds the Arpa LM in memory, and provides interfaces for LM
    operations, such as GetNgramLogprob().

    LmState is an auxiliary class that computes the relative pointers for
    ConstArpaLmBuilder and ConstArpaLm. It will only be called once during the
    building process, so it doesn't have to be very efficient.

    In summary, the general building process is as follows:
    1. In ConstArpaLmBuilder, read in the Arpa format LM. While reading, we keep
       in memory something like this:
         std::unordered_map<std::vector<int32>,
                            LmState*, VectorHasher<int32> > seq_to_state_;
       The map helps us to convert n-gram entries into LmState (including
       setting up the parent-children relationship, see above about LmState).
       Note that at this stage, we don't work on the relative pointers yet.
    2. In ConstArpaLmBuilder, create a sorted vector from <seq_to_state_>
         std::vector<std::pair<std::vector<int32>*, LmState*> > sorted_vec;
       Note, only LmState with non-zero MemSize() should be put into the sorted
       vector, and we sort it lexicographically according to the word.
    3. In ConstArpaLmBuilder, update the address for each LmState, relative to
       the first LmState in the sorted vector (i.e. assume the first LmState has
       address 0, and work out the rest LmState address using the MemSize() of
       each LmState).
    4. In ConstArpaLmBuilder, create a memory block for all the LmStates (after
       sorting and updating the address). This includes <lm_state_> that stores
       all the LmStates in an int32 array, <unigram_states_> that keeps the
       address of unigram LmStates, <overflow_buffer_> that keeps the address
       of LmState whose address differs too much from the parent address. See
       above how we handle the leaf case.
    5. With the information in step 4, create the class ConstArpaLm.
*/

// Forward declaration of Auxiliary struct ArpaLine.
struct ArpaLine;

union Int32AndFloat {
  int32 i;
  float f;

  Int32AndFloat() {}
  Int32AndFloat(int32 input_i) : i(input_i) {}
  Int32AndFloat(float input_f) : f(input_f) {}
};

class ConstArpaLm {
 public:

  // Default constructor, will be used if you are going to load the ConstArpaLm
  // format language model from disk.
  ConstArpaLm() {
    lm_states_ = NULL;
    unigram_states_ = NULL;
    overflow_buffer_ = NULL;
    memory_assigned_ = false;
    initialized_ = false;
    ngram_order_ = 0;
  }

  // Special constructor, will be used when you initialize ConstArpaLm from
  // scratch through this constructor.
  ConstArpaLm(const int32 bos_symbol, const int32 eos_symbol,
              const int32 unk_symbol, const int32 ngram_order,
              const int32 num_words, const int32 overflow_buffer_size,
              const int64 lm_states_size, int32** unigram_states,
              int32** overflow_buffer, int32* lm_states) :
      bos_symbol_(bos_symbol), eos_symbol_(eos_symbol),
      unk_symbol_(unk_symbol), ngram_order_(ngram_order),
      num_words_(num_words), overflow_buffer_size_(overflow_buffer_size),
      lm_states_size_(lm_states_size), unigram_states_(unigram_states),
      overflow_buffer_(overflow_buffer), lm_states_(lm_states) {
    KALDI_ASSERT(unigram_states_ != NULL);
    KALDI_ASSERT(overflow_buffer_ != NULL);
    KALDI_ASSERT(lm_states_ != NULL);
    KALDI_ASSERT(ngram_order_ > 0);
    KALDI_ASSERT(bos_symbol_ < num_words_ && bos_symbol_ > 0);
    KALDI_ASSERT(eos_symbol_ < num_words_ && eos_symbol_ > 0);
    KALDI_ASSERT(unk_symbol_ < num_words_ &&
                 (unk_symbol_ > 0 || unk_symbol_ == -1));
    lm_states_end_ = lm_states_ + lm_states_size_ - 1;
    memory_assigned_ = false;
    initialized_ = true;
  }

  ~ConstArpaLm() {
    if (memory_assigned_) {
      delete[] lm_states_;
      delete[] unigram_states_;
      delete[] overflow_buffer_;
    }
  }

  // Reads the ConstArpaLm format language model. It calls ReadInternal() or
  // ReadInternalOldFormat() to do the actual reading.
  void Read(std::istream &is, bool binary);

  // Writes the language model in ConstArpaLm format.
  void Write(std::ostream &os, bool binary) const;

  // Creates Arpa format language model from ConstArpaLm format, and writes it
  // to output stream. This will be useful in testing.
  void WriteArpa(std::ostream &os) const;

  // Wrapper of GetNgramLogprobRecurse. It first maps possible out-of-vocabulary
  // words to <unk>, if <unk> is defined, and then calls GetNgramLogprobRecurse.
  float GetNgramLogprob(const int32 word, const std::vector<int32>& hist) const;

  // Returns true if the history word sequence <hist> has successor, which means
  // <hist> will be a state in the FST format language model.
  bool HistoryStateExists(const std::vector<int32>& hist) const;

  int32 BosSymbol() const { return bos_symbol_; }
  int32 EosSymbol() const { return eos_symbol_; }
  int32 UnkSymbol() const { return unk_symbol_; }
  int32 NgramOrder() const { return ngram_order_; }
  bool Initialized() const { return initialized_; }

 private:
  // Function that loads data from stream to the class.
  void ReadInternal(std::istream &is, bool binary);

  // Function that loads data from stream to the class. This is a deprecated one
  // that handles the old on-disk format. We keep this for back-compatibility
  // purpose. We have modified the Write() function so for all the new on-disk
  // format, ReadInternal() will be called.
  void ReadInternalOldFormat(std::istream &is, bool binary);

  // Loops up n-gram probability for given word sequence. Backoff is handled by
  // recursively calling this function.
  float GetNgramLogprobRecurse(const int32 word,
                               const std::vector<int32>& hist) const;

  // Given a word sequence, find the address of the corresponding LmState.
  // Returns NULL if no corresponding LmState is found.
  //
  // If the word sequence exists in n-gram language model, but it is a leaf and
  // is not an unigram, we still return NULL, since there is no LmState struct
  // reserved for this sequence.
  int32* GetLmState(const std::vector<int32>& seq) const;

  // Given a pointer to the parent, find the child_info that corresponds to
  // given word. The parent has the following structure:
  // struct LmState {
  //   float logprob;
  //   float backoff_logprob;
  //   int32 num_children;
  //   std::pair<int32, int32> [] children;
  // }
  // It returns false if the child is not found.
  bool GetChildInfo(const int32 word, int32* parent, int32* child_info) const;

  // Decodes <child_info> to get log probability and child LmState. In the leaf
  // case, only <logprob> will be returned, and <child_address> will be NULL.
  void DecodeChildInfo(const int32 child_info, int32* parent,
                       int32** child_lm_state, float* logprob) const;

  void WriteArpaRecurse(int32* lm_state,
                        const std::vector<int32>& seq,
                        std::vector<ArpaLine> *output) const;

  // We assign memory in Read(). If it is called, we have to release memory in
  // the destructor.
  bool memory_assigned_;

  // Makes sure that the language model has been loaded before using it.
  bool initialized_;

  // Integer corresponds to <s>.
  int32 bos_symbol_;

  // Integer corresponds to </s>.
  int32 eos_symbol_;

  // Integer corresponds to unknown-word. -1 if no unknown-word symbol is
  // provided.
  int32 unk_symbol_;

  // N-gram order of the language model.
  int32 ngram_order_;

  // Index of largest word-id plus one. It defines the end of <unigram_states_>
  // array.
  int32 num_words_;

  // Number of entries in the overflow buffer for pointers that couldn't be
  // represented as a 30-bit relative index.
  int32 overflow_buffer_size_;

  // Size of the <lm_states_> array, which will be needed by I/O.
  int64 lm_states_size_;

  // Points to the end of <lm_states_>. We use this information to check if
  // there is any illegal visit to the un-reserved memory.
  int32* lm_states_end_;

  // Loopup table for pointers of unigrams. The pointer could be NULL, for
  // example for those words that are in words.txt, but not in the language
  // model.
  int32** unigram_states_;

  // Technically a 32-bit number cannot represent a possibly 64-bit pointer. We
  // therefore use "relative" address instead of "absolute" address, which will
  // be a small number most of the time. This buffer is for the case where the
  // relative address has more than 30-bits.
  int32** overflow_buffer_;

  // Memory chunk that contains the actual LmStates. One LmState has the
  // following structure:
  //
  // struct LmState {
  //   float logprob;
  //   float backoff_logprob;
  //   int32 num_children;
  //   std::pair<int32, int32> [] children;
  // }
  //
  // Note that the floating point representation has 4 bytes, int32 also has 4
  // bytes, therefore one LmState will occupy the following number of bytes:
  //
  // x = 1 + 1 + 1 + 2 * children.size() = 3 + 2 * children.size()
  int32* lm_states_;
};

/**
 This class wraps a ConstArpaLm format language model with the interface defined
 in DeterministicOnDemandFst.
 */
class ConstArpaLmDeterministicFst
  : public fst::DeterministicOnDemandFst<fst::StdArc> {
 public:
  typedef fst::StdArc::Weight Weight;
  typedef fst::StdArc::StateId StateId;
  typedef fst::StdArc::Label Label;

  explicit ConstArpaLmDeterministicFst(const ConstArpaLm& lm);

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual StateId Start() { return start_state_; }

  // We cannot use "const" because the pure virtual function in the interface is
  // not const.
  virtual Weight Final(StateId s);

  virtual bool GetArc(StateId s, Label ilabel, fst::StdArc* oarc);

 private:
  typedef unordered_map<std::vector<Label>,
                        StateId, VectorHasher<Label> > MapType;
  StateId start_state_;
  MapType wseq_to_state_;
  std::vector<std::vector<Label> > state_to_wseq_;
  const ConstArpaLm& lm_;
};

// Reads in an Arpa format language model and converts it into ConstArpaLm
// format. We assume that the words in the input Arpa format language model have
// been converted into integers.
bool BuildConstArpaLm(const ArpaParseOptions& options,
                      const std::string& arpa_rxfilename,
                      const std::string& const_arpa_wxfilename);

}  // namespace kaldi

#endif  // KALDI_LM_CONST_ARPA_LM_H_
