// lat/word-align-lattice-lexicon-test.cc

// Copyright    2015  Johns Hopkins University (Author: Daniel Povey)

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

#include "lat/determinize-lattice-pruned.h"
#include "fstext/lattice-utils.h"
#include "fstext/fst-test-utils.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "lat/lattice-functions-transition-model.h"
#include "hmm/hmm-test-utils.h"
#include "lat/word-align-lattice-lexicon.h"

#include <random>

namespace kaldi {

// This function generates a lexicon in the same format that
// WordAlignLatticeLexicon uses: (original-word-id), (new-word-id), (phone-seq).
void GenerateLexicon(const std::vector<int32> &phones,
                     bool allow_zero_words,
                     bool allow_empty_word,
                     bool allow_multiple_prons,
                     std::vector<std::vector<int32> > *lexicon) {
  KALDI_ASSERT(!phones.empty());
  lexicon->clear();
  int32 num_words = RandInt(1, 20);
  for (int32 word = 1; word <= num_words; word++) {
    int32 num_prons = RandInt(1, (allow_multiple_prons ? 2 : 1));
    bool is_zero_word = allow_zero_words && (RandInt(1, 5) == 1);

    for (int32 j = 0; j < num_prons; j++) {
      // don't allow empty pron if this word isn't labeled in the lattice (zero word,
      // like optional silence).  This doesn't make sense.
      int32 pron_length = RandInt(((allow_empty_word && !is_zero_word) ? 0 : 1),
                                  4);
      std::vector<int32> this_entry;
      this_entry.push_back(is_zero_word ? 0 : word);
      this_entry.push_back(word);
      for (int32 p = 0; p < pron_length; p++)
        this_entry.push_back(phones[RandInt(0, phones.size() - 1)]);
      lexicon->push_back(this_entry);
    }
  }
  SortAndUniq(lexicon);

  // randomize the order.
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(lexicon->begin(), lexicon->end(), g);


  for (size_t i = 0; i < lexicon->size(); i++) {
    if ((*lexicon)[i].size() > 2) {
      // ok, this lexicon has at least one nonempty word: potentially OK.  Do
      // further check that the info object doesn't complain.
      try {
        WordAlignLatticeLexiconInfo info(*lexicon);
        return;  // OK, we're satisfied with this lexicon.
      } catch (...) {
        break;  // will re-try, see below.
      }
    }
  }
  // there were no nonempty words in the lexicon -> try again.
  // recursing is the easiest way.
  GenerateLexicon(phones, allow_zero_words, allow_empty_word, allow_multiple_prons,
                  lexicon);


}


static void PrintLexicon(const std::vector<std::vector<int32> > &lexicon) {
  KALDI_LOG << "Lexicon is: ";
  for (size_t i = 0; i < lexicon.size(); i++) {
    KALDI_ASSERT(lexicon[i].size() >= 2);
    const std::vector<int32> &entry = lexicon[i];
    std::cerr << entry[0] << "\t" << entry[1] << "\t";
    for (size_t j = 2; j < entry.size(); j++)
      std::cerr << entry[j] << " ";
    std::cerr << "\n";
  }
}

static void PrintWordsAndPhones(const std::vector<int32> &words,
                                const std::vector<int32> &phones) {
  std::ostringstream word_str, phone_str;
  for (size_t i = 0; i < words.size(); i++)
    word_str << words[i] << " ";
  for (size_t i = 0; i < phones.size(); i++)
    phone_str << phones[i] << " ";
  KALDI_LOG << "Word-sequence is: " << word_str.str();
  KALDI_LOG << "Phone-sequence is: " << phone_str.str();
}


// generates a phone and word sequence together from the lexicon.  Not
// guaranteed nonempty.
void GenerateWordAndPhoneSequence(std::vector<std::vector<int32> > &lexicon,
                                  std::vector<int32> *phone_seq,
                                  std::vector<int32> *word_seq) {
  int32 num_words = RandInt(0, 5);
  phone_seq->clear();
  word_seq->clear();
  for (int32 i = 0; i < num_words; i++) {
    const std::vector<int32> &lexicon_entry =
        lexicon[RandInt(0, lexicon.size() - 1)];
    // the zeroth element of 'lexicon_entry' is how it appears in
    // the lattice prior to word alignment.
    int32 word = lexicon_entry[0];
    if (word != 0) word_seq->push_back(word);
    // add everything from position 2 in the lexicon entry, to the
    // phone sequence.
    phone_seq->insert(phone_seq->end(),
                      lexicon_entry.begin() + 2,
                      lexicon_entry.end());
  }
}



void GenerateCompactLatticeRandomly(const std::vector<int32> &alignment,
                                    const std::vector<int32> &words,
                                    CompactLattice *clat) {
  clat->DeleteStates();
  clat->AddState();
  clat->SetStart(0);
  int32 cur_state = 0;
  size_t word_start = 0, alignment_start = 0,
      num_words = words.size(), num_transition_ids = alignment.size();
  for (; word_start < num_words; word_start++) {
    int32 word = words[word_start];
    int32 ali_length = RandInt(0, num_transition_ids - alignment_start);
    std::vector<int32> this_ali(ali_length);
    for (int32 i = 0; i < ali_length; i++)
      this_ali[i] = alignment[alignment_start + i];
    alignment_start += ali_length;
    CompactLatticeWeight weight(LatticeWeight::One(), this_ali);
    int32 ilabel = word;
    int32 next_state = clat->AddState();
    CompactLatticeArc arc(ilabel, ilabel, weight, next_state);
    clat->AddArc(cur_state, arc);
    cur_state = next_state;
  }
  if (alignment_start < alignment.size()) {
    int32 ali_length = num_transition_ids - alignment_start;
    std::vector<int32> this_ali(ali_length);
    for (int32 i = 0; i < ali_length; i++)
      this_ali[i] = alignment[alignment_start + i];
    alignment_start += ali_length;
    CompactLatticeWeight weight(LatticeWeight::One(), this_ali);
    int32 ilabel = 0;
    int32 next_state = clat->AddState();
    CompactLatticeArc arc(ilabel, ilabel, weight, next_state);
    clat->AddArc(cur_state, arc);
    cur_state = next_state;
  }
  clat->SetFinal(cur_state, CompactLatticeWeight::One());
}



void TestWordAlignLatticeLexicon() {
  ContextDependency *ctx_dep;
  TransitionModel *trans_model = GenRandTransitionModel(&ctx_dep);
  bool allow_zero_words = true;
  bool allow_empty_word = true;
  bool allow_multiple_prons = true;

  const std::vector<int32> &phones = trans_model->GetPhones();
  std::vector<std::vector<int32> > lexicon;
  GenerateLexicon(phones, allow_zero_words, allow_empty_word,
                  allow_multiple_prons, &lexicon);

  std::vector<int32> phone_seq;
  std::vector<int32> word_seq;
  while (phone_seq.empty())
    GenerateWordAndPhoneSequence(lexicon, &phone_seq, &word_seq);

  PrintLexicon(lexicon);
  PrintWordsAndPhones(word_seq, phone_seq);

  std::vector<int32> alignment;
  bool reorder = (RandInt(0, 1) == 0);
  GenerateRandomAlignment(*ctx_dep, *trans_model, reorder,
                          phone_seq, &alignment);

  CompactLattice clat;
  GenerateCompactLatticeRandomly(alignment, word_seq, &clat);

  KALDI_LOG << "clat is ";
  WriteCompactLattice(std::cerr, false, clat);

  WordAlignLatticeLexiconOpts opts;
  WordAlignLatticeLexiconInfo lexicon_info(lexicon);
  opts.reorder = reorder;
  CompactLattice aligned_clat;
  bool allow_duplicate_paths = true;
  bool ans = WordAlignLatticeLexicon(clat, *trans_model, lexicon_info, opts,
                                     &aligned_clat);
  if (ans) { // We only test if it succeeded.
      if (!TestWordAlignedLattice(lexicon_info, *trans_model, clat, aligned_clat,
                                  allow_duplicate_paths)) {
          KALDI_WARN << "Lattice failed test (activated because --test=true). "
                     << "Probable code error, please contact Kaldi maintainers.";
          ans = false;
      }
  }

  KALDI_LOG << "Aligned clat is ";
  WriteCompactLattice(std::cerr, false, aligned_clat);
  KALDI_ASSERT(ans);

  Lattice lat;
  ConvertLattice(clat, &lat);
  int32 n = 1000;  // a maximum.
  Lattice nbest_lat;
  std::vector<Lattice> nbest_lats;
  fst::ShortestPath(lat, &nbest_lat, n);
  fst::ConvertNbestToVector(nbest_lat, &nbest_lats);
  KALDI_LOG << "Word-aligned lattice has " << nbest_lats.size() << " paths.";

  delete ctx_dep;
  delete trans_model;
}

} // end namespace kaldi

int main() {
  for (int32 i = 0; i < 3; i++)
    kaldi::TestWordAlignLatticeLexicon();
  std::cout << "Tests succeeded\n";
}

