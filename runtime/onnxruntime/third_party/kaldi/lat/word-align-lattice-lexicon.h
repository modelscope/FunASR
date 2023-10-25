// lat/word-align-lattice-lexicon.h

// Copyright 2013 Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_LAT_WORD_ALIGN_LATTICE_LEXICON_H_
#define KALDI_LAT_WORD_ALIGN_LATTICE_LEXICON_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "hmm/transition-model.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {

/** Read the lexicon in the special format required for word alignment.  Each line has
   a series of integers on it (at least two on each line), representing:

   <old-word-id> <new-word-id> [<phone-id-1> [<phone-id-2> ... ] ]

   Here, <old-word-id> is the word-id that appears in the lattice before alignment, and
   <new-word-id> is the word-is that should appear in the lattice after alignment.  This
   is mainly useful when the lattice may have no symbol for the optional-silence arcs
   (so <old-word-id> would equal zero), but we want it to be output with a symbol on those
   arcs (so <new-word-id> would be nonzero).
   If the silence should not be added to the lattice, both <old-word-id> and <new-word-id>
   may be zero.

   This function is very simple: it just reads in a series of lines from a text file,
   each with at least two integers on them.
*/
bool ReadLexiconForWordAlign (std::istream &is,
                              std::vector<std::vector<int32> > *lexicon);



/// This class extracts some information from the lexicon and stores it
/// in a suitable form for the word-alignment code to use.
class WordAlignLatticeLexiconInfo {
 public:
  WordAlignLatticeLexiconInfo(const std::vector<std::vector<int32> > &lexicon);

  /// Returns true if this lexicon-entry can appear, intepreted as
  /// (output-word phone1 phone2 ...).  This is just used in testing code.
  bool IsValidEntry(const std::vector<int32> &entry) const;

  /// Purely for the testing code, we map words into equivalence classes derived
  /// from the mappings in the first two fields of each line in the lexicon.  This
  /// function maps from each word-id to the lowest member of its equivalence class.
  int32 EquivalenceClassOf(int32 word) const;
 protected:
  friend class LatticeLexiconWordAligner;

  void UpdateViabilityMap(const std::vector<int32> &lexicon_entry);
  void UpdateLexiconMap(const std::vector<int32> &lexicon_entry);
  void UpdateNumPhonesMap(const std::vector<int32> &lexicon_entry);
  void UpdateEquivalenceMap(const std::vector<std::vector<int32> > &lexicon);

  void FinalizeViabilityMap(); // sorts the vectors.

  /// The type ViabilityMap maps from sequences of phones (excluding the empty
  /// sequence), to the sets of all word-labels [on the input lattice] that
  /// could correspond to phone sequences that start with s [but are longer than
  /// s].  The sets of word-labels are represented as sorted vectors of int32
  /// Note: the zero word-label is included here.  This is used in a kind
  /// of co-accessibility test, to see whether it is worth extending this state
  /// by traversing arcs in the input lattice.
  typedef unordered_map<std::vector<int32>,
                        std::vector<int32>,
                        VectorHasher<int32> > ViabilityMap;

  /// This is a map from a vector (orig-word-symbol phone1 phone2 ... ) to
  /// the new word-symbol.  [todo: make sure the new word-symbol is always nonzero.]
  typedef unordered_map<std::vector<int32>, int32,
                        VectorHasher<int32> > LexiconMap;

  /// This is a map from the word-id (as present in the original lattice)
  /// to the minimum and maximum #phones of lexicon entries for that word.
  /// It helps improve efficiency.
  typedef unordered_map<int32, std::pair<int32, int32> > NumPhonesMap;

  /// This is used only in testing code; it defines a mapping from a word
  /// to the primary member of that word's equivalence-class.
  typedef unordered_map<int32, int32> EquivalenceMap;

  // The following three variables represent various types of information
  // gathered from the lexicon.
  LexiconMap lexicon_map_;
  NumPhonesMap num_phones_map_;
  ViabilityMap viability_map_;

  // As lexicon_map but in reverse sense w.r.t. words [we only
  // do this for asymmetric entries.]  Used only in testing code.
  LexiconMap reverse_lexicon_map_;

  // This is used only in testing code; it defines a mapping from a word
  // to the primary member of that word's equivalence-class.  If an index
  // is not present in the map, it's assumed to map to itself.
  EquivalenceMap equivalence_map_;
};


struct WordAlignLatticeLexiconOpts {
  int32 partial_word_label;
  bool reorder;
  BaseFloat max_expand;

  WordAlignLatticeLexiconOpts(): partial_word_label(0), reorder(true),
                                 max_expand(-1.0) { }

  void Register(OptionsItf *opts) {
    opts->Register("partial-word-label", &partial_word_label, "Numeric id of "
                   "word symbol that is to be used for arcs in the word-aligned "
                   "lattice corresponding to partial words at the end of "
                   "\"forced-out\" utterances (zero is OK)");
    opts->Register("reorder", &reorder, "True if the lattices were generated "
                   "from graphs that had the --reorder option true, relating to "
                   "reordering self-loops (typically true)");
    opts->Register("max-expand", &max_expand, "If >0.0, the maximum ratio "
                   "by which we allow the lattice-alignment code to increase the #states "
                   "in a lattice (vs. the phone-aligned lattice) before we fail and "
                   "refuse to align the lattice.  This is helpful in order to "
                   "prevent 'pathological' lattices from causing the program to "
                   "exhaust memory.  Actual max-states is 1000 + max-expand * "
                   "orig-num-states.");
  }
};


/// Align lattice so that each arc has the transition-ids on it
/// that correspond to the word that is on that arc.  [May also have
/// epsilon arcs for optional silences.]
/// Returns true if everything was OK, false if there was any kind of
/// error including when the the lattice seems to have been "forced out"
/// (did not reach end state, resulting in partial word at end).
bool WordAlignLatticeLexicon(const CompactLattice &lat,
                             const TransitionInformation &tmodel,
                             const WordAlignLatticeLexiconInfo &lexicon_info,
                             const WordAlignLatticeLexiconOpts &opts,
                             CompactLattice *lat_out);

} // namespace kaldi
#endif
