// lat/word-align-lattice.h

// Copyright 2009-2012  Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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

#ifndef KALDI_LAT_WORD_ALIGN_LATTICE_H_
#define KALDI_LAT_WORD_ALIGN_LATTICE_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "itf/transition-information.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {


struct WordBoundaryInfoOpts {
  // Note: use of this structure
  // is deprecated, see WordBoundaryInfoNewOpts.
  
  // Note: this structure (and the code in word-align-lattice.{h,cc}
  // makes stronger assumptions than the rest of the Kaldi toolkit:
  // that is, it assumes you have word-position-dependent phones,
  // with disjoint subsets of phones for (word-begin, word-end,
  // word-internal, word-begin-and-end), and of course silence,
  // which is assumed not to be inside a word [it will just print
  // a warning if it is, though, and should give the right output
  // as long as it's not at the beginning or end of a word].

  std::string wbegin_phones;
  std::string wend_phones;
  std::string wbegin_and_end_phones;
  std::string winternal_phones;
  std::string silence_phones;
  int32 silence_label;
  int32 partial_word_label;
  bool reorder;
  bool silence_may_be_word_internal;
  bool silence_has_olabels;
  
  WordBoundaryInfoOpts(): silence_label(0), partial_word_label(0),
                          reorder(true), silence_may_be_word_internal(false),
                          silence_has_olabels(false) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("wbegin-phones", &wbegin_phones, "Colon-separated list of "
                   "numeric ids of phones that begin a word");
    opts->Register("wend-phones", &wend_phones, "Colon-separated list of "
                   "numeric ids of phones that end a word");
    opts->Register("winternal-phones", &winternal_phones, "Colon-separated list "
                   "of numeric ids of phones that are internal to a word");
    opts->Register("wbegin-and-end-phones", &wbegin_and_end_phones, "Colon-separated "
                   "list of numeric ids of phones that are used for "
                   "single-phone words.");
    opts->Register("silence-phones", &silence_phones, "Colon-separated list of "
                   "numeric ids of phones that are used for silence (and other "
                   "non-word events such as noise - anything that doesn't have "
                   "a corresponding symbol in the lexicon.");
    opts->Register("silence-label", &silence_label, "Numeric id of word symbol "
                   "that is to be used for silence arcs in the word-aligned "
                   "lattice (zero is OK)");
    opts->Register("partial-word-label", &partial_word_label, "Numeric id of "
                   "word symbol that is to be used for arcs in the word-aligned "
                   "lattice corresponding to partial words at the end of "
                   "\"forced-out\" utterances (zero is OK)");
    opts->Register("reorder", &reorder, "True if the lattices were generated "
                   "from graphs that had the --reorder option true, relating to "
                   "reordering self-loops (typically true)");
    opts->Register("silence-may-be-word-internal", &silence_may_be_word_internal,
                   "If true, silence may appear inside words' prons (but not at begin/end!)\n");
    opts->Register("silence-has-olabels", &silence_has_olabels, 
                   "If true, silence phones have output labels in the lattice, just\n"
                   "like regular words.  [This means you can't have un-labeled silences]");
  }
};


// This structure is to be used for newer code, from s5 scripts on.
struct WordBoundaryInfoNewOpts {
  int32 silence_label;
  int32 partial_word_label;
  bool reorder;
  
  WordBoundaryInfoNewOpts(): silence_label(0), partial_word_label(0),
                             reorder(true) { }
  
  void Register(OptionsItf *opts) {
    opts->Register("silence-label", &silence_label, "Numeric id of word symbol "
                   "that is to be used for silence arcs in the word-aligned "
                   "lattice (zero is OK)");
    opts->Register("partial-word-label", &partial_word_label, "Numeric id of "
                   "word symbol that is to be used for arcs in the word-aligned "
                   "lattice corresponding to partial words at the end of "
                   "\"forced-out\" utterances (zero is OK)");
    opts->Register("reorder", &reorder, "True if the lattices were generated "
                   "from graphs that had the --reorder option true, relating to "
                   "reordering self-loops (typically true)");
  }
};


struct WordBoundaryInfo {
  // This initializer will be deleted eventually.
  WordBoundaryInfo(const WordBoundaryInfoOpts &opts); // Initialize from
  // options class.  Note: this throws.  Don't try to catch this error
  // and continue; catching errors thrown from initializers is dangerous.
  // Note: the following vectors are initialized from the corresponding
  // options strings in the options class, but if silence_may_be_word_internal=true
  // or silence_has_olabels=true, we modify them as needed to make
  // silence phones behave in this way.

  // This initializer is to be used in future.
  WordBoundaryInfo(const WordBoundaryInfoNewOpts &opts);
  WordBoundaryInfo(const WordBoundaryInfoNewOpts &opts,
                   std::string word_boundary_file);

  void Init(std::istream &stream);

  enum PhoneType {
    kNoPhone = 0,
    kWordBeginPhone,
    kWordEndPhone,
    kWordBeginAndEndPhone,
    kWordInternalPhone,
    kNonWordPhone // non-word phones are typically silence phones; but the point
    // is that there is
    // no word label associated with them in the lattice.  If a silence phone
    // had a word label with it, we'd have to call it kWordBeginAndEndPhone.
  };
  PhoneType TypeOfPhone(int32 p) const {
    if ((p < 0 || p > phone_to_type.size()))
      KALDI_ERR << "Phone " << p << " was not specified in "
          "word-boundary file (or options)";
    return phone_to_type[p];
  }
  
  std::vector<PhoneType> phone_to_type;

  int32 silence_label; // The integer label we give to silence words.
  // (May be zero).
  int32 partial_word_label; // The label we give to partially
  // formed words that we might get at the end of the utterance
  // if the lattice was "forced out" (no end state was reached).

  bool reorder; // True if the "reordering" of self-loops versus
  // forward-transition was done during graph creation (will
  // normally be true.

 private:
  // This is to be removed eventually, when we all move to s5 scripts.
  void SetOptions(const std::string int_list, PhoneType phone_type);
};

/// Align lattice so that each arc has the transition-ids on it
/// that correspond to the word that is on that arc.  [May also have
/// epsilon arcs for optional silences.]
/// Returns true if everything was OK, false if some kind of
/// error was detected (e.g. the words didn't have the kinds of
/// sequences we would expect if the WordBoundaryInfo was
/// correct).  Note: we don't expect silence inside words,
/// or empty words (words with no phones), and we expect
/// the word to start with a wbegin_phone, to end with
/// a wend_phone, and to possibly have winternal_phones
/// inside (or to consist of just one wbegin_and_end_phone).
/// Note: if it returns false, it doesn't mean the lattice
/// that the output is necessarily bad: it might just be that
/// the lattice was "forced out" as the end-state was not
/// reached during decoding, and in this case the output might
/// be usable.
///  If max_states > 0, if this code detects that the #states
/// of the output will be greater than max_states, it will
/// abort the computation, return false and produce an empty
/// lattice out.
bool WordAlignLattice(const CompactLattice &lat,
                      const TransitionInformation &tmodel,
                      const WordBoundaryInfo &info,
                      int32 max_states,
                      CompactLattice *lat_out);



/// This function is designed to crash if something went wrong with the
/// word-alignment of the lattice.  It verifies
/// that arcs are of 4 types:
///   properly-aligned word arcs, with a word label.
///   partial-word arcs, with the partial-word label.
///   silence arcs, with the silence label.
void TestWordAlignedLattice(const CompactLattice &lat,
                            const TransitionInformation &tmodel,
                            const WordBoundaryInfo &info,
                            const CompactLattice &aligned_lat);

} // end namespace kaldi
#endif
