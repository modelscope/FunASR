// lat/phone-align-lattice.h

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

#ifndef KALDI_LAT_PHONE_ALIGN_LATTICE_H_
#define KALDI_LAT_PHONE_ALIGN_LATTICE_H_
#include <fst/fstlib.h>
#include <fst/fst-decl.h>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fstext/fstext-lib.h"
#include "itf/transition-information.h"
#include "lat/kaldi-lattice.h"

namespace kaldi {


struct PhoneAlignLatticeOptions {
  bool reorder;
  bool remove_epsilon;
  bool replace_output_symbols;
  PhoneAlignLatticeOptions(): reorder(true),
                              remove_epsilon(true),
                              replace_output_symbols(false) { }
  void Register(OptionsItf *opts) {
    opts->Register("reorder", &reorder, "True if lattice was created from HCLG with "
                   "--reorder=true option.");
    opts->Register("remove-epsilon", &remove_epsilon, "If true, removes epsilons from "
                   "the phone lattice; if replace-output-symbols==false, this will "
                   "mean that an arc can have multiple phones on it.");
    opts->Register("replace-output-symbols", &replace_output_symbols, "If true, "
                   "the output symbols (typically words) will be replaced with "
                   "phones.");
  }
};

/// Outputs a lattice in which the arcs correspond exactly to sequences of
/// phones, so the boundaries between the arcs correspond to the boundaries
/// between phones If remove-epsilon == false and replace-output-symbols ==
/// false, but an arc may have >1 phone on it, but the boundaries will still
/// correspond with the boundaries between phones.  Note: it's possible
/// to have arcs with words on them but no transition-ids at all.  Returns true if
/// everything was OK, false if some kind of error was detected (e.g. the
/// "reorder" option was incorrectly specified.)
bool PhoneAlignLattice(const CompactLattice &lat,
                       const TransitionInformation &tmodel,
                       const PhoneAlignLatticeOptions &opts,
                       CompactLattice *lat_out);


} // end namespace kaldi
#endif
