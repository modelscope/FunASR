// lat/push-lattice-test.cc

// Copyright 2013  Johns Hopkins University (author: Daniel Povey)

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


#include "lat/kaldi-lattice.h"
#include "lat/push-lattice.h"
#include "fstext/rand-fst.h"


namespace kaldi {
using namespace fst;

CompactLattice *RandCompactLattice() {
  RandFstOptions opts;
  opts.acyclic = true;
  Lattice *fst = fst::RandPairFst<LatticeArc>(opts);
  CompactLattice *cfst = new CompactLattice;
  ConvertLattice(*fst, cfst);
  delete fst;
  return cfst;
}

void TestPushCompactLatticeStrings() {
  CompactLattice *clat = RandCompactLattice();
  CompactLattice clat2(*clat);
  PushCompactLatticeStrings(&clat2);
  KALDI_ASSERT(fst::RandEquivalent(*clat, clat2, 5, 0.001, Rand(), 10));
  for (CompactLatticeArc::StateId s = 0; s < clat2.NumStates(); s++) {
    if (s == 0)
      continue; // We don't check state zero, as the "leftover string" stays
                // there.
    int32 first_label = -1;
    bool ok = false;
    bool first_label_set = false;
    for (ArcIterator<CompactLattice> aiter(clat2, s); !aiter.Done();
         aiter.Next()) {
      if (aiter.Value().weight.String().size() == 0) {
        ok = true;
      } else {
        int32 this_label = aiter.Value().weight.String().front();
        if (first_label_set) {
          if (this_label != first_label) ok = true;
        } else {
          first_label = this_label;
          first_label_set = true;
        }
      }
    }
    if (clat2.Final(s) != CompactLatticeWeight::Zero()) {
      if (clat2.Final(s).String().size() == 0) ok = true;
      else {
        int32 this_label = clat2.Final(s).String().front();
        if (first_label_set && this_label != first_label) ok = true;
      }
    }
    KALDI_ASSERT(ok);
  }
  delete clat;
}

void TestPushCompactLatticeWeights() {
  CompactLattice *clat = RandCompactLattice();
  CompactLattice clat2(*clat);
  PushCompactLatticeWeights(&clat2);
  KALDI_ASSERT(fst::RandEquivalent(*clat, clat2, 5, 0.001, Rand(), 10));
  for (CompactLatticeArc::StateId s = 0; s < clat2.NumStates(); s++) {
    if (s == 0)
      continue; // We don't check state zero, as the "leftover string" stays
                // there.
    LatticeWeight sum = clat2.Final(s).Weight();
    for (ArcIterator<CompactLattice> aiter(clat2, s); !aiter.Done();
         aiter.Next()) {
      sum = Plus(sum, aiter.Value().weight.Weight());
    }
    if (!ApproxEqual(sum, LatticeWeight::One())) {
      {
        fst::FstPrinter<CompactLatticeArc> printer(clat2, NULL, NULL,
                                                   NULL, true, true, "\t");
        printer.Print(&std::cerr, "<unknown>");
      }
      {
        fst::FstPrinter<CompactLatticeArc> printer(*clat, NULL, NULL,
                                                   NULL, true, true, "\t");
        printer.Print(&std::cerr, "<unknown>");
      }
      KALDI_ERR << "Bad lattice being pushed.";
    }
  }
  delete clat;
}



} // end namespace kaldi

int main() {
  using namespace kaldi;
  using kaldi::int32;
  for (int32 i = 0; i < 15; i++) {
    TestPushCompactLatticeStrings();
    TestPushCompactLatticeWeights();
  }
  KALDI_LOG << "Success.";
}
