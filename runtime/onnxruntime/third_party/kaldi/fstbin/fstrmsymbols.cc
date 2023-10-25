// fstbin/fstrmsymbols.cc

// Copyright 2009-2011  Microsoft Corporation

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


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

namespace fst {
// we can move these functions elsewhere later, if they are needed in other
// places.

template<class Arc, class I>
void RemoveArcsWithSomeInputSymbols(const std::vector<I> &symbols_in,
                                    VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;

  kaldi::ConstIntegerSet<I> symbol_set(symbols_in);

  StateId num_states = fst->NumStates();
  StateId dead_state = fst->AddState();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<VectorFst<Arc> > iter(fst, s);
         !iter.Done(); iter.Next()) {
      if (symbol_set.count(iter.Value().ilabel) != 0) {
        Arc arc = iter.Value();
        arc.nextstate = dead_state;
        iter.SetValue(arc);
      }
    }
  }
  // Connect() will actually remove the arcs, and the dead state.
  Connect(fst);
  if (fst->NumStates() == 0)
    KALDI_WARN << "After Connect(), fst was empty.";
}

template<class Arc, class I>
void PenalizeArcsWithSomeInputSymbols(const std::vector<I> &symbols_in,
                                      float penalty,
                                      VectorFst<Arc> *fst) {
  typedef typename Arc::StateId StateId;
  typedef typename Arc::Label Label;
  typedef typename Arc::Weight Weight;

  Weight penalty_weight(penalty);

  kaldi::ConstIntegerSet<I> symbol_set(symbols_in);

  StateId num_states = fst->NumStates();
  for (StateId s = 0; s < num_states; s++) {
    for (MutableArcIterator<VectorFst<Arc> > iter(fst, s);
         !iter.Done(); iter.Next()) {
      if (symbol_set.count(iter.Value().ilabel) != 0) {
        Arc arc = iter.Value();
        arc.weight = Times(arc.weight, penalty_weight);
        iter.SetValue(arc);
      }
    }
  }
}

}


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    bool apply_to_output = false;
    bool remove_arcs = false;
    float penalty = -std::numeric_limits<BaseFloat>::infinity();

    const char *usage =
        "With no options, replaces a subset of symbols with epsilon, wherever\n"
        "they appear on the input side of an FST."
        "With --remove-arcs=true, will remove arcs that contain these symbols\n"
        "on the input\n"
        "With --penalty=<float>, will add the specified penalty to the\n"
        "cost of any arc that has one of the given symbols on its input side\n"
        "In all cases, the option --apply-to-output=true (or for\n"
        "back-compatibility, --remove-from-output=true) makes this apply\n"
        "to the output side.\n"
        "\n"
        "Usage:  fstrmsymbols [options] <in-disambig-list>  [<in.fst> [<out.fst>]]\n"
        "E.g:  fstrmsymbols in.list  < in.fst > out.fst\n"
        "<in-disambig-list> is an rxfilename specifying a file containing list of integers\n"
        "representing symbols, in text form, one per line.\n";

    ParseOptions po(usage);
    po.Register("remove-from-output", &apply_to_output, "If true, this applies to symbols "
                "on the output, not the input, side.  (For back compatibility; use "
                "--apply-to-output insead)");
    po.Register("apply-to-output", &apply_to_output, "If true, this applies to symbols "
                "on the output, not the input, side.");
    po.Register("remove-arcs", &remove_arcs, "If true, instead of converting the symbol "
                "to <eps>, remove the arcs.");
    po.Register("penalty", &penalty, "If specified, instead of converting "
                "the symbol to <eps>, penalize the arc it is on by adding this "
                "value to its cost.");


    po.Read(argc, argv);

    if (remove_arcs &&
        penalty != -std::numeric_limits<BaseFloat>::infinity())
      KALDI_ERR << "--remove-arc and --penalty options are mutually exclusive";

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string disambig_rxfilename = po.GetArg(1),
        fst_rxfilename = po.GetOptArg(2),
        fst_wxfilename = po.GetOptArg(3);

    VectorFst<StdArc> *fst = CastOrConvertToVectorFst(
        ReadFstKaldiGeneric(fst_rxfilename));

    std::vector<int32> disambig_in;
    if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_in))
      KALDI_ERR << "fstrmsymbols: Could not read disambiguation symbols from "
                << (disambig_rxfilename == "" ? "standard input" : disambig_rxfilename);

    if (apply_to_output) Invert(fst);
    if (remove_arcs) {
      RemoveArcsWithSomeInputSymbols(disambig_in, fst);
    } else if (penalty != -std::numeric_limits<BaseFloat>::infinity()) {
      PenalizeArcsWithSomeInputSymbols(disambig_in, penalty, fst);
    } else {
      RemoveSomeInputSymbols(disambig_in, fst);
    }
    if (apply_to_output) Invert(fst);

    WriteFstKaldi(*fst, fst_wxfilename);

    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

/* some test examples:

 ( echo "0 0 1 1"; echo " 0 0 3 2"; echo "0 0"; ) | fstcompile | fstrmsymbols "echo 3; echo  4|" | fstprint
 # should produce:
 # 0   0   1   1
 # 0   0   0   2
 # 0

 ( echo "0 0 1 1"; echo " 0 0 3 2"; echo "0 0"; ) | fstcompile | fstrmsymbols --apply-to-output=true "echo 2; echo 3|" | fstprint
 # should produce:
 # 0   0   1   1
 # 0   0   3   0
 # 0


 ( echo "0 0 1 1"; echo " 0 0 3 2"; echo "0 0"; ) | fstcompile | fstrmsymbols --remove-arcs=true  "echo 3; echo  4|" | fstprint
 # should produce:
 # 0   0   1   1
 # 0

 ( echo "0 0 1 1"; echo " 0 0 3 2"; echo "0 0"; ) | fstcompile | fstrmsymbols --penalty=2 "echo 3; echo 4; echo 5|" | fstprint
# should produce:
 # 0   0   1   1
 # 0   0   3   2   2
 # 0

*/
