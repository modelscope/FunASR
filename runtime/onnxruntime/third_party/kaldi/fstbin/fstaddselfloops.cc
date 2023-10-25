// fstbin/fstaddselfloops.cc

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
#include "fst/fstlib.h"
#include "fstext/determinize-star.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"
#include "util/parse-options.h"
#include "util/simple-io-funcs.h"
/* some test examples:
  pushd ~/tmpdir
 ( echo 3; echo  4) > in.list
 ( echo 5; echo  6) > out.list
 ( echo "0 0 0 0"; echo "0 0" ) | fstcompile | fstaddselfloops in.list out.list | fstprint
 ( echo "0 1 0 1"; echo " 0 2 1 0"; echo "1 0"; echo "2 0"; ) | fstcompile | fstaddselfloops in.list out.list | fstprint
*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Adds self-loops to states of an FST to propagate disambiguation symbols through it\n"
        "They are added on each final state and each state with non-epsilon output symbols\n"
        "on at least one arc out of the state.  Useful in conjunction with predeterminize\n"
        "\n"
        "Usage:  fstaddselfloops in-disambig-list out-disambig-list  [in.fst [out.fst] ]\n"
        "E.g:  fstaddselfloops in.list out.list < in.fst > withloops.fst\n"
        "in.list and out.list are lists of integers, one per line, of the\n"
        "same length.\n";

    ParseOptions po(usage);
    po.Read(argc, argv);

    if (po.NumArgs() < 2 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string disambig_in_rxfilename = po.GetArg(1),
        disambig_out_rxfilename = po.GetArg(2),
        fst_in_filename = po.GetOptArg(3),
        fst_out_filename = po.GetOptArg(4);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    std::vector<int32> disambig_in;
    if (!ReadIntegerVectorSimple(disambig_in_rxfilename, &disambig_in))
      KALDI_ERR << "fstaddselfloops: Could not read disambiguation symbols from "
                 << kaldi::PrintableRxfilename(disambig_in_rxfilename);

    std::vector<int32> disambig_out;
    if (!ReadIntegerVectorSimple(disambig_out_rxfilename, &disambig_out))
      KALDI_ERR << "fstaddselfloops: Could not read disambiguation symbols from "
                << kaldi::PrintableRxfilename(disambig_out_rxfilename);

    if (disambig_in.size() != disambig_out.size())
      KALDI_ERR << "fstaddselfloops: mismatch in size of disambiguation symbols";

    AddSelfLoops(fst, disambig_in, disambig_out);

    WriteFstKaldi(*fst, fst_out_filename);

    delete fst;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
  return 0;
}

