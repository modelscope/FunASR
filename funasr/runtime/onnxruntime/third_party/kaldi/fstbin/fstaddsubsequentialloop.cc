// fstbin/fstaddsubsequentialloop.cc

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
#include "util/kaldi-io.h"
#include "util/parse-options.h"
#include "util/text-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"
#include "fstext/kaldi-fst-io.h"



/* some test  examples:
 ( echo "0 0 0 0"; echo "0 0" ) | fstcompile | fstaddsubsequentialloop 1 | fstprint
 ( echo "0 1 0 0"; echo " 0 2 0 0"; echo "1 0"; echo "2 0"; ) | fstcompile | fstaddsubsequentialloop 1 | fstprint
*/

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Minimizes FST after encoding [this algorithm applicable to all FSTs in tropical semiring]\n"
        "\n"
        "Usage:  fstaddsubsequentialloop subseq_sym [in.fst [out.fst] ]\n"
        "E.g.:   fstaddsubsequentialloop 52 < LG.fst > LG_sub.fst\n";

    float delta = kDelta;
    ParseOptions po(usage);
    po.Register("delta", &delta,
                "Delta likelihood used for quantization of weights");
    po.Read(argc, argv);

    if (po.NumArgs() < 1 || po.NumArgs() > 3) {
      po.PrintUsage();
      exit(1);
    }

    int32 subseq_sym;
    if (!ConvertStringToInteger(po.GetArg(1), &subseq_sym))
      KALDI_ERR << "Invalid subsequential symbol "<<po.GetArg(1);

    std::string fst_in_filename = po.GetOptArg(2);

    std::string fst_out_filename = po.GetOptArg(3);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    int32 h = HighestNumberedInputSymbol(*fst);
    if (subseq_sym <= h) {
      std::cerr << "fstaddsubsequentialloop.cc: subseq symbol does not seem right, "<<subseq_sym<<" <= "<<h<<'\n';
    }

    AddSubsequentialLoop(subseq_sym, fst);

    WriteFstKaldi(*fst, fst_out_filename);
    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

