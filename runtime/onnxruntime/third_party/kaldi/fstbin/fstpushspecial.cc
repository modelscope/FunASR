// fstbin/fstpushspecial.cc

// Copyright 2012  Daniel Povey

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
#include "fstext/push-special.h"
#include "fstext/kaldi-fst-io.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Pushes weights in an FST such that all the states\n"
        "in the FST have arcs and final-probs with weights that\n"
        "sum to the same amount (viewed as being in the log semiring).\n"
        "Thus, the \"extra weight\" is distributed throughout the FST.\n"
        "Tolerance parameter --delta controls how exact this is, and the\n"
        "speed.\n"
        "\n"
        "Usage:  fstpushspecial [options] [in.fst [out.fst] ]\n";

    BaseFloat delta = kDelta;
    ParseOptions po(usage);
    po.Register("delta", &delta, "Delta cost: after pushing, all states will "
                "have a total weight that differs from the average by no more "
                "than this.");
    po.Read(argc, argv);

    if (po.NumArgs() > 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_filename = po.GetOptArg(1),
        fst_out_filename = po.GetOptArg(2);

    VectorFst<StdArc> *fst = ReadFstKaldi(fst_in_filename);

    PushSpecial(fst, delta);

    WriteFstKaldi(*fst, fst_out_filename);
    delete fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
