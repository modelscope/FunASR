// fstbin/fstrand.cc

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
#include "fstext/rand-fst.h"
#include "time.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

int main(int argc, char *argv[]) {
  try {
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Generate random FST\n"
        "\n"
        "Usage:  fstrand [out.fst]\n";

    srand(time(NULL));
    RandFstOptions opts;


    kaldi::ParseOptions po(usage);
    po.Register("allow-empty", &opts.allow_empty,
                "If true, we may generate an empty FST.");

    po.Read(argc, argv);

    if (po.NumArgs() > 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_out_filename = po.GetOptArg(1);

    VectorFst <StdArc> *rand_fst = RandFst<StdArc>(opts);

    WriteFstKaldi(*rand_fst, fst_out_filename);
    delete rand_fst;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

