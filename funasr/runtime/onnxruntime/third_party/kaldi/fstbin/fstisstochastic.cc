// fstbin/fstisstochastic.cc

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
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"

// e.g. of test:
// echo " 0 0" | fstcompile | fstisstochastic
// should return 0 and print "0 0" [meaning, min and
// max weight are one = exp(0)]
// echo " 0 1" | fstcompile | fstisstochastic
// should  return 1, not stochastic, and print 1 1
// (echo "0 0 0 0 0.693147 "; echo "0 1 0 0 0.693147 "; echo "1 0" ) | fstcompile | fstisstochastic
// should return 0, stochastic; it prints "0 -1.78e-07" for me
// (echo "0 0 0 0 0.693147 "; echo "0 1 0 0 0.693147 "; echo "1 0" ) | fstcompile | fstisstochastic --test-in-log=false
// should return 1, not stochastic in tropical; it prints "0 0.693147" for me
// (echo "0 0 0 0 0 "; echo "0 1 0 0 0 "; echo "1 0" ) | fstcompile | fstisstochastic --test-in-log=false
// should return 0, stochastic in tropical; it prints "0 0" for me
// (echo "0 0 0 0 0.693147 "; echo "0 1 0 0 0.693147 "; echo "1 0" ) | fstcompile | fstisstochastic --test-in-log=false --delta=1
// returns 0 even though not stochastic because we gave it an absurdly large delta.

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Checks whether an FST is stochastic and exits with success if so.\n"
        "Prints out maximum error (in log units).\n"
        "\n"
        "Usage:  fstisstochastic [ in.fst ]\n";

    float delta = 0.01;
    bool test_in_log = true;

    ParseOptions po(usage);
    po.Register("delta", &delta, "Maximum error to accept.");
    po.Register("test-in-log", &test_in_log, "Test stochasticity in log semiring.");
    po.Read(argc, argv);

    if (po.NumArgs() > 1) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_in_filename = po.GetOptArg(1);

    Fst<StdArc> *fst = ReadFstKaldiGeneric(fst_in_filename);

    bool ans;
    StdArc::Weight min, max;
    if (test_in_log)  ans = IsStochasticFstInLog(*fst, delta, &min, &max);
    else ans = IsStochasticFst(*fst, delta, &min, &max);

    std::cout << min.Value() << " " << max.Value() << '\n';
    delete fst;
    if (ans) return 0;  // success;
    else return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
