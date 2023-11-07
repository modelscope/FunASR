// fstbin/fsts-project.cc

// Copyright 2016  Johns Hopkins University (Authors: Jan "Yenda" Trmal)

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
#include "fstext/fstext-utils.h"
#include "fstext/kaldi-fst-io.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    typedef kaldi::int32 int32;
    typedef kaldi::uint64 uint64;

    const char *usage =
        "Reads kaldi archive of FSTs; for each element, performs the project\n"
        "operation either on input (default) or on the output (if the option\n"
        "--project-output is true).\n"
        "\n"
        "Usage: fsts-project [options] <fsts-rspecifier> <fsts-wspecifier>\n"
        " e.g.: fsts-project ark:train.fsts ark,t:train.fsts\n"
        "\n"
        "see also: fstproject (from the OpenFst toolkit)\n";

    ParseOptions po(usage);

    bool project_output = false;

    po.Register("project-output", &project_output,
                "If true, project output vs input");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier = po.GetArg(1),
        fsts_wspecifier = po.GetArg(2);


    SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
    TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);

    int32 n_done = 0;
    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      VectorFst<StdArc> fst(fst_reader.Value());

      Project(&fst, project_output ? PROJECT_OUTPUT : PROJECT_INPUT);

      fst_writer.Write(key, fst);
      n_done++;
    }

    KALDI_LOG << "Projected " << n_done << " FSTs";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
