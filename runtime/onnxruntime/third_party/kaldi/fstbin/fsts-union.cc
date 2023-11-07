// fstbin/fsts-union.cc

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
        "Reads a kaldi archive of FSTs. Performs the FST operation union on\n"
        "all fsts sharing the same key. Assumes the archive is sorted by key.\n"
        "\n"
        "Usage: fsts-union [options] <fsts-rspecifier> <fsts-wspecifier>\n"
        " e.g.: fsts-union ark:keywords_tmp.fsts ark,t:keywords.fsts\n"
        "\n"
        "see also: fstunion (from the OpenFst toolkit)\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier = po.GetArg(1),
        fsts_wspecifier = po.GetArg(2);


    SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
    TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);

    int32 n_out_done = 0,
          n_in_done = 0;
    std::string res_key = "";
    VectorFst<StdArc> res_fst;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      VectorFst<StdArc> fst(fst_reader.Value());

      n_in_done++;
      if (key == res_key) {
        fst::Union(&res_fst, fst);
      } else {
        if (res_key != "") {
          VectorFst<StdArc> out_fst;
          fst::Determinize(res_fst, &out_fst);
          fst::Minimize(&out_fst);
          fst::RmEpsilon(&out_fst);
          fst_writer.Write(res_key, out_fst);
          n_out_done++;
        }
        res_fst = fst;
        res_key = key;
      }
    }
    if (res_key != "") {
      VectorFst<StdArc> out_fst;
      fst::Determinize(res_fst, &out_fst);
      fst::Minimize(&out_fst);
      fst::RmEpsilon(&out_fst);
      fst_writer.Write(res_key, out_fst);
      n_out_done++;
    }

    KALDI_LOG << "Applied fst union on " << n_in_done
              << " FSTs, produced " <<  n_out_done << " FSTs";
    return (n_out_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
