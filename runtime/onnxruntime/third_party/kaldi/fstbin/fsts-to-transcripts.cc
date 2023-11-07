// fstbin/fsts-to-transcripts.cc

// Copyright 2012-2013  Johns Hopkins University (Authors: Guoguo Chen,
//                                                         Daniel Povey)

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
        "Reads a table of FSTs; for each element, finds the best path and \n"
        "prints out the output-symbol sequence (if --output-side=true), or \n"
        "input-symbol sequence otherwise.\n"
        "\n"
        "Usage:\n"
        " fsts-to-transcripts [options] <fsts-rspecifier>"
        " <transcriptions-wspecifier>\n"
        "e.g.:\n"
        " fsts-to-transcripts ark:train.fsts ark,t:train.text\n";

    ParseOptions po(usage);

    bool output_side = true;

    po.Register("output-side", &output_side, "If true, extract the symbols on "
                "the output side of the FSTs, else the input side.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string fst_rspecifier = po.GetArg(1),
        transcript_wspecifier = po.GetArg(2);


    SequentialTableReader<VectorFstHolder> fst_reader(fst_rspecifier);
    Int32VectorWriter transcript_writer(transcript_wspecifier);

    int32 n_done = 0, n_err = 0;
    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();
      const VectorFst<StdArc> &fst = fst_reader.Value();


      VectorFst<StdArc> shortest_path;
      ShortestPath(fst, &shortest_path);  // the OpenFst algorithm ShortestPath.

      if (shortest_path.NumStates() == 0) {
        KALDI_WARN << "Input FST (after shortest path) was empty. Producing "
                   << "no output for key " << key;
        n_err++;
        continue;
      }

      std::vector<int32> transcript;
      bool ans;
      if (output_side) ans = fst::GetLinearSymbolSequence<StdArc, int32>(
              shortest_path, NULL, &transcript, NULL);
      else
        ans = fst::GetLinearSymbolSequence<StdArc, int32>(
          shortest_path, &transcript, NULL, NULL);
      if (!ans) {
        KALDI_ERR << "GetLinearSymbolSequence returned false (code error);";
      }
      transcript_writer.Write(key, transcript);
      n_done++;
    }

    KALDI_LOG << "Converted " << n_done << " FSTs, " << n_err << " with errors";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
