// fstbin/fsts-concat.cc

// Copyright 2016  Johns Hopkins University (Authors: Jan "Yenda" Trmal)
//           2018  Soapbox Labs (Author: Karel Vesely)

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
        "Reads kaldi archives with FSTs. Concatenates the fsts from all the rspecifiers.\n"
        "The fsts to concatenate must have same key. The sequencing is given by the position of arguments.\n"
        "\n"
        "Usage: fsts-concat [options] <fsts-rspecifier1> <fsts-rspecifier2> ... <fsts-wspecifier>\n"
        " e.g.: fsts-concat scp:fsts1.scp scp:fsts2.scp ... ark:fsts_out.ark\n"
        "\n"
        "see also: fstconcat (from the OpenFst toolkit)\n";

    ParseOptions po(usage);

    po.Read(argc, argv);

    if (po.NumArgs() < 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string fsts_rspecifier = po.GetArg(1),
        fsts_wspecifier = po.GetArg(po.NumArgs());

    SequentialTableReader<VectorFstHolder> fst_reader(fsts_rspecifier);
    std::vector<RandomAccessTableReader<VectorFstHolder>*> fst_readers;
    TableWriter<VectorFstHolder> fst_writer(fsts_wspecifier);

    for (int32 i = 2; i < po.NumArgs(); i++)
      fst_readers.push_back(new RandomAccessTableReader<VectorFstHolder>(po.GetArg(i)));
    const int32 num_fst_readers = fst_readers.size();

    int32 n_done = 0,
          n_skipped = 0;

    for (; !fst_reader.Done(); fst_reader.Next()) {
      std::string key = fst_reader.Key();

      // Check that the key exists in all 'fst_readers'.
      bool skip_key = false;
      for (int32 i = 0; i < num_fst_readers; i++) {
        if (!fst_readers[i]->HasKey(key)) {
          KALDI_WARN << "Skipping '" << key << "'"
            << " due to missing the fst in " << (i+2) << "th <rspecifier> : "
            << "'" << po.GetArg(i+2) << "'";
          skip_key = true;
        }
      }
      if (skip_key) {
        n_skipped++;
        continue;
      }

      // Concatenate!
      VectorFst<StdArc> fst_out = fst_readers.back()->Value(key);
      // Loop from (last-1) to first, as 'prepending' the fsts is faster,
      // see: http://www.openfst.org/twiki/bin/view/FST/ConcatDoc
      for (int32 i = num_fst_readers-2; i >= 0; i--) {
        fst::Concat(fst_readers[i]->Value(key), &fst_out);
      }
      // Finally, prepend the fst from the 'Sequential' reader.
      fst::Concat(fst_reader.Value(), &fst_out);

      // Write the output.
      fst_writer.Write(key, fst_out);
      n_done++;
    }

    // Cleanup.
    for (int32 i = 0; i < num_fst_readers; i++)
      delete fst_readers[i];
    fst_readers.clear();

    KALDI_LOG << "Produced " << n_done << " FSTs by concatenating " << po.NumArgs()-1
      << " streams " << "(" << n_skipped << " keys skipped).";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
