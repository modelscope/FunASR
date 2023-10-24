// lmbin/arpa-to-const-arpa.cc

// Copyright 2014  Guoguo Chen

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
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

#include <string>

#include "lm/const-arpa-lm.h"
#include "util/parse-options.h"

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  try {
    const char *usage  =
        "Converts an Arpa format language model into ConstArpaLm format,\n"
        "which is an in-memory representation of the pre-built Arpa language\n"
        "model. The output language model can then be read in by a program\n"
        "that wants to rescore lattices. We assume that the words in the\n"
        "input arpa language model has been converted to integers.\n"
        "\n"
        "The program is used jointly with utils/map_arpa_lm.pl to build\n"
        "ConstArpaLm format language model. We first map the words in an Arpa\n"
        "format language model to integers using utils/map_arpa_m.pl, and\n"
        "then use this program to build a ConstArpaLm format language model.\n"
        "\n"
        "Usage: arpa-to-const-arpa [opts] <input-arpa> <const-arpa>\n"
        " e.g.: arpa-to-const-arpa --bos-symbol=1 --eos-symbol=2 \\\n"
        "                          arpa.txt const_arpa";

    kaldi::ParseOptions po(usage);

    ArpaParseOptions options;
    options.Register(&po);

    // Ideally, these registrations would be in ArpaParseOptions, but some
    // programs want integers and other want symbols, so we register them
    // outside instead.
    po.Register("unk-symbol", &options.unk_symbol,
                "Integer corresponds to unknown-word in language model. -1 if "
                "no such word is provided.");
    po.Register("bos-symbol", &options.bos_symbol,
                "Integer corresponds to <s>. You must set this to your actual "
                "BOS integer.");
    po.Register("eos-symbol", &options.eos_symbol,
                "Integer corresponds to </s>. You must set this to your actual "
                "EOS integer.");

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    if (options.bos_symbol == -1 || options.eos_symbol == -1) {
      KALDI_ERR << "Please set --bos-symbol and --eos-symbol.";
      exit(1);
    }

    std::string arpa_rxfilename = po.GetArg(1),
        const_arpa_wxfilename = po.GetOptArg(2);

    bool ans = BuildConstArpaLm(options, arpa_rxfilename,
                                const_arpa_wxfilename);
    if (ans)
      return 0;
    else
      return 1;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
