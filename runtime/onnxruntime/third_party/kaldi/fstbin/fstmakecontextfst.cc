// fstbin/fstmakecontextfst.cc

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
#include "util/common-utils.h"
#include "fst/fstlib.h"
#include "fstext/fstext-utils.h"
#include "fstext/context-fst.h"
#include "fstext/kaldi-fst-io.h"


/* for example of testing setup, see fstmakecontextsymbols.cc */

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    using namespace fst;
    using kaldi::int32;

    const char *usage =
        "Constructs a context FST with a specified context-width and context-position.\n"
        "Outputs the context FST, and a file in Kaldi format that describes what the\n"
        "input labels mean.  Note: this is very inefficient if there are a lot of phones,\n"
        "better to use fstcomposecontext instead\n"
        "\n"
        "Usage:  fstmakecontextfst <phones-symbol-table> <subsequential-symbol> <ilabels-output-file> [<out-fst>]\n"
        "E.g.:   fstmakecontextfst phones.txt 42 ilabels.sym > C.fst\n";

    bool binary = true;  // binary output to ilabels_output_file.
    std::string disambig_rxfilename, disambig_wxfilename;
    int32 context_width = 3, central_position = 1;

    ParseOptions po(usage);
    po.Register("read-disambig-syms", &disambig_rxfilename,
                "List of disambiguation symbols to read");
    po.Register("write-disambig-syms", &disambig_wxfilename,
                "List of disambiguation symbols to write");
    po.Register("context-size", &context_width, "Size of phonetic context window");
    po.Register("central-position", &central_position,
                "Designated central position in context window");
    po.Register("binary", &binary,
                "Write ilabels output file in binary Kaldi format");

    po.Read(argc, argv);

    if (po.NumArgs() < 3 || po.NumArgs() > 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string phones_symtab_filename = po.GetArg(1);
    int32 subseq_sym;
    if (!ConvertStringToInteger(po.GetArg(2), &subseq_sym))
      KALDI_ERR << "Invalid subsequential symbol " << po.GetArg(2);
    std::string ilabels_out_filename = po.GetArg(3);
    std::string fst_out_filename = po.GetOptArg(4);


    std::vector<kaldi::int32> phone_syms;
    {
      fst::SymbolTable *phones_symtab = NULL;
      {  // read phone symbol table.
        std::ifstream is(phones_symtab_filename.c_str());
        phones_symtab = fst::SymbolTable::ReadText(is, phones_symtab_filename);
        if (!phones_symtab) KALDI_ERR << "Could not read phones symbol-table file "<<phones_symtab_filename;
      }
      GetSymbols(*phones_symtab,
                 false,  // don't include eps,
                 &phone_syms);
      delete phones_symtab;
    }

    if ( (disambig_wxfilename != "") && (disambig_rxfilename == "") )
      KALDI_ERR << "fstmakecontextfst: cannot specify --write-disambig-syms if "
          "not specifying --read-disambig-syms\n";

    std::vector<int32> disambig_in;
    if (disambig_rxfilename != "") {
      if (!ReadIntegerVectorSimple(disambig_rxfilename, &disambig_in))
        KALDI_ERR << "fstcomposecontext: Could not read disambiguation symbols from "
                  << PrintableRxfilename(disambig_rxfilename);
    }

    if (std::binary_search(phone_syms.begin(), phone_syms.end(), subseq_sym)
       || std::binary_search(disambig_in.begin(), disambig_in.end(), subseq_sym))
      KALDI_ERR << "Invalid subsequential symbol " << subseq_sym
                << ", already a phone or disambiguation symbol.";

    // 'loop_fst' will be an acceptor FST with single (initial and final) state, with
    // a loop for each phone and disambiguation symbol.
    StdVectorFst loop_fst;
    loop_fst.AddState();  // Add state zero.
    loop_fst.SetStart(0);
    loop_fst.SetFinal(0, TropicalWeight::One());
    for (size_t i = 0; i < phone_syms.size(); i++) {
      int32 sym = phone_syms[i];
      loop_fst.AddArc(0, StdArc(sym, sym, TropicalWeight::One(), 0));
    }

    std::vector<std::vector<int32> > ilabels;
    VectorFst<StdArc> context_fst;

    ComposeContext(disambig_in, context_width, central_position,
                   &loop_fst, &context_fst, &ilabels, true);

    WriteFstKaldi(context_fst, fst_out_filename);

    WriteILabelInfo(Output(ilabels_out_filename, binary).Stream(),
                    binary, ilabels);

    if (disambig_wxfilename != "") {
      std::vector<int32> disambig_out;
      for (size_t i = 0; i < ilabels.size(); i++)
        if (ilabels[i].size() == 1 && ilabels[i][0] <= 0)
          disambig_out.push_back(static_cast<int32>(i));
      if (!WriteIntegerVectorSimple(disambig_wxfilename, disambig_out))
        KALDI_ERR << "fstcomposecontext: Could not write disambiguation symbols to "
                  << PrintableWxfilename(disambig_wxfilename);
    }
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
