// bin/arpa2fst.cc
//
// Copyright 2009-2011  Gilles Boulianne.
//
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

#include "lm/arpa-lm-compiler.h"
#include "util/kaldi-io.h"
#include "util/parse-options.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage  =
        "Convert an ARPA format language model into an FST\n"
        "Usage: arpa2fst [opts] <input-arpa> <output-fst>\n"
        " e.g.: arpa2fst --disambig-symbol=#0 --read-symbol-table="
        "data/lang/words.txt lm/input.arpa G.fst\n\n"
        "Note: When called without switches, the output G.fst will contain\n"
        "an embedded symbol table. This is compatible with the way a previous\n"
        "version of arpa2fst worked.\n";

    ParseOptions po(usage);

    ArpaParseOptions options;
    options.Register(&po);

    // Option flags.
    std::string bos_symbol = "<s>";
    std::string eos_symbol = "</s>";
    std::string disambig_symbol;
    std::string read_syms_filename;
    std::string write_syms_filename;
    bool keep_symbols = false;
    bool ilabel_sort = true;

    po.Register("bos-symbol", &bos_symbol,
                "Beginning of sentence symbol");
    po.Register("eos-symbol", &eos_symbol,
                "End of sentence symbol");
    po.Register("disambig-symbol", &disambig_symbol,
                "Disambiguator. If provided (e. g. #0), used on input side of "
                "backoff links, and <s> and </s> are replaced with epsilons");
    po.Register("read-symbol-table", &read_syms_filename,
                "Use existing symbol table");
    po.Register("write-symbol-table", &write_syms_filename,
                "Write generated symbol table to a file");
    po.Register("keep-symbols", &keep_symbols,
                "Store symbol table with FST. Symbols always saved to FST if "
                "symbol tables are neither read or written (otherwise symbols "
                "would be lost entirely)");
    po.Register("ilabel-sort", &ilabel_sort,
                "Ilabel-sort the output FST");

    po.Read(argc, argv);

    if (po.NumArgs() != 1 && po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string arpa_rxfilename = po.GetArg(1),
        fst_wxfilename = po.GetOptArg(2);

    int64 disambig_symbol_id = 0;

    fst::SymbolTable* symbols;
    if (!read_syms_filename.empty()) {
      // Use existing symbols. Required symbols must be in the table.
      kaldi::Input kisym(read_syms_filename);
      symbols = fst::SymbolTable::ReadText(
          kisym.Stream(), PrintableWxfilename(read_syms_filename));
      if (symbols == NULL)
        KALDI_ERR << "Could not read symbol table from file "
                  << read_syms_filename;

      options.oov_handling = ArpaParseOptions::kSkipNGram;
      if (!disambig_symbol.empty()) {
        disambig_symbol_id = symbols->Find(disambig_symbol);
        if (disambig_symbol_id == -1) // fst::kNoSymbol
          KALDI_ERR << "Symbol table " << read_syms_filename
                    << " has no symbol for " << disambig_symbol;
      }
    } else {
      // Create a new symbol table and populate it from ARPA file.
      symbols = new fst::SymbolTable(PrintableWxfilename(fst_wxfilename));
      options.oov_handling = ArpaParseOptions::kAddToSymbols;
      symbols->AddSymbol("<eps>", 0);
      if (!disambig_symbol.empty()) {
        disambig_symbol_id = symbols->AddSymbol(disambig_symbol);
      }
    }

    // Add or use existing BOS and EOS.
    options.bos_symbol = symbols->AddSymbol(bos_symbol);
    options.eos_symbol = symbols->AddSymbol(eos_symbol);

    // If producing new (not reading existing) symbols and not saving them,
    // need to keep symbols with FST, otherwise they would be lost.
    if (read_syms_filename.empty() && write_syms_filename.empty())
      keep_symbols = true;

    // Actually compile LM.
    KALDI_ASSERT (symbols != NULL);
    ArpaLmCompiler lm_compiler(options, disambig_symbol_id, symbols);
    {
      Input ki(arpa_rxfilename);
      lm_compiler.Read(ki.Stream());
    }

    // Sort the FST in-place if requested by options.
    if (ilabel_sort) {
      fst::ArcSort(lm_compiler.MutableFst(), fst::StdILabelCompare());
    }

    // Write symbols if requested.
    if (!write_syms_filename.empty()) {
      kaldi::Output kosym(write_syms_filename, false);
      symbols->WriteText(kosym.Stream());
    }

    // Write LM FST.
    bool write_binary = true, write_header = false;
    kaldi::Output kofst(fst_wxfilename, write_binary, write_header);
    fst::FstWriteOptions wopts(PrintableWxfilename(fst_wxfilename));
    wopts.write_isymbols = wopts.write_osymbols = keep_symbols;
    lm_compiler.Fst().Write(kofst.Stream(), wopts);

    delete symbols;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
