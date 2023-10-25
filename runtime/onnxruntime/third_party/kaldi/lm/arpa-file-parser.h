// lm/arpa-file-parser.h

// Copyright 2014  Guoguo Chen
// Copyright 2016  Smart Action Company LLC (kkm)

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

#ifndef KALDI_LM_ARPA_FILE_PARSER_H_
#define KALDI_LM_ARPA_FILE_PARSER_H_

#include <fst/fst-decl.h>

#include <string>
#include <vector>

#include "base/kaldi-types.h"
#include "itf/options-itf.h"

namespace kaldi {

/**
  Options that control ArpaFileParser
*/
struct ArpaParseOptions {
  enum OovHandling {
    kRaiseError,     ///< Abort on OOV words
    kAddToSymbols,   ///< Add novel words to the symbol table.
    kReplaceWithUnk,  ///< Replace OOV words with <unk>.
    kSkipNGram       ///< Skip n-gram with OOV word and continue.
  };

  ArpaParseOptions():
      bos_symbol(-1), eos_symbol(-1), unk_symbol(-1),
      oov_handling(kRaiseError), max_warnings(30) { }

  void Register(OptionsItf *opts) {
    // Registering only the max_warnings count, since other options are
    // treated differently by client programs: some want integer symbols,
    // while other are passed words in their command line.
    opts->Register("max-arpa-warnings", &max_warnings,
                   "Maximum warnings to report on ARPA parsing, "
                   "0 to disable, -1 to show all");
  }

  int32 bos_symbol;  ///< Symbol for <s>, Required non-epsilon.
  int32 eos_symbol;  ///< Symbol for </s>, Required non-epsilon.
  int32 unk_symbol;  ///< Symbol for <unk>, Required for kReplaceWithUnk.
  OovHandling oov_handling;  ///< How to handle OOV words in the file.
  int32 max_warnings;  ///< Maximum warnings to report, <0 unlimited.
};

/**
   A parsed n-gram from ARPA LM file.
*/
struct NGram {
  NGram() : logprob(0.0), backoff(0.0) { }
  std::vector<int32> words;  ///< Symbols in left to right order.
  float logprob;             ///< Log-prob of the n-gram.
  float backoff;             ///< log-backoff weight of the n-gram.
                             ///< Defaults to zero if not specified.
};

/**
    ArpaFileParser is an abstract base class for ARPA LM file conversion.

    See ConstArpaLmBuilder and ArpaLmCompiler for usage examples.
*/
class ArpaFileParser {
 public:
  /// Constructs the parser with the given options and optional symbol table.
  /// If symbol table is provided, then the file should contain text n-grams,
  /// and the words are mapped to symbols through it. bos_symbol and
  /// eos_symbol in the options structure must be valid symbols in the table,
  /// and so must be unk_symbol if provided. The table is not owned by the
  /// parser, but may be augmented, if oov_handling is set to kAddToSymbols.
  /// If symbol table is a null pointer, the file should contain integer
  /// symbol values, and oov_handling has no effect. bos_symbol and eos_symbol
  /// must be valid symbols still.
  ArpaFileParser(const ArpaParseOptions& options, fst::SymbolTable* symbols);
  virtual ~ArpaFileParser();

  /// Read ARPA LM file from a stream.
  void Read(std::istream &is);

  /// Parser options.
  const ArpaParseOptions& Options() const { return options_; }

 protected:
  /// Override called before reading starts. This is the point to prepare
  /// any state in the derived class.
  virtual void ReadStarted() { }

  /// Override function called to signal that ARPA header with the expected
  /// number of n-grams has been read, and ngram_counts() is now valid.
  virtual void HeaderAvailable() { }

  /// Pure override that must be implemented to process current n-gram. The
  /// n-grams are sent in the file order, which guarantees that all
  /// (k-1)-grams are processed before the first k-gram is.
  virtual void ConsumeNGram(const NGram&) = 0;

  /// Override function called after the last n-gram has been consumed.
  virtual void ReadComplete() { }

  /// Read-only access to symbol table. Not owned, do not make public.
  const fst::SymbolTable* Symbols() const { return symbols_; }

  /// Inside ConsumeNGram(), provides the current line number.
  int32 LineNumber() const { return line_number_; }

  /// Inside ConsumeNGram(), returns a formatted reference to the line being
  /// compiled, to print out as part of diagnostics.
  std::string LineReference() const;

  /// Increments warning count, and returns true if a warning should be
  /// printed or false if the count has exceeded the set maximum.
  bool ShouldWarn();

  /// N-gram counts. Valid from the point when HeaderAvailable() is called.
  const std::vector<int32>& NgramCounts() const { return ngram_counts_; }

 private:
  ArpaParseOptions options_;
  fst::SymbolTable* symbols_;  // the pointer is not owned here.
  int32 line_number_;
  uint32 warning_count_;
  std::string current_line_;
  std::vector<int32> ngram_counts_;
};

}  // namespace kaldi

#endif  // KALDI_LM_ARPA_FILE_PARSER_H_
