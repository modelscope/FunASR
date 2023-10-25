// lm/arpa-file-parser.cc

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

#include <fst/fstlib.h>

#include <sstream>

#include "base/kaldi-error.h"
#include "base/kaldi-math.h"
#include "lm/arpa-file-parser.h"
#include "util/text-utils.h"

namespace kaldi {

ArpaFileParser::ArpaFileParser(const ArpaParseOptions& options,
                               fst::SymbolTable* symbols)
    : options_(options), symbols_(symbols),
      line_number_(0), warning_count_(0) {
}

ArpaFileParser::~ArpaFileParser() {
}

void TrimTrailingWhitespace(std::string *str) {
  str->erase(str->find_last_not_of(" \n\r\t") + 1);
}

void ArpaFileParser::Read(std::istream &is) {
  // Argument sanity checks.
  if (options_.bos_symbol <= 0 || options_.eos_symbol <= 0 ||
      options_.bos_symbol == options_.eos_symbol)
    KALDI_ERR << "BOS and EOS symbols are required, must not be epsilons, and "
              << "differ from each other. Given:"
              << " BOS=" << options_.bos_symbol
              << " EOS=" << options_.eos_symbol;
  if (symbols_ != NULL &&
      options_.oov_handling == ArpaParseOptions::kReplaceWithUnk &&
      (options_.unk_symbol <= 0 ||
       options_.unk_symbol == options_.bos_symbol ||
       options_.unk_symbol == options_.eos_symbol))
    KALDI_ERR << "When symbol table is given and OOV mode is kReplaceWithUnk, "
              << "UNK symbol is required, must not be epsilon, and "
              << "differ from both BOS and EOS symbols. Given:"
              << " UNK=" << options_.unk_symbol
              << " BOS=" << options_.bos_symbol
              << " EOS=" << options_.eos_symbol;
  if (symbols_ != NULL && symbols_->Find(options_.bos_symbol).empty())
    KALDI_ERR << "BOS symbol must exist in symbol table";
  if (symbols_ != NULL && symbols_->Find(options_.eos_symbol).empty())
    KALDI_ERR << "EOS symbol must exist in symbol table";
  if (symbols_ != NULL && options_.unk_symbol > 0 &&
      symbols_->Find(options_.unk_symbol).empty())
    KALDI_ERR << "UNK symbol must exist in symbol table";

  ngram_counts_.clear();
  line_number_ = 0;
  warning_count_ = 0;
  current_line_.clear();

#define PARSE_ERR KALDI_ERR << LineReference() << ": "

  // Give derived class an opportunity to prepare its state.
  ReadStarted();

  // Processes "\data\" section.
  bool keyword_found = false;
  while (++line_number_, getline(is, current_line_) && !is.eof()) {
    if (current_line_.find_first_not_of(" \t\n\r") == std::string::npos) {
      continue;
    }

    TrimTrailingWhitespace(&current_line_);

    // Continue skipping lines until the \data\ marker alone on a line is found.
    if (!keyword_found) {
      if (current_line_ == "\\data\\") {
        KALDI_LOG << "Reading \\data\\ section.";
        keyword_found = true;
      }
      continue;
    }

    if (current_line_[0] == '\\') break;

    // Enters "\data\" section, and looks for patterns like "ngram 1=1000",
    // which means there are 1000 unigrams.
    std::size_t equal_symbol_pos = current_line_.find("=");
    if (equal_symbol_pos != std::string::npos)
      // Guaranteed spaces around the "=".
      current_line_.replace(equal_symbol_pos, 1, " = ");
    std::vector<std::string> col;
    SplitStringToVector(current_line_, " \t", true, &col);
    if (col.size() == 4 && col[0] == "ngram" && col[2] == "=") {
      int32 order, ngram_count = 0;
      if (!ConvertStringToInteger(col[1], &order) ||
          !ConvertStringToInteger(col[3], &ngram_count)) {
        PARSE_ERR << "cannot parse ngram count";
      }
      if (ngram_counts_.size() <= order) {
        ngram_counts_.resize(order);
      }
      ngram_counts_[order - 1] = ngram_count;
    } else {
      KALDI_WARN << LineReference()
                 << ": uninterpretable line in \\data\\ section";
    }
  }

  if (ngram_counts_.size() == 0)
    PARSE_ERR << "\\data\\ section missing or empty.";

  // Signal that grammar order and n-gram counts are known.
  HeaderAvailable();

  NGram ngram;
  ngram.words.reserve(ngram_counts_.size());

  // Processes "\N-grams:" section.
  for (int32 cur_order = 1; cur_order <= ngram_counts_.size(); ++cur_order) {
    // Skips n-grams with zero count.
    if (ngram_counts_[cur_order - 1] == 0)
      KALDI_WARN << "Zero ngram count in ngram order " << cur_order
                 << "(look for 'ngram " << cur_order << "=0' in the \\data\\ "
                 << " section). There is possibly a problem with the file.";

    // Must be looking at a \k-grams: directive at this point.
    std::ostringstream keyword;
    keyword << "\\" << cur_order << "-grams:";
    if (current_line_ != keyword.str()) {
      PARSE_ERR << "invalid directive, expecting '" << keyword.str() << "'";
    }
    KALDI_LOG << "Reading " << current_line_ << " section.";

    int32 ngram_count = 0;
    while (++line_number_, getline(is, current_line_) && !is.eof()) {
      if (current_line_.find_first_not_of(" \n\t\r") == std::string::npos) {
        continue;
      }
      if (current_line_[0] == '\\') {
        TrimTrailingWhitespace(&current_line_);
        std::ostringstream next_keyword;
        next_keyword << "\\" << cur_order + 1 << "-grams:";
        if ((current_line_ != next_keyword.str()) &&
            (current_line_ != "\\end\\")) {
          if (ShouldWarn()) {
            KALDI_WARN << "ignoring possible directive '" << current_line_
                       << "' expecting '" << next_keyword.str() << "'";

            if (warning_count_ > 0 &&
                warning_count_ > static_cast<uint32>(options_.max_warnings)) {
              KALDI_WARN << "Of " << warning_count_ << " parse warnings, "
                         << options_.max_warnings << " were reported. "
                         << "Run program with --max-arpa-warnings=-1 "
                         << "to see all warnings";
            }
          }
        } else {
          break;
        }
      }

      std::vector<std::string> col;
      SplitStringToVector(current_line_, " \t", true, &col);

      if (col.size() < 1 + cur_order ||
          col.size() > 2 + cur_order ||
          (cur_order == ngram_counts_.size() && col.size() != 1 + cur_order)) {
        PARSE_ERR << "Invalid n-gram data line";
      }
      ++ngram_count;

      // Parse out n-gram logprob and, if present, backoff weight.
      if (!ConvertStringToReal(col[0], &ngram.logprob)) {
        PARSE_ERR << "invalid n-gram logprob '" << col[0] << "'";
      }
      ngram.backoff = 0.0;
      if (col.size() > cur_order + 1) {
        if (!ConvertStringToReal(col[cur_order + 1], &ngram.backoff))
          PARSE_ERR << "invalid backoff weight '" << col[cur_order + 1] << "'";
      }
      // Convert to natural log.
      ngram.logprob *= M_LN10;
      ngram.backoff *= M_LN10;

      ngram.words.resize(cur_order);
      bool skip_ngram = false;
      for (int32 index = 0; !skip_ngram && index < cur_order; ++index) {
        int32 word;
        if (symbols_) {
          // Symbol table provided, so symbol labels are expected.
          if (options_.oov_handling == ArpaParseOptions::kAddToSymbols) {
            word = symbols_->AddSymbol(col[1 + index]);
          } else {
            word = symbols_->Find(col[1 + index]);
            if (word == -1) { // fst::kNoSymbol
              switch (options_.oov_handling) {
                case ArpaParseOptions::kReplaceWithUnk:
                  word = options_.unk_symbol;
                  break;
                case ArpaParseOptions::kSkipNGram:
                  if (ShouldWarn())
                    KALDI_WARN << LineReference() << " skipped: word '"
                               << col[1 + index] << "' not in symbol table";
                  skip_ngram = true;
                  break;
                default:
                  PARSE_ERR << "word '"  << col[1 + index]
                            << "' not in symbol table";
              }
            }
          }
        } else {
          // Symbols not provided, LM file should contain integers.
          if (!ConvertStringToInteger(col[1 + index], &word) || word < 0) {
            PARSE_ERR << "invalid symbol '" << col[1 + index] << "'";
          }
        }
        // Whichever way we got it, an epsilon is invalid.
        if (word == 0) {
          PARSE_ERR << "epsilon symbol '" << col[1 + index]
                    << "' is illegal in ARPA LM";
        }
        ngram.words[index] = word;
      }
      if (!skip_ngram) {
        ConsumeNGram(ngram);
      }
    }
    if (ngram_count > ngram_counts_[cur_order - 1]) {
      PARSE_ERR << "header said there would be " << ngram_counts_[cur_order - 1]
                << " n-grams of order " << cur_order
                << ", but we saw more already.";
    }
  }

  if (current_line_ != "\\end\\") {
    PARSE_ERR << "invalid or unexpected directive line, expecting \\end\\";
  }

  if (warning_count_ > 0 &&
      warning_count_ > static_cast<uint32>(options_.max_warnings)) {
    KALDI_WARN << "Of " << warning_count_ << " parse warnings, "
               << options_.max_warnings << " were reported. Run program with "
               << "--max-arpa-warnings=-1 to see all warnings";
  }

  current_line_.clear();
  ReadComplete();

#undef PARSE_ERR
}

std::string ArpaFileParser::LineReference() const {
  std::ostringstream ss;
  ss << "line " << line_number_ << " [" << current_line_ << "]";
  return ss.str();
}

bool ArpaFileParser::ShouldWarn() {
  return (warning_count_ != -1) &&
    (++warning_count_ <= static_cast<uint32>(options_.max_warnings));
}

}  // namespace kaldi
