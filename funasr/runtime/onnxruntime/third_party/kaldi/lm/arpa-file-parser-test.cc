// lm/arpa-file-parser-test.cc

// Copyright 2016  Smart Action Company LLC (kkm)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//  http://www.apache.org/licenses/LICENSE-2.0

// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

/**
 * @file arpa-file-parser-test.cc
 * @brief Unit tests for language model code.
 */

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "base/kaldi-common.h"
#include "fst/fstlib.h"
#include "lm/arpa-file-parser.h"

namespace kaldi {
namespace {

const int kMaxOrder = 3;

struct NGramTestData {
  int32 line_number;
  float logprob;
  int32 words[kMaxOrder];
  float backoff;
};

std::ostream& operator<<(std::ostream &os, const NGramTestData &data) {
  std::ios::fmtflags saved_state(os.flags());
  os << std::fixed << std::setprecision(6);

  os << data.logprob << ' ';
  for (int i = 0; i < kMaxOrder; ++i) os << data.words[i] << ' ';
  os << data.backoff << " // Line " << data.line_number;

  os.flags(saved_state);
  return os;
}

// This does not own the array pointer, and uset to simplify passing expected
// result to TestableArpaFileParser::Verify.
template <class T>
struct CountedArray {
  template <size_t N>
  CountedArray(T(&array)[N]) : array(array), count(N) { }
  const T *array;
  const size_t count;
};

template <class T, size_t N>
inline CountedArray<T> MakeCountedArray(T(&array)[N]) {
  return CountedArray<T>(array);
}

class TestableArpaFileParser : public ArpaFileParser {
 public:
  TestableArpaFileParser(const ArpaParseOptions &options,
                         fst::SymbolTable *symbols)
      : ArpaFileParser(options, symbols),
        header_available_(false),
        read_complete_(false),
        last_order_(0) { }
  void Validate(CountedArray<int32> counts, CountedArray<NGramTestData> ngrams);

 private:
  // ArpaFileParser overrides.
  virtual void HeaderAvailable();
  virtual void ConsumeNGram(const NGram& ngram);
  virtual void ReadComplete();

  bool header_available_;
  bool read_complete_;
  int32 last_order_;
  std::vector<NGramTestData> ngrams_;
};

void TestableArpaFileParser::HeaderAvailable() {
  KALDI_ASSERT(!header_available_);
  KALDI_ASSERT(!read_complete_);
  header_available_ = true;
  KALDI_ASSERT(NgramCounts().size() <= kMaxOrder);
}

void TestableArpaFileParser::ConsumeNGram(const NGram& ngram) {
  KALDI_ASSERT(header_available_);
  KALDI_ASSERT(!read_complete_);
  KALDI_ASSERT(ngram.words.size() <= NgramCounts().size());
  KALDI_ASSERT(ngram.words.size() >= last_order_);
  last_order_ = ngram.words.size();

  NGramTestData entry = { 0 };
  entry.line_number = LineNumber();
  entry.logprob = ngram.logprob;
  entry.backoff = ngram.backoff;
  std::copy(ngram.words.begin(), ngram.words.end(), entry.words);
  ngrams_.push_back(entry);
}

void TestableArpaFileParser::ReadComplete() {
  KALDI_ASSERT(header_available_);
  KALDI_ASSERT(!read_complete_);
  read_complete_ = true;
}

bool CompareNgrams(const NGramTestData &actual,
                   NGramTestData expected) {
  expected.logprob *= Log(10.0);
  expected.backoff *= Log(10.0);
  if (actual.line_number != expected.line_number
      || !std::equal(actual.words, actual.words + kMaxOrder,
                     expected.words)
      || !ApproxEqual(actual.logprob, expected.logprob)
      || !ApproxEqual(actual.backoff, expected.backoff)) {
    KALDI_WARN << "Actual n-gram [" << actual
               << "] differs from expected [" << expected << "]";
    return false;
  }
  return true;
}

void TestableArpaFileParser::Validate(
    CountedArray<int32> expect_counts,
    CountedArray<NGramTestData> expect_ngrams) {
  // This needs better disagnostics probably.
  KALDI_ASSERT(NgramCounts().size() == expect_counts.count);
  KALDI_ASSERT(std::equal(NgramCounts().begin(), NgramCounts().end(),
                          expect_counts.array));

  KALDI_ASSERT(ngrams_.size() == expect_ngrams.count);
  // auto mpos = std::mismatch(ngrams_.begin(), ngrams_.end(),
  //                           expect_ngrams.array, CompareNgrams);
  // if (mpos.first != ngrams_.end())
  //   KALDI_ERR << "Maismatch at index " << mpos.first - ngrams_.begin();
  // TODO: auto above requres C++11, and I cannot spell out the type!!!
  KALDI_ASSERT(std::equal(ngrams_.begin(), ngrams_.end(),
                          expect_ngrams.array, CompareNgrams));
}

// Read integer LM (no symbols) with log base conversion.
void ReadIntegerLmLogconvExpectSuccess() {
  KALDI_LOG << "ReadIntegerLmLogconvExpectSuccess()";

  static std::string integer_lm = "\
\\data\\\n\
ngram 1=4\n\
ngram 2=2\n\
ngram 3=2\n\
\n\
\\1-grams:\n\
-5.2\t4\t-3.3\n\
-3.4\t5\n\
0\t1\t-2.5\n\
-4.3\t2\n\
\n\
\\2-grams:\n\
-1.4\t4 5\t-3.2\n\
-1.3\t1 4\t-4.2\n\
\n\
\\3-grams:\n\
-0.3\t1 4 5\n\
-0.2\t4 5 2\n\
\n\
\\end\\";

  int32 expect_counts[] = { 4, 2, 2 };
  NGramTestData expect_ngrams[] = {
    {  7, -5.2, { 4, 0, 0 }, -3.3 },
    {  8, -3.4, { 5, 0, 0 },  0.0 },
    {  9,  0.0, { 1, 0, 0 }, -2.5 },
    { 10, -4.3, { 2, 0, 0 },  0.0 },

    { 13, -1.4, { 4, 5, 0 }, -3.2 },
    { 14, -1.3, { 1, 4, 0 }, -4.2 },

    { 17, -0.3, { 1, 4, 5 },  0.0 },
    { 18, -0.2, { 4, 5, 2 },  0.0 } };

  ArpaParseOptions options;
  options.bos_symbol = 1;
  options.eos_symbol = 2;

  TestableArpaFileParser parser(options, NULL);
  std::istringstream stm(integer_lm, std::ios_base::in);
  parser.Read(stm);
  parser.Validate(MakeCountedArray(expect_counts),
                  MakeCountedArray(expect_ngrams));
}

// \xCE\xB2 = UTF-8 for Greek beta, to churn some UTF-8 cranks.
static std::string symbolic_lm = "\
We also allow random text coming before the \\data\\\n\
section marker. Even this is ok:\n\
\n\
\\1-grams:\n\
\n\
and should be ignored before the \\data\\ marker\n\
is seen alone by itself on a line.\n\
\n\
\\data\\\n\
ngram 1=4\n\
ngram 2=2\n\
ngram 3=2\n\
\n\
\\1-grams: \n\
-5.2\ta\t-3.3\n\
-3.4\t\xCE\xB2\n\
0.0\t<s>\t-2.5\n\
-4.3\t</s>\n\
\n\
\\2-grams:\t\n\
-1.5\ta \xCE\xB2\t-3.2\n\
-1.3\t<s> a\t-4.2\n\
\n\
\\3-grams:\n\
-0.3\t<s> a \xCE\xB2\n\
-0.2\t<s> a </s>\n\
\\end\\";

// Symbol table that is created with predefined test symbols, "a" but no "b".
class TestSymbolTable : public fst::SymbolTable {
 public:
  TestSymbolTable() {
    AddSymbol("<eps>", 0);
    AddSymbol("<s>", 1);
    AddSymbol("</s>", 2);
    AddSymbol("<unk>", 3);
    AddSymbol("a", 4);
  }
};

// Full expected result shared between ReadSymbolicLmNoOovImpl and
// ReadSymbolicLmWithOovAddToSymbols().
NGramTestData expect_symbolic_full[] = {
  { 15, -5.2, { 4, 0, 0 }, -3.3 },
  { 16, -3.4, { 5, 0, 0 },  0.0 },
  { 17,  0.0, { 1, 0, 0 }, -2.5 },
  { 18, -4.3, { 2, 0, 0 },  0.0 },

  { 21, -1.5, { 4, 5, 0 }, -3.2 },
  { 22, -1.3, { 1, 4, 0 }, -4.2 },

  { 25, -0.3, { 1, 4, 5 },  0.0 },
  { 26, -0.2, { 1, 4, 2 },  0.0 } };

// This is run with all possible oov setting and yields same result.
void ReadSymbolicLmNoOovImpl(ArpaParseOptions::OovHandling oov) {
  int32 expect_counts[] = { 4, 2, 2 };
  TestSymbolTable symbols;
  symbols.AddSymbol("\xCE\xB2", 5);

  ArpaParseOptions options;
  options.bos_symbol = 1;
  options.eos_symbol = 2;
  options.unk_symbol = 3;
  options.oov_handling = oov;
  TestableArpaFileParser parser(options, &symbols);
  std::istringstream stm(symbolic_lm, std::ios_base::in);
  parser.Read(stm);
  parser.Validate(MakeCountedArray(expect_counts),
                  MakeCountedArray(expect_symbolic_full));
  KALDI_ASSERT(symbols.NumSymbols() == 6);
}

void ReadSymbolicLmNoOovTests() {
  KALDI_LOG << "ReadSymbolicLmNoOovImpl(kRaiseError)";
  ReadSymbolicLmNoOovImpl(ArpaParseOptions::kRaiseError);
  KALDI_LOG << "ReadSymbolicLmNoOovImpl(kAddToSymbols)";
  ReadSymbolicLmNoOovImpl(ArpaParseOptions::kAddToSymbols);
  KALDI_LOG << "ReadSymbolicLmNoOovImpl(kReplaceWithUnk)";
  ReadSymbolicLmNoOovImpl(ArpaParseOptions::kReplaceWithUnk);
  KALDI_LOG << "ReadSymbolicLmNoOovImpl(kSkipNGram)";
  ReadSymbolicLmNoOovImpl(ArpaParseOptions::kSkipNGram);
}

// This is run with all possible oov setting and yields same result.
void ReadSymbolicLmWithOovImpl(
    ArpaParseOptions::OovHandling oov,
    CountedArray<NGramTestData> expect_ngrams,
    fst::SymbolTable* symbols) {
  int32 expect_counts[] = { 4, 2, 2 };
  ArpaParseOptions options;
  options.bos_symbol = 1;
  options.eos_symbol = 2;
  options.unk_symbol = 3;
  options.oov_handling = oov;
  TestableArpaFileParser parser(options, symbols);
  std::istringstream stm(symbolic_lm, std::ios_base::in);
  parser.Read(stm);
  parser.Validate(MakeCountedArray(expect_counts), expect_ngrams);
}

void ReadSymbolicLmWithOovAddToSymbols() {
  TestSymbolTable symbols;
  ReadSymbolicLmWithOovImpl(ArpaParseOptions::kAddToSymbols,
                            MakeCountedArray(expect_symbolic_full),
                            &symbols);
  KALDI_ASSERT(symbols.NumSymbols() == 6);
  KALDI_ASSERT(symbols.Find("\xCE\xB2") == 5);
}

void ReadSymbolicLmWithOovReplaceWithUnk() {
  NGramTestData expect_symbolic_unk_b[] = {
    { 15, -5.2, { 4, 0, 0 }, -3.3 },
    { 16, -3.4, { 3, 0, 0 },  0.0 },
    { 17,  0.0, { 1, 0, 0 }, -2.5 },
    { 18, -4.3, { 2, 0, 0 },  0.0 },

    { 21, -1.5, { 4, 3, 0 }, -3.2 },
    { 22, -1.3, { 1, 4, 0 }, -4.2 },

    { 25, -0.3, { 1, 4, 3 },  0.0 },
    { 26, -0.2, { 1, 4, 2 },  0.0 } };

  TestSymbolTable symbols;
  ReadSymbolicLmWithOovImpl(ArpaParseOptions::kReplaceWithUnk,
                            MakeCountedArray(expect_symbolic_unk_b),
                            &symbols);
  KALDI_ASSERT(symbols.NumSymbols() == 5);
}

void ReadSymbolicLmWithOovSkipNGram() {
  NGramTestData expect_symbolic_no_b[] = {
    { 15, -5.2, { 4, 0, 0 }, -3.3 },
    { 17,  0.0, { 1, 0, 0 }, -2.5 },
    { 18, -4.3, { 2, 0, 0 },  0.0 },

    { 22, -1.3, { 1, 4, 0 }, -4.2 },

    { 26, -0.2, { 1, 4, 2 },  0.0 } };

  TestSymbolTable symbols;
  ReadSymbolicLmWithOovImpl(ArpaParseOptions::kSkipNGram,
                            MakeCountedArray(expect_symbolic_no_b),
                            &symbols);
  KALDI_ASSERT(symbols.NumSymbols() == 5);
}

void ReadSymbolicLmWithOovTests() {
  KALDI_LOG << "ReadSymbolicLmWithOovAddToSymbols()";
  ReadSymbolicLmWithOovAddToSymbols();
  KALDI_LOG << "ReadSymbolicLmWithOovReplaceWithUnk()";
  ReadSymbolicLmWithOovReplaceWithUnk();
  KALDI_LOG << "ReadSymbolicLmWithOovSkipNGram()";
  ReadSymbolicLmWithOovSkipNGram();
}

}  // namespace
}  // namespace kaldi

int main(int argc, char *argv[]) {
  kaldi::ReadIntegerLmLogconvExpectSuccess();
  kaldi::ReadSymbolicLmNoOovTests();
  kaldi::ReadSymbolicLmWithOovTests();
}
