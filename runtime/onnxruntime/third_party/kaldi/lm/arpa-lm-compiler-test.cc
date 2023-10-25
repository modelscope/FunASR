// lm/arpa-lm-compiler-test.cc

// Copyright 2009-2011 Gilles Boulianne
// Copyright 2016 Smart Action LLC (kkm)

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

#include <iostream>
#include <string>
#include <sstream>

#include "base/kaldi-error.h"
#include "base/kaldi-math.h"
#include "lm/arpa-lm-compiler.h"
#include "util/kaldi-io.h"

namespace kaldi {

// Predefine some symbol values, because any integer is as good than any other.
enum {
  kEps = 0,
  kDisambig,
  kBos, kEos,
};

// Number of random sentences for coverage test.
static const int kRandomSentences = 50;

// Creates an FST that generates any sequence of symbols taken from given
// symbol table. The FST is then associated with the symbol table.
static fst::StdVectorFst* CreateGenFst(bool seps, const fst::SymbolTable* pst) {
  fst::StdVectorFst* genFst = new fst::StdVectorFst;
  genFst->SetInputSymbols(pst);
  genFst->SetOutputSymbols(pst);

  fst::StdArc::StateId midId   = genFst->AddState();
  if (!seps) {
    fst::StdArc::StateId initId  = genFst->AddState();
    fst::StdArc::StateId finalId = genFst->AddState();
    genFst->SetStart(initId);
    genFst->SetFinal(finalId, fst::StdArc::Weight::One());
    genFst->AddArc(initId, fst::StdArc(kBos, kBos, 0, midId));
    genFst->AddArc(midId,  fst::StdArc(kEos, kEos, 0, finalId));
  } else {
    genFst->SetStart(midId);
    genFst->SetFinal(midId, fst::StdArc::Weight::One());
  }

  // Add a loop for each symbol in the table except the four special ones.
  fst::SymbolTableIterator si(*pst);
  for (si.Reset(); !si.Done(); si.Next()) {
    if (si.Value() == kBos || si.Value() == kEos ||
        si.Value() == kEps || si.Value() == kDisambig)
      continue;
    genFst->AddArc(midId, fst::StdArc(si.Value(), si.Value(),
                                      fst::StdArc::Weight::One(), midId));
  }
  return genFst;
}

// Compile given ARPA file.
ArpaLmCompiler* Compile(bool seps, const std::string &infile) {
  ArpaParseOptions options;
  fst::SymbolTable symbols;
  // Use spaces on special symbols, so we rather fail than read them by mistake.
  symbols.AddSymbol(" <eps>", kEps);
  symbols.AddSymbol(" #0", kDisambig);
  options.bos_symbol = symbols.AddSymbol("<s>", kBos);
  options.eos_symbol = symbols.AddSymbol("</s>", kEos);
  options.oov_handling = ArpaParseOptions::kAddToSymbols;

  // Tests in this form cannot be run with epsilon substitution, unless every
  // random path is also fitted with a #0-transducing self-loop.
  ArpaLmCompiler* lm_compiler =
      new ArpaLmCompiler(options,
                         seps ? kDisambig : 0,
                         &symbols);
  {
    Input ki(infile);
    lm_compiler->Read(ki.Stream());
  }
  return lm_compiler;
}

// Add a state to an FSA after last_state, add a form last_state to the new
// state, and return the new state.
fst::StdArc::StateId AddToChainFsa(fst::StdMutableFst* fst,
                                   fst::StdArc::StateId last_state,
                                   int64 symbol) {
  fst::StdArc::StateId next_state  = fst->AddState();
  fst->AddArc(last_state, fst::StdArc(symbol, symbol, 0, next_state));
  return next_state;
}

// Add a disambiguator-generating self loop to every state of an FST.
void AddSelfLoops(fst::StdMutableFst* fst) {
  for (fst::StateIterator<fst::StdMutableFst> siter(*fst);
       !siter.Done(); siter.Next()) {
    fst->AddArc(siter.Value(),
                fst::StdArc(kEps, kDisambig, 0, siter.Value()));
  }
}

// Compiles infile and then runs kRandomSentences random coverage tests on the
// compiled FST.
bool CoverageTest(bool seps, const std::string &infile) {
  // Compile ARPA model.
  ArpaLmCompiler* lm_compiler = Compile(seps, infile);

  // Create an FST that generates any sequence of symbols taken from the model
  // output.
  fst::StdVectorFst* genFst =
      CreateGenFst(seps, lm_compiler->Fst().OutputSymbols());

  int num_successes = 0;
  for (int32 i = 0; i < kRandomSentences; ++i) {
    // Generate a random sentence FST.
    fst::StdVectorFst sentence;
    RandGen(*genFst, &sentence);
    if (seps)
      AddSelfLoops(&sentence);

    fst::ArcSort(lm_compiler->MutableFst(), fst::StdOLabelCompare());

    // The past must successfully compose with the LM FST.
    fst::StdVectorFst composition;
    Compose(sentence, lm_compiler->Fst(), &composition);
    if (composition.Start() != fst::kNoStateId)
      ++num_successes;
  }

  delete genFst;
  delete lm_compiler;

  bool ok = num_successes == kRandomSentences;
  if (!ok) {
    KALDI_WARN << "Coverage test failed on " << infile << ": composed "
               << num_successes << "/" << kRandomSentences;
  }
  return ok;
}

bool ScoringTest(bool seps, const std::string &infile, const std::string& sentence,
                 float expected) {
  ArpaLmCompiler* lm_compiler = Compile(seps, infile);
  const fst::SymbolTable* symbols = lm_compiler->Fst().InputSymbols();

  // Create a sentence FST for scoring.
  fst::StdVectorFst sentFst;
  fst::StdArc::StateId state = sentFst.AddState();
  sentFst.SetStart(state);
  if (!seps) {
    state = AddToChainFsa(&sentFst, state, kBos);
  }
  std::stringstream ss(sentence);
  std::string word;
  while (ss >> word) {
    int64 word_sym = symbols->Find(word);
    KALDI_ASSERT(word_sym != -1);
    state = AddToChainFsa(&sentFst, state, word_sym);
  }
  if (!seps) {
    state = AddToChainFsa(&sentFst, state, kEos);
  }
  if (seps) {
    AddSelfLoops(&sentFst);
  }
  sentFst.SetFinal(state, 0);
  sentFst.SetOutputSymbols(symbols);

  // Do the composition and extract final weight.
  fst::StdVectorFst composed;
  fst::Compose(sentFst, lm_compiler->Fst(), &composed);
  delete lm_compiler;

  if (composed.Start() == fst::kNoStateId) {
    KALDI_WARN << "Test sentence " << sentence << " did not compose "
               << "with the language model FST\n";
    return false;
  }

  std::vector<fst::StdArc::Weight> shortest;
  fst::ShortestDistance(composed, &shortest, true);
  float actual = shortest[composed.Start()].Value();

  bool ok = ApproxEqual(expected, actual);
  if (!ok) {
    KALDI_WARN << "Scored " << sentence << " in " << infile
               << ": Expected=" << expected << " actual=" << actual;
  }
  return ok;
}

bool ThrowsExceptionTest(bool seps, const std::string &infile) {
  try {
    // Make memory cleanup easy in both cases of try-catch block.
    std::unique_ptr<ArpaLmCompiler> compiler(Compile(seps, infile));
    return false;
  } catch (const KaldiFatalError&) {
    return true;
  }
}

}  // namespace kaldi

bool RunAllTests(bool seps) {
  bool ok = true;
  ok &= kaldi::CoverageTest(seps, "test_data/missing_backoffs.arpa");
  ok &= kaldi::CoverageTest(seps, "test_data/unused_backoffs.arpa");
  ok &= kaldi::CoverageTest(seps, "test_data/input.arpa");

  ok &= kaldi::ScoringTest(seps, "test_data/input.arpa", "b b b a", 59.2649);
  ok &= kaldi::ScoringTest(seps, "test_data/input.arpa", "a b", 4.36082);

  ok &= kaldi::ThrowsExceptionTest(seps, "test_data/missing_bos.arpa");

  if (!ok) {
    KALDI_WARN << "Tests " << (seps ? "with" : "without")
               << " epsilon substitution FAILED";
  }
  return ok;
}

int main(int argc, char *argv[]) {
  bool ok = true;

  ok &= RunAllTests(false);  // Without disambiguators (old behavior).
  ok &= RunAllTests(true);   // With epsilon substitution (new behavior).

  if (ok) {
    KALDI_LOG << "All tests passed";
    return 0;
  } else {
    KALDI_WARN << "Test FAILED";
    return 1;
  }
}
