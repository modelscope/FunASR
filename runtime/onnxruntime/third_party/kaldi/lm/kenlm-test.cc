#ifdef HAVE_KENLM
#include "util/text-utils.h"
#include "util/kaldi-io.h"
#include "lm/kenlm.h"

namespace kaldi {

void UnitTestKenLm() {
  // construct symbol_to_symbol_id to map word string to kaldi symbol index,
  // in practice, hypothesis is already intergerized, no need for this mapping.
  std::unordered_map<std::string, int32> symbol_to_symbol_id;
  std::ifstream symbol_table_stream("test_data/words.txt");
  std::string line;
  while (std::getline(symbol_table_stream, line)) {
    std::vector<std::string> fields;
    SplitStringToVector(line, " ", true, &fields);
    if (fields.size() == 2) {
      std::string symbol = fields[0];
      uint32 symbol_id = -1;
      ConvertStringToInteger(fields[1], &symbol_id);
      symbol_to_symbol_id[symbol] = symbol_id;
    }
  }

  // open testing stream, one sentence per line, in raw text form
  std::ifstream is("test_data/sentences.txt");

  KenLm lm;
  lm.Load("test_data/lm.kenlm", "test_data/words.txt");
  KenLmDeterministicOnDemandFst<fst::StdArc> lm_fst(&lm);

  std::string sentence;
  while(std::getline(is, sentence)) {
    std::vector<std::string> words;
    SplitStringToVector(sentence, " ", true, &words);
    words.push_back("</s>");

    // 1. test KenLm interface: this is only for test purpose,
    //    you should not use kenlm this way in Kaldi.
    std::string sentence_log = "[KENLM]";

    KenLm::State state[2];
    KenLm::State* istate = &state[0];
    KenLm::State* ostate = &state[1];

    lm.SetStateToBeginOfSentence(istate);

    for (int i = 0; i < words.size(); i++) {
      std::string word = words[i];
      BaseFloat log10_word_score = lm.Score(istate, lm.GetWordIndex(word), ostate);
      sentence_log += " " + word + 
                      "[" + std::to_string(lm.GetWordIndex(word)) + "]=" + 
                      std::to_string(-log10_word_score * M_LN10); //convert to -ln()
      std::swap(istate, ostate);
    }
    KALDI_LOG << sentence_log;

    // 2. test Fst wrapper interface (KenLmDeterministicFst),
    //    this is the recommanded way to interact with Kaldi's Fst framework.
    sentence_log = "(KALDI)";
    KenLmDeterministicOnDemandFst<fst::StdArc>::StateId s = lm_fst.Start();
    for (int i = 0; i < words.size(); i++) {
      int32 symbol_id = symbol_to_symbol_id[words[i]];
      sentence_log += " " + words[i] + "(" + std::to_string(symbol_id) + ")=";
      if (words[i] == "</s>") {
        sentence_log += std::to_string(lm_fst.Final(s).Value());
      } else {
        fst::StdArc arc;
        lm_fst.GetArc(s, symbol_id, &arc);
        s = arc.nextstate;
        sentence_log += std::to_string(arc.weight.Value());
      }
    }
    KALDI_LOG << sentence_log;
  }
}
} // namespace kaldi

int main(int argc, char *argv[]) {
  using namespace kaldi;
  UnitTestKenLm();
  return 0;
}
#endif