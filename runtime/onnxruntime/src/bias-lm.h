#ifndef BIAS_LM_
#define BIAS_LM_
#include <assert.h>
#include "util.h"
#include "fst/fstlib.h"
#include "phone-set.h"
#include "vocab.h"
#include "util/text-utils.h"
#include <yaml-cpp/yaml.h>
#ifdef _WIN32
#include "win_func.h"
#endif
// node type
#define ROOT_NODE 0
#define VALUE_ZERO 0.0f

namespace funasr {
typedef fst::StdArc Arc;
typedef typename Arc::StateId StateId;
typedef typename Arc::Weight Weight;
typedef typename Arc::Label Label;
typedef typename fst::SortedMatcher<fst::StdVectorFst> Matcher;
typedef typename fst::ArcIterator<fst::StdVectorFst> ArcIterator;

class Node {
 public:
  Node() : score_(0.0f), is_final_(false), back_off_(-1) {}
  float score_;
  bool is_final_;
  StateId back_off_;
};

class BiasLmOption {
 public:
  BiasLmOption() : incre_bias_(20.0f), scale_(1.0f) {}
  float incre_bias_;
  float scale_;
};

class BiasLm {
 public:
  BiasLm(const string &hws_file, const string &cfg_file, 
    const PhoneSet& phn_set, const Vocab& vocab) :
    phn_set_(phn_set), vocab_(vocab) {
    std::string line;
    std::ifstream ifs_hws(hws_file.c_str());
    std::vector<float> custom_weight;
    std::vector<std::vector<int>> split_id_vec;

    struct timeval start, end;
    gettimeofday(&start, nullptr);

    LoadCfgFromYaml(cfg_file.c_str(), opt_);
    while (getline(ifs_hws, line)) {
      Trim(&line);
      if (line.empty()) {
        continue;
      }
      float score = 1.0f;
      bool is_oov = false;
      std::vector<std::string> text;
      std::vector<std::string> split_str;
      std::vector<int> split_id;
      SplitStringToVector(line, "\t", true, &text);
      if (text.size() > 1) {
        score = std::stof(text[1]);
      }
      SplitChiEngCharacters(text[0], split_str);
      for (auto &str : split_str) {
        std::vector<string> lex_vec;
        std::string lex_str = vocab_.Word2Lex(str);
        SplitStringToVector(lex_str, " ", true, &lex_vec);
        for (auto &token : lex_vec) {
          split_id.push_back(phn_set_.String2Id(token));
          if (!phn_set_.Find(token)) {
            is_oov = true;
            break;
          }
        }
      }
      if (!is_oov) {
        split_id_vec.push_back(split_id);
        custom_weight.push_back(score);
      }
    }
    BuildGraph(split_id_vec, custom_weight);
    ifs_hws.close();

    gettimeofday(&end, nullptr);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Build bias lm takes " << (double)modle_init_micros / 1000000 << " s";
  }

  BiasLm(unordered_map<string, int> &hws_map, int inc_bias,
    const PhoneSet& phn_set, const Vocab& vocab) :
    phn_set_(phn_set), vocab_(vocab) {
    std::vector<float> custom_weight;
    std::vector<std::vector<int>> split_id_vec;

    struct timeval start, end;
    gettimeofday(&start, nullptr);
    opt_.incre_bias_ = inc_bias;
    for (const pair<string, int>& kv : hws_map) {
      float score = 1.0f;
      bool is_oov = false;
      std::vector<std::string> text;
      std::vector<std::string> split_str;
      std::vector<int> split_id;
      score = kv.second;
      SplitChiEngCharacters(kv.first, split_str);
      for (auto &str : split_str) {
        std::vector<string> lex_vec;
        std::string lex_str = vocab_.Word2Lex(str);
        SplitStringToVector(lex_str, " ", true, &lex_vec);
        for (auto &token : lex_vec) {
          split_id.push_back(phn_set_.String2Id(token));
          if (!phn_set_.Find(token)) {
            is_oov = true;
            break;
          }
        }
      }
      if (!is_oov) {
        split_id_vec.push_back(split_id);
        custom_weight.push_back(score);
      }
    }
    BuildGraph(split_id_vec, custom_weight);

    gettimeofday(&end, nullptr);
    long seconds = (end.tv_sec - start.tv_sec);
    long modle_init_micros = ((seconds * 1000000) + end.tv_usec) - (start.tv_usec);
    LOG(INFO) << "Build bias lm takes " << (double)modle_init_micros / 1000000 << " s";
  }

  void BuildGraph(std::vector<std::vector<int>> &vec, std::vector<float> &wts);
  float BiasLmScore(const StateId &cur_state, const Label &lab, Label &new_state);
  void VocabIdToPhnIdVector(int vocab_id, std::vector<int> &phn_ids);
  void LoadCfgFromYaml(const char* filename, BiasLmOption &opt);
  std::string GetPhoneLabel(int phone_id);
 private:
  const PhoneSet& phn_set_;
  const Vocab& vocab_;
  std::unique_ptr<fst::StdVectorFst> graph_ = nullptr;
  std::vector<Node> node_list_;
  BiasLmOption opt_;
};
} // namespace funasr
#endif // BIAS_LM_
