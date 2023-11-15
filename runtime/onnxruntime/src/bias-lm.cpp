#include "bias-lm.h"
#ifdef _WIN32
#include "fst-types.cc"
#endif
namespace funasr {
void print(std::queue<StateId> &q) {
  std::queue<StateId> data = q;
  while (!data.empty())
  {
    cout << data.front() << " ";
    data.pop();
  }
  cout << endl;
}

void BiasLm::LoadCfgFromYaml(const char* filename, BiasLmOption &opt) {
  YAML::Node config;
  try {
    config = YAML::LoadFile(filename);
  } catch(exception const &e) {
    LOG(INFO) << "Error loading file, yaml file error or not exist.";
    exit(-1);
  }
  try {
    YAML::Node bias_lm_conf = config["bias_lm_conf"];
    opt_.incre_bias_ = bias_lm_conf["increment_weight"].as<float>();
  } catch(exception const &e) {
  }
}

void BiasLm::BuildGraph(std::vector<std::vector<int>> &split_id_vec,
  std::vector<float> &custom_weight) {
  if (split_id_vec.empty()) {
    LOG(INFO) << "Skip building biaslm graph, hotword not exits.";
    return ; 
  }
  assert(split_id_vec.size() == custom_weight.size());
  // Build prefix tree
  std::unique_ptr<fst::StdVectorFst> prefix_tree(new fst::StdVectorFst());
  StateId start_state = prefix_tree->AddState();
  prefix_tree->SetStart(start_state);
  int id = 0;
  for (auto& x : split_id_vec) {
    StateId state = start_state;
    StateId next_state = state;
    float w = custom_weight[id++];
    std::vector<int> split_id = x;
    for (int j = 0; j < split_id.size(); j++) {
      next_state = prefix_tree->AddState();
      if (j == split_id.size() - 1) {
        prefix_tree->SetFinal(next_state, w);
      }
      prefix_tree->AddArc(state, Arc(split_id[j], split_id[j], opt_.incre_bias_, next_state));
      state = next_state;
    }
  }
  graph_ = std::unique_ptr<fst::StdVectorFst>(new fst::StdVectorFst());
  fst::Determinize(*prefix_tree, graph_.get());

  int num_node = graph_->NumStates();
  node_list_.resize(num_node);
  for (auto& x : split_id_vec) {
    StateId cur_state = 0;
    StateId next_state = 0;
    std::vector<int> split_id = x;
    for (int j = 0; j < split_id.size(); j++) {
      Matcher matcher(*graph_, fst::MATCH_INPUT);
      matcher.SetState(cur_state);
      if (matcher.Find(split_id[j])) {
        next_state = matcher.Value().nextstate;
        if (graph_->Final(next_state) != Weight::Zero()) {
          node_list_[next_state].is_final_ = true;
        }
        node_list_[next_state].score_ = opt_.incre_bias_ * (j + 1);
        cur_state = next_state;
      }
    }
  }
  
  // Build Aho-Corasick Automata
  std::queue<StateId> q;
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  // Back off state of all child nodes of the root node points to the root node
  for (ArcIterator aiter(*graph_, start_state); !aiter.Done(); aiter.Next()) {
    const Arc& arc = aiter.Value();
    node_list_[arc.nextstate].back_off_ = start_state;
    float back_off_score = (node_list_[arc.nextstate].is_final_ ? 0 :
      node_list_[start_state].score_ - node_list_[arc.nextstate].score_);
    graph_->AddArc(arc.nextstate, Arc(0, 0, back_off_score, start_state));
    q.push(arc.nextstate);
  }
  while (!q.empty()) {
    StateId state_id = q.front();
    q.pop();
    for (ArcIterator aiter(*graph_, state_id); !aiter.Done(); aiter.Next()) {
      const Arc& arc = aiter.Value();
      StateId next_state = arc.nextstate;
      StateId temp_state = node_list_[state_id].back_off_;
      if (next_state == start_state || next_state == temp_state) { 
        continue; 
      }
      while (true) {
        matcher.SetState(temp_state);
        if (matcher.Find(arc.ilabel)) {
          node_list_[next_state].back_off_ = matcher.Value().nextstate;
          break;
        } else if (temp_state == start_state) {
          node_list_[next_state].back_off_ = start_state;
          break;
        }
        temp_state = node_list_[temp_state].back_off_;
      }
      float back_off_score = (node_list_[next_state].is_final_ ? 0 :
        node_list_[node_list_[next_state].back_off_].score_ -
        node_list_[next_state].score_);
      graph_->AddArc(next_state, Arc(0, 0, back_off_score, 
        node_list_[next_state].back_off_));
      q.push(next_state);
    }
  }
  fst::ArcSort(graph_.get(), fst::StdILabelCompare());
  //graph_->Write("graph.final.fst");
}

float BiasLm::BiasLmScore(const StateId &his_state, const Label &lab, Label &new_state) {
  if (lab < 1 || lab > phn_set_.Size() || !graph_) { return VALUE_ZERO; }
  StateId cur_state = his_state;
  StateId next_state;
  float score = VALUE_ZERO;
  Matcher matcher(*graph_, fst::MATCH_INPUT);
  while (true) {
    StateId prev_state = cur_state;
    matcher.SetState(cur_state);
    if (matcher.Find(lab)) {
      next_state = matcher.Value().nextstate;
      score += matcher.Value().weight.Value();
      if (node_list_[next_state].is_final_) {
        score = score + graph_->Final(next_state).Value();
      }
      cur_state = next_state;
      break;
    } else {
      ArcIterator aiter(*graph_, cur_state);
      const Arc& arc = aiter.Value();
      if (arc.ilabel == 0) {
        score += arc.weight.Value();
        next_state = arc.nextstate;
        cur_state = next_state;
      }
      if (prev_state == ROOT_NODE && cur_state == ROOT_NODE) {
        break;
      }
    }
  }
  new_state = cur_state;
  return score;
}

void BiasLm::VocabIdToPhnIdVector(int vocab_id, std::vector<int> &phn_ids) {
  bool is_oov = false;
  phn_ids.clear();
  std::string word = vocab_.Id2String(vocab_id);
  std::vector<std::string> phn_vec;
  Utf8ToCharset(word, phn_vec);
  for (auto& phn : phn_vec) {
    if (!phn_set_.Find(phn)) {
      is_oov = true;
      break;
    } else {
      phn_ids.push_back(phn_set_.String2Id(phn));
    }
  }
  if (is_oov) { phn_ids.clear(); }
}

std::string BiasLm::GetPhoneLabel(int phone_id) {
  if (phone_id < 0 || phone_id >= phn_set_.Size()) { return ""; }
  return phn_set_.Id2String(phone_id);
}
}
