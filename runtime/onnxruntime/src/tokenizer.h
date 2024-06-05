/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once
#include <yaml-cpp/yaml.h>
#include "cppjieba/DictTrie.hpp"
#include "cppjieba/HMMModel.hpp"
#include "cppjieba/Jieba.hpp"
#include "nlohmann/json.hpp"

namespace funasr {
class CTokenizer {
private:

	bool  m_ready = false;
	vector<string>   m_id2token,m_id2punc;
	map<string, int>  m_token2id,m_punc2id;

	cppjieba::DictTrie *jieba_dict_trie_=nullptr;
    cppjieba::HMMModel *jieba_model_=nullptr;
	cppjieba::Jieba jieba_processor_;

public:

	CTokenizer(const char* sz_yamlfile);
	CTokenizer();
	~CTokenizer();
	bool OpenYaml(const char* sz_yamlfile);
	bool OpenYaml(const char* sz_yamlfile, const char* token_file);
	void ReadYaml(const YAML::Node& node);
	vector<string> Id2String(vector<int> input);
	vector<int> String2Ids(vector<string> input);
	int String2Id(string input);
	vector<string> Id2Punc(vector<int> input);
	string Id2Punc(int n_punc_id);
	vector<int> Punc2Ids(vector<string> input);
	vector<string> SplitChineseString(const string& str_info);
	vector<string> SplitChineseJieba(const string& str_info);
	void StrSplit(const string& str, const char split, vector<string>& res);
	void Tokenize(const char* str_info, vector<string>& str_out, vector<int>& id_out);
	bool IsPunc(string& Punc);
	bool seg_jieba = false;
	void SetJiebaRes(cppjieba::DictTrie *dict, cppjieba::HMMModel *hmm);
	void JiebaInit(std::string punc_config);
};

} // namespace funasr
