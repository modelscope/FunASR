/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once
#include <yaml-cpp/yaml.h>

class CTokenizer {
private:

	bool  m_ready = false;
	vector<string>   m_id2token,m_id2punc;
	map<string, int>  m_token2id,m_punc2id;

public:

	CTokenizer(const char* sz_yamlfile);
	CTokenizer();
	bool OpenYaml(const char* sz_yamlfile);
	void ReadYaml(const YAML::Node& node);
	vector<string> Id2String(vector<int> input);
	vector<int> String2Ids(vector<string> input);
	int String2Id(string input);
	vector<string> Id2Punc(vector<int> input);
	string Id2Punc(int n_punc_id);
	vector<int> Punc2Ids(vector<string> input);
	vector<string> SplitChineseString(const string& str_info);
	void StrSplit(const string& str, const char split, vector<string>& res);
	void Tokenize(const char* str_info, vector<string>& str_out, vector<int>& id_out);

};
