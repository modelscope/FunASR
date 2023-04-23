#pragma once
#include "yaml-cpp/yaml.h"

class CTokenizer {
private:

	bool  m_Ready = false;
	vector<string>   m_ID2Token,m_ID2Punc;
	map<string, int>  m_Token2ID,m_Punc2ID;

public:

	CTokenizer(const char* szYmlFile);
	CTokenizer();
	bool OpenYaml(const char* szYmlFile);
	void read_yml(const YAML::Node& node);
	vector<string> ID2String(vector<int> Input);
	vector<int> String2IDs(vector<string> Input);
	int String2ID(string Input);
	vector<string> ID2Punc(vector<int> Input);
	string ID2Punc(int nPuncID);
	vector<int> Punc2IDs(vector<string> Input);
	vector<string> SplitChineseString(const string& strInfo);
	void strSplit(const string& str, const char split, vector<string>& res);
	void Tokenize(const char* strInfo, vector<string>& strOut, vector<int>& IDOut);

};
