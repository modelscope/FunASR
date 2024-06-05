 /**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"

namespace funasr {
CTokenizer::CTokenizer(const char* sz_yamlfile):m_ready(false)
{
	OpenYaml(sz_yamlfile);
}

CTokenizer::CTokenizer():m_ready(false)
{
}

CTokenizer::~CTokenizer()
{
	if (jieba_dict_trie_){
		delete jieba_dict_trie_;
	}
	if (jieba_model_){
    	delete jieba_model_;
	}
}

void CTokenizer::SetJiebaRes(cppjieba::DictTrie *dict, cppjieba::HMMModel *hmm) {
	jieba_processor_.SetJiebaRes(dict, hmm);
}

void CTokenizer::JiebaInit(std::string punc_config){
    if (seg_jieba){
        std::string model_path = punc_config.substr(0, punc_config.length() - (sizeof(PUNC_CONFIG_NAME)-1));
        std::string jieba_dict_file = PathAppend(model_path, JIEBA_DICT);
        std::string jieba_hmm_file = PathAppend(model_path, JIEBA_HMM_MODEL);
        std::string jieba_userdict_file = PathAppend(model_path, JIEBA_USERDICT);
		try{
        	jieba_dict_trie_ = new cppjieba::DictTrie(jieba_dict_file, jieba_userdict_file);
			LOG(INFO) << "Successfully load file from " << jieba_dict_file << ", " << jieba_userdict_file;
		}catch(exception const &e){
			LOG(ERROR) << "Error loading file, Jieba dict file error or not exist.";
			exit(-1);
		}

		try{
        	jieba_model_ = new cppjieba::HMMModel(jieba_hmm_file);
			LOG(INFO) << "Successfully load model from " << jieba_hmm_file;
		}catch(exception const &e){
			LOG(ERROR) << "Error loading file, Jieba hmm file error or not exist.";
			exit(-1);
		}

        SetJiebaRes(jieba_dict_trie_, jieba_model_);
    }else {
        jieba_dict_trie_ = nullptr;
        jieba_model_ = nullptr;
    }
}

void CTokenizer::ReadYaml(const YAML::Node& node) 
{
	if (node.IsMap()) 
	{//��map��
		for (auto it = node.begin(); it != node.end(); ++it) 
		{
			ReadYaml(it->second);
		}
	}
	if (node.IsSequence()) {//��������
		for (size_t i = 0; i < node.size(); ++i) {
			ReadYaml(node[i]);
		}
	}
	if (node.IsScalar()) {//�Ǳ�����
		LOG(INFO) << node.as<string>();
	}
}

bool CTokenizer::OpenYaml(const char* sz_yamlfile)
{
	YAML::Node m_Config;
	try{
		m_Config = YAML::LoadFile(sz_yamlfile);
	}catch(exception const &e){
        LOG(INFO) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

	try
	{
		YAML::Node conf_seg_jieba = m_Config["seg_jieba"];
        if (conf_seg_jieba.IsDefined()){
            seg_jieba = conf_seg_jieba.as<bool>();
        }

		auto Tokens = m_Config["token_list"];
		if (Tokens.IsSequence())
		{
			for (size_t i = 0; i < Tokens.size(); ++i) 
			{
				if (Tokens[i].IsScalar())
				{
					m_id2token.push_back(Tokens[i].as<string>());
					m_token2id.insert(make_pair<string, int>(Tokens[i].as<string>(), i));
				}
			}
		}
		auto Puncs = m_Config["punc_list"];
		if (Puncs.IsSequence())
		{
			for (size_t i = 0; i < Puncs.size(); ++i)
			{
				if (Puncs[i].IsScalar())
				{ 
					m_id2punc.push_back(Puncs[i].as<string>());
					m_punc2id.insert(make_pair<string, int>(Puncs[i].as<string>(), i));
				}
			}
		}
	}
	catch (YAML::BadFile& e) {
		LOG(ERROR) << "Read error!";
		return  false;
	}
	m_ready = true;
	return m_ready;
}

bool CTokenizer::OpenYaml(const char* sz_yamlfile, const char* token_file)
{
	YAML::Node m_Config;
	try{
		m_Config = YAML::LoadFile(sz_yamlfile);
	}catch(exception const &e){
        LOG(INFO) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

	try
	{
		YAML::Node conf_seg_jieba = m_Config["seg_jieba"];
        if (conf_seg_jieba.IsDefined()){
            seg_jieba = conf_seg_jieba.as<bool>();
        }

		auto Puncs = m_Config["model_conf"]["punc_list"];
		if (Puncs.IsSequence())
		{
			for (size_t i = 0; i < Puncs.size(); ++i)
			{
				if (Puncs[i].IsScalar())
				{ 
					m_id2punc.push_back(Puncs[i].as<string>());
					m_punc2id.insert(make_pair<string, int>(Puncs[i].as<string>(), i));
				}
			}
		}

		nlohmann::json json_array;
		std::ifstream file(token_file);
		if (file.is_open()) {
			file >> json_array;
			file.close();
		} else {
			LOG(INFO) << "Error loading token file, token file error or not exist.";
			return  false;
		}

		int i = 0;
		for (const auto& element : json_array) {
			m_id2token.push_back(element);
			m_token2id[element] = i;
			i++;
		}
	}
	catch (YAML::BadFile& e) {
		LOG(ERROR) << "Read error!";
		return  false;
	}
	m_ready = true;
	return m_ready;
}

vector<string> CTokenizer::Id2String(vector<int> input)
{
	vector<string> result;
	for (auto& item : input)
	{
		result.push_back(m_id2token[item]);
	}
	return result;
}

int CTokenizer::String2Id(string input)
{
	int nID = 0; // <blank>
	if (m_token2id.find(input) != m_token2id.end())
		nID=(m_token2id[input]);
	else
		nID=(m_token2id[UNK_CHAR]);
	return nID;
}

vector<int> CTokenizer::String2Ids(vector<string> input)
{
	vector<int> result;
	for (auto& item : input)
	{	
		transform(item.begin(), item.end(), item.begin(), ::tolower);
		if (m_token2id.find(item) != m_token2id.end())
			result.push_back(m_token2id[item]);
		else
			result.push_back(m_token2id[UNK_CHAR]);
	}
	return result;
}

vector<string> CTokenizer::Id2Punc(vector<int> input)
{
	vector<string> result;
	for (auto& item : input)
	{
		result.push_back(m_id2punc[item]);
	}
	return result;
}

string CTokenizer::Id2Punc(int n_punc_id)
{
	return m_id2punc[n_punc_id];
}

vector<int> CTokenizer::Punc2Ids(vector<string> input)
{
	vector<int> result;
	for (auto& item : input)
	{
		result.push_back(m_punc2id[item]);
	}
	return result;
}

bool CTokenizer::IsPunc(string& Punc)
{
	if (m_punc2id.find(Punc) != m_punc2id.end())
		return true;
	else
		return false;
}

vector<string> CTokenizer::SplitChineseString(const string & str_info)
{
	vector<string> list;
	int strSize = str_info.size();
	int i = 0;

	while (i < strSize) {
		int len = 1;
		for (int j = 0; j < 6 && (str_info[i] & (0x80 >> j)); j++) {
			len = j + 1;
		}
		list.push_back(str_info.substr(i, len));
		i += len;
	}
	return list;
}

vector<string> CTokenizer::SplitChineseJieba(const string & str_info)
{
	vector<string> list;
	jieba_processor_.Cut(str_info, list, false);

	return list;
}

void CTokenizer::StrSplit(const string& str, const char split, vector<string>& res)
{
	if (str == "")
	{
		return;
	}
	string&& strs = str + split;
	size_t pos = strs.find(split);

	while (pos != string::npos)
	{
		res.emplace_back(strs.substr(0, pos));
		strs = move(strs.substr(pos + 1, strs.size()));
		pos = strs.find(split);
	}
}

void CTokenizer::Tokenize(const char* str_info, vector<string> & str_out, vector<int> & id_out)
{
	vector<string>  strList;
	StrSplit(str_info,' ', strList);
	string current_eng,current_chinese;
	for (auto& item : strList)
	{
		current_eng = "";
		current_chinese = "";
		for (auto& ch : item)
		{
			if (!(ch& 0x80))
			{ // Ӣ��
				if (current_chinese.size() > 0)
				{
					// for utf-8 chinese
					vector<string> chineseList;
					if(seg_jieba){
						chineseList = SplitChineseJieba(current_chinese);
					}else{
						chineseList = SplitChineseString(current_chinese);
					}
					str_out.insert(str_out.end(), chineseList.begin(),chineseList.end());
					current_chinese = "";
				}
				current_eng += ch;
			}
			else
			{
				if (current_eng.size() > 0)
				{
					str_out.push_back(current_eng);
					current_eng = "";
				}
				current_chinese += ch;
			}
		}
		if (current_chinese.size() > 0)
		{
			// for utf-8 chinese
			vector<string> chineseList;
			if(seg_jieba){
				chineseList = SplitChineseJieba(current_chinese);
			}else{
				chineseList = SplitChineseString(current_chinese);
			}
			str_out.insert(str_out.end(), chineseList.begin(), chineseList.end());
			current_chinese = "";
		}
		if (current_eng.size() > 0)
		{
			str_out.push_back(current_eng);
		}
	}
	id_out= String2Ids(str_out);
}

} // namespace funasr