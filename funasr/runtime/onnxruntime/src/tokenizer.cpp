 #include "precomp.h"

CTokenizer::CTokenizer(const char* szYmlFile):m_Ready(false)
{
	OpenYaml(szYmlFile);
}

CTokenizer::CTokenizer():m_Ready(false)
{
}

void CTokenizer::read_yml(const YAML::Node& node) 
{
	if (node.IsMap()) 
	{//��map��
		for (auto it = node.begin(); it != node.end(); ++it) 
		{
			read_yml(it->second);
		}
	}
	if (node.IsSequence()) {//��������
		for (size_t i = 0; i < node.size(); ++i) {
			read_yml(node[i]);
		}
	}
	if (node.IsScalar()) {//�Ǳ�����
		cout << node.as<string>() << endl;
	}
}

bool CTokenizer::OpenYaml(const char* szYmlFile)
{
	YAML::Node m_Config = YAML::LoadFile(szYmlFile);
	if (m_Config.IsNull())
		return false;
	try
	{
		auto Tokens = m_Config["token_list"];
		if (Tokens.IsSequence())
		{
			for (size_t i = 0; i < Tokens.size(); ++i) 
			{
				if (Tokens[i].IsScalar())
				{
					m_ID2Token.push_back(Tokens[i].as<string>());
					m_Token2ID.insert(make_pair<string, int>(Tokens[i].as<string>(), i));
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
					m_ID2Punc.push_back(Puncs[i].as<string>());
					m_Punc2ID.insert(make_pair<string, int>(Puncs[i].as<string>(), i));
				}
			}
		}
	}
	catch (YAML::BadFile& e) {
		std::cout << "read error!" << std::endl;
		return  false;
	}
	m_Ready = true;
	return m_Ready;
}

vector<string> CTokenizer::ID2String(vector<int> Input)
{
	vector<string> result;
	for (auto& item : Input)
	{
		result.push_back(m_ID2Token[item]);
	}
	return result;
}

int CTokenizer::String2ID(string Input)
{
	int nID = 0; // <blank>
	if (m_Token2ID.find(Input) != m_Token2ID.end())
		nID=(m_Token2ID[Input]);
	else
		nID=(m_Token2ID[UNK_CHAR]);
	return nID;
}

vector<int> CTokenizer::String2IDs(vector<string> Input)
{
	vector<int> result;
	for (auto& item : Input)
	{	
		transform(item.begin(), item.end(), item.begin(), ::tolower);
		if (m_Token2ID.find(item) != m_Token2ID.end())
			result.push_back(m_Token2ID[item]);
		else
			result.push_back(m_Token2ID[UNK_CHAR]);
	}
	return result;
}

vector<string> CTokenizer::ID2Punc(vector<int> Input)
{
	vector<string> result;
	for (auto& item : Input)
	{
		result.push_back(m_ID2Punc[item]);
	}
	return result;
}

string CTokenizer::ID2Punc(int nPuncID)
{
	return m_ID2Punc[nPuncID];
}

vector<int> CTokenizer::Punc2IDs(vector<string> Input)
{
	vector<int> result;
	for (auto& item : Input)
	{
		result.push_back(m_Punc2ID[item]);
	}
	return result;
}

vector<string> CTokenizer::SplitChineseString(const string & strInfo)
{
	vector<string> list;
	int strSize = strInfo.size();
	int i = 0;

	while (i < strSize) {
		int len = 1;
		for (int j = 0; j < 6 && (strInfo[i] & (0x80 >> j)); j++) {
			len = j + 1;
		}
		list.push_back(strInfo.substr(i, len));
		i += len;
	}
	return list;
}

void CTokenizer::strSplit(const string& str, const char split, vector<string>& res)
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

 void CTokenizer::Tokenize(const char* strInfo, vector<string> & strOut, vector<int> & IDOut)
{
	vector<string>  strList;
	strSplit(strInfo,' ', strList);
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
					auto chineseList = SplitChineseString(current_chinese);
					strOut.insert(strOut.end(), chineseList.begin(),chineseList.end());
					current_chinese = "";
				}
				current_eng += ch;
			}
			else
			{
				if (current_eng.size() > 0)
				{
					strOut.push_back(current_eng);
					current_eng = "";
				}
				current_chinese += ch;
			}
		}
		if (current_chinese.size() > 0)
		{
			auto chineseList = SplitChineseString(current_chinese);
			strOut.insert(strOut.end(), chineseList.begin(), chineseList.end());
			current_chinese = "";
		}
		if (current_eng.size() > 0)
		{
			strOut.push_back(current_eng);
		}
	}
	IDOut= String2IDs(strOut);
}
