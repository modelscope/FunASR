/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once 

class CTTransformer {
/**
 * Author: Speech Lab of DAMO Academy, Alibaba Group
 * CT-Transformer: Controllable time-delay transformer for real-time punctuation prediction and disfluency detection
 * https://arxiv.org/pdf/2003.01309.pdf
*/

private:

	CTokenizer m_tokenizer;
	vector<string> m_strInputNames, m_strOutputNames;
	vector<const char*> m_szInputNames;
	vector<const char*> m_szOutputNames;

	std::shared_ptr<Ort::Session> m_session;
    Ort::Env env_;
    Ort::SessionOptions session_options;
public:

	CTTransformer();
	void InitPunc(const std::string &punc_model, const std::string &punc_config, int thread_num);
	~CTTransformer();
	vector<int>  Infer(vector<int64_t> input_data);
	string AddPunc(const char* sz_input);
};
