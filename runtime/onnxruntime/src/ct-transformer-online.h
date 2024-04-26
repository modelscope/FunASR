/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#pragma once 

namespace funasr {
class CTTransformerOnline : public PuncModel {
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

	CTTransformerOnline();
	void InitPunc(const std::string &punc_model, const std::string &punc_config, const std::string &token_file, int thread_num);
	~CTTransformerOnline();
	vector<int>  Infer(vector<int32_t> input_data, int nCacheSize);
	string AddPunc(const char* sz_input, vector<string> &arr_cache, std::string language="zh-cn");
	void Transport(vector<float>& In, int nRows, int nCols);
	void VadMask(int size, int vad_pos,vector<float>& Result);
	void Triangle(int text_length, vector<float>& Result);
};
} // namespace funasr