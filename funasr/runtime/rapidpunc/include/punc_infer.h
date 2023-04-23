#pragma once 

class CRapidPuncOnnx {
private:

	CRpTokenizer m_Tokenizer;


	vector<const char*> m_szInputNames;
	vector<const char*> m_szOutputNames;
public:


	CRapidPuncOnnx(const char* szModelDir, int nNumThread);

	~CRapidPuncOnnx();

	void LoadModel(const std::string& model_dir, int nNumThread);
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "CRapidPuncOnnx");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();
	Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
	Ort::Session* m_session;
	vector<int>  Infer(vector<int64_t> InputData);
	string AddPunc(const char* szInput);
};