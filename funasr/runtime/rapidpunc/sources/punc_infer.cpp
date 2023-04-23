#include "precomp.h"

/**

name: input
type: int64[batch_size,feats_length]

name: text_lengths
type: int32[batch_size]
**/




CRapidPuncOnnx::CRapidPuncOnnx(const char* szModelDir, int nNumThread)
{

    for (size_t i=0; i< INPUT_NUM; i++)
    {
        m_szInputNames.push_back(INPUT_NAMES[i]);
    }

    m_szOutputNames.push_back(OUTPUT_NAME);

    LoadModel(szModelDir, nNumThread);
}

CRapidPuncOnnx::~CRapidPuncOnnx()
{


	if (m_session)
	{
		delete m_session;
		m_session = nullptr;
	}

}


void CRapidPuncOnnx::LoadModel(const std::string& model_dir, int nNumThread)
{

	sessionOptions.SetInterOpNumThreads(nNumThread);
	sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);


	string strModelPath = model_dir + MODEL_FILE;

	string strYamlPath = model_dir + YAML_FILE;
#ifdef _WIN32
	std::wstring detPath = strToWstr(strModelPath);
	m_session = new Ort::Session(env, detPath.c_str(), sessionOptions);
#else
	m_session = new Ort::Session(env, strModelPath.c_str(), sessionOptions);
#endif


	m_Tokenizer.OpenYaml(strYamlPath.c_str());

}



/*
     # Search for the last Period/QuestionMark as cache
        if mini_sentence_i < len(mini_sentences) - 1:
            sentenceEnd = -1
            last_comma_index = -1
            for i in range(len(punctuations) - 2, 1, -1):
                if self.punc_list[punctuations[i]] == "。" or self.punc_list[punctuations[i]] == "？":
                    sentenceEnd = i
                    break
                if last_comma_index < 0 and self.punc_list[punctuations[i]] == "，":
                    last_comma_index = i

            if sentenceEnd < 0 and len(mini_sentence) > cache_pop_trigger_limit and last_comma_index >= 0:
                # The sentence it too long, cut off at a comma.
                sentenceEnd = last_comma_index
                punctuations[sentenceEnd] = self.period
            cache_sent = mini_sentence[sentenceEnd + 1:]
            cache_sent_id = mini_sentence_id[sentenceEnd + 1:].tolist()
            mini_sentence = mini_sentence[0:sentenceEnd + 1]
            punctuations = punctuations[0:sentenceEnd + 1]

        new_mini_sentence_punc += [int(x) for x in punctuations]
        words_with_punc = []
        for i in range(len(mini_sentence)):
            if i > 0:
                if len(mini_sentence[i][0].encode()) == 1 and len(mini_sentence[i - 1][0].encode()) == 1:
                    mini_sentence[i] = " " + mini_sentence[i]
            words_with_punc.append(mini_sentence[i])
            if self.punc_list[punctuations[i]] != "_":
                words_with_punc.append(self.punc_list[punctuations[i]])
        new_mini_sentence += "".join(words_with_punc)
        # Add Period for the end of the sentence
        new_mini_sentence_out = new_mini_sentence
        new_mini_sentence_punc_out = new_mini_sentence_punc
        if mini_sentence_i == len(mini_sentences) - 1:
            if new_mini_sentence[-1] == "，" or new_mini_sentence[-1] == "、":
                new_mini_sentence_out = new_mini_sentence[:-1] + "。"
                new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
            elif new_mini_sentence[-1] != "。" and new_mini_sentence[-1] != "？":
                new_mini_sentence_out = new_mini_sentence + "。"
                new_mini_sentence_punc_out = new_mini_sentence_punc[:-1] + [self.period]
*/



string CRapidPuncOnnx::AddPunc(const char* szInput)
{

    string strResult;
    vector<string> strOut;
    vector<int> InputData;
    m_Tokenizer.Tokenize(szInput, strOut, InputData); 

    int nTotalBatch = ceil((float)InputData.size() / TOKEN_LEN);
    int nCurBatch = -1;
    int nSentEnd = -1, nLastCommaIndex = -1;
    vector<int64_t> RemainIDs; // 
    vector<string> RemainStr; //
    vector<int> NewPunctuation; //
    vector<string> NewString; //
    vector<string> NewSentenceOut;
    vector<int> NewPuncOut;
    int nDiff = 0;
    for (size_t i = 0; i < InputData.size(); i += TOKEN_LEN)
    {
        nDiff = (i + TOKEN_LEN) < InputData.size() ? (0) : (i + TOKEN_LEN - InputData.size());
        vector<int64_t> InputIDs(InputData.begin() + i, InputData.begin() + i + TOKEN_LEN - nDiff);
        vector<string> InputStr(strOut.begin() + i, strOut.begin() + i + TOKEN_LEN - nDiff);

        InputIDs.insert(InputIDs.begin(), RemainIDs.begin(), RemainIDs.end()); // RemainIDs+InputIDs;
        InputStr.insert(InputStr.begin(), RemainStr.begin(), RemainStr.end()); // RemainStr+InputStr;

        auto Punction = Infer(InputIDs);

        nCurBatch = i / TOKEN_LEN;

        
        if (nCurBatch < nTotalBatch - 1) // not the last minisetence
        {
            nSentEnd = -1;
            nLastCommaIndex = -1;
            for (int nIndex = Punction.size() - 2; nIndex > 0; nIndex--)
            {
                if (m_Tokenizer.ID2Punc(Punction[nIndex]) == m_Tokenizer.ID2Punc(PERIOD_INDEX) || m_Tokenizer.ID2Punc(Punction[nIndex]) == m_Tokenizer.ID2Punc(QUESTION_INDEX))
                {
                    nSentEnd = nIndex;
                    break;
                }

                if (nLastCommaIndex < 0 && m_Tokenizer.ID2Punc(Punction[nIndex]) == m_Tokenizer.ID2Punc(COMMA_INDEX))
                {
                    nLastCommaIndex = nIndex;
                }
            }

            if (nSentEnd < 0 && InputStr.size() > CACHE_POP_TRIGGER_LIMIT && nLastCommaIndex > 0)
            {
                nSentEnd = nLastCommaIndex;
                Punction[nSentEnd] = PERIOD_INDEX;
            }

            RemainStr.assign(InputStr.begin() + nSentEnd + 1, InputStr.end());
            RemainIDs.assign(InputIDs.begin() + nSentEnd + 1, InputIDs.end());

            InputStr.assign(InputStr.begin(), InputStr.begin() + nSentEnd + 1);  // minit_sentence
            Punction.assign(Punction.begin(), Punction.begin() + nSentEnd + 1);

        }
        
        NewPunctuation.insert(NewPunctuation.end(), Punction.begin(), Punction.end());
        vector<string> WordWithPunc;

        for (int i = 0; i < InputStr.size(); i++)
        {

            if (i > 0 && !(InputStr[i][0] & 0x80) && (i + 1) <InputStr.size() && !(InputStr[i+1][0] & 0x80))// 中间的英文？
            {
                InputStr[i] = InputStr[i]+ " ";
            }
            WordWithPunc.push_back(InputStr[i]);

            if (Punction[i] != NOTPUNC_INDEX) // 下划线
            {
                WordWithPunc.push_back(m_Tokenizer.ID2Punc(Punction[i]));

            }

        }

        NewString.insert(NewString.end(), WordWithPunc.begin(), WordWithPunc.end()); // new_mini_sentence += "".join(words_with_punc)

         NewSentenceOut = NewString;
         NewPuncOut = NewPunctuation;
        // last mini sentence
        if(nCurBatch == nTotalBatch - 1)
        {
            if (NewString[NewString.size() - 1] == m_Tokenizer.ID2Punc(COMMA_INDEX) || NewString[NewString.size() - 1] == m_Tokenizer.ID2Punc(DUN_INDEX))
            {
                NewSentenceOut.assign(NewString.begin(), NewString.end() - 1);
                NewSentenceOut.push_back(m_Tokenizer.ID2Punc(PERIOD_INDEX));

                NewPuncOut.assign(NewPunctuation.begin(), NewPunctuation.end() - 1);
                NewPuncOut.push_back(PERIOD_INDEX);

            }
            else if (NewString[NewString.size() - 1] == m_Tokenizer.ID2Punc(PERIOD_INDEX) && NewString[NewString.size() - 1] == m_Tokenizer.ID2Punc(QUESTION_INDEX))
            {
                NewSentenceOut = NewString;
                NewSentenceOut.push_back(m_Tokenizer.ID2Punc(PERIOD_INDEX));
                NewPuncOut = NewPunctuation;
                NewPuncOut.push_back(PERIOD_INDEX);
            }

        }

    }

    for (auto& item : NewSentenceOut)
        strResult += item;
 
    return strResult;
}



vector<int>  CRapidPuncOnnx::Infer(vector<int64_t> InputData)
{

    Ort::RunOptions run_option;

    vector<int> punction;
    std::array<int64_t, 2> input_shape_{ 1,(int64_t)InputData.size()};
    Ort::Value onnx_input = Ort::Value::CreateTensor<int64_t>(m_memoryInfo,
        InputData.data(),
        InputData.size(),
        input_shape_.data(),
        input_shape_.size());

    std::array<int32_t,1> text_lengths{ (int32_t)InputData.size() };
    std::array<int64_t,1> text_lengths_dim{ 1 };
    Ort::Value onnx_text_lengths = Ort::Value::CreateTensor(
        m_memoryInfo,
        text_lengths.data(),
        text_lengths.size() * sizeof(int32_t),
        text_lengths_dim.data(),
        text_lengths_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_input));
    input_onnx.emplace_back(std::move(onnx_text_lengths));
        
    try {

        auto outputTensor = m_session->Run(run_option, m_szInputNames.data(), input_onnx.data(), m_szInputNames.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();


        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float * floatData = outputTensor[0].GetTensorMutableData<float>();

        for (int i = 0; i < outputCount; i += CANDIDATE_NUM)
        {
            int index = argmax(floatData + i, floatData + i + CANDIDATE_NUM-1);
            punction.push_back(index);
        }
       
    }
    catch (...)
    {

        
    }


    return punction;
}



