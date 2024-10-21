/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"

namespace funasr {
CTTransformerOnline::CTTransformerOnline()
:env_(ORT_LOGGING_LEVEL_ERROR, ""),session_options{}
{
}

void CTTransformerOnline::InitPunc(const std::string &punc_model, const std::string &punc_config, const std::string &token_file, int thread_num){
    session_options.SetIntraOpNumThreads(thread_num);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options.DisableCpuMemArena();

    try{
        m_session = std::make_unique<Ort::Session>(env_, ORTSTRING(punc_model).c_str(), session_options);
        LOG(INFO) << "Successfully load model from " << punc_model;
    }
    catch (std::exception const &e) {
        LOG(ERROR) << "Error when load punc onnx model: " << e.what();
        exit(-1);
    }
    // read inputnames outputnames
    GetInputNames(m_session.get(), m_strInputNames, m_szInputNames);
    GetOutputNames(m_session.get(), m_strOutputNames, m_szOutputNames);

	m_tokenizer.OpenYaml(punc_config.c_str(), token_file.c_str());
}

CTTransformerOnline::~CTTransformerOnline()
{
}

string CTTransformerOnline::AddPunc(const char* sz_input, vector<string> &arr_cache, std::string language)
{
    string strResult;
    vector<string> strOut;
    vector<int> InputData;
    string strText; //full_text
    strText = accumulate(arr_cache.begin(), arr_cache.end(), strText);
    strText += sz_input;  // full_text = precache + text  
    m_tokenizer.Tokenize(strText.c_str(), strOut, InputData);

    int nTotalBatch = ceil((float)InputData.size() / TOKEN_LEN);
    int nCurBatch = -1;
    int nSentEnd = -1, nLastCommaIndex = -1;
    vector<int32_t> RemainIDs; // 
    vector<string> RemainStr; //
    vector<int>     new_mini_sentence_punc; //          sentence_punc_list = []
    vector<string> sentenceOut; // sentenceOut
    vector<string> sentence_punc_list,sentence_words_list,sentence_punc_list_out; // sentence_words_list = []
    
    int nSkipNum = 0;
    int nDiff = 0;
    for (size_t i = 0; i < InputData.size(); i += TOKEN_LEN)
    {
        nDiff = (i + TOKEN_LEN) < InputData.size() ? (0) : (i + TOKEN_LEN - InputData.size());
        vector<int32_t> InputIDs(InputData.begin() + i, InputData.begin() + i + (TOKEN_LEN - nDiff));
        vector<string> InputStr(strOut.begin() + i, strOut.begin() + i + (TOKEN_LEN - nDiff));
        InputIDs.insert(InputIDs.begin(), RemainIDs.begin(), RemainIDs.end()); // RemainIDs+InputIDs;
        InputStr.insert(InputStr.begin(), RemainStr.begin(), RemainStr.end()); // RemainStr+InputStr;

        auto Punction = Infer(InputIDs, arr_cache.size());
        nCurBatch = i / TOKEN_LEN;
        if (nCurBatch < nTotalBatch - 1) // not the last minisetence
        {
            nSentEnd = -1;
            nLastCommaIndex = -1;
            for (int nIndex = Punction.size() - 2; nIndex > 0; nIndex--)
            {
                if (m_tokenizer.Id2Punc(Punction[nIndex]) == m_tokenizer.Id2Punc(PERIOD_INDEX) || m_tokenizer.Id2Punc(Punction[nIndex]) == m_tokenizer.Id2Punc(QUESTION_INDEX))
                {
                    nSentEnd = nIndex;
                    break;
                }
                if (nLastCommaIndex < 0 && m_tokenizer.Id2Punc(Punction[nIndex]) == m_tokenizer.Id2Punc(COMMA_INDEX))
                {
                    nLastCommaIndex = nIndex;
                }
            }
            if (nSentEnd < 0 && InputStr.size() > CACHE_POP_TRIGGER_LIMIT && nLastCommaIndex > 0)
            {
                nSentEnd = nLastCommaIndex;
                Punction[nSentEnd] = PERIOD_INDEX;
            }
            RemainStr.assign(InputStr.begin() + (nSentEnd + 1), InputStr.end());
            RemainIDs.assign(InputIDs.begin() + (nSentEnd + 1), InputIDs.end());
            InputStr.assign(InputStr.begin(), InputStr.begin() + (nSentEnd + 1));  // minit_sentence
            Punction.assign(Punction.begin(), Punction.begin() + (nSentEnd + 1));
        }
        
        for (auto& item : Punction)  
        {
            sentence_punc_list.push_back(m_tokenizer.Id2Punc(item));
        }

        sentence_words_list.insert(sentence_words_list.end(), InputStr.begin(), InputStr.end());

        new_mini_sentence_punc.insert(new_mini_sentence_punc.end(), Punction.begin(), Punction.end());
    }    
    vector<string> WordWithPunc;
    for (int i = 0; i < sentence_words_list.size(); i++) // for i in range(0, len(sentence_words_list)):
    {
        if (!(sentence_words_list[i][0] & 0x80) && (i + 1) < sentence_words_list.size() && !(sentence_words_list[i + 1][0] & 0x80))
        {
            sentence_words_list[i] = " " + sentence_words_list[i];
        }
        if (nSkipNum < arr_cache.size())  //    if skip_num < len(cache):
            nSkipNum++;
        else
            WordWithPunc.push_back(sentence_words_list[i]);

        if (nSkipNum >= arr_cache.size())
        {
            sentence_punc_list_out.push_back(sentence_punc_list[i]);
            if (sentence_punc_list[i] != NOTPUNC)
            {
                WordWithPunc.push_back(sentence_punc_list[i]);
            }
        }
    }

    sentenceOut.insert(sentenceOut.end(), WordWithPunc.begin(), WordWithPunc.end()); //
    nSentEnd = -1;
    for (int i = sentence_punc_list.size() - 2; i > 0; i--)
    {
        if (new_mini_sentence_punc[i] == PERIOD_INDEX || new_mini_sentence_punc[i] == QUESTION_INDEX)
        {
            nSentEnd = i;
            break;
        }
    }
    arr_cache.assign(sentence_words_list.begin() + (nSentEnd + 1), sentence_words_list.end());

    if (sentenceOut.size() > 0 && m_tokenizer.IsPunc(sentenceOut[sentenceOut.size() - 1]))
    {
        sentenceOut.assign(sentenceOut.begin(), sentenceOut.end() - 1);
        sentence_punc_list_out[sentence_punc_list_out.size() - 1] = m_tokenizer.Id2Punc(NOTPUNC_INDEX);
    }
    return accumulate(sentenceOut.begin(), sentenceOut.end(), string(""));
}

vector<int> CTTransformerOnline::Infer(vector<int32_t> input_data, int nCacheSize)
{
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    vector<int> punction;
    std::array<int64_t, 2> input_shape_{ 1, (int64_t)input_data.size()};
    Ort::Value onnx_input = Ort::Value::CreateTensor(
        m_memoryInfo,
        input_data.data(),
        input_data.size() * sizeof(int32_t),
        input_shape_.data(),
        input_shape_.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    std::array<int32_t,1> text_lengths{ (int32_t)input_data.size() };
    std::array<int64_t,1> text_lengths_dim{ 1 };
    Ort::Value onnx_text_lengths = Ort::Value::CreateTensor<int32_t>(
        m_memoryInfo,
        text_lengths.data(),
        text_lengths.size(),
        text_lengths_dim.data(),
        text_lengths_dim.size()); //, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);

    //vad_mask
    // vector<float> arVadMask,arSubMask;
    vector<float> arVadMask;
    int nTextLength = input_data.size();

    VadMask(nTextLength, nCacheSize, arVadMask);
    // Triangle(nTextLength, arSubMask);
    std::array<int64_t, 4> VadMask_Dim{ 1,1, nTextLength ,nTextLength };
    Ort::Value onnx_vad_mask = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        arVadMask.data(),
        arVadMask.size(), // * sizeof(float),
        VadMask_Dim.data(),
        VadMask_Dim.size()); // , ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
    //sub_masks
    
    std::array<int64_t, 4> SubMask_Dim{ 1,1, nTextLength ,nTextLength };
    Ort::Value onnx_sub_mask = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
        arVadMask.data(),
        arVadMask.size(),
        SubMask_Dim.data(),
        SubMask_Dim.size()); // , ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_input));
    input_onnx.emplace_back(std::move(onnx_text_lengths));
    input_onnx.emplace_back(std::move(onnx_vad_mask));
    input_onnx.emplace_back(std::move(onnx_sub_mask));
        
    try {
        auto outputTensor = m_session->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), m_szInputNames.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float * floatData = outputTensor[0].GetTensorMutableData<float>();

        for (int i = 0; i < outputCount; i += CANDIDATE_NUM)
        {
            int index = Argmax(floatData + i, floatData + i + CANDIDATE_NUM-1);
            punction.push_back(index);
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR) << "Error when run punc onnx forword: " << (e.what());
    }
    return punction;
}

void CTTransformerOnline::VadMask(int nSize, int vad_pos, vector<float>& Result)
{
    Result.resize(0);
    Result.assign(nSize * nSize, 1);
    if (vad_pos <= 0 || vad_pos >= nSize)
    {
        return;
    }
    for (int i = 0; i < vad_pos-1; i++)
    {
        for (int j = vad_pos; j < nSize; j++)
        {
            Result[i * nSize + j] = 0.0f;
        }
    }
}

void CTTransformerOnline::Triangle(int text_length, vector<float>& Result)
{
    Result.resize(0);
    Result.assign(text_length * text_length,1); // generate a zeros: text_length x text_length

    for (int i = 0; i < text_length; i++) // rows
    {
        for (int j = i+1; j<text_length; j++) //cols
        {
            Result[i * text_length + j] = 0.0f;
        }

    }
    //Transport(Result, text_length, text_length);
}

void CTTransformerOnline::Transport(vector<float>& In,int nRows, int nCols)
{
    vector<float> Out;
    Out.resize(nRows * nCols);
    int i = 0;
    for (int j = 0; j < nCols; j++) {
        for (; i < nRows * nCols; i++) {
            Out[i] = In[j + nCols * (i % nRows)];
            if ((i + 1) % nRows == 0) {
                i++;
                break;
            }
        }
    }
    In = Out;
}

} // namespace funasr
