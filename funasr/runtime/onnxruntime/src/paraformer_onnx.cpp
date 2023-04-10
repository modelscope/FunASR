#include "precomp.h"

using namespace std;
using namespace paraformer;

ModelImp::ModelImp(const char* path,int nNumThread, bool quantize)
{
    string model_path;
    string cmvn_path;
    string config_path;

    if(quantize)
    {
        model_path = pathAppend(path, "model_quant.onnx");
    }else{
        model_path = pathAppend(path, "model.onnx");
    }
    cmvn_path = pathAppend(path, "am.mvn");
    config_path = pathAppend(path, "config.yaml");

    fe = new FeatureExtract(3);

    //sessionOptions.SetInterOpNumThreads(1);
    sessionOptions.SetIntraOpNumThreads(nNumThread);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
    wstring wstrPath = strToWstr(model_path);
    m_session = new Ort::Session(env, wstrPath.c_str(), sessionOptions);
#else
    m_session = new Ort::Session(env, model_path.c_str(), sessionOptions);
#endif

    string strName;
    getInputName(m_session, strName);
    m_strInputNames.push_back(strName.c_str());
    getInputName(m_session, strName,1);
    m_strInputNames.push_back(strName);
    
    getOutputName(m_session, strName);
    m_strOutputNames.push_back(strName);
    getOutputName(m_session, strName,1);
    m_strOutputNames.push_back(strName);

    for (auto& item : m_strInputNames)
        m_szInputNames.push_back(item.c_str());
    for (auto& item : m_strOutputNames)
        m_szOutputNames.push_back(item.c_str());
    vocab = new Vocab(config_path.c_str());
    load_cmvn(cmvn_path.c_str());
}

ModelImp::~ModelImp()
{
    if(fe)
        delete fe;
    if (m_session)
    {
        delete m_session;
        m_session = nullptr;
    }
    if(vocab)
        delete vocab;
}

void ModelImp::reset()
{
    fe->reset();
}

void ModelImp::apply_lfr(Tensor<float>*& din)
{
    int mm = din->size[2];
    int ll = ceil(mm / 6.0);
    Tensor<float>* tmp = new Tensor<float>(ll, 560);
    int out_offset = 0;
    for (int i = 0; i < ll; i++) {
        for (int j = 0; j < 7; j++) {
            int idx = i * 6 + j - 3;
            if (idx < 0) {
                idx = 0;
            }
            if (idx >= mm) {
                idx = mm - 1;
            }
            memcpy(tmp->buff + out_offset, din->buff + idx * 80,
                sizeof(float) * 80);
            out_offset += 80;
        }
    }
    delete din;
    din = tmp;
}

void ModelImp::load_cmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    string line;

    while (getline(cmvn_stream, line)) {
        istringstream iss(line);
        vector<string> line_item{istream_iterator<string>{iss}, istream_iterator<string>{}};
        if (line_item[0] == "<AddShift>") {
            getline(cmvn_stream, line);
            istringstream means_lines_stream(line);
            vector<string> means_lines{istream_iterator<string>{means_lines_stream}, istream_iterator<string>{}};
            if (means_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < means_lines.size() - 1; j++) {
                    means_list.push_back(stof(means_lines[j]));
                }
                continue;
            }
        }
        else if (line_item[0] == "<Rescale>") {
            getline(cmvn_stream, line);
            istringstream vars_lines_stream(line);
            vector<string> vars_lines{istream_iterator<string>{vars_lines_stream}, istream_iterator<string>{}};
            if (vars_lines[0] == "<LearnRateCoef>") {
                for (int j = 3; j < vars_lines.size() - 1; j++) {
                    vars_list.push_back(stof(vars_lines[j])*scale);
                }
                continue;
            }
        }
    }
}

void ModelImp::apply_cmvn(Tensor<float>* din)
{
    const float* var;
    const float* mean;
    var = vars_list.data();
    mean= means_list.data();

    int m = din->size[2];
    int n = din->size[3];

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i * n + j;
            din->buff[idx] = (din->buff[idx] + mean[j]) * var[j];
        }
    }
}

string ModelImp::greedy_search(float * in, int nLen )
{
    vector<int> hyps;
    int Tmax = nLen;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        findmax(in + i * 8404, 8404, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab->vector2stringV2(hyps);
}

string ModelImp::forward(float* din, int len, int flag)
{

    Tensor<float>* in;
    fe->insert(din, len, flag);
    fe->fetch(in);
    apply_lfr(in);
    apply_cmvn(in);
    Ort::RunOptions run_option;

    std::array<int64_t, 3> input_shape_{ in->size[0],in->size[2],in->size[3] };
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(m_memoryInfo,
        in->buff,
        in->buff_size,
        input_shape_.data(),
        input_shape_.size());

    std::vector<int32_t> feats_len{ in->size[2] };
    std::vector<int64_t> feats_len_dim{ 1 };
    Ort::Value onnx_feats_len = Ort::Value::CreateTensor(
        m_memoryInfo,
        feats_len.data(),
        feats_len.size() * sizeof(int32_t),
        feats_len_dim.data(),
        feats_len_dim.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32);
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_feats_len));

    string result;
    try {

        auto outputTensor = m_session->Run(run_option, m_szInputNames.data(), input_onnx.data(), m_szInputNames.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();


        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        auto encoder_out_lens = outputTensor[1].GetTensorMutableData<int64_t>();
        result = greedy_search(floatData, *encoder_out_lens);
    }
    catch (...)
    {
        result = "";
    }


    if(in)
        delete in;

    return result;
}

string ModelImp::forward_chunk(float* din, int len, int flag)
{

    printf("Not Imp!!!!!!\n");
    return "Hello";
}

string ModelImp::rescoring()
{
    printf("Not Imp!!!!!!\n");
    return "Hello";
}
