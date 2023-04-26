/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"

using namespace std;
using namespace paraformer;

Paraformer::Paraformer(std::map<std::string, std::string>& model_path,int thread_num)
:env_(ORT_LOGGING_LEVEL_ERROR, "paraformer"),session_options{}{

    // VAD model
    if(model_path.find(VAD_MODEL_PATH) != model_path.end()){
        use_vad = true;
        string vad_model_path;
        string vad_cmvn_path;
        string vad_config_path;
    
        try{
            vad_model_path = model_path.at(VAD_MODEL_PATH);
            vad_cmvn_path = model_path.at(VAD_CMVN_PATH);
            vad_config_path = model_path.at(VAD_CONFIG_PATH);
        }catch(const out_of_range& e){
            LOG(ERROR) << "Error when read "<< VAD_CMVN_PATH << " or " << VAD_CONFIG_PATH <<" :" << e.what();
            exit(0);
        }
        vad_handle = make_unique<FsmnVad>();
        vad_handle->InitVad(vad_model_path, vad_cmvn_path, vad_config_path);
    }

    // AM model
    if(model_path.find(AM_MODEL_PATH) != model_path.end()){
        string am_model_path;
        string am_cmvn_path;
        string am_config_path;
    
        try{
            am_model_path = model_path.at(AM_MODEL_PATH);
            am_cmvn_path = model_path.at(AM_CMVN_PATH);
            am_config_path = model_path.at(AM_CONFIG_PATH);
        }catch(const out_of_range& e){
            LOG(ERROR) << "Error when read "<< AM_CONFIG_PATH << " or " << AM_CMVN_PATH <<" :" << e.what();
            exit(0);
        }
        InitAM(am_model_path, am_cmvn_path, am_config_path, thread_num);
    }

    // PUNC model
    if(model_path.find(PUNC_MODEL_PATH) != model_path.end()){
        use_punc = true;
        string punc_model_path;
        string punc_config_path;
    
        try{
            punc_model_path = model_path.at(PUNC_MODEL_PATH);
            punc_config_path = model_path.at(PUNC_CONFIG_PATH);
        }catch(const out_of_range& e){
            LOG(ERROR) << "Error when read "<< PUNC_CONFIG_PATH <<" :" << e.what();
            exit(0);
        }

        punc_handle = make_unique<CTTransformer>();
        punc_handle->InitPunc(punc_model_path, punc_config_path, thread_num);
    }
}

void Paraformer::InitAM(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    // knf options
    fbank_opts.frame_opts.dither = 0;
    fbank_opts.mel_opts.num_bins = 80;
    fbank_opts.frame_opts.samp_freq = MODEL_SAMPLE_RATE;
    fbank_opts.frame_opts.window_type = "hamming";
    fbank_opts.frame_opts.frame_shift_ms = 10;
    fbank_opts.frame_opts.frame_length_ms = 25;
    fbank_opts.energy_floor = 0;
    fbank_opts.mel_opts.debug_mel = false;
    // fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts);

    // session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(thread_num);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options.DisableCpuMemArena();

    try {
        m_session = std::make_unique<Ort::Session>(env_, am_model.c_str(), session_options);
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(0);
    }

    string strName;
    GetInputName(m_session.get(), strName);
    m_strInputNames.push_back(strName.c_str());
    GetInputName(m_session.get(), strName,1);
    m_strInputNames.push_back(strName);
    
    GetOutputName(m_session.get(), strName);
    m_strOutputNames.push_back(strName);
    GetOutputName(m_session.get(), strName,1);
    m_strOutputNames.push_back(strName);

    for (auto& item : m_strInputNames)
        m_szInputNames.push_back(item.c_str());
    for (auto& item : m_strOutputNames)
        m_szOutputNames.push_back(item.c_str());
    vocab = new Vocab(am_config.c_str());
    LoadCmvn(am_cmvn.c_str());
}

Paraformer::~Paraformer()
{
    if(vocab)
        delete vocab;
}

void Paraformer::Reset()
{
}

vector<std::vector<int>> Paraformer::VadSeg(std::vector<float>& pcm_data){
    return vad_handle->Infer(pcm_data);
}

string Paraformer::AddPunc(const char* sz_input){
    return punc_handle->AddPunc(sz_input);
}

vector<float> Paraformer::FbankKaldi(float sample_rate, const float* waves, int len) {
    knf::OnlineFbank fbank_(fbank_opts);
    fbank_.AcceptWaveform(sample_rate, waves, len);
    //fbank_->InputFinished();
    int32_t frames = fbank_.NumFramesReady();
    int32_t feature_dim = fbank_opts.mel_opts.num_bins;
    vector<float> features(frames * feature_dim);
    float *p = features.data();

    for (int32_t i = 0; i != frames; ++i) {
        const float *f = fbank_.GetFrame(i);
        std::copy(f, f + feature_dim, p);
        p += feature_dim;
    }

    return features;
}

void Paraformer::LoadCmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filename;
        exit(0);
    }
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

string Paraformer::GreedySearch(float * in, int n_len,  int64_t token_nums)
{
    vector<int> hyps;
    int Tmax = n_len;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        FindMax(in + i * token_nums, token_nums, max_val, max_idx);
        hyps.push_back(max_idx);
    }

    return vocab->Vector2StringV2(hyps);
}

vector<float> Paraformer::ApplyLfr(const std::vector<float> &in) 
{
    int32_t in_feat_dim = fbank_opts.mel_opts.num_bins;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_window_size) / lfr_window_shift + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_window_size;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_window_shift * in_feat_dim;
    }

    return out;
  }

  void Paraformer::ApplyCmvn(std::vector<float> *v)
  {
    int32_t dim = means_list.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + means_list[k]) * vars_list[k];
      }

      p += dim;
    }
  }

string Paraformer::Forward(float* din, int len, int flag)
{

    int32_t in_feat_dim = fbank_opts.mel_opts.num_bins;
    std::vector<float> wav_feats = FbankKaldi(MODEL_SAMPLE_RATE, din, len);
    wav_feats = ApplyLfr(wav_feats);
    ApplyCmvn(&wav_feats);

    int32_t feat_dim = lfr_window_size*in_feat_dim;
    int32_t num_frames = wav_feats.size() / feat_dim;

#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

    const int64_t input_shape_[3] = {1, num_frames, feat_dim};
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(m_memoryInfo,
        wav_feats.data(),
        wav_feats.size(),
        input_shape_,
        3);

    const int64_t paraformer_length_shape[1] = {1};
    std::vector<int32_t> paraformer_length;
    paraformer_length.emplace_back(num_frames);
    Ort::Value onnx_feats_len = Ort::Value::CreateTensor<int32_t>(
          m_memoryInfo, paraformer_length.data(), paraformer_length.size(), paraformer_length_shape, 1);
    
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_feats_len));

    string result;
    try {
        auto outputTensor = m_session->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), input_onnx.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        auto encoder_out_lens = outputTensor[1].GetTensorMutableData<int64_t>();
        result = GreedySearch(floatData, *encoder_out_lens, outputShape[2]);
    }
    catch (std::exception const &e)
    {
        printf(e.what());
    }

    return result;
}

string Paraformer::ForwardChunk(float* din, int len, int flag)
{

    printf("Not Imp!!!!!!\n");
    return "Hello";
}

string Paraformer::Rescoring()
{
    printf("Not Imp!!!!!!\n");
    return "Hello";
}
