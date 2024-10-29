/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"
#include "sensevoice-small.h"
#include <cstddef>

using namespace std;
namespace funasr {

SenseVoiceSmall::SenseVoiceSmall()
:use_hotword(false),
 env_(ORT_LOGGING_LEVEL_ERROR, "sensevoice"),session_options_{} {
}

// offline
void SenseVoiceSmall::InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){
    LoadConfigFromYaml(am_config.c_str());
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = asr_sample_rate;
    fbank_opts_.frame_opts.window_type = window_type;
    fbank_opts_.frame_opts.frame_shift_ms = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms = frame_length;
    fbank_opts_.energy_floor = 0;
    fbank_opts_.mel_opts.debug_mel = false;

    // session_options_.SetInterOpNumThreads(1);
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options_.DisableCpuMemArena();

    try {
        m_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(am_model).c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(-1);
    }

    GetInputNames(m_session_.get(), m_strInputNames, m_szInputNames);
    GetOutputNames(m_session_.get(), m_strOutputNames, m_szOutputNames);
    vocab = new Vocab(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());
}

// online
void SenseVoiceSmall::InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){
    
    LoadOnlineConfigFromYaml(am_config.c_str());
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = asr_sample_rate;
    fbank_opts_.frame_opts.window_type = window_type;
    fbank_opts_.frame_opts.frame_shift_ms = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms = frame_length;
    fbank_opts_.energy_floor = 0;
    fbank_opts_.mel_opts.debug_mel = false;

    // session_options_.SetInterOpNumThreads(1);
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options_.DisableCpuMemArena();

    try {
        encoder_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(en_model).c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << en_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am encoder model: " << e.what();
        exit(-1);
    }

    try {
        decoder_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(de_model).c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << de_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am decoder model: " << e.what();
        exit(-1);
    }

    // encoder
    string strName;
    GetInputName(encoder_session_.get(), strName);
    en_strInputNames.push_back(strName.c_str());
    GetInputName(encoder_session_.get(), strName,1);
    en_strInputNames.push_back(strName);
    
    GetOutputName(encoder_session_.get(), strName);
    en_strOutputNames.push_back(strName);
    GetOutputName(encoder_session_.get(), strName,1);
    en_strOutputNames.push_back(strName);
    GetOutputName(encoder_session_.get(), strName,2);
    en_strOutputNames.push_back(strName);

    for (auto& item : en_strInputNames)
        en_szInputNames_.push_back(item.c_str());
    for (auto& item : en_strOutputNames)
        en_szOutputNames_.push_back(item.c_str());

    // decoder
    int de_input_len = 4 + fsmn_layers;
    int de_out_len = 2 + fsmn_layers;
    for(int i=0;i<de_input_len; i++){
        GetInputName(decoder_session_.get(), strName, i);
        de_strInputNames.push_back(strName.c_str());
    }

    for(int i=0;i<de_out_len; i++){
        GetOutputName(decoder_session_.get(), strName,i);
        de_strOutputNames.push_back(strName);
    }

    for (auto& item : de_strInputNames)
        de_szInputNames_.push_back(item.c_str());
    for (auto& item : de_strOutputNames)
        de_szOutputNames_.push_back(item.c_str());

    online_vocab = new Vocab(token_file.c_str());
    phone_set_ = new PhoneSet(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());
}

// 2pass
void SenseVoiceSmall::InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, 
    const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, const std::string &online_token_file, int thread_num){
    // online
    InitAsr(en_model, de_model, am_cmvn, am_config, online_token_file, thread_num);

    // offline
    try {
        m_session_ = std::make_unique<Ort::Session>(env_, ORTSTRING(am_model).c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(-1);
    }

    GetInputNames(m_session_.get(), m_strInputNames, m_szInputNames);
    GetOutputNames(m_session_.get(), m_strOutputNames, m_szOutputNames);
    vocab = new Vocab(token_file.c_str());
}

void SenseVoiceSmall::LoadOnlineConfigFromYaml(const char* filename){

    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(ERROR) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

    try{
        YAML::Node frontend_conf = config["frontend_conf"];
        YAML::Node encoder_conf = config["encoder_conf"];
        YAML::Node decoder_conf = config["decoder_conf"];
        YAML::Node predictor_conf = config["predictor_conf"];

        this->window_type = frontend_conf["window"].as<string>();
        this->n_mels = frontend_conf["n_mels"].as<int>();
        this->frame_length = frontend_conf["frame_length"].as<int>();
        this->frame_shift = frontend_conf["frame_shift"].as<int>();
        this->lfr_m = frontend_conf["lfr_m"].as<int>();
        this->lfr_n = frontend_conf["lfr_n"].as<int>();

        this->encoder_size = encoder_conf["output_size"].as<int>();
        this->fsmn_dims = encoder_conf["output_size"].as<int>();

        this->fsmn_layers = decoder_conf["num_blocks"].as<int>();
        this->fsmn_lorder = decoder_conf["kernel_size"].as<int>()-1;

        this->cif_threshold = predictor_conf["threshold"].as<double>();
        this->tail_alphas = predictor_conf["tail_threshold"].as<double>();

        this->asr_sample_rate = frontend_conf["fs"].as<int>();


    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

void SenseVoiceSmall::LoadConfigFromYaml(const char* filename){

    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(ERROR) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

    try{
        YAML::Node frontend_conf = config["frontend_conf"];
        YAML::Node encoder_conf = config["encoder_conf"];

        this->window_type = frontend_conf["window"].as<string>();
        this->n_mels = frontend_conf["n_mels"].as<int>();
        this->frame_length = frontend_conf["frame_length"].as<int>();
        this->frame_shift = frontend_conf["frame_shift"].as<int>();
        this->lfr_m = frontend_conf["lfr_m"].as<int>();
        this->lfr_n = frontend_conf["lfr_n"].as<int>();

        this->encoder_size = encoder_conf["output_size"].as<int>();
        this->fsmn_dims = encoder_conf["output_size"].as<int>();

        this->asr_sample_rate = frontend_conf["fs"].as<int>();
    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

SenseVoiceSmall::~SenseVoiceSmall()
{
    if(vocab){
        delete vocab;
    }
    if(online_vocab){
        delete online_vocab;
    }
    if(lm_vocab){
        delete lm_vocab;
    }
    if(seg_dict){
        delete seg_dict;
    }
    if(phone_set_){
        delete phone_set_;
    }
}

void SenseVoiceSmall::StartUtterance()
{
}

void SenseVoiceSmall::EndUtterance()
{
}

void SenseVoiceSmall::Reset()
{
}

void SenseVoiceSmall::FbankKaldi(float sample_rate, const float* waves, int len, std::vector<std::vector<float>> &asr_feats) {
    knf::OnlineFbank fbank_(fbank_opts_);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());

    int32_t frames = fbank_.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float *frame = fbank_.GetFrame(i);
        std::vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
        asr_feats.emplace_back(frame_vector);
    }
}

void SenseVoiceSmall::LoadCmvn(const char *filename)
{
    ifstream cmvn_stream(filename);
    if (!cmvn_stream.is_open()) {
        LOG(ERROR) << "Failed to open file: " << filename;
        exit(-1);
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
                    means_list_.push_back(stof(means_lines[j]));
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
                    vars_list_.push_back(stof(vars_lines[j])*scale);
                }
                continue;
            }
        }
    }
}

string SenseVoiceSmall::CTCSearch(float * in, std::vector<int32_t> paraformer_length, std::vector<int64_t> outputShape)
{
    std::string unicodeChar = "▁";
    int32_t vocab_size = outputShape[2];

    std::vector<int64_t> tokens;
    std::string text="";
    int32_t prev_id = -1;
    for (int32_t t = 0; t != paraformer_length[0]; ++t) {
        auto y = std::distance(
            static_cast<const float *>(in),
            std::max_element(
                static_cast<const float *>(in),
                static_cast<const float *>(in) + vocab_size));
        in += vocab_size;

        if (y != blank_id && y != prev_id) {
            tokens.push_back(y);
        }
        prev_id = y;
    }
    string str_lang = "";
    string str_emo = "";
    string str_event = "";
    string str_itn = "";
    if(tokens.size() >=3){
        str_lang  = vocab->Id2String(tokens[0]);
        str_emo   = vocab->Id2String(tokens[1]);
        str_event = vocab->Id2String(tokens[2]);
        str_itn = vocab->Id2String(tokens[3]);
    }

    for(int32_t i = 4; i < tokens.size(); ++i){
        string word = vocab->Id2String(tokens[i]);
        size_t found = word.find(unicodeChar);
        if(found != std::string::npos){
            text += " " + word.substr(3);
        }else{
            text += word;
        }
    }
    if(str_itn == "<|withitn|>"){
        if(str_lang == "<|zh|>"){
            text += "。";
        }else{
            text += ".";
        }
    }

    return str_lang + str_emo + str_event + " " + text;
}

string SenseVoiceSmall::GreedySearch(float * in, int n_len,  int64_t token_nums, bool is_stamp, std::vector<float> us_alphas, std::vector<float> us_cif_peak)
{
    vector<int> hyps;
    int Tmax = n_len;
    for (int i = 0; i < Tmax; i++) {
        int max_idx;
        float max_val;
        FindMax(in + i * token_nums, token_nums, max_val, max_idx);
        hyps.push_back(max_idx);
    }
    if(!is_stamp){
        return online_vocab->Vector2StringV2(hyps, language);
    }else{
        std::vector<string> char_list;
        std::vector<std::vector<float>> timestamp_list;
        std::string res_str;
        online_vocab->Vector2String(hyps, char_list);
        std::vector<string> raw_char(char_list);
        TimestampOnnx(us_alphas, us_cif_peak, char_list, res_str, timestamp_list);

        return PostProcess(raw_char, timestamp_list);
    }
}

void SenseVoiceSmall::LfrCmvn(std::vector<std::vector<float>> &asr_feats) {

    std::vector<std::vector<float>> out_feats;
    int T = asr_feats.size();
    int T_lrf = ceil(1.0 * T / lfr_n);

    // Pad frames at start(copy first frame)
    for (int i = 0; i < (lfr_m - 1) / 2; i++) {
        asr_feats.insert(asr_feats.begin(), asr_feats[0]);
    }
    // Merge lfr_m frames as one,lfr_n frames per window
    T = T + (lfr_m - 1) / 2;
    std::vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        if (lfr_m <= T - i * lfr_n) {
            for (int j = 0; j < lfr_m; j++) {
                p.insert(p.end(), asr_feats[i * lfr_n + j].begin(), asr_feats[i * lfr_n + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            // Fill to lfr_m frames at last window if less than lfr_m frames  (copy last frame)
            int num_padding = lfr_m - (T - i * lfr_n);
            for (int j = 0; j < (asr_feats.size() - i * lfr_n); j++) {
                p.insert(p.end(), asr_feats[i * lfr_n + j].begin(), asr_feats[i * lfr_n + j].end());
            }
            for (int j = 0; j < num_padding; j++) {
                p.insert(p.end(), asr_feats[asr_feats.size() - 1].begin(), asr_feats[asr_feats.size() - 1].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        }
    }
    // Apply cmvn
    for (auto &out_feat: out_feats) {
        for (int j = 0; j < means_list_.size(); j++) {
            out_feat[j] = (out_feat[j] + means_list_[j]) * vars_list_[j];
        }
    }
    asr_feats = out_feats;
}

std::vector<std::vector<float>> SenseVoiceSmall::CompileHotwordEmbedding(std::string &hotwords) {
    int embedding_dim = encoder_size;
    std::vector<std::vector<float>> hw_emb;
    std::vector<float> vec(embedding_dim, 0);
    hw_emb.push_back(vec);
    return hw_emb;
}

std::vector<std::string> SenseVoiceSmall::Forward(float** din, int* len, bool input_finished, std::string svs_lang, bool svs_itn, int batch_in)
{
    std::vector<std::string> results;
    string result="";
    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;

    if(batch_in != 1){
        results.push_back(result);
        return results;
    }

    std::vector<std::vector<float>> asr_feats;
    FbankKaldi(asr_sample_rate, din[0], len[0], asr_feats);
    if(asr_feats.size() == 0){
        results.push_back(result);
        return results;
    }
    LfrCmvn(asr_feats);
    int32_t feat_dim = lfr_m*in_feat_dim;
    int32_t num_frames = asr_feats.size();

    std::vector<float> wav_feats;
    for (const auto &frame_feat: asr_feats) {
        wav_feats.insert(wav_feats.end(), frame_feat.begin(), frame_feat.end());
    }

    //lid textnorm
    int svs_lid = 0;
    int svs_itnid = 15;
    if(lid_map.find(svs_lang) != lid_map.end()){
        svs_lid = lid_map[svs_lang];
    }
    if(svs_itn){
        svs_itnid = 14;
    }

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

    const int64_t lid_shape[1] = {1};
    std::vector<int32_t> lid_length;
    lid_length.emplace_back(svs_lid);
    Ort::Value onnx_lid = Ort::Value::CreateTensor<int32_t>(
          m_memoryInfo, lid_length.data(), lid_length.size(), lid_shape, 1);

    const int64_t textnorm_shape[1] = {1};
    std::vector<int32_t> textnorm_length;
    textnorm_length.emplace_back(svs_itnid);
    Ort::Value onnx_itn = Ort::Value::CreateTensor<int32_t>(
          m_memoryInfo, textnorm_length.data(), textnorm_length.size(), textnorm_shape, 1);

    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_feats));
    input_onnx.emplace_back(std::move(onnx_feats_len));
    input_onnx.emplace_back(std::move(onnx_lid));
    input_onnx.emplace_back(std::move(onnx_itn));

    try {
        auto outputTensor = m_session_->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), input_onnx.size(), m_szOutputNames.data(), m_szOutputNames.size());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        result = CTCSearch(floatData, paraformer_length, outputShape);
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }

    results.push_back(result);
    return results;
}

string SenseVoiceSmall::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
