/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"
#include "paraformer.h"
#include "encode_converter.h"
#include <cstddef>

using namespace std;
namespace funasr {

Paraformer::Paraformer()
:use_hotword(false),
 env_(ORT_LOGGING_LEVEL_ERROR, "paraformer"),session_options_{},
 hw_env_(ORT_LOGGING_LEVEL_ERROR, "paraformer_hw"),hw_session_options{} {
}

// offline
void Paraformer::InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = MODEL_SAMPLE_RATE;
    fbank_opts_.frame_opts.window_type = window_type;
    fbank_opts_.frame_opts.frame_shift_ms = frame_shift;
    fbank_opts_.frame_opts.frame_length_ms = frame_length;
    fbank_opts_.energy_floor = 0;
    fbank_opts_.mel_opts.debug_mel = false;
    // fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts);

    // session_options_.SetInterOpNumThreads(1);
    session_options_.SetIntraOpNumThreads(thread_num);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options_.DisableCpuMemArena();

    try {
        m_session_ = std::make_unique<Ort::Session>(env_, am_model.c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(-1);
    }

    string strName;
    GetInputName(m_session_.get(), strName);
    m_strInputNames.push_back(strName.c_str());
    GetInputName(m_session_.get(), strName,1);
    m_strInputNames.push_back(strName);
    if (use_hotword) {
        GetInputName(m_session_.get(), strName, 2);
        m_strInputNames.push_back(strName);
    }
    
    size_t numOutputNodes = m_session_->GetOutputCount();
    for(int index=0; index<numOutputNodes; index++){
        GetOutputName(m_session_.get(), strName, index);
        m_strOutputNames.push_back(strName);
    }

    for (auto& item : m_strInputNames)
        m_szInputNames.push_back(item.c_str());
    for (auto& item : m_strOutputNames)
        m_szOutputNames.push_back(item.c_str());
    vocab = new Vocab(am_config.c_str());
    LoadCmvn(am_cmvn.c_str());
}

// online
void Paraformer::InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    
    LoadOnlineConfigFromYaml(am_config.c_str());
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = MODEL_SAMPLE_RATE;
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
        encoder_session_ = std::make_unique<Ort::Session>(env_, en_model.c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << en_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am encoder model: " << e.what();
        exit(-1);
    }

    try {
        decoder_session_ = std::make_unique<Ort::Session>(env_, de_model.c_str(), session_options_);
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

    vocab = new Vocab(am_config.c_str());
    LoadCmvn(am_cmvn.c_str());
}

// 2pass
void Paraformer::InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    // online
    InitAsr(en_model, de_model, am_cmvn, am_config, thread_num);

    // offline
    try {
        m_session_ = std::make_unique<Ort::Session>(env_, am_model.c_str(), session_options_);
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am onnx model: " << e.what();
        exit(-1);
    }

    string strName;
    GetInputName(m_session_.get(), strName);
    m_strInputNames.push_back(strName.c_str());
    GetInputName(m_session_.get(), strName,1);
    m_strInputNames.push_back(strName);
    
    GetOutputName(m_session_.get(), strName);
    m_strOutputNames.push_back(strName);
    GetOutputName(m_session_.get(), strName,1);
    m_strOutputNames.push_back(strName);

    for (auto& item : m_strInputNames)
        m_szInputNames.push_back(item.c_str());
    for (auto& item : m_strOutputNames)
        m_szOutputNames.push_back(item.c_str());
}

void Paraformer::LoadOnlineConfigFromYaml(const char* filename){

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

    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

void Paraformer::InitHwCompiler(const std::string &hw_model, int thread_num) {
    hw_session_options.SetIntraOpNumThreads(thread_num);
    hw_session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    hw_session_options.DisableCpuMemArena();

    try {
        hw_m_session = std::make_unique<Ort::Session>(hw_env_, hw_model.c_str(), hw_session_options);
        LOG(INFO) << "Successfully load model from " << hw_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load hw compiler onnx model: " << e.what();
        exit(-1);
    }

    string strName;
    GetInputName(hw_m_session.get(), strName);
    hw_m_strInputNames.push_back(strName.c_str());
    //GetInputName(hw_m_session.get(), strName,1);
    //hw_m_strInputNames.push_back(strName);
    
    GetOutputName(hw_m_session.get(), strName);
    hw_m_strOutputNames.push_back(strName);

    for (auto& item : hw_m_strInputNames)
        hw_m_szInputNames.push_back(item.c_str());
    for (auto& item : hw_m_strOutputNames)
        hw_m_szOutputNames.push_back(item.c_str());
    // if init hotword compiler is called, this is a hotword paraformer model
    use_hotword = true;
}

void Paraformer::InitSegDict(const std::string &seg_dict_model) {
    seg_dict = new SegDict(seg_dict_model.c_str());
}

Paraformer::~Paraformer()
{
    if(vocab)
        delete vocab;
    if(seg_dict)
        delete seg_dict;
}

void Paraformer::Reset()
{
}

vector<float> Paraformer::FbankKaldi(float sample_rate, const float* waves, int len) {
    knf::OnlineFbank fbank_(fbank_opts_);
    std::vector<float> buf(len);
    for (int32_t i = 0; i != len; ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank_.AcceptWaveform(sample_rate, buf.data(), buf.size());
    //fbank_->InputFinished();
    int32_t frames = fbank_.NumFramesReady();
    int32_t feature_dim = fbank_opts_.mel_opts.num_bins;
    vector<float> features(frames * feature_dim);
    float *p = features.data();
    //std::cout << "samples " << len << std::endl;
    //std::cout << "fbank frames " << frames << std::endl;
    //std::cout << "fbank dim " << feature_dim << std::endl;
    //std::cout << "feature size " << features.size() << std::endl;

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

string Paraformer::GreedySearch(float * in, int n_len,  int64_t token_nums, bool is_stamp, std::vector<float> us_alphas, std::vector<float> us_cif_peak)
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
        return vocab->Vector2StringV2(hyps);
    }else{
        std::vector<string> char_list;
        std::vector<std::vector<float>> timestamp_list;
        std::string res_str;
        vocab->Vector2String(hyps, char_list);
        std::vector<string> raw_char(char_list);
        TimestampOnnx(us_alphas, us_cif_peak, char_list, res_str, timestamp_list);

        return PostProcess(raw_char, timestamp_list);
    }
}

string Paraformer::PostProcess(std::vector<string> &raw_char, std::vector<std::vector<float>> &timestamp_list){
    std::vector<std::vector<float>> timestamp_merge;
    int i;
    list<string> words;
    int is_pre_english = false;
    int pre_english_len = 0;
    int is_combining = false;
    string combine = "";

    float begin=-1;
    for (i=0; i<raw_char.size(); i++){
        string word = raw_char[i];
        // step1 space character skips
        if (word == "<s>" || word == "</s>" || word == "<unk>")
            continue;
        // step2 combie phoneme to full word
        {
            int sub_word = !(word.find("@@") == string::npos);
            // process word start and middle part
            if (sub_word) {
                combine += word.erase(word.length() - 2);
                if(!is_combining){
                    begin = timestamp_list[i][0];
                }
                is_combining = true;
                continue;
            }
            // process word end part
            else if (is_combining) {
                combine += word;
                is_combining = false;
                word = combine;
                combine = "";
            }
        }

        // step3 process english word deal with space , turn abbreviation to upper case
        {
            // input word is chinese, not need process 
            if (vocab->IsChinese(word)) {
                words.push_back(word);
                timestamp_merge.emplace_back(timestamp_list[i]);
                is_pre_english = false;
            }
            // input word is english word
            else {
                // pre word is chinese
                if (!is_pre_english) {
                    // word[0] = word[0] - 32;
                    words.push_back(word);
                    begin = (begin==-1)?timestamp_list[i][0]:begin;
                    std::vector<float> vec = {begin, timestamp_list[i][1]};
                    timestamp_merge.emplace_back(vec);
                    begin = -1;
                    pre_english_len = word.size();
                }
                // pre word is english word
                else {
                    // single letter turn to upper case
                    // if (word.size() == 1) {
                    //     word[0] = word[0] - 32;
                    // }

                    if (pre_english_len > 1) {
                        words.push_back(" ");
                        words.push_back(word);
                        begin = (begin==-1)?timestamp_list[i][0]:begin;
                        std::vector<float> vec = {begin, timestamp_list[i][1]};
                        timestamp_merge.emplace_back(vec);
                        begin = -1;
                        pre_english_len = word.size();
                    }
                    else {
                        // if (word.size() > 1) {
                        //     words.push_back(" ");
                        // }
                        words.push_back(" ");
                        words.push_back(word);
                        begin = (begin==-1)?timestamp_list[i][0]:begin;
                        std::vector<float> vec = {begin, timestamp_list[i][1]};
                        timestamp_merge.emplace_back(vec);
                        begin = -1;
                        pre_english_len = word.size();
                    }
                }
                is_pre_english = true;
            }
        }
    }
    string stamp_str="";
    for (i=0; i<timestamp_merge.size(); i++) {
        stamp_str += std::to_string(timestamp_merge[i][0]);
        stamp_str += ", ";
        stamp_str += std::to_string(timestamp_merge[i][1]);
        if(i!=timestamp_merge.size()-1){
            stamp_str += ",";
        }
    }

    stringstream ss;
    for (auto it = words.begin(); it != words.end(); it++) {
        ss << *it;
    }

    return ss.str()+" | "+stamp_str;
}

void Paraformer::TimestampOnnx(std::vector<float>& us_alphas,
                                std::vector<float> us_cif_peak, 
                                std::vector<string>& char_list, 
                                std::string &res_str, 
                                std::vector<std::vector<float>> &timestamp_vec, 
                                float begin_time, 
                                float total_offset){
    if (char_list.empty()) {
        return ;
    }

    const float START_END_THRESHOLD = 5.0;
    const float MAX_TOKEN_DURATION = 30.0;
    const float TIME_RATE = 10.0 * 6 / 1000 / 3;
    // 3 times upsampled, cif_peak is flattened into a 1D array
    std::vector<float> cif_peak = us_cif_peak;
    int num_frames = cif_peak.size();
    if (char_list.back() == "</s>") {
        char_list.pop_back();
    }
    if (char_list.empty()) {
        return ;
    }
    vector<vector<float>> timestamp_list;
    vector<string> new_char_list;
    vector<float> fire_place;
    // for bicif model trained with large data, cif2 actually fires when a character starts
    // so treat the frames between two peaks as the duration of the former token
    for (int i = 0; i < num_frames; i++) {
        if (cif_peak[i] > 1.0 - 1e-4) {
            fire_place.push_back(i + total_offset);
        }
    }
    int num_peak = fire_place.size();
    if(num_peak != (int)char_list.size() + 1){
        float sum = std::accumulate(us_alphas.begin(), us_alphas.end(), 0.0f);
        float scale = sum/((int)char_list.size() + 1);
        if(scale == 0){
            return;
        }
        cif_peak.clear();
        sum = 0.0;
        for(auto &alpha:us_alphas){
            alpha = alpha/scale;
            sum += alpha;
            cif_peak.emplace_back(sum);
            if(sum>=1.0 - 1e-4){
                sum -=(1.0 - 1e-4);
            }            
        }

        fire_place.clear();
        for (int i = 0; i < num_frames; i++) {
            if (cif_peak[i] > 1.0 - 1e-4) {
                fire_place.push_back(i + total_offset);
            }
        }
    }
    
    num_peak = fire_place.size();
    if(fire_place.size() == 0){
        return;
    }

    // begin silence
    if (fire_place[0] > START_END_THRESHOLD) {
        new_char_list.push_back("<sil>");
        timestamp_list.push_back({0.0, fire_place[0] * TIME_RATE});
    }

    // tokens timestamp
    for (int i = 0; i < num_peak - 1; i++) {
        new_char_list.push_back(char_list[i]);
        if (i == num_peak - 2 || MAX_TOKEN_DURATION < 0 || fire_place[i + 1] - fire_place[i] < MAX_TOKEN_DURATION) {
            timestamp_list.push_back({fire_place[i] * TIME_RATE, fire_place[i + 1] * TIME_RATE});
        } else {
            // cut the duration to token and sil of the 0-weight frames last long
            float _split = fire_place[i] + MAX_TOKEN_DURATION;
            timestamp_list.push_back({fire_place[i] * TIME_RATE, _split * TIME_RATE});
            timestamp_list.push_back({_split * TIME_RATE, fire_place[i + 1] * TIME_RATE});
            new_char_list.push_back("<sil>");
        }
    }

    // tail token and end silence
    if(timestamp_list.size()==0){
        LOG(ERROR)<<"timestamp_list's size is 0!";
        return;
    }
    if (num_frames - fire_place.back() > START_END_THRESHOLD) {
        float _end = (num_frames + fire_place.back()) / 2.0;
        timestamp_list.back()[1] = _end * TIME_RATE;
        timestamp_list.push_back({_end * TIME_RATE, num_frames * TIME_RATE});
        new_char_list.push_back("<sil>");
    } else {
        timestamp_list.back()[1] = num_frames * TIME_RATE;
    }

    if (begin_time) {  // add offset time in model with vad
        for (auto& timestamp : timestamp_list) {
            timestamp[0] += begin_time / 1000.0;
            timestamp[1] += begin_time / 1000.0;
        }
    }

    assert(new_char_list.size() == timestamp_list.size());

    for (int i = 0; i < (int)new_char_list.size(); i++) {
        res_str += new_char_list[i] + " " + to_string(timestamp_list[i][0]) + " " + to_string(timestamp_list[i][1]) + ";";
    }

    for (int i = 0; i < (int)new_char_list.size(); i++) {
        if(new_char_list[i] != "<sil>"){
            timestamp_vec.push_back(timestamp_list[i]);
        }
    }
}

vector<float> Paraformer::ApplyLfr(const std::vector<float> &in) 
{
    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;
    int32_t in_num_frames = in.size() / in_feat_dim;
    int32_t out_num_frames =
        (in_num_frames - lfr_m) / lfr_n + 1;
    int32_t out_feat_dim = in_feat_dim * lfr_m;

    std::vector<float> out(out_num_frames * out_feat_dim);

    const float *p_in = in.data();
    float *p_out = out.data();

    for (int32_t i = 0; i != out_num_frames; ++i) {
      std::copy(p_in, p_in + out_feat_dim, p_out);

      p_out += out_feat_dim;
      p_in += lfr_n * in_feat_dim;
    }

    return out;
  }

  void Paraformer::ApplyCmvn(std::vector<float> *v)
  {
    int32_t dim = means_list_.size();
    int32_t num_frames = v->size() / dim;

    float *p = v->data();

    for (int32_t i = 0; i != num_frames; ++i) {
      for (int32_t k = 0; k != dim; ++k) {
        p[k] = (p[k] + means_list_[k]) * vars_list_[k];
      }

      p += dim;
    }
  }

string Paraformer::Forward(float* din, int len, bool input_finished, const std::vector<std::vector<float>> &hw_emb)
{

    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;
    std::vector<float> wav_feats = FbankKaldi(MODEL_SAMPLE_RATE, din, len);
    wav_feats = ApplyLfr(wav_feats);
    ApplyCmvn(&wav_feats);

    int32_t feat_dim = lfr_m*in_feat_dim;
    int32_t num_frames = wav_feats.size() / feat_dim;
    //std::cout << "feat in: " << num_frames << " " << feat_dim << std::endl;

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

    std::vector<float> embedding;
    try{
        if (use_hotword) {
            if(hw_emb.size()<=0){
                LOG(ERROR) << "hw_emb is null";
                return "";
            }
            //PrintMat(hw_emb, "input_clas_emb");
            const int64_t hotword_shape[3] = {1, hw_emb.size(), hw_emb[0].size()};
            embedding.reserve(hw_emb.size() * hw_emb[0].size());
            for (auto item : hw_emb) {
                embedding.insert(embedding.end(), item.begin(), item.end());
            }
            //LOG(INFO) << "hotword shape " << hotword_shape[0] << " " << hotword_shape[1] << " " << hotword_shape[2] << " size " << embedding.size();
            Ort::Value onnx_hw_emb = Ort::Value::CreateTensor<float>(
                m_memoryInfo, embedding.data(), embedding.size(), hotword_shape, 3);

            input_onnx.emplace_back(std::move(onnx_hw_emb));
        }
    }catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
        return "";
    }

    string result="";
    try {
        auto outputTensor = m_session_->Run(Ort::RunOptions{nullptr}, m_szInputNames.data(), input_onnx.data(), input_onnx.size(), m_szOutputNames.data(), m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();
        //LOG(INFO) << "paraformer out shape " << outputShape[0] << " " << outputShape[1] << " " << outputShape[2];

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>();
        auto encoder_out_lens = outputTensor[1].GetTensorMutableData<int64_t>();
        // timestamp
        if(outputTensor.size() == 4){
            std::vector<int64_t> us_alphas_shape = outputTensor[2].GetTensorTypeAndShapeInfo().GetShape();
            float* us_alphas_data = outputTensor[2].GetTensorMutableData<float>();
            std::vector<float> us_alphas(us_alphas_shape[1]);
            for (int i = 0; i < us_alphas_shape[1]; i++) {
                us_alphas[i] = us_alphas_data[i];
            }

            std::vector<int64_t> us_peaks_shape = outputTensor[3].GetTensorTypeAndShapeInfo().GetShape();
            float* us_peaks_data = outputTensor[3].GetTensorMutableData<float>();
            std::vector<float> us_peaks(us_peaks_shape[1]);
            for (int i = 0; i < us_peaks_shape[1]; i++) {
                us_peaks[i] = us_peaks_data[i];
            }
            result = GreedySearch(floatData, *encoder_out_lens, outputShape[2], true, us_alphas, us_peaks);
        }else{
            result = GreedySearch(floatData, *encoder_out_lens, outputShape[2]);
        }
//         int pos = 0;
//         std::vector<std::vector<float>> logits;
//         for (int j = 0; j < outputShape[1]; j++)
//         {
//             std::vector<float> vec_token;
//             vec_token.insert(vec_token.begin(), floatData + pos, floatData + pos + outputShape[2]);
//             logits.push_back(vec_token);
//             pos += outputShape[2];
//         }
//         //PrintMat(logits, "logits_out");
//         result = GreedySearch(floatData, *encoder_out_lens, outputShape[2]);
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }

    return result;
}


std::vector<std::vector<float>> Paraformer::CompileHotwordEmbedding(std::string &hotwords) {
    int embedding_dim = encoder_size;
    std::vector<std::vector<float>> hw_emb;
    if (!use_hotword) {
        std::vector<float> vec(embedding_dim, 0);
        hw_emb.push_back(vec);
        return hw_emb;
    }
    int max_hotword_len = 10;
    std::vector<int32_t> hotword_matrix;
    std::vector<int32_t> lengths;
    int hotword_size = 1;
    int real_hw_size = 0;
    if (!hotwords.empty()) {
      std::vector<std::string> hotword_array = split(hotwords, ' ');
      hotword_size = hotword_array.size() + 1;
      hotword_matrix.reserve(hotword_size * max_hotword_len);
      for (auto hotword : hotword_array) {
        std::vector<std::string> chars;
        if (EncodeConverter::IsAllChineseCharactor((const U8CHAR_T*)hotword.c_str(), hotword.size())) {
          KeepChineseCharacterAndSplit(hotword, chars);
        } else {
          // for english
          std::vector<std::string> words = split(hotword, ' ');
          for (auto word : words) {
            std::vector<string> tokens = seg_dict->GetTokensByWord(word);
            chars.insert(chars.end(), tokens.begin(), tokens.end());
          }
        }
        if(chars.size()==0){
            continue;
        }
        std::vector<int32_t> hw_vector(max_hotword_len, 0);
        int vector_len = std::min(max_hotword_len, (int)chars.size());
        for (int i=0; i<chars.size(); i++) {
          std::cout << chars[i] << " ";
          hw_vector[i] = vocab->GetIdByToken(chars[i]);
        }
        std::cout << std::endl;
        lengths.push_back(vector_len);
        real_hw_size += 1;
        hotword_matrix.insert(hotword_matrix.end(), hw_vector.begin(), hw_vector.end());
      }
      hotword_size = real_hw_size + 1;
    }
    std::vector<int32_t> blank_vec(max_hotword_len, 0);
    blank_vec[0] = 1;
    hotword_matrix.insert(hotword_matrix.end(), blank_vec.begin(), blank_vec.end());
    lengths.push_back(1);

#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif

    const int64_t input_shape_[2] = {hotword_size, max_hotword_len};
    Ort::Value onnx_hotword = Ort::Value::CreateTensor<int32_t>(m_memoryInfo,
        (int32_t*)hotword_matrix.data(),
        hotword_size * max_hotword_len,
        input_shape_,
        2);
    LOG(INFO) << "clas shape " << hotword_size << " " << max_hotword_len << std::endl;
    
    std::vector<Ort::Value> input_onnx;
    input_onnx.emplace_back(std::move(onnx_hotword));

    std::vector<std::vector<float>> result;
    try {
        auto outputTensor = hw_m_session->Run(Ort::RunOptions{nullptr}, hw_m_szInputNames.data(), input_onnx.data(), input_onnx.size(), hw_m_szOutputNames.data(), hw_m_szOutputNames.size());
        std::vector<int64_t> outputShape = outputTensor[0].GetTensorTypeAndShapeInfo().GetShape();

        int64_t outputCount = std::accumulate(outputShape.begin(), outputShape.end(), 1, std::multiplies<int64_t>());
        float* floatData = outputTensor[0].GetTensorMutableData<float>(); // shape [max_hotword_len, hotword_size, dim]
        // get embedding by real hotword length
        assert(outputShape[0] == max_hotword_len);
        assert(outputShape[1] == hotword_size);
        embedding_dim = outputShape[2];

        for (int j = 0; j < hotword_size; j++)
        {
            int start_pos = hotword_size * (lengths[j] - 1) * embedding_dim + j * embedding_dim;
            std::vector<float> embedding;
            embedding.insert(embedding.begin(), floatData + start_pos, floatData + start_pos + embedding_dim);
            result.push_back(embedding);
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }
    //PrintMat(result, "clas_embedding_output");
    return result;
}

string Paraformer::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
