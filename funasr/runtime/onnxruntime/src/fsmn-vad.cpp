/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include <fstream>
#include "precomp.h"

void FsmnVad::InitVad(const std::string &vad_model, const std::string &vad_cmvn, const std::string &vad_config) {
    session_options_.SetIntraOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    session_options_.DisableCpuMemArena();

    ReadModel(vad_model.c_str());
    LoadCmvn(vad_cmvn.c_str());
    LoadConfigFromYaml(vad_config.c_str());
    InitCache();
}

void FsmnVad::LoadConfigFromYaml(const char* filename){

    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(ERROR) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

    try{
        YAML::Node frontend_conf = config["frontend_conf"];
        YAML::Node post_conf = config["vad_post_conf"];

        this->vad_sample_rate_ = frontend_conf["fs"].as<int>();
        this->vad_silence_duration_ =  post_conf["max_end_silence_time"].as<int>();
        this->vad_max_len_ = post_conf["max_single_segment_time"].as<int>();
        this->vad_speech_noise_thres_ = post_conf["speech_noise_thres"].as<double>();

        fbank_opts.frame_opts.dither = frontend_conf["dither"].as<float>();
        fbank_opts.mel_opts.num_bins = frontend_conf["n_mels"].as<int>();
        fbank_opts.frame_opts.samp_freq = (float)vad_sample_rate_;
        fbank_opts.frame_opts.window_type = frontend_conf["window"].as<string>();
        fbank_opts.frame_opts.frame_shift_ms = frontend_conf["frame_shift"].as<float>();
        fbank_opts.frame_opts.frame_length_ms = frontend_conf["frame_length"].as<float>();
        fbank_opts.energy_floor = 0;
        fbank_opts.mel_opts.debug_mel = false;
    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

void FsmnVad::ReadModel(const char* vad_model) {
    try {
        vad_session_ = std::make_shared<Ort::Session>(
                env_, vad_model, session_options_);
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load vad onnx model: " << e.what();
        exit(0);
    }
    GetInputOutputInfo(vad_session_, &vad_in_names_, &vad_out_names_);
}

void FsmnVad::GetInputOutputInfo(
        const std::shared_ptr<Ort::Session> &session,
        std::vector<const char *> *in_names, std::vector<const char *> *out_names) {
    Ort::AllocatorWithDefaultOptions allocator;
    // Input info
    int num_nodes = session->GetInputCount();
    in_names->resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetInputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session->GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        std::stringstream shape;
        for (auto j: node_dims) {
            shape << j;
            shape << " ";
        }
        // LOG(INFO) << "\tInput " << i << " : name=" << name.get() << " type=" << type
        //           << " dims=" << shape.str();
        (*in_names)[i] = name.get();
        name.release();
    }
    // Output info
    num_nodes = session->GetOutputCount();
    out_names->resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) {
        std::unique_ptr<char, Ort::detail::AllocatedFree> name = session->GetOutputNameAllocated(i, allocator);
        Ort::TypeInfo type_info = session->GetOutputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        ONNXTensorElementDataType type = tensor_info.GetElementType();
        std::vector<int64_t> node_dims = tensor_info.GetShape();
        std::stringstream shape;
        for (auto j: node_dims) {
            shape << j;
            shape << " ";
        }
        // LOG(INFO) << "\tOutput " << i << " : name=" << name.get() << " type=" << type
        //           << " dims=" << shape.str();
        (*out_names)[i] = name.get();
        name.release();
    }
}


void FsmnVad::Forward(
        const std::vector<std::vector<float>> &chunk_feats,
        std::vector<std::vector<float>> *out_prob) {
    Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    int num_frames = chunk_feats.size();
    const int feature_dim = chunk_feats[0].size();

    //  2. Generate input nodes tensor
    // vad node { batch,frame number,feature dim }
    const int64_t vad_feats_shape[3] = {1, num_frames, feature_dim};
    std::vector<float> vad_feats;
    for (const auto &chunk_feat: chunk_feats) {
        vad_feats.insert(vad_feats.end(), chunk_feat.begin(), chunk_feat.end());
    }
    Ort::Value vad_feats_ort = Ort::Value::CreateTensor<float>(
            memory_info, vad_feats.data(), vad_feats.size(), vad_feats_shape, 3);
    
    // 3. Put nodes into onnx input vector
    std::vector<Ort::Value> vad_inputs;
    vad_inputs.emplace_back(std::move(vad_feats_ort));
    // 4 caches
    // cache node {batch,128,19,1}
    const int64_t cache_feats_shape[4] = {1, 128, 19, 1};
    for (int i = 0; i < in_cache_.size(); i++) {
      vad_inputs.emplace_back(std::move(Ort::Value::CreateTensor<float>(
              memory_info, in_cache_[i].data(), in_cache_[i].size(), cache_feats_shape, 4)));
    }
  
    // 4. Onnx infer
    std::vector<Ort::Value> vad_ort_outputs;
    try {
        vad_ort_outputs = vad_session_->Run(
                Ort::RunOptions{nullptr}, vad_in_names_.data(), vad_inputs.data(),
                vad_inputs.size(), vad_out_names_.data(), vad_out_names_.size());
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when run vad onnx forword: " << (e.what());
        exit(0);
    }

    // 5. Change infer result to output shapes
    float *logp_data = vad_ort_outputs[0].GetTensorMutableData<float>();
    auto type_info = vad_ort_outputs[0].GetTensorTypeAndShapeInfo();

    int num_outputs = type_info.GetShape()[1];
    int output_dim = type_info.GetShape()[2];
    out_prob->resize(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        (*out_prob)[i].resize(output_dim);
        memcpy((*out_prob)[i].data(), logp_data + i * output_dim,
               sizeof(float) * output_dim);
    }
  
    // get 4 caches outputs,each size is 128*19
    for (int i = 1; i < 5; i++) {
      float* data = vad_ort_outputs[i].GetTensorMutableData<float>();
      memcpy(in_cache_[i-1].data(), data, sizeof(float) * 128*19);
    }
}

void FsmnVad::FbankKaldi(float sample_rate, std::vector<std::vector<float>> &vad_feats,
                         const std::vector<float> &waves) {
    knf::OnlineFbank fbank(fbank_opts);

    fbank.AcceptWaveform(sample_rate, &waves[0], waves.size());
    int32_t frames = fbank.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float *frame = fbank.GetFrame(i);
        std::vector<float> frame_vector(frame, frame + fbank_opts.mel_opts.num_bins);
        vad_feats.emplace_back(frame_vector);
    }
}

void FsmnVad::LoadCmvn(const char *filename)
{
    try{
        using namespace std;
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
                        // vars_list.push_back(stof(vars_lines[j])*scale);
                        vars_list.push_back(stof(vars_lines[j]));
                    }
                    continue;
                }
            }
        }
    }catch(std::exception const &e) {
        LOG(ERROR) << "Error when load vad cmvn : " << e.what();
        exit(0);
    }
}

std::vector<std::vector<float>> &FsmnVad::LfrCmvn(std::vector<std::vector<float>> &vad_feats) {

    std::vector<std::vector<float>> out_feats;
    int T = vad_feats.size();
    int T_lrf = ceil(1.0 * T / lfr_n);

    // Pad frames at start(copy first frame)
    for (int i = 0; i < (lfr_m - 1) / 2; i++) {
        vad_feats.insert(vad_feats.begin(), vad_feats[0]);
    }
    // Merge lfr_m frames as one,lfr_n frames per window
    T = T + (lfr_m - 1) / 2;
    std::vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        if (lfr_m <= T - i * lfr_n) {
            for (int j = 0; j < lfr_m; j++) {
                p.insert(p.end(), vad_feats[i * lfr_n + j].begin(), vad_feats[i * lfr_n + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            // Fill to lfr_m frames at last window if less than lfr_m frames  (copy last frame)
            int num_padding = lfr_m - (T - i * lfr_n);
            for (int j = 0; j < (vad_feats.size() - i * lfr_n); j++) {
                p.insert(p.end(), vad_feats[i * lfr_n + j].begin(), vad_feats[i * lfr_n + j].end());
            }
            for (int j = 0; j < num_padding; j++) {
                p.insert(p.end(), vad_feats[vad_feats.size() - 1].begin(), vad_feats[vad_feats.size() - 1].end());
            }
            out_feats.emplace_back(p);
        }
    }
    // Apply cmvn
    for (auto &out_feat: out_feats) {
        for (int j = 0; j < means_list.size(); j++) {
            out_feat[j] = (out_feat[j] + means_list[j]) * vars_list[j];
        }
    }
    vad_feats = out_feats;
    return vad_feats;
}

std::vector<std::vector<int>>
FsmnVad::Infer(const std::vector<float> &waves) {
    std::vector<std::vector<float>> vad_feats;
    std::vector<std::vector<float>> vad_probs;
    FbankKaldi(vad_sample_rate_, vad_feats, waves);
    vad_feats = LfrCmvn(vad_feats);
    Forward(vad_feats, &vad_probs);

    E2EVadModel vad_scorer = E2EVadModel();
    std::vector<std::vector<int>> vad_segments;
    vad_segments = vad_scorer(vad_probs, waves, true, false, vad_silence_duration_, vad_max_len_,
                              vad_speech_noise_thres_, vad_sample_rate_);
    return vad_segments;
}

void FsmnVad::InitCache(){
  std::vector<float> cache_feats(128 * 19 * 1, 0);
  for (int i=0;i<4;i++){
    in_cache_.emplace_back(cache_feats);
  }
};

void FsmnVad::Reset(){
  in_cache_.clear();
  InitCache();
};

void FsmnVad::Test() {
}

FsmnVad::FsmnVad():env_(ORT_LOGGING_LEVEL_ERROR, ""),session_options_{} {
}
