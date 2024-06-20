/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"
#include "paraformer-torch.h"
#include "encode_converter.h"
#include <cstddef>

using namespace std;
namespace funasr {

ParaformerTorch::ParaformerTorch()
:use_hotword(false){
}

// offline
void ParaformerTorch::InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){
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

    vocab = new Vocab(token_file.c_str());
	phone_set_ = new PhoneSet(token_file.c_str());
    LoadCmvn(am_cmvn.c_str());

    torch::DeviceType device = at::kCPU;
    #ifdef USE_GPU
    if (!torch::cuda::is_available()) {
        LOG(ERROR) << "CUDA is not available! Please check your GPU settings";
        exit(-1);
    } else {
        LOG(INFO) << "CUDA is available, running on GPU";
        device = at::kCUDA;
    }
    #endif
    #ifdef USE_IPEX
    torch::jit::setTensorExprFuserEnabled(false);
    #endif

    try {
        torch::jit::script::Module model = torch::jit::load(am_model, device);
        model_ = std::make_shared<TorchModule>(std::move(model)); 
        LOG(INFO) << "Successfully load model from " << am_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am model: " << am_model << e.what();
        exit(-1);
    }
}

void ParaformerTorch::InitLm(const std::string &lm_file, 
                        const std::string &lm_cfg_file, 
                        const std::string &lex_file) {
    try {
        lm_ = std::shared_ptr<fst::Fst<fst::StdArc>>(
            fst::Fst<fst::StdArc>::Read(lm_file));
        if (lm_){
            lm_vocab = new Vocab(lm_cfg_file.c_str(), lex_file.c_str());
            LOG(INFO) << "Successfully load lm file " << lm_file;
        }else{
            LOG(ERROR) << "Failed to load lm file " << lm_file;
        }
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load lm file: " << e.what();
        exit(0);
    }
}

void ParaformerTorch::LoadConfigFromYaml(const char* filename){

    YAML::Node config;
    try{
        config = YAML::LoadFile(filename);
    }catch(exception const &e){
        LOG(ERROR) << "Error loading file, yaml file error or not exist.";
        exit(-1);
    }

    try{
        YAML::Node frontend_conf = config["frontend_conf"];
        this->asr_sample_rate = frontend_conf["fs"].as<int>();

        YAML::Node lang_conf = config["lang"];
        if (lang_conf.IsDefined()){
            language = lang_conf.as<string>();
        }
    }catch(exception const &e){
        LOG(ERROR) << "Error when load argument from vad config YAML.";
        exit(-1);
    }
}

void ParaformerTorch::InitHwCompiler(const std::string &hw_model, int thread_num) {
    // TODO
    use_hotword = true;
}

void ParaformerTorch::InitSegDict(const std::string &seg_dict_model) {
    seg_dict = new SegDict(seg_dict_model.c_str());
}

ParaformerTorch::~ParaformerTorch()
{
    if(vocab){
        delete vocab;
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

void ParaformerTorch::StartUtterance()
{
}

void ParaformerTorch::EndUtterance()
{
}

void ParaformerTorch::Reset()
{
}

void ParaformerTorch::FbankKaldi(float sample_rate, const float* waves, int len, std::vector<std::vector<float>> &asr_feats) {
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

void ParaformerTorch::LoadCmvn(const char *filename)
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

string ParaformerTorch::GreedySearch(float * in, int n_len,  int64_t token_nums, bool is_stamp, std::vector<float> us_alphas, std::vector<float> us_cif_peak)
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
        return vocab->Vector2StringV2(hyps, language);
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

string ParaformerTorch::BeamSearch(WfstDecoder* &wfst_decoder, float *in, int len, int64_t token_nums)
{
  return wfst_decoder->Search(in, len, token_nums);
}

string ParaformerTorch::FinalizeDecode(WfstDecoder* &wfst_decoder,
                                  bool is_stamp, std::vector<float> us_alphas, std::vector<float> us_cif_peak)
{
  return wfst_decoder->FinalizeDecode(is_stamp, us_alphas, us_cif_peak);
}

void ParaformerTorch::LfrCmvn(std::vector<std::vector<float>> &asr_feats) {

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

std::vector<std::string> ParaformerTorch::Forward(float** din, int* len, bool input_finished, const std::vector<std::vector<float>> &hw_emb, void* decoder_handle, int batch_in)
{
    WfstDecoder* wfst_decoder = (WfstDecoder*)decoder_handle;
    int32_t in_feat_dim = fbank_opts_.mel_opts.num_bins;
    int32_t feature_dim = lfr_m*in_feat_dim;

    std::vector<vector<float>> feats_batch;
    std::vector<int32_t> paraformer_length;
    int max_size = 0;
    int max_frames = 0;
    for(int index=0; index<batch_in; index++){
        std::vector<std::vector<float>> asr_feats;
        FbankKaldi(asr_sample_rate, din[index], len[index], asr_feats);
        if(asr_feats.size() != 0){
            LfrCmvn(asr_feats);
        }
        int32_t num_frames  = asr_feats.size();
        paraformer_length.emplace_back(num_frames);
        if(max_size < asr_feats.size()*feature_dim){
            max_size = asr_feats.size()*feature_dim;
            max_frames = num_frames;
        }

        std::vector<float> flattened;
        for (const auto& sub_vector : asr_feats) {
            flattened.insert(flattened.end(), sub_vector.begin(), sub_vector.end());
        }
        feats_batch.emplace_back(flattened);
    }

    torch::NoGradGuard no_grad;
    model_->eval();
    // padding
    std::vector<float> all_feats(batch_in * max_frames * feature_dim);
    for(int index=0; index<batch_in; index++){
        feats_batch[index].resize(max_size);
        std::memcpy(&all_feats[index * max_frames * feature_dim], feats_batch[index].data(),
                        max_frames * feature_dim * sizeof(float));
    }
    torch::Tensor feats =
        torch::from_blob(all_feats.data(),
                {batch_in, max_frames, feature_dim}, torch::kFloat).contiguous();
    torch::Tensor feat_lens = torch::from_blob(paraformer_length.data(),
                        {batch_in}, torch::kInt32);

    // 2. forward
    #ifdef USE_GPU
    feats = feats.to(at::kCUDA);
    feat_lens = feat_lens.to(at::kCUDA);
    #endif
    std::vector<torch::jit::IValue> inputs = {feats, feat_lens};

    vector<std::string> results;
    try {
        auto outputs = model_->forward(inputs).toTuple()->elements();
        torch::Tensor am_scores;
        torch::Tensor valid_token_lens;
        #ifdef USE_GPU
        am_scores = outputs[0].toTensor().to(at::kCPU);
        valid_token_lens = outputs[1].toTensor().to(at::kCPU);
        #else
        am_scores = outputs[0].toTensor();
        valid_token_lens = outputs[1].toTensor();
        #endif
        // timestamp
        for(int index=0; index<batch_in; index++){
            string result="";
            if(outputs.size() == 4){
                torch::Tensor us_alphas_tensor;
                torch::Tensor us_peaks_tensor;
                #ifdef USE_GPU
                us_alphas_tensor = outputs[2].toTensor().to(at::kCPU);
                us_peaks_tensor = outputs[3].toTensor().to(at::kCPU);
                #else
                us_alphas_tensor = outputs[2].toTensor();
                us_peaks_tensor = outputs[3].toTensor();
                #endif

                float* us_alphas_data = us_alphas_tensor[index].data_ptr<float>();
                std::vector<float> us_alphas(paraformer_length[index]);
                for (int i = 0; i < us_alphas.size(); i++) {
                    us_alphas[i] = us_alphas_data[i];
                }

                float* us_peaks_data = us_peaks_tensor[index].data_ptr<float>();
                std::vector<float> us_peaks(paraformer_length[index]);
                for (int i = 0; i < us_peaks.size(); i++) {
                    us_peaks[i] = us_peaks_data[i];
                }
                if (lm_ == nullptr) {
                    result = GreedySearch(am_scores[index].data_ptr<float>(), valid_token_lens[index].item<int>(), am_scores.size(2), true, us_alphas, us_peaks);
                } else {
                    result = BeamSearch(wfst_decoder, am_scores[index].data_ptr<float>(), valid_token_lens[index].item<int>(), am_scores.size(2));
                    if (input_finished) {
                        result = FinalizeDecode(wfst_decoder, true, us_alphas, us_peaks);
                    }
                }
            }else{
                if (lm_ == nullptr) {
                    result = GreedySearch(am_scores[index].data_ptr<float>(), valid_token_lens[index].item<int>(), am_scores.size(2));
                } else {
                    result = BeamSearch(wfst_decoder, am_scores[index].data_ptr<float>(), valid_token_lens[index].item<int>(), am_scores.size(2));
                    if (input_finished) {
                        result = FinalizeDecode(wfst_decoder);
                    }
                }
            }
            results.push_back(result);
			if (wfst_decoder){
				wfst_decoder->StartUtterance();
			}
        }
    }
    catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }

    return results;
}

std::vector<std::vector<float>> ParaformerTorch::CompileHotwordEmbedding(std::string &hotwords) {
    // TODO
    std::vector<std::vector<float>> result(1, std::vector<float>(10, 0.0f));
    return result;
}

Vocab* ParaformerTorch::GetVocab()
{
    return vocab;
}

Vocab* ParaformerTorch::GetLmVocab()
{
    return lm_vocab;
}

PhoneSet* ParaformerTorch::GetPhoneSet()
{
    return phone_set_;
}

string ParaformerTorch::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
