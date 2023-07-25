/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"

using namespace std;

namespace funasr {

ParaformerOnline::ParaformerOnline()
:env_(ORT_LOGGING_LEVEL_ERROR, "ParaformerOnline"),session_options{}{
}

void ParaformerOnline::InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num){
    // knf options
    fbank_opts_.frame_opts.dither = 0;
    fbank_opts_.mel_opts.num_bins = n_mels;
    fbank_opts_.frame_opts.samp_freq = MODEL_SAMPLE_RATE;
    fbank_opts_.frame_opts.window_type = "hamming";
    fbank_opts_.frame_opts.frame_shift_ms = 10;
    fbank_opts_.frame_opts.frame_length_ms = 25;
    fbank_opts_.energy_floor = 0;
    fbank_opts_.mel_opts.debug_mel = false;
    // fbank_ = std::make_unique<knf::OnlineFbank>(fbank_opts_);

    // session_options.SetInterOpNumThreads(1);
    session_options.SetIntraOpNumThreads(thread_num);
    session_options.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // DisableCpuMemArena can improve performance
    session_options.DisableCpuMemArena();

    try {
        encoder_session = std::make_unique<Ort::Session>(env_, en_model.c_str(), session_options);
        LOG(INFO) << "Successfully load model from " << en_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am encoder model: " << e.what();
        exit(0);
    }

    try {
        decoder_session = std::make_unique<Ort::Session>(env_, de_model.c_str(), session_options);
        LOG(INFO) << "Successfully load model from " << de_model;
    } catch (std::exception const &e) {
        LOG(ERROR) << "Error when load am decoder model: " << e.what();
        exit(0);
    }

    // encoder
    string strName;
    GetInputName(encoder_session.get(), strName);
    en_strInputNames.push_back(strName.c_str());
    GetInputName(encoder_session.get(), strName,1);
    en_strInputNames.push_back(strName);
    
    GetOutputName(encoder_session.get(), strName);
    en_strOutputNames.push_back(strName);
    GetOutputName(encoder_session.get(), strName,1);
    en_strOutputNames.push_back(strName);
    GetOutputName(encoder_session.get(), strName,2);
    en_strOutputNames.push_back(strName);

    for (auto& item : en_strInputNames)
        en_szInputNames.push_back(item.c_str());
    for (auto& item : en_strOutputNames)
        en_szOutputNames.push_back(item.c_str());

    // decoder
    int de_input_len = 4 + fsmn_layers;
    int de_out_len = 2 + fsmn_layers;
    for(int i=0;i<de_input_len; i++){
        GetInputName(decoder_session.get(), strName, i);
        de_strInputNames.push_back(strName.c_str());
    }

    for(int i=0;i<de_out_len; i++){
        GetOutputName(decoder_session.get(), strName,i);
        de_strOutputNames.push_back(strName);
    }

    for (auto& item : de_strInputNames)
        de_szInputNames.push_back(item.c_str());
    for (auto& item : de_strOutputNames)
        de_szOutputNames.push_back(item.c_str());

    vocab = new Vocab(am_config.c_str());
    LoadCmvn(am_cmvn.c_str());

    InitCache();
}

void ParaformerOnline::FbankKaldi(float sample_rate, std::vector<std::vector<float>> &wav_feats,
                               std::vector<float> &waves) {
    knf::OnlineFbank fbank(fbank_opts_);
    // cache merge
    waves.insert(waves.begin(), input_cache_.begin(), input_cache_.end());
    int frame_number = ComputeFrameNum(waves.size(), frame_sample_length_, frame_shift_sample_length_);
    // Send the audio after the last frame shift position to the cache
    input_cache_.clear();
    input_cache_.insert(input_cache_.begin(), waves.begin() + frame_number * frame_shift_sample_length_, waves.end());
    if (frame_number == 0) {
        return;
    }
    // Delete audio that haven't undergone fbank processing
    waves.erase(waves.begin() + (frame_number - 1) * frame_shift_sample_length_ + frame_sample_length_, waves.end());

    std::vector<float> buf(waves.size());
    for (int32_t i = 0; i != waves.size(); ++i) {
        buf[i] = waves[i] * 32768;
    }
    fbank.AcceptWaveform(sample_rate, buf.data(), buf.size());
    // fbank.AcceptWaveform(sample_rate, &waves[0], waves.size());
    int32_t frames = fbank.NumFramesReady();
    for (int32_t i = 0; i != frames; ++i) {
        const float *frame = fbank.GetFrame(i);
        vector<float> frame_vector(frame, frame + fbank_opts_.mel_opts.num_bins);
        wav_feats.emplace_back(frame_vector);
    }
}

void ParaformerOnline::ExtractFeats(float sample_rate, vector<std::vector<float>> &wav_feats,
                                 vector<float> &waves, bool input_finished) {
  FbankKaldi(sample_rate, wav_feats, waves);
  // cache deal & online lfr,cmvn
  if (wav_feats.size() > 0) {
    if (!reserve_waveforms_.empty()) {
      waves.insert(waves.begin(), reserve_waveforms_.begin(), reserve_waveforms_.end());
    }
    if (lfr_splice_cache_.empty()) {
      for (int i = 0; i < (lfr_m - 1) / 2; i++) {
        lfr_splice_cache_.emplace_back(wav_feats[0]);
      }
    }
    if (wav_feats.size() + lfr_splice_cache_.size() >= lfr_m) {
      wav_feats.insert(wav_feats.begin(), lfr_splice_cache_.begin(), lfr_splice_cache_.end());
      int frame_from_waves = (waves.size() - frame_sample_length_) / frame_shift_sample_length_ + 1;
      int minus_frame = reserve_waveforms_.empty() ? (lfr_m - 1) / 2 : 0;
      int lfr_splice_frame_idxs = OnlineLfrCmvn(wav_feats, input_finished);
      int reserve_frame_idx = lfr_splice_frame_idxs - minus_frame;
      reserve_waveforms_.clear();
      reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                waves.begin() + reserve_frame_idx * frame_shift_sample_length_,
                                waves.begin() + frame_from_waves * frame_shift_sample_length_);
      int sample_length = (frame_from_waves - 1) * frame_shift_sample_length_ + frame_sample_length_;
      waves.erase(waves.begin() + sample_length, waves.end());
    } else {
      reserve_waveforms_.clear();
      reserve_waveforms_.insert(reserve_waveforms_.begin(),
                                waves.begin() + frame_sample_length_ - frame_shift_sample_length_, waves.end());
      lfr_splice_cache_.insert(lfr_splice_cache_.end(), wav_feats.begin(), wav_feats.end());
    }
  } else {
    if (input_finished) {
      if (!reserve_waveforms_.empty()) {
        waves = reserve_waveforms_;
      }
      wav_feats = lfr_splice_cache_;
      OnlineLfrCmvn(wav_feats, input_finished);
    }
  }
  if(input_finished){
      ResetCache();
  }
}

int ParaformerOnline::OnlineLfrCmvn(vector<vector<float>> &wav_feats, bool input_finished) {
    vector<vector<float>> out_feats;
    int T = wav_feats.size();
    int T_lrf = ceil((T - (lfr_m - 1) / 2) / (float)lfr_n);
    int lfr_splice_frame_idxs = T_lrf;
    vector<float> p;
    for (int i = 0; i < T_lrf; i++) {
        if (lfr_m <= T - i * lfr_n) {
            for (int j = 0; j < lfr_m; j++) {
                p.insert(p.end(), wav_feats[i * lfr_n + j].begin(), wav_feats[i * lfr_n + j].end());
            }
            out_feats.emplace_back(p);
            p.clear();
        } else {
            if (input_finished) {
                int num_padding = lfr_m - (T - i * lfr_n);
                for (int j = 0; j < (wav_feats.size() - i * lfr_n); j++) {
                    p.insert(p.end(), wav_feats[i * lfr_n + j].begin(), wav_feats[i * lfr_n + j].end());
                }
                for (int j = 0; j < num_padding; j++) {
                    p.insert(p.end(), wav_feats[wav_feats.size() - 1].begin(), wav_feats[wav_feats.size() - 1].end());
                }
                out_feats.emplace_back(p);
            } else {
                lfr_splice_frame_idxs = i;
                break;
            }
        }
    }
    lfr_splice_frame_idxs = std::min(T - 1, lfr_splice_frame_idxs * lfr_n);
    lfr_splice_cache_.clear();
    lfr_splice_cache_.insert(lfr_splice_cache_.begin(), wav_feats.begin() + lfr_splice_frame_idxs, wav_feats.end());

    // Apply cmvn
    for (auto &out_feat: out_feats) {
        for (int j = 0; j < means_list_.size(); j++) {
            out_feat[j] = (out_feat[j] + means_list_[j]) * vars_list_[j];
        }
    }
    wav_feats = out_feats;
    return lfr_splice_frame_idxs;
}

void ParaformerOnline::GetPosEmb(std::vector<std::vector<float>> &wav_feats, int timesteps, int feat_dim)
{
    int start_idx = start_idx_cache_;
    start_idx_cache_ += timesteps;
    int mm = start_idx_cache_;

    int i;
    float scale = -0.0330119726594128;

    std::vector<float> tmp(mm*feat_dim);

    for (i = 0; i < feat_dim/2; i++) {
        float tmptime = exp(i * scale);
        int j;
        for (j = 0; j < mm; j++) {
            int sin_idx = j * feat_dim + i;
            int cos_idx = j * feat_dim + i + feat_dim/2;
            float coe = tmptime * (j + 1);
            tmp[sin_idx] = sin(coe);
            tmp[cos_idx] = cos(coe);
        }
    }

    for (i = start_idx; i < start_idx + timesteps; i++) {
        for (int j = 0; j < feat_dim; j++) {
            wav_feats[i-start_idx][j] += tmp[i*feat_dim+j];
        }
    }
}

void ParaformerOnline::CifSearch(std::vector<std::vector<float>> hidden, std::vector<float> alphas, bool is_final, std::vector<std::vector<float>>& list_frame)
{
    int hidden_size = 0;
    if(hidden.size() > 0){
        hidden_size = hidden[0].size();
    }
    // cache
    int i,j;
    int chunk_size_pre = chunk_size[0];
    for (i = 0; i < chunk_size_pre; i++)
        alphas[i] = 0.0;

    int chunk_size_suf = std::accumulate(chunk_size.begin(), chunk_size.end()-1, 0);
    for (i = chunk_size_suf; i < alphas.size(); i++){
        alphas[i] = 0.0;
    }

    if(hidden_cache_.size()>0){
        hidden.insert(hidden.begin(), hidden_cache_.begin(), hidden_cache_.end());
        alphas.insert(alphas.begin(), alphas_cache_.begin(), alphas_cache_.end());
        hidden_cache_.clear();
        alphas_cache_.clear();
    }
    
    if (is_last_chunk) { // TODD: finish final part
        std::vector<float> tail_hidden(hidden_size, 0);
        hidden.emplace_back(tail_hidden);
        alphas.emplace_back(tail_alphas);
    }

    float intergrate = 0.0;
    int len_time = alphas.size();
    std::vector<float> frames(hidden_size, 0);
    std::vector<float> list_fire;

    for (i = 0; i < len_time; i++) {
        float alpha = alphas[i];
        if (alpha + intergrate < cif_threshold) {
            intergrate += alpha;
            list_fire.emplace_back(intergrate);
            for (j = 0; j < hidden_size; j++) {
                frames[j] += alpha * hidden[i][j];
            }
        } else {
            for (j = 0; j < hidden_size; j++) {
                frames[j] += (cif_threshold - intergrate) * hidden[i][j];
            }
            std::vector<float> frames_cp(frames);
            list_frame.emplace_back(frames_cp);
            intergrate += alpha;
            list_fire.emplace_back(intergrate);
            intergrate -= cif_threshold;
            for (j = 0; j < hidden_size; j++) {
                frames[j] = intergrate * hidden[i][j];
            }
        }
    }

    // cache
    alphas_cache_.emplace_back(intergrate);
    if (intergrate > 0.0) {
        std::vector<float> hidden_cache(hidden_size, 0);
        for (i = 0; i < hidden_size; i++) {
            hidden_cache[i] = frames[i] / intergrate;
        }
        hidden_cache_.emplace_back(hidden_cache);
    } else {
        std::vector<float> frames_cp(frames);
        hidden_cache_.emplace_back(frames_cp);
    }
}

void ParaformerOnline::InitCache(){
    
    start_idx_cache_ = 0;
    is_first_chunk = true;
    is_last_chunk = false;
    hidden_cache_.clear();
    alphas_cache_.clear();
    feats_cache_.clear();
    fsmn_caches_.clear();

    // cif cache
    std::vector<float> hidden_cache(encoder_size, 0);
    hidden_cache_.emplace_back(hidden_cache);
    alphas_cache_.emplace_back(0);
    
    // feats
    std::vector<float> feat_cache(feat_dims, 0);
    for(int i=0; i<(chunk_size[0]+chunk_size[2]); i++){
        feats_cache_.emplace_back(feat_cache);
    }

    // fsmn cache
    std::vector<float> tmp(fsmn_lorder, 0);
    std::vector<std::vector<float>> tmp_dims;
    for(int i=0; i<fsmn_dims; i++){
        tmp_dims.emplace_back(tmp);
    }
    std::vector<std::vector<std::vector<float>>> fsmn_cache;
    fsmn_cache.emplace_back(tmp_dims);
    for(int i=0; i<fsmn_layers; i++){
        fsmn_caches_.emplace_back(fsmn_cache);
    }
};

void ParaformerOnline::Reset()
{
    InitCache();
}

void ParaformerOnline::ResetCache() {
    reserve_waveforms_.clear();
    input_cache_.clear();
    lfr_splice_cache_.clear();
}

void ParaformerOnline::AddOverlapChunk(std::vector<std::vector<float>> &wav_feats, bool input_finished){
    wav_feats.insert(wav_feats.begin(), feats_cache_.begin(), feats_cache_.end());
    if(input_finished){
        feats_cache_.clear();
        feats_cache_.insert(feats_cache_.begin(), wav_feats.end()-chunk_size[0], wav_feats.end());
        if(!is_last_chunk){
            int padding_length = std::accumulate(chunk_size.begin(), chunk_size.end(), 0) - wav_feats.size();
            std::vector<float> tmp(feat_dims, 0);
            for(int i=0; i<padding_length; i++){
                wav_feats.emplace_back(feat_dims);
            }
        }
    }else{
        feats_cache_.clear();
        feats_cache_.insert(feats_cache_.begin(), wav_feats.end()-chunk_size[0]-chunk_size[2], wav_feats.end());        
    }
}

string ParaformerOnline::ForwardChunk(std::vector<std::vector<float>> &chunk_feats, bool input_finished)
{
    int32_t num_frames = chunk_feats.size();

#ifdef _WIN_X86
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
        Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    const int64_t input_shape_[3] = {1, num_frames, feat_dims};
    std::vector<float> wav_feats;
    for (const auto &chunk_feat: chunk_feats) {
        wav_feats.insert(wav_feats.end(), chunk_feat.begin(), chunk_feat.end());
    }
    Ort::Value onnx_feats = Ort::Value::CreateTensor<float>(
        m_memoryInfo,
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
    
    auto encoder_tensor = encoder_session->Run(Ort::RunOptions{nullptr}, en_szInputNames.data(), input_onnx.data(), input_onnx.size(), en_szOutputNames.data(), en_szOutputNames.size());
    // get enc_vec
    std::vector<int64_t> enc_shape = encoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
    float* enc_data = encoder_tensor[0].GetTensorMutableData<float>();
    std::vector<std::vector<float>> enc_vec(enc_shape[1], std::vector<float>(enc_shape[2]));
    for (int i = 0; i < enc_shape[1]; i++) {
        for (int j = 0; j < enc_shape[2]; j++) {
            enc_vec[i][j] = enc_data[i * enc_shape[2] + j];
        }
    }

    // get alpha_vec
    std::vector<int64_t> alpha_shape = encoder_tensor[2].GetTensorTypeAndShapeInfo().GetShape();
    float* alpha_data = encoder_tensor[2].GetTensorMutableData<float>();
    std::vector<float> alpha_vec(alpha_shape[1]);
    for (int i = 0; i < alpha_shape[1]; i++) {
        alpha_vec[i] = alpha_data[i];
    }

    std::vector<std::vector<float>> list_frame;
    CifSearch(enc_vec, alpha_vec, input_finished, list_frame);

    string result;
    if(list_frame.size()>0){
        std::vector<Ort::Value> decoder_onnx;
        // enc
        const int64_t enc_shape_[3] = {1, enc_vec.size(), enc_vec[0].size()};
        std::vector<float> enc_input;
        for (const auto &enc_vec_: enc_vec) {
            enc_input.insert(enc_input.end(), enc_vec_.begin(), enc_vec_.end());
        }
        Ort::Value onnx_enc = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            enc_input.data(),
            enc_input.size(),
            enc_shape_,
            3);
        decoder_onnx.emplace_back(std::move(onnx_enc));

        // enc_lens, encoder_tensor[1]
        decoder_onnx.emplace_back(std::move(encoder_tensor[1]));

        // acoustic_embeds
        const int64_t emb_shape_[3] = {1, list_frame.size(), list_frame[0].size()};
        std::vector<float> emb_input;
        for (const auto &list_frame_: list_frame) {
            emb_input.insert(emb_input.end(), list_frame_.begin(), list_frame_.end());
        }
        Ort::Value onnx_emb = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            emb_input.data(),
            emb_input.size(),
            emb_shape_,
            3);
        decoder_onnx.emplace_back(std::move(onnx_emb));

        // acoustic_embeds_len
        const int64_t emb_length_shape[1] = {1};
        std::vector<int32_t> emb_length;
        emb_length.emplace_back(list_frame.size());
        Ort::Value onnx_emb_len = Ort::Value::CreateTensor<int32_t>(
            m_memoryInfo, emb_length.data(), emb_length.size(), emb_length_shape, 1);
        decoder_onnx.emplace_back(std::move(onnx_emb_len));

        // fsmn_caches
        std::vector<std::vector<float>> fsmn_inputs;
        for (auto &fsmn_cache_s3: fsmn_caches_) {
            std::vector<float> fsmn_input;
            for (auto &fsmn_cache_s2: fsmn_cache_s3) {
                for (auto &fsmn_cache_s1: fsmn_cache_s2){
                    fsmn_input.insert(fsmn_input.end(), fsmn_cache_s1.begin(), fsmn_cache_s1.end());
                }
            }
            fsmn_inputs.emplace_back(fsmn_input);
        }
        const int64_t fsmn_shape_[3] = {1, fsmn_dims, fsmn_lorder};

        for(int l=0; l<fsmn_layers; l++){
            Ort::Value onnx_fsmn_cache = Ort::Value::CreateTensor<float>(
                m_memoryInfo,
                fsmn_inputs[l].data(),
                fsmn_inputs[l].size(),
                fsmn_shape_,
                3);
            decoder_onnx.emplace_back(std::move(onnx_fsmn_cache));
        }
        std::vector<int64_t> fsmn_shape__ = decoder_onnx[4].GetTensorTypeAndShapeInfo().GetShape();
        float* fsmn_data = decoder_onnx[4].GetTensorMutableData<float>();

        auto decoder_tensor = decoder_session->Run(Ort::RunOptions{nullptr}, de_szInputNames.data(), decoder_onnx.data(), decoder_onnx.size(), de_szOutputNames.data(), de_szOutputNames.size());
        for(int l=0;l<fsmn_layers;l++){
            std::vector<int64_t> fsmn_shape = decoder_tensor[2+l].GetTensorTypeAndShapeInfo().GetShape();
            float* fsmn_data = decoder_tensor[2+l].GetTensorMutableData<float>();
            for(int b=0; b<fsmn_shape[0]; b++){
                for(int dim1=0; dim1<fsmn_shape[1]; dim1++){
                    for(int dim2=fsmn_shape[2]-fsmn_lorder; dim2<fsmn_shape[2]; dim2++){
                        fsmn_caches_[l][b][dim1][dim2-(fsmn_shape[2]-fsmn_lorder)] = fsmn_data[b*fsmn_shape[1]*fsmn_shape[2]+dim1*fsmn_shape[2]+dim2];
                    }

                }
            }
        }

        std::vector<int64_t> decoder_shape = decoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
        float* floatData = decoder_tensor[0].GetTensorMutableData<float>();
        result = GreedySearch(floatData, list_frame.size(), decoder_shape[2]);

    }

    return result;
}

string ParaformerOnline::Forward(float* din, int len, bool input_finished)
{
    std::vector<std::vector<float>> wav_feats;
    std::vector<float> waves(din, din+len);
    
    string result="";
    if(len <16*60 && input_finished && !is_first_chunk){
        is_last_chunk = true;
        wav_feats = feats_cache_;
        result = ForwardChunk(wav_feats, is_last_chunk);
        // reset
        ResetCache();
        Reset();
        return result;
    }
    ExtractFeats(MODEL_SAMPLE_RATE, wav_feats, waves, input_finished);
    if(wav_feats.size() == 0){
        return result;
    }
    double factor = std::sqrt(encoder_size);
    for (auto& row : wav_feats) {
        for (auto& val : row) {
            val *= factor;
        }
    }

    GetPosEmb(wav_feats, wav_feats.size(), wav_feats[0].size());
    if(input_finished){
        if(wav_feats.size()+chunk_size[2] <= chunk_size[1]){
            is_last_chunk = true;
            AddOverlapChunk(wav_feats, input_finished);
        }else{
            // first chunk
            std::vector<std::vector<float>> first_chunk;
            first_chunk.insert(first_chunk.begin(), wav_feats.begin(), wav_feats.begin()+chunk_size[1]);
            AddOverlapChunk(first_chunk, input_finished);
            string str_first_chunk = ForwardChunk(first_chunk, is_last_chunk);

            // last chunk
            is_last_chunk = true;
            std::vector<std::vector<float>> last_chunk;
            last_chunk.insert(last_chunk.begin(), wav_feats.end()-(wav_feats.size()+chunk_size[2]-chunk_size[1]), wav_feats.end());
            AddOverlapChunk(last_chunk, input_finished);
            string str_last_chunk = ForwardChunk(last_chunk, is_last_chunk);

            result = str_first_chunk+str_last_chunk;
            // reset
            ResetCache();
            Reset();
            return result;
        }
    }else{
        AddOverlapChunk(wav_feats, input_finished);
    }

    result = ForwardChunk(wav_feats, is_last_chunk);
    if(input_finished){
        // reset
        ResetCache();
        Reset();
    }

    return result;
}

string ParaformerOnline::GreedySearch(float * in, int n_len,  int64_t token_nums)
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

ParaformerOnline::~ParaformerOnline()
{
    if(vocab)
        delete vocab;
}

void ParaformerOnline::LoadCmvn(const char *filename)
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

string ParaformerOnline::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
