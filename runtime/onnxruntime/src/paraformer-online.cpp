/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/

#include "precomp.h"

using namespace std;

namespace funasr {

ParaformerOnline::ParaformerOnline(Model* offline_handle, std::vector<int> chunk_size, std::string model_type)
:offline_handle_(std::move(offline_handle)),chunk_size(chunk_size),session_options_{}{
    if(model_type == MODEL_PARA){
        Paraformer* para_handle = dynamic_cast<Paraformer*>(offline_handle_);
        InitOnline(
        para_handle->fbank_opts_,
        para_handle->encoder_session_,
        para_handle->decoder_session_,
        para_handle->en_szInputNames_,
        para_handle->en_szOutputNames_,
        para_handle->de_szInputNames_,
        para_handle->de_szOutputNames_,
        para_handle->means_list_,
        para_handle->vars_list_,
        para_handle->frame_length,
        para_handle->frame_shift,
        para_handle->n_mels,
        para_handle->lfr_m,
        para_handle->lfr_n,
        para_handle->encoder_size,
        para_handle->fsmn_layers,
        para_handle->fsmn_lorder,
        para_handle->fsmn_dims,
        para_handle->cif_threshold,
        para_handle->tail_alphas);
    }else if(model_type == MODEL_SVS){
        SenseVoiceSmall* svs_handle = dynamic_cast<SenseVoiceSmall*>(offline_handle_);
        InitOnline(
        svs_handle->fbank_opts_,
        svs_handle->encoder_session_,
        svs_handle->decoder_session_,
        svs_handle->en_szInputNames_,
        svs_handle->en_szOutputNames_,
        svs_handle->de_szInputNames_,
        svs_handle->de_szOutputNames_,
        svs_handle->means_list_,
        svs_handle->vars_list_,
        svs_handle->frame_length,
        svs_handle->frame_shift,
        svs_handle->n_mels,
        svs_handle->lfr_m,
        svs_handle->lfr_n,
        svs_handle->encoder_size,
        svs_handle->fsmn_layers,
        svs_handle->fsmn_lorder,
        svs_handle->fsmn_dims,
        svs_handle->cif_threshold,
        svs_handle->tail_alphas);
    }
    InitCache();
}

void ParaformerOnline::InitOnline(
        knf::FbankOptions &fbank_opts,
        std::shared_ptr<Ort::Session> &encoder_session,
        std::shared_ptr<Ort::Session> &decoder_session,
        vector<const char*> &en_szInputNames,
        vector<const char*> &en_szOutputNames,
        vector<const char*> &de_szInputNames,
        vector<const char*> &de_szOutputNames,
        vector<float> &means_list,
        vector<float> &vars_list,
        int frame_length_,
        int frame_shift_,
        int n_mels_,
        int lfr_m_,
        int lfr_n_,
        int encoder_size_,
        int fsmn_layers_,
        int fsmn_lorder_,
        int fsmn_dims_,
        float cif_threshold_,
        float tail_alphas_){
    fbank_opts_ = fbank_opts;
    encoder_session_ = encoder_session;
    decoder_session_ = decoder_session;
    en_szInputNames_ = en_szInputNames;
    en_szOutputNames_ = en_szOutputNames;
    de_szInputNames_ = de_szInputNames;
    de_szOutputNames_ = de_szOutputNames;
    means_list_ = means_list;
    vars_list_ = vars_list;

    frame_length = frame_length_;
    frame_shift = frame_shift_;
    n_mels = n_mels_;
    lfr_m = lfr_m_;
    lfr_n = lfr_n_;
    encoder_size = encoder_size_;
    fsmn_layers = fsmn_layers_;
    fsmn_lorder = fsmn_lorder_;
    fsmn_dims = fsmn_dims_;
    cif_threshold = cif_threshold_;
    tail_alphas = tail_alphas_;

    // other vars
    sqrt_factor = std::sqrt(encoder_size);
    for(int i=0; i<fsmn_lorder*fsmn_dims; i++){
        fsmn_init_cache_.emplace_back(0);
    }
    chunk_len = chunk_size[1]*frame_shift*lfr_n*offline_handle_->GetAsrSampleRate()/1000;

    frame_sample_length_ = offline_handle_->GetAsrSampleRate() / 1000 * frame_length;
    frame_shift_sample_length_ = offline_handle_->GetAsrSampleRate() / 1000 * frame_shift;

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
            int reserve_frame_idx = std::abs(lfr_splice_frame_idxs - minus_frame);
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
            if(wav_feats.size() == 0){
                LOG(ERROR) << "wav_feats's size is 0";
            }else{
                OnlineLfrCmvn(wav_feats, input_finished);
            }
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
                p.clear();
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
    try{
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
        
        if (is_last_chunk) {
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
    }catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
    }
}

void ParaformerOnline::InitCache(){

    start_idx_cache_ = 0;
    is_first_chunk = true;
    is_last_chunk = false;
    hidden_cache_.clear();
    alphas_cache_.clear();
    feats_cache_.clear();
    decoder_onnx.clear();

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
#ifdef _WIN_X86
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
#else
    Ort::MemoryInfo m_memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    const int64_t fsmn_shape_[3] = {1, fsmn_dims, fsmn_lorder};
    for(int l=0; l<fsmn_layers; l++){
        Ort::Value onnx_fsmn_cache = Ort::Value::CreateTensor<float>(
            m_memoryInfo,
            fsmn_init_cache_.data(),
            fsmn_init_cache_.size(),
            fsmn_shape_,
            3);
        decoder_onnx.emplace_back(std::move(onnx_fsmn_cache));
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
    string result;
    try{
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
        
        auto encoder_tensor = encoder_session_->Run(Ort::RunOptions{nullptr}, en_szInputNames_.data(), input_onnx.data(), input_onnx.size(), en_szOutputNames_.data(), en_szOutputNames_.size());

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

        
        if(list_frame.size()>0){
            // enc
            decoder_onnx.insert(decoder_onnx.begin(), std::move(encoder_tensor[0]));
            // enc_lens
            decoder_onnx.insert(decoder_onnx.begin()+1, std::move(encoder_tensor[1]));

            // acoustic_embeds
            const int64_t emb_shape_[3] = {1, (int64_t)list_frame.size(), (int64_t)list_frame[0].size()};
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
            decoder_onnx.insert(decoder_onnx.begin()+2, std::move(onnx_emb));

            // acoustic_embeds_len
            const int64_t emb_length_shape[1] = {1};
            std::vector<int32_t> emb_length;
            emb_length.emplace_back(list_frame.size());
            Ort::Value onnx_emb_len = Ort::Value::CreateTensor<int32_t>(
                m_memoryInfo, emb_length.data(), emb_length.size(), emb_length_shape, 1);
            decoder_onnx.insert(decoder_onnx.begin()+3, std::move(onnx_emb_len));

            auto decoder_tensor = decoder_session_->Run(Ort::RunOptions{nullptr}, de_szInputNames_.data(), decoder_onnx.data(), decoder_onnx.size(), de_szOutputNames_.data(), de_szOutputNames_.size());
            // fsmn cache
            try{
                decoder_onnx.clear();
            }catch (std::exception const &e)
            {
                LOG(ERROR)<<e.what();
                return result;
            }
            for(int l=0;l<fsmn_layers;l++){
                decoder_onnx.emplace_back(std::move(decoder_tensor[2+l]));
            }

            std::vector<int64_t> decoder_shape = decoder_tensor[0].GetTensorTypeAndShapeInfo().GetShape();
            float* float_data = decoder_tensor[0].GetTensorMutableData<float>();
            result = offline_handle_->GreedySearch(float_data, list_frame.size(), decoder_shape[2]);
        }
    }catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
        return result;
    }
    return result;
}

string ParaformerOnline::Forward(float* din, int len, bool input_finished, const std::vector<std::vector<float>> &hw_emb, void* wfst_decoder)
{
    std::vector<std::vector<float>> wav_feats;
    std::vector<float> waves(din, din+len);

    string result="";
    try{
        if(len <16*60 && input_finished && !is_first_chunk){
            is_last_chunk = true;
            wav_feats = feats_cache_;
            result = ForwardChunk(wav_feats, is_last_chunk);
            // reset
            ResetCache();
            Reset();
            return result;
        }
        if(is_first_chunk){
            is_first_chunk = false;
        }
        ExtractFeats(offline_handle_->GetAsrSampleRate(), wav_feats, waves, input_finished);
        if(wav_feats.size() == 0){
            return result;
        }
        
        for (auto& row : wav_feats) {
            for (auto& val : row) {
                val *= sqrt_factor;
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
                first_chunk.insert(first_chunk.begin(), wav_feats.begin(), wav_feats.end());
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
    }catch (std::exception const &e)
    {
        LOG(ERROR)<<e.what();
        return result;
    }

    return result;
}

ParaformerOnline::~ParaformerOnline()
{
}

string ParaformerOnline::Rescoring()
{
    LOG(ERROR)<<"Not Imp!!!!!!";
    return "";
}
} // namespace funasr
