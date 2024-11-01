/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once

#include "precomp.h"

namespace funasr {

    class ParaformerOnline : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * ParaformerOnline: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:

        void FbankKaldi(float sample_rate, std::vector<std::vector<float>> &wav_feats,
                std::vector<float> &waves);
        int OnlineLfrCmvn(vector<vector<float>> &wav_feats, bool input_finished);
        void GetPosEmb(std::vector<std::vector<float>> &wav_feats, int timesteps, int feat_dim);
        void CifSearch(std::vector<std::vector<float>> hidden, std::vector<float> alphas, bool is_final, std::vector<std::vector<float>> &list_frame);

        static int ComputeFrameNum(int sample_length, int frame_sample_length, int frame_shift_sample_length) {
            int frame_num = static_cast<int>((sample_length - frame_sample_length) / frame_shift_sample_length + 1);
            if (frame_num >= 1 && sample_length >= frame_sample_length)
                return frame_num;
            else
                return 0;
        }
        void InitOnline(
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
            float tail_alphas_);

        void StartUtterance()
        {
        }
        
        void EndUtterance()
        {
        }
        
        Model* offline_handle_ = nullptr;
        // from offline_handle_
        knf::FbankOptions fbank_opts_;
        std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
        std::shared_ptr<Ort::Session> decoder_session_ = nullptr;
        Ort::SessionOptions session_options_;
        vector<const char*> en_szInputNames_;
        vector<const char*> en_szOutputNames_;
        vector<const char*> de_szInputNames_;
        vector<const char*> de_szOutputNames_;
        vector<float> means_list_;
        vector<float> vars_list_;
        // configs from offline_handle_
        int frame_length = 25;
        int frame_shift = 10;
        int n_mels = 80;
        int lfr_m = PARA_LFR_M;
        int lfr_n = PARA_LFR_N;
        int encoder_size = 512;
        int fsmn_layers = 16;
        int fsmn_lorder = 10;
        int fsmn_dims = 512;
        float cif_threshold = 1.0;
        float tail_alphas = 0.45;

        // configs
        int feat_dims = lfr_m*n_mels;
        std::vector<int> chunk_size = {5,10,5};        
        int frame_sample_length_ = MODEL_SAMPLE_RATE / 1000 * frame_length;
        int frame_shift_sample_length_ = MODEL_SAMPLE_RATE / 1000 * frame_shift;

        // The reserved waveforms by fbank
        std::vector<float> reserve_waveforms_;
        // waveforms reserved after last shift position
        std::vector<float> input_cache_;
        // lfr reserved cache
        std::vector<std::vector<float>> lfr_splice_cache_;
        // position index cache
        int start_idx_cache_ = 0;
        // cif alpha
        std::vector<float> alphas_cache_;
        std::vector<std::vector<float>> hidden_cache_;
        std::vector<std::vector<float>> feats_cache_;
        // fsmn init caches
        std::vector<float> fsmn_init_cache_;
        std::vector<Ort::Value> decoder_onnx;

        bool is_first_chunk = true;
        bool is_last_chunk = false;
        double sqrt_factor;

    public:
        ParaformerOnline(Model* offline_handle, std::vector<int> chunk_size, std::string model_type=MODEL_PARA);
        ~ParaformerOnline();
        void Reset();
        void ResetCache();
        void InitCache();
        void ExtractFeats(float sample_rate, vector<vector<float>> &wav_feats, vector<float> &waves, bool input_finished);
        void AddOverlapChunk(std::vector<std::vector<float>> &wav_feats, bool input_finished);
        
        string ForwardChunk(std::vector<std::vector<float>> &wav_feats, bool input_finished);
        string Forward(float* din, int len, bool input_finished, const std::vector<std::vector<float>> &hw_emb={{0.0}}, void* wfst_decoder=nullptr);
        string Rescoring();

        int GetAsrSampleRate() { return offline_handle_->GetAsrSampleRate(); };

        // 2pass
        std::string online_res;
        int chunk_len;
    };

} // namespace funasr
