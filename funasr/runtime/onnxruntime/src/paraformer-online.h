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
        //std::unique_ptr<knf::OnlineFbank> fbank_;
        knf::FbankOptions fbank_opts_;

        Vocab* vocab = nullptr;
        vector<float> means_list_;
        vector<float> vars_list_;
        // const float scale = 22.6274169979695;
        const float scale = 1;
        int32_t lfr_window_size = 7;
        int32_t lfr_window_shift = 6;

        void LoadCmvn(const char *filename);
        string GreedySearch( float* in, int n_len, int64_t token_nums);

        std::shared_ptr<Ort::Session> encoder_session = nullptr;
        std::shared_ptr<Ort::Session> decoder_session = nullptr;
        Ort::Env env_;
        Ort::SessionOptions session_options;

        vector<string> en_strInputNames, en_strOutputNames;
        vector<const char*> en_szInputNames;
        vector<const char*> en_szOutputNames;

        vector<string> de_strInputNames, de_strOutputNames;
        vector<const char*> de_szInputNames;
        vector<const char*> de_szOutputNames;

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
        void LoadConfigFromYaml(const char* filename);

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
        std::vector<std::vector<std::vector<std::vector<float>>>> fsmn_caches_;

        bool is_first_chunk = true;
        bool is_last_chunk = false;
        
        // configs
        string window_type = "hamming";
        int frame_length = 25;
        int frame_shift = 10;
        int frame_sample_length_ = MODEL_SAMPLE_RATE / 1000 * frame_length;
        int frame_shift_sample_length_ = MODEL_SAMPLE_RATE / 1000 * frame_shift;
        int n_mels = 80;
        int lfr_m = PARA_LFR_M;
        int lfr_n = PARA_LFR_N;
        std::vector<int> chunk_size = {5,10,5};
        int encoder_size = 512;
        int fsmn_layers = 16;
        int fsmn_lorder = 10;
        int fsmn_dims = 512;
        int feat_dims = lfr_m*n_mels;
        float cif_threshold = 1.0;
        float tail_alphas = 0.45;

    public:
        ParaformerOnline();
        ~ParaformerOnline();
        void InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        void Reset();
        void ResetCache();
        void InitCache();
        void ExtractFeats(float sample_rate, vector<vector<float>> &wav_feats, vector<float> &waves, bool input_finished);
        void AddOverlapChunk(std::vector<std::vector<float>> &wav_feats, bool input_finished);
        
        string ForwardChunk(std::vector<std::vector<float>> &wav_feats, bool input_finished);
        string Forward(float* din, int len, bool input_finished);
        string Rescoring();
    };

} // namespace funasr
