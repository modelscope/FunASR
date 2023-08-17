/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once

#include "precomp.h"

namespace funasr {

    class Paraformer : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:
        Vocab* vocab = nullptr;
        //const float scale = 22.6274169979695;
        const float scale = 1.0;

        void LoadOnlineConfigFromYaml(const char* filename);
        void LoadCmvn(const char *filename);
        vector<float> ApplyLfr(const vector<float> &in);
        void ApplyCmvn(vector<float> *v);

    public:
        Paraformer();
        ~Paraformer();
        void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        // online
        void InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        // 2pass
        void InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, int thread_num);
        void Reset();
        vector<float> FbankKaldi(float sample_rate, const float* waves, int len);
        string Forward(float* din, int len, bool input_finished=true);
        string GreedySearch( float* in, int n_len, int64_t token_nums, bool is_stamp=false, std::vector<float> us_alphas={0}, std::vector<float> us_cif_peak={0});
        void TimestampOnnx(std::vector<float> &us_alphas, vector<float> us_cif_peak, vector<string>& char_list, std::string &res_str, 
                           vector<vector<float>> &timestamp_list, float begin_time = 0.0, float total_offset = -1.5);
        string PostProcess(std::vector<string> &raw_char, std::vector<std::vector<float>> &timestamp_list);
        string Rescoring();

        knf::FbankOptions fbank_opts_;
        vector<float> means_list_;
        vector<float> vars_list_;
        int lfr_m = PARA_LFR_M;
        int lfr_n = PARA_LFR_N;

        // paraformer-offline
        std::shared_ptr<Ort::Session> m_session_ = nullptr;
        Ort::Env env_;
        Ort::SessionOptions session_options_;

        vector<string> m_strInputNames, m_strOutputNames;
        vector<const char*> m_szInputNames;
        vector<const char*> m_szOutputNames;

        // paraformer-online
        std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
        std::shared_ptr<Ort::Session> decoder_session_ = nullptr;
        vector<string> en_strInputNames, en_strOutputNames;
        vector<const char*> en_szInputNames_;
        vector<const char*> en_szOutputNames_;
        vector<string> de_strInputNames, de_strOutputNames;
        vector<const char*> de_szInputNames_;
        vector<const char*> de_szOutputNames_;
        
        string window_type = "hamming";
        int frame_length = 25;
        int frame_shift = 10;
        int n_mels = 80;
        int encoder_size = 512;
        int fsmn_layers = 16;
        int fsmn_lorder = 10;
        int fsmn_dims = 512;
        float cif_threshold = 1.0;
        float tail_alphas = 0.45;


    };

} // namespace funasr
