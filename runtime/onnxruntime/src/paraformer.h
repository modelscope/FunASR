/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once

#include "precomp.h"
#include "fst/fstlib.h"
#include "fst/symbol-table.h"
#include "bias-lm.h"
#include "phone-set.h"

namespace funasr {

    class Paraformer : public Model {
    /**
     * Author: Speech Lab of DAMO Academy, Alibaba Group
     * Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
     * https://arxiv.org/pdf/2206.08317.pdf
    */
    private:
        Vocab* vocab = nullptr;
        Vocab* lm_vocab = nullptr;
        SegDict* seg_dict = nullptr;
        PhoneSet* phone_set_ = nullptr;
        //const float scale = 22.6274169979695;
        const float scale = 1.0;

        void LoadConfigFromYaml(const char* filename);
        void LoadOnlineConfigFromYaml(const char* filename);
        void LoadCmvn(const char *filename);
        void LfrCmvn(std::vector<std::vector<float>> &asr_feats);

        std::shared_ptr<Ort::Session> hw_m_session = nullptr;
        Ort::Env hw_env_;
        Ort::SessionOptions hw_session_options;
        vector<string> hw_m_strInputNames, hw_m_strOutputNames;
        vector<const char*> hw_m_szInputNames;
        vector<const char*> hw_m_szOutputNames;
        bool use_hotword;

    public:
        Paraformer();
        ~Paraformer();
        void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num);
        // online
        void InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num);
        // 2pass
        void InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num);
        void InitHwCompiler(const std::string &hw_model, int thread_num);
        void InitSegDict(const std::string &seg_dict_model);
        std::vector<std::vector<float>> CompileHotwordEmbedding(std::string &hotwords);
        void Reset();
        void FbankKaldi(float sample_rate, const float* waves, int len, std::vector<std::vector<float>> &asr_feats);
        std::vector<std::string> Forward(float** din, int* len, bool input_finished=true, const std::vector<std::vector<float>> &hw_emb={{0.0}}, void* wfst_decoder=nullptr, int batch_in=1);
        string GreedySearch( float* in, int n_len, int64_t token_nums,
                             bool is_stamp=false, std::vector<float> us_alphas={0}, std::vector<float> us_cif_peak={0});

        string Rescoring();
        string GetLang(){return language;};
        int GetAsrSampleRate() { return asr_sample_rate; };
        int GetBatchSize() {return batch_size_;};
        void StartUtterance();
        void EndUtterance();
        void InitLm(const std::string &lm_file, const std::string &lm_cfg_file, const std::string &lex_file);
        string BeamSearch(WfstDecoder* &wfst_decoder, float* in, int n_len, int64_t token_nums);
        string FinalizeDecode(WfstDecoder* &wfst_decoder,
                          bool is_stamp=false, std::vector<float> us_alphas={0}, std::vector<float> us_cif_peak={0});
        Vocab* GetVocab();
        Vocab* GetLmVocab();
        PhoneSet* GetPhoneSet();
		
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

        std::string language="zh-cn";

        // paraformer-online
        std::shared_ptr<Ort::Session> encoder_session_ = nullptr;
        std::shared_ptr<Ort::Session> decoder_session_ = nullptr;
        vector<string> en_strInputNames, en_strOutputNames;
        vector<const char*> en_szInputNames_;
        vector<const char*> en_szOutputNames_;
        vector<string> de_strInputNames, de_strOutputNames;
        vector<const char*> de_szInputNames_;
        vector<const char*> de_szOutputNames_;

        // lm
        std::shared_ptr<fst::Fst<fst::StdArc>> lm_ = nullptr;

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
        int asr_sample_rate = MODEL_SAMPLE_RATE;
        int batch_size_ = 1;
    };

} // namespace funasr
