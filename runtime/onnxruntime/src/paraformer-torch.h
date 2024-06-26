/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
 * MIT License  (https://opensource.org/licenses/MIT)
*/
#pragma once
#define C10_USE_GLOG
#include <torch/serialize.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torch/csrc/jit/passes/tensorexpr_fuser.h>
#include "precomp.h"
#include "fst/fstlib.h"
#include "fst/symbol-table.h"
#include "bias-lm.h"
#include "phone-set.h"

namespace funasr {

    class ParaformerTorch : public Model {
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
        void LoadCmvn(const char *filename);
        void LfrCmvn(std::vector<std::vector<float>> &asr_feats);

        using TorchModule = torch::jit::script::Module;
        std::shared_ptr<TorchModule> model_ = nullptr;
        std::shared_ptr<TorchModule> hw_model_ = nullptr;
        std::vector<torch::Tensor> encoder_outs_;
        bool use_hotword;

    public:
        ParaformerTorch();
        ~ParaformerTorch();
        void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num);
        void InitHwCompiler(const std::string &hw_model, int thread_num);
        void InitSegDict(const std::string &seg_dict_model);
        std::vector<std::vector<float>> CompileHotwordEmbedding(std::string &hotwords);
        void Reset();
        void FbankKaldi(float sample_rate, const float* waves, int len, std::vector<std::vector<float>> &asr_feats);
        void WarmUp();
        std::vector<std::string> Forward(float** din, int* len, bool input_finished=true, const std::vector<std::vector<float>> &hw_emb={{0.0}}, void* wfst_decoder=nullptr, int batch_in=1);
        string GreedySearch( float* in, int n_len, int64_t token_nums,
                             bool is_stamp=false, std::vector<float> us_alphas={0}, std::vector<float> us_cif_peak={0});

        string Rescoring();
        string GetLang(){return language;};
        int GetAsrSampleRate() { return asr_sample_rate; };
        void SetBatchSize(int batch_size) {batch_size_ = batch_size;};
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
        std::string language="zh-cn";

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
