#include "precomp.h"

namespace funasr {
OfflineStream::OfflineStream(std::map<std::string, std::string>& model_path, int thread_num, bool use_gpu, int batch_size)
{
    // VAD model
    if(model_path.find(VAD_DIR) != model_path.end()){
        string vad_model_path;
        string vad_cmvn_path;
        string vad_config_path;
    
        vad_model_path = PathAppend(model_path.at(VAD_DIR), MODEL_NAME);
        if(model_path.find(VAD_QUANT) != model_path.end() && model_path.at(VAD_QUANT) == "true"){
            vad_model_path = PathAppend(model_path.at(VAD_DIR), QUANT_MODEL_NAME);
        }
        vad_cmvn_path = PathAppend(model_path.at(VAD_DIR), VAD_CMVN_NAME);
        vad_config_path = PathAppend(model_path.at(VAD_DIR), VAD_CONFIG_NAME);
        if (access(vad_model_path.c_str(), F_OK) != 0 ||
            access(vad_cmvn_path.c_str(), F_OK) != 0 ||
            access(vad_config_path.c_str(), F_OK) != 0 )
        {
            LOG(INFO) << "VAD model file is not exist, skip load vad model.";
        }else{
            vad_handle = make_unique<FsmnVad>();
            vad_handle->InitVad(vad_model_path, vad_cmvn_path, vad_config_path, thread_num);
            use_vad = true;
        }
    }

    // AM model
    if(model_path.find(MODEL_DIR) != model_path.end()){
        string am_model_path;
        string am_cmvn_path;
        string am_config_path;
        string token_path;
        string hw_cpu_model_path;
        string hw_gpu_model_path;
        string seg_dict_path;
    
        if(use_gpu){
            #ifdef USE_GPU
            asr_handle = make_unique<ParaformerTorch>();
            asr_handle->SetBatchSize(batch_size);
            #else
            LOG(ERROR) <<"GPU is not supported! CPU will be used! If you want to use GPU, please add -DGPU=ON when cmake";
            asr_handle = make_unique<Paraformer>();
            use_gpu = false;
            #endif
        }else{
            if (model_path.at(MODEL_DIR).find(MODEL_SVS) != std::string::npos)
            {
                asr_handle = make_unique<SenseVoiceSmall>();
                model_type = MODEL_SVS;
            }else{
                asr_handle = make_unique<Paraformer>();
            }
        }

        bool enable_hotword = false;
        hw_cpu_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_EB_NAME);
        hw_gpu_model_path = PathAppend(model_path.at(MODEL_DIR), TORCH_MODEL_EB_NAME);
        seg_dict_path = PathAppend(model_path.at(MODEL_DIR), MODEL_SEG_DICT);
        if (access(hw_cpu_model_path.c_str(), F_OK) == 0) { // if model_eb.onnx exist, hotword enabled
          enable_hotword = true;
          asr_handle->InitHwCompiler(hw_cpu_model_path, thread_num);
          asr_handle->InitSegDict(seg_dict_path);
        }
        if (use_gpu && access(hw_gpu_model_path.c_str(), F_OK) == 0) { // if model_eb.torchscript exist, hotword enabled
          enable_hotword = true;
          asr_handle->InitHwCompiler(hw_gpu_model_path, thread_num);
          asr_handle->InitSegDict(seg_dict_path);
        }

        am_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
        if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
            am_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
        }
        if(use_gpu){
            am_model_path = PathAppend(model_path.at(MODEL_DIR), TORCH_MODEL_NAME);
            if(model_path.find(BLADEDISC) != model_path.end() && model_path.at(BLADEDISC) == "true"){
                am_model_path = PathAppend(model_path.at(MODEL_DIR), BLADE_MODEL_NAME);
            }
        }

        am_cmvn_path = PathAppend(model_path.at(MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(MODEL_DIR), AM_CONFIG_NAME);
        token_path = PathAppend(model_path.at(MODEL_DIR), TOKEN_PATH);

        asr_handle->InitAsr(am_model_path, am_cmvn_path, am_config_path, token_path, thread_num);
    }

    // Lm resource
    if (model_path.find(LM_DIR) != model_path.end() && model_path.at(LM_DIR) != "") {
        string fst_path, lm_config_path, lex_path;
        fst_path = PathAppend(model_path.at(LM_DIR), LM_FST_RES);
        lm_config_path = PathAppend(model_path.at(LM_DIR), LM_CONFIG_NAME);
        lex_path = PathAppend(model_path.at(LM_DIR), LEX_PATH);
        if (access(lex_path.c_str(), F_OK) != 0 )
        {
            LOG(ERROR) << "Lexicon.txt file is not exist, please use the latest version. Skip load LM model.";
        }else{
            asr_handle->InitLm(fst_path, lm_config_path, lex_path);
        }
    }

    // PUNC model
    if(model_path.find(PUNC_DIR) != model_path.end()){
        string punc_model_path;
        string punc_config_path;
        string token_path;
    
        punc_model_path = PathAppend(model_path.at(PUNC_DIR), MODEL_NAME);
        if(model_path.find(PUNC_QUANT) != model_path.end() && model_path.at(PUNC_QUANT) == "true"){
            punc_model_path = PathAppend(model_path.at(PUNC_DIR), QUANT_MODEL_NAME);
        }
        punc_config_path = PathAppend(model_path.at(PUNC_DIR), PUNC_CONFIG_NAME);
        token_path = PathAppend(model_path.at(PUNC_DIR), TOKEN_PATH);

        if (access(punc_model_path.c_str(), F_OK) != 0 ||
            access(punc_config_path.c_str(), F_OK) != 0 ||
            access(token_path.c_str(), F_OK) != 0)
        {
            LOG(INFO) << "PUNC model file is not exist, skip load punc model.";
        }else{
            punc_handle = make_unique<CTTransformer>();
            punc_handle->InitPunc(punc_model_path, punc_config_path, token_path, thread_num);
            use_punc = true;
        }
    }
#if !defined(__APPLE__)
    // Optional: ITN, here we just support language_type=MandarinEnglish
    if(model_path.find(ITN_DIR) != model_path.end() && model_path.at(ITN_DIR) != ""){
        string itn_tagger_path = PathAppend(model_path.at(ITN_DIR), ITN_TAGGER_NAME);
        string itn_verbalizer_path = PathAppend(model_path.at(ITN_DIR), ITN_VERBALIZER_NAME);

        if (access(itn_tagger_path.c_str(), F_OK) != 0 ||
            access(itn_verbalizer_path.c_str(), F_OK) != 0 )
        {
            LOG(INFO) << "ITN model file is not exist, skip load ITN model.";
        }else{
            itn_handle = make_unique<ITNProcessor>();
            itn_handle->InitITN(itn_tagger_path, itn_verbalizer_path, thread_num);
            use_itn = true;
        }
    }
#endif
    if(model_type == MODEL_SVS){
        use_itn = false;
        use_punc = false;
    }
}

OfflineStream *CreateOfflineStream(std::map<std::string, std::string>& model_path, int thread_num, bool use_gpu, int batch_size)
{
    OfflineStream *mm;
    mm = new OfflineStream(model_path, thread_num, use_gpu, batch_size);
    return mm;
}

} // namespace funasr
