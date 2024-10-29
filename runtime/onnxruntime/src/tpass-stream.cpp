#include "precomp.h"

namespace funasr {
TpassStream::TpassStream(std::map<std::string, std::string>& model_path, int thread_num)
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
    if(model_path.find(OFFLINE_MODEL_DIR) != model_path.end() && model_path.find(ONLINE_MODEL_DIR) != model_path.end()){
        // 2pass
        string am_model_path;
        string en_model_path;
        string de_model_path;
        string am_cmvn_path;
        string am_config_path;
        string token_path;
        string online_token_path;
        string hw_compile_model_path;
        string seg_dict_path;
        
        if (model_path.at(MODEL_DIR).find(MODEL_SVS) != std::string::npos)
        {
            asr_handle = make_unique<SenseVoiceSmall>();
            model_type = MODEL_SVS;
        }else{
            asr_handle = make_unique<Paraformer>();
        }

        bool enable_hotword = false;
        hw_compile_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_EB_NAME);
        seg_dict_path = PathAppend(model_path.at(MODEL_DIR), MODEL_SEG_DICT);
        if ((access(hw_compile_model_path.c_str(), F_OK) == 0) && 
            (access(seg_dict_path.c_str(), F_OK) == 0)) { // if model_eb.onnx exist, hotword enabled
          enable_hotword = true;
          asr_handle->InitHwCompiler(hw_compile_model_path, thread_num);
          asr_handle->InitSegDict(seg_dict_path);
        }

        am_model_path = PathAppend(model_path.at(OFFLINE_MODEL_DIR), MODEL_NAME);
        en_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), ENCODER_NAME);
        de_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), DECODER_NAME);
        online_token_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), TOKEN_PATH);
        if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
            am_model_path = PathAppend(model_path.at(OFFLINE_MODEL_DIR), QUANT_MODEL_NAME);
            en_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), QUANT_ENCODER_NAME);
            de_model_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), QUANT_DECODER_NAME);
        }
        am_cmvn_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), AM_CMVN_NAME);
        am_config_path = PathAppend(model_path.at(ONLINE_MODEL_DIR), AM_CONFIG_NAME);
        token_path = PathAppend(model_path.at(MODEL_DIR), TOKEN_PATH);

        asr_handle->InitAsr(am_model_path, en_model_path, de_model_path, am_cmvn_path, am_config_path, token_path, online_token_path, thread_num);
    }else{
        LOG(ERROR) <<"Can not find offline-model-dir or online-model-dir";
        exit(-1);
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
            punc_online_handle = make_unique<CTTransformerOnline>();
            punc_online_handle->InitPunc(punc_model_path, punc_config_path, token_path, thread_num);
            use_punc = true;
        }
    }
#if !defined(__APPLE__)
    // Optional: ITN, here we just support language_type=MandarinEnglish
    if(model_path.find(ITN_DIR) != model_path.end()){
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
      
}

TpassStream *CreateTpassStream(std::map<std::string, std::string>& model_path, int thread_num)
{
    TpassStream *mm;
    mm = new TpassStream(model_path, thread_num);
    return mm;
}
} // namespace funasr
