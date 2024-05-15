#include "precomp.h"

namespace funasr {
PuncModel *CreatePuncModel(std::map<std::string, std::string>& model_path, int thread_num, PUNC_TYPE type)
{
    PuncModel *mm;
    if (type==PUNC_OFFLINE){
        mm = new CTTransformer();
    }else if(type==PUNC_ONLINE){
        mm = new CTTransformerOnline();
    }else{
        LOG(ERROR) << "Wrong PUNC TYPE";
        exit(-1);
    }
    string punc_model_path;
    string punc_config_path;
    string token_file;

    punc_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
    if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
        punc_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
    }
    punc_config_path = PathAppend(model_path.at(MODEL_DIR), PUNC_CONFIG_NAME);
    token_file = PathAppend(model_path.at(MODEL_DIR), TOKEN_PATH);

    mm->InitPunc(punc_model_path, punc_config_path, token_file, thread_num);
    return mm;
}

} // namespace funasr
