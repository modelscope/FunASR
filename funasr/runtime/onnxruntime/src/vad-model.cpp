#include "precomp.h"

namespace funasr {
VadModel *CreateVadModel(std::map<std::string, std::string>& model_path, int thread_num)
{
    VadModel *mm;
    mm = new FsmnVad();

    string vad_model_path;
    string vad_cmvn_path;
    string vad_config_path;

    vad_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
    if(model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true"){
        vad_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
    }
    vad_cmvn_path = PathAppend(model_path.at(MODEL_DIR), VAD_CMVN_NAME);
    vad_config_path = PathAppend(model_path.at(MODEL_DIR), VAD_CONFIG_NAME);

    mm->InitVad(vad_model_path, vad_cmvn_path, vad_config_path, thread_num);
    return mm;
}

} // namespace funasr