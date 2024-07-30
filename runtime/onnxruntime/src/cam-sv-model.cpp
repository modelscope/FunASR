#include "precomp.h"

namespace funasr
{

    SvModel *CreateSVModel(std::map<std::string, std::string> &model_path, int thread_num)
    {
        SvModel *mm;
        mm = new CamPPlusSv();

        string vad_model_path;
        string vad_cmvn_path;
        string vad_config_path;

        vad_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
        if (model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true")
        {
            vad_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
        }
        vad_cmvn_path = PathAppend(model_path.at(MODEL_DIR), SV_CMVN_NAME);
        vad_config_path = PathAppend(model_path.at(MODEL_DIR), SV_CONFIG_NAME);

        mm->InitSv(vad_model_path, vad_cmvn_path, vad_config_path, thread_num);
        return mm;
    }

    // SvModel *CreateAndInferSvModel(std::map<std::string, std::string> &model_path, int thread_num, std::vector<float> wave)
    SvModel *CreateAndInferSvModel(std::map<std::string, std::string> &model_path, int thread_num)
    {
        SvModel *mm;
        mm = new CamPPlusSv();

        string vad_model_path;
        string vad_cmvn_path;
        string vad_config_path;

        vad_model_path = PathAppend(model_path.at(SV_DIR), MODEL_NAME);
        if (model_path.find(SV_QUANT) != model_path.end() && model_path.at(SV_QUANT) == "true")
        {
            vad_model_path = PathAppend(model_path.at(SV_DIR), QUANT_MODEL_NAME);
        }
        vad_cmvn_path = PathAppend(model_path.at(SV_DIR), SV_CMVN_NAME);
        vad_config_path = PathAppend(model_path.at(SV_DIR), SV_CONFIG_NAME);

        mm->InitSv(vad_model_path, vad_cmvn_path, vad_config_path, thread_num);

        return mm;
    }

    // std::vector<std::vector<float>> InferSvModel(std::map<std::string, std::string> &model_path, int thread_num, std::vector<float> wave)
    // {
    //     SvModel *mm;
    //     mm = new CamPPlusSv();

    //     string vad_model_path;
    //     string vad_cmvn_path;
    //     string vad_config_path;

    //     vad_model_path = PathAppend(model_path.at(MODEL_DIR), MODEL_NAME);
    //     if (model_path.find(QUANTIZE) != model_path.end() && model_path.at(QUANTIZE) == "true")
    //     {
    //         vad_model_path = PathAppend(model_path.at(MODEL_DIR), QUANT_MODEL_NAME);
    //     }
    //     vad_cmvn_path = PathAppend(model_path.at(MODEL_DIR), SV_CMVN_NAME);
    //     vad_config_path = PathAppend(model_path.at(MODEL_DIR), SV_CONFIG_NAME);

    //     mm->InitSv(vad_model_path, vad_cmvn_path, vad_config_path, thread_num);
    //     std::vector<std::vector<float>> result = mm->Infer(wave);
    //     delete mm;
    //     return result;
    // }

} // namespace funasr