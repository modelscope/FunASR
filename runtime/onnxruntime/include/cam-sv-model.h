
#pragma  once

#include <string>
#include <map>
#include <vector>

// #define SV_CMVN_NAME "sv.mvn"
// #define SV_CONFIG_NAME "config.yaml"
// #define SV_MODEL_DIR "sv-dir"
// #define SV_QUANT "sv_quant"
namespace funasr {
class SvModel {
  public:
    virtual ~SvModel(){};
    virtual void InitSv(const std::string &model, const std::string &cmvn, const std::string &config, int thread_num)=0;
    virtual std::vector<std::vector<float>> Infer(std::vector<float> &waves)=0;
    float threshold=0.45;
};

SvModel *CreateSvModel(std::map<std::string, std::string>& model_path, int thread_num);

SvModel *CreateAndInferSvModel(std::map<std::string, std::string>& model_path, int thread_num);
// std::vector<std::vector<float>> InferSvModel(std::map<std::string, std::string>& model_path, int thread_num, std::vector<float>wave);

} // namespace funasr
