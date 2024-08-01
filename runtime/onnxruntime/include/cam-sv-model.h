
#pragma  once

#include <string>
#include <map>
#include <vector>

namespace funasr {
class SvModel {
  public:
    virtual ~SvModel(){};    
    virtual void InitSv(const std::string &model, const std::string &config, int thread_num)=0;
    virtual std::vector<std::vector<float>> Infer(std::vector<float> &waves)=0;
    float threshold=0.40;
};

SvModel *CreateSVModel(std::map<std::string, std::string>& model_path, int thread_num);
} // namespace funasr
