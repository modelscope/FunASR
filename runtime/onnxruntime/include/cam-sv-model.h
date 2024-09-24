
#pragma  once

#include <string>
#include <map>
#include <vector>
#include "com-define.h"

namespace funasr {
class SvModel {
  public:
    virtual ~SvModel(){};    
    virtual void InitSv(const std::string &model, const std::string &config, int thread_num)=0;
    virtual std::vector<std::vector<float>> Infer(sv_segment vad_seg)=0;
    float threshold=0.40;
};

SvModel *CreateSVModel(std::map<std::string, std::string>& model_path, int thread_num);
} // namespace funasr
