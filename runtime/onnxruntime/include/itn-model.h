
#ifndef ITN_MODEL_H
#define ITN_MODEL_H

#include <string>
#include <map>
#include <vector>

namespace funasr {
class ITNModel {
  public:
    virtual ~ITNModel(){};
    virtual void InitITN(const std::string &itn_tagger, const std::string &itn_verbalizer, int thread_num)=0;
    virtual std::string Normalize(const std::string& input){return "";};
};

ITNModel *CreateITNModel(std::map<std::string, std::string>& model_path, int thread_num);

} // namespace funasr
#endif
