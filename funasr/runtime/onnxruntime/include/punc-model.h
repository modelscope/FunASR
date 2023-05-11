
#ifndef PUNC_MODEL_H
#define PUNC_MODEL_H

#include <string>
#include <map>
#include <vector>

namespace funasr {
class PuncModel {
  public:
    virtual ~PuncModel(){};
	  virtual void InitPunc(const std::string &punc_model, const std::string &punc_config, int thread_num)=0;
	  virtual std::vector<int>  Infer(std::vector<int32_t> input_data)=0;
	  virtual std::string AddPunc(const char* sz_input)=0;
};

PuncModel *CreatePuncModel(std::map<std::string, std::string>& model_path, int thread_num);
} // namespace funasr
#endif
