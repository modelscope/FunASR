
#ifndef PUNC_MODEL_H
#define PUNC_MODEL_H

#include <string>
#include <map>
#include <vector>
#include "funasrruntime.h"

namespace funasr {
class PuncModel {
  public:
    virtual ~PuncModel(){};
	  virtual void InitPunc(const std::string &punc_model, const std::string &punc_config, int thread_num)=0;
	  virtual std::string AddPunc(const char* sz_input){};
	  virtual std::string AddPunc(const char* sz_input, std::vector<std::string>& arr_cache){};
};

PuncModel *CreatePuncModel(std::map<std::string, std::string>& model_path, int thread_num, PUNC_TYPE type=PUNC_OFFLINE);
} // namespace funasr
#endif
