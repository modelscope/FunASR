
#ifndef MODEL_H
#define MODEL_H

#include <string>
#include <map>
#include "funasrruntime.h"
#include "vocab.h"
#include "phone-set.h"
#include "fst/fstlib.h"
#include "fst/symbol-table.h"
namespace funasr {
class Model {
  public:
    virtual ~Model(){};
    virtual void StartUtterance() = 0;
    virtual void EndUtterance() = 0;
    virtual void Reset() = 0;
    virtual string GreedySearch(float* in, int n_len, int64_t token_nums, bool is_stamp=false, std::vector<float> us_alphas={0}, std::vector<float> us_cif_peak={0}){return "";};
    virtual void InitAsr(const std::string &am_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){};
    virtual void InitAsr(const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, const std::string &am_config, const std::string &token_file, int thread_num){};
    virtual void InitAsr(const std::string &am_model, const std::string &en_model, const std::string &de_model, const std::string &am_cmvn, 
      const std::string &am_config, const std::string &token_file, const std::string &online_token_file, int thread_num){};
    virtual void InitLm(const std::string &lm_file, const std::string &lm_config, const std::string &lex_file){};
    virtual void InitFstDecoder(){};
    virtual std::string Forward(float *din, int len, bool input_finished, const std::vector<std::vector<float>> &hw_emb={{0.0}}, void* wfst_decoder=nullptr){return "";};
    virtual std::vector<std::string> Forward(float** din, int* len, bool input_finished, const std::vector<std::vector<float>> &hw_emb={{0.0}}, void* wfst_decoder=nullptr, int batch_in=1)
      {return std::vector<string>();};
    virtual std::vector<std::string> Forward(float** din, int* len, bool input_finished, std::string svs_lang="auto", bool svs_itn=false, int batch_in=1)
      {return std::vector<string>();};
    virtual std::string Rescoring() = 0;
    virtual void InitHwCompiler(const std::string &hw_model, int thread_num){};
    virtual void InitSegDict(const std::string &seg_dict_model){};
    virtual std::vector<std::vector<float>> CompileHotwordEmbedding(std::string &hotwords){return std::vector<std::vector<float>>();};
    virtual std::string GetLang(){return "";};
    virtual int GetAsrSampleRate() = 0;
    virtual void SetBatchSize(int batch_size) {};
    virtual int GetBatchSize() {return 0;};
    virtual Vocab* GetVocab() {return nullptr;};
    virtual Vocab* GetLmVocab() {return nullptr;};
    virtual PhoneSet* GetPhoneSet() {return nullptr;};
};

Model *CreateModel(std::map<std::string, std::string>& model_path, int thread_num=1, ASR_TYPE type=ASR_OFFLINE);
Model *CreateModel(void* asr_handle, std::vector<int> chunk_size);

} // namespace funasr
#endif
