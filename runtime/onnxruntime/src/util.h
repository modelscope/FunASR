#ifndef UTIL_H
#define UTIL_H
#include <vector>
#include <memory>
#include <unordered_map>
#include <deque>
#include "tensor.h"

using namespace std;

namespace funasr {
typedef unsigned short          U16CHAR_T;
extern float *LoadParams(const char *filename);

extern void SaveDataFile(const char *filename, void *data, uint32_t len);
extern void Relu(Tensor<float> *din);
extern void Swish(Tensor<float> *din);
extern void Sigmoid(Tensor<float> *din);
extern void DoubleSwish(Tensor<float> *din);

extern void Softmax(float *din, int mask, int len);

extern void LogSoftmax(float *din, int len);
extern int ValAlign(int val, int align);
extern void DispParams(float *din, int size);

extern void BasicNorm(Tensor<float> *&din, float norm);

extern void FindMax(float *din, int len, float &max_val, int &max_idx);

extern void Glu(Tensor<float> *din, Tensor<float> *dout);

string PathAppend(const string &p1, const string &p2);
bool is_target_file(const std::string& filename, const std::string target);

void KeepChineseCharacterAndSplit(const std::string &input_str,
                                  std::vector<std::string> &chinese_characters);
void SplitChiEngCharacters(const std::string &input_str,
                                  std::vector<std::string> &characters);
void TimestampAdd(std::deque<string> &alignment_str1, std::string str_word);
vector<vector<int>> ParseTimestamps(const std::string& str);
bool TimestampIsDigit(U16CHAR_T &u16);
bool TimestampIsAlpha(U16CHAR_T &u16);
bool TimestampIsPunctuation(U16CHAR_T &u16);
bool TimestampIsPunctuation(const std::string& str);
void TimestampSplitChiEngCharacters(const std::string &input_str,
                                  std::vector<std::string> &characters);
std::string VectorToString(const std::vector<std::vector<int>>& vec, bool out_empty=true);                                  
std::string TimestampSmooth(std::string &text, std::string &text_itn, std::string &str_time);
std::string TimestampSentence(std::string &text, std::string &str_time);
std::vector<std::string> split(const std::string &s, char delim);
std::vector<std::string> SplitStr(const std::string &s, string delimiter);

template<typename T>
void PrintMat(const std::vector<std::vector<T>> &mat, const std::string &name);
void Trim(std::string *str);
size_t Utf8ToCharset(const std::string &input, std::vector<std::string> &output);
void SplitStringToVector(const std::string &full, const char *delim,
                         bool omit_empty_strings,
                         std::vector<std::string> *out);
string PostProcess(std::vector<string> &raw_char,
                   std::vector<std::vector<float>> &timestamp_list);
void TimestampOnnx( std::vector<float>& us_alphas,
                    std::vector<float> us_cif_peak, 
                    std::vector<string>& char_list, 
                    std::string &res_str, 
                    std::vector<std::vector<float>> &timestamp_vec, 
                    float begin_time = 0.0, 
                    float total_offset = -1.5);
bool IsTargetFile(const std::string& filename, const std::string target);
void ExtractHws(string hws_file, unordered_map<string, int> &hws_map);
void ExtractHws(string hws_file, unordered_map<string, int> &hws_map, string& nn_hotwords_);
} // namespace funasr
#endif
