
#include "precomp.h"

namespace funasr {
float *LoadParams(const char *filename)
{

    FILE *fp;
    fp = fopen(filename, "rb");
    fseek(fp, 0, SEEK_END);
    uint32_t nFileLen = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    float *params_addr = (float *)AlignedMalloc(32, nFileLen);
    int n = fread(params_addr, 1, nFileLen, fp);
    fclose(fp);

    return params_addr;
}

int ValAlign(int val, int align)
{
    float tmp = ceil((float)val / (float)align) * (float)align;
    return (int)tmp;
}

void DispParams(float *din, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        printf("%f ", din[i]);
    }
    printf("\n");
}
void SaveDataFile(const char *filename, void *data, uint32_t len)
{
    FILE *fp;
    fp = fopen(filename, "wb+");
    fwrite(data, 1, len, fp);
    fclose(fp);
}

void BasicNorm(Tensor<float> *&din, float norm)
{

    int Tmax = din->size[2];

    int i, j;
    for (i = 0; i < Tmax; i++) {
        float sum = 0;
        for (j = 0; j < 512; j++) {
            int ii = i * 512 + j;
            sum += din->buff[ii] * din->buff[ii];
        }
        float mean = sqrt(sum / 512 + norm);
        for (j = 0; j < 512; j++) {
            int ii = i * 512 + j;
            din->buff[ii] = din->buff[ii] / mean;
        }
    }
}

void FindMax(float *din, int len, float &max_val, int &max_idx)
{
    int i;
    max_val = -INFINITY;
    max_idx = -1;
    for (i = 0; i < len; i++) {
        if (din[i] > max_val) {
            max_val = din[i];
            max_idx = i;
        }
    }
}

string PathAppend(const string &p1, const string &p2)
{

    char sep = '/';
    string tmp = p1;

#ifdef _WIN32
    sep = '\\';
#endif

    if (p1[p1.length()-1] != sep) { // Need to add a
        tmp += sep;               // path separator
        return (tmp + p2);
    } else
        return (p1 + p2);
}

void Relu(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = val < 0 ? 0 : val;
    }
}

void Swish(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = val / (1 + exp(-val));
    }
}

void Sigmoid(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = 1 / (1 + exp(-val));
    }
}

void DoubleSwish(Tensor<float> *din)
{
    int i;
    for (i = 0; i < din->buff_size; i++) {
        float val = din->buff[i];
        din->buff[i] = val / (1 + exp(-val + 1));
    }
}

void Softmax(float *din, int mask, int len)
{
    float *tmp = (float *)malloc(mask * sizeof(float));
    int i;
    float sum = 0;
    float max = -INFINITY;

    for (i = 0; i < mask; i++) {
        max = max < din[i] ? din[i] : max;
    }

    for (i = 0; i < mask; i++) {
        tmp[i] = exp(din[i] - max);
        sum += tmp[i];
    }
    for (i = 0; i < mask; i++) {
        din[i] = tmp[i] / sum;
    }
    free(tmp);
    for (i = mask; i < len; i++) {
        din[i] = 0;
    }
}

void LogSoftmax(float *din, int len)
{
    float *tmp = (float *)malloc(len * sizeof(float));
    int i;
    float sum = 0;
    for (i = 0; i < len; i++) {
        tmp[i] = exp(din[i]);
        sum += tmp[i];
    }
    for (i = 0; i < len; i++) {
        din[i] = log(tmp[i] / sum);
    }
    free(tmp);
}

void Glu(Tensor<float> *din, Tensor<float> *dout)
{
    int mm = din->buff_size / 1024;
    int i, j;
    for (i = 0; i < mm; i++) {
        for (j = 0; j < 512; j++) {
            int in_off = i * 1024 + j;
            int out_off = i * 512 + j;
            float a = din->buff[in_off];
            float b = din->buff[in_off + 512];
            dout->buff[out_off] = a / (1 + exp(-b));
        }
    }
}

bool is_target_file(const std::string& filename, const std::string target) {
    std::size_t pos = filename.find_last_of(".");
    if (pos == std::string::npos) {
        return false;
    }
    std::string extension = filename.substr(pos + 1);
    return (extension == target);
}

void KeepChineseCharacterAndSplit(const std::string &input_str,
                                  std::vector<std::string> &chinese_characters) {
  chinese_characters.resize(0);
  std::vector<U16CHAR_T> u16_buf;
  u16_buf.resize(std::max(u16_buf.size(), input_str.size() + 1));
  U16CHAR_T* pu16 = u16_buf.data();
  U8CHAR_T * pu8 = (U8CHAR_T*)input_str.data();
  size_t ilen = input_str.size();
  size_t len = EncodeConverter::Utf8ToUtf16(pu8, ilen, pu16, ilen + 1);
  for (size_t i = 0; i < len; i++) {
    if (EncodeConverter::IsChineseCharacter(pu16[i])) {
      U8CHAR_T u8buf[4];
      size_t n = EncodeConverter::Utf16ToUtf8(pu16 + i, u8buf);
      u8buf[n] = '\0';
      chinese_characters.push_back((const char*)u8buf);
    }
  }
}

std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> elems;
  std::stringstream ss(s);
  std::string item;
  while(std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
  return elems;
}

template<typename T>
void PrintMat(const std::vector<std::vector<T>> &mat, const std::string &name) {
  std::cout << name << ":" << std::endl;
  for (auto item : mat) {
    for (auto item_ : item) {
      std::cout << item_ << " ";
    }
    std::cout << std::endl;
  }
}
} // namespace funasr
