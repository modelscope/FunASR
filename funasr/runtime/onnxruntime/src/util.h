

#ifndef UTIL_H
#define UTIL_H

using namespace std;

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

#endif
