/*
 *    Bsed on https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v4/cuda/wkv_op.cpp
   Function signatures were modified based on https://github.com/huggingface/transformers/blob/main/src/transformers/kernels/rwkv/wkv_op.cpp

 */

#include <torch/extension.h>

void cuda_forward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y);

void cuda_backward(int B, int T, int C, float *w, float *u, float *k, float *v, float *y, float *gy, float *gw, float *gu, float *gk, float *gv);

void forward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y) {
  const int B = k.size(0);
  const int T = k.size(1);
  const int C = k.size(2);

  cuda_forward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>());
}

void backward(torch::Tensor &w, torch::Tensor &u, torch::Tensor &k, torch::Tensor &v, torch::Tensor &y, torch::Tensor &gy, torch::Tensor &gw, torch::Tensor &gu, torch::Tensor &gk, torch::Tensor &gv) {
  const int B = k.size(0);
  const int T = k.size(1);
  const int C = k.size(2);

  cuda_backward(B, T, C, w.data_ptr<float>(), u.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), y.data_ptr<float>(), gy.data_ptr<float>(), gw.data_ptr<float>(), gu.data_ptr<float>(), gk.data_ptr<float>(), gv.data_ptr<float>());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &forward, "wkv forward");
    m.def("backward", &backward, "wkv backward");
}

TORCH_LIBRARY(wkv_decoder, m) {
    m.def("forward", forward);
    m.def("backward", backward);
}
