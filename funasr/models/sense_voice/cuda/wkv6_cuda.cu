#include <stdio.h>
#include <assert.h>
#include "ATen/ATen.h"
typedef at::BFloat16 bf16;
// typedef float bf16;

template <typename F>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u,
                               F *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h*_N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];
    float state[_N_] = {0};

    __syncthreads();
    u[i] = float(_u[i]);
    __syncthreads();

    for (int t = b*T*C + h*_N_ + i; t < (b+1)*T*C + h*_N_ + i; t += C)
    {
        __syncthreads();
        w[i] = exp(_w[t]);
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        __syncthreads();

        const float v = float(_v[t]);
        float y = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j+=4)
        {
            const float4& r_ = (float4&)(r[j]);
            const float4& k_ = (float4&)(k[j]);
            const float4& w_ = (float4&)(w[j]);
            const float4& u_ = (float4&)(u[j]);
            float4& s = (float4&)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = F(y);
    }
}

template <typename F>
__global__ void kernel_backward_111(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gr, F *__restrict__ const _gk, F *__restrict__ const _gv, F *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h*_N_;

    __shared__ float u_[_N_];
    __shared__ float r[_N_], k[_N_], v[_N_], w_[_N_], gy[_N_];
    __syncthreads();
    u_[i] = float(_u[i]);
    __syncthreads();

    const float u = u_[i];

    float state[_N_] = {0}, scccc[_N_] = {0}, sdddd[_N_] = {0};

    const int t_0 = b*T*C + h*_N_ + i;
    const int t_T_1 = t_0 + (T-1)*C;
    const int t_T = t_0 + T*C;

    float gu = 0;
    for (int t = t_0; t < t_T; t += C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float k = float(_k[t]);
        const float w = exp(_w[t]);
        float gr = 0, gu_ = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = state[j];
            float x = k * v[j];

            gr += (u * x + s) * gy[j];
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = F(gr);
        gu += float(_r[t]) * gu_;
    }
    _gu[b*C + h*_N_ + i] = F(gu);

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        v[i] = float(_v[t]);
        gy[i] = float(_gy[t]);
        __syncthreads();

        const float rr = float(_r[t]);
        const float w = exp(_w[t]);
        float gk = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = rr * gy[j];
            
            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = F(gk);
    }

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        r[i] = float(_r[t]);
        k[i] = float(_k[t]);
        w_[i] = exp(_w[t]);
        __syncthreads();

        const float gyy = float(_gy[t]);
        float gv = 0;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = sdddd[j];
            float x = gyy * r[j];
            
            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = F(gv);
    }
}

template <typename F>
__global__ void kernel_backward_222(const int B, const int T, const int C, const int H,
    const F *__restrict__ const _r, const F *__restrict__ const _k, const F *__restrict__ const _v, const float *__restrict__ _w, const F *__restrict__ _u, const F *__restrict__ const _gy,
    F *__restrict__ const _gw)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    float saaaa[_N_] = {0}, sbbbb[_T_-2] = {0}, scccc[_N_] = {0};

    const int t_0 = b*T*C + h*_N_ + i;
    const int t_1 = t_0 + C;
    const int t_2 = t_0 + 2*C;
    const int t_T_1 = t_0 + (T-1)*C;

    for (int t = t_T_1; t > t_1; t -= C)
    {
        __syncthreads();
        gy[i] = float(_gy[t]);
        v[i] = float(_v[t-2*C]);
        __syncthreads();

        const float r = float(_r[t]);
        const float w = exp(_w[t-C]);
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = saaaa[j];
            float x = r * gy[j];
            s = (s + x) * w;
            sum += s * v[j];
        }
        sbbbb[(t-t_2)/C] = sum * float(_k[t-2*C]);
    }

    float sss = sbbbb[0];
    _gw[t_0] = 0;
    _gw[t_1] = F(sss * _w[t_1]);

    for (int t = t_2; t < t_T_1; t += C)
    {
        __syncthreads();
        gy[i] = float(_gy[t]);
        v[i] = float(_v[t-2*C]);
        __syncthreads();

        const float w = exp(_w[t-C]);
        const float k = float(_k[t-2*C]);
        float sum = 0.0f;

        #pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float& s = scccc[j];
            float x = k * v[j];
            s = (s + x) * w;
            sum += s * gy[j];
        }
        sss += sbbbb[(t-t_1)/C] - (sum * float(_r[t]));
        _gw[t] = F(sss * _w[t]);
    }
    _gw[t_T_1] = 0;
}

void cuda_forward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *y)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_forward<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, y);
}

void cuda_backward(int B, int T, int C, int H, bf16 *r, bf16 *k, bf16 *v, float *w, bf16 *u, bf16 *gy, bf16 *gr, bf16 *gk, bf16 *gv, bf16 *gw, bf16 *gu)
{
    assert(H*_N_ == C);
    assert(_N_%4 == 0);
    kernel_backward_111<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gu);
    kernel_backward_222<<<dim3(B * H), dim3(_N_)>>>(B, T, C, H, r, k, v, w, u, gy, gw);
}
