/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */
//
//  attention.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "attention.hpp"

#if defined(__APPLE__)
#include <Accelerate/Accelerate.h>
#else
#include "cblas.h"
#endif

#include "online_utils.hpp"
//#include "matrix.h"

//using namespace QNN_NS;
namespace paraformer_online {

// (size_r, size_c) -> (size_c, size_r)
float* T(const float* matrix, int size_r, int size_c)
{
    float* new_matrix = new float[size_r * size_c]{ 0 };
#if defined(__APPLE__)
    vDSP_mtrans(matrix, 1, new_matrix, 1, size_c, size_r);
#else
    
    int row(0), col(0);

    for (; row < size_r; row++) {
        for (; col < size_c; col++)
            new_matrix[col * size_r + row] = matrix[row * size_c + col];
        col = 0;
    }
#endif
    // delete matrix;

    return new_matrix;
}

void conv_compute(Matrix &bottom_blob, Matrix &top_blob, Matrix &weight_data) {
    int kernel_h = 1;
    int kernel_w = 11;
    
    int stride_h = 1;
    int stride_w = 1;
    
    int in_channel = 512;
    int groups = 512;
    int out_channel = 512;
    
    int h = bottom_blob.h;
    int w = bottom_blob.w;

    int outh = (h - kernel_h) / stride_h + 1;
    int outw = (w - kernel_w) / stride_w + 1;

    int k_offset = kernel_h * kernel_w;

    std::vector<int> _space_ofs(k_offset);
    int* space_ofs = &_space_ofs[0];
    {
        int p1 = 0;
        int p2 = 0;
        int gap = w - kernel_w;
        for (int i = 0; i < kernel_h; i++) {
            for (int j = 0; j < kernel_w; j++) {
                space_ofs[p1] = p2;
                p1++;
                p2++;
            }
            p2 += gap;
        }
    }

    top_blob.h = outh;
    top_blob.w = outw;
    top_blob.c = out_channel;
    top_blob.data = (float *)malloc(outh*outw*out_channel*sizeof(float));
    
    if (in_channel == groups && groups == out_channel) {
        for (int g = 0; g < groups; g++) {
            float* outptr = (float *)top_blob.data + g*top_blob.h*top_blob.w;
            float *kptr = weight_data.data + k_offset * g;
            float *m = (float *)bottom_blob.data + bottom_blob.h*bottom_blob.w*g;

            for (int i = 0; i < outh; i++) {
                for (int j = 0; j < outw; j++) {
                    float sum = 0.f;
                    float *sptr = (float *)m + (i*stride_h)*bottom_blob.w + j * stride_w;

                    for (int k = 0; k < k_offset; k++) {
                        float val = sptr[space_ofs[k]];
                        float w = kptr[k];
                        sum += val * w;
                    }

                    outptr[j] = sum;
                }

                outptr += outw;
            }
        }
    }
}

MultiHeadedAttentionSANM::MultiHeadedAttentionSANM(EncSelfAttnParams *params) : params(params) {
    
}

MultiHeadedAttentionSANM::~MultiHeadedAttentionSANM() {
    
}

void MultiHeadedAttentionSANM::forward_qkv(Tensor<float> *din, Tensor<float> *dout,
                                     float *weight, float *bias)
{
    int mm = din->size[2];
    int nn = din->size[3];
    int i;
    int offset = 0;
    for (i = 0; i < mm; i++) {
        memcpy(dout->buff + offset, bias, sizeof(float) * 1536);
        offset += 1536;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, 1536, nn, 1, din->buff, nn, weight, nn, 1, dout->buff, 1536);
}

void MultiHeadedAttentionSANM::forward_fsmn(Tensor<float> *din, int *conv_im2col)
{

    int mm = din->size[2];
    int v_offset = 0;

    Tensor<float> blasin(mm, 11);
    int i, j;

    for (i = 0; i < 512; i++) {
        for (j = 0; j < mm * 11; j++) {
            int tmp_idx = conv_im2col[j];
            if (tmp_idx == -1)
                blasin.buff[j] = 0;
            else
                blasin.buff[j] = din->buff[tmp_idx + v_offset];
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, mm, 1, 11, 1,
                    blasin.buff, 11, params->fsmn_block_weight + i * 11, 1, 1,
                    din->buff + v_offset, 512);

        v_offset++;
    }
}

void MultiHeadedAttentionSANM::forward(Tensor<float> *&feats, int *conv_im2col) {
    int Tmax = feats->size[2];
    Tensor<float> qkv(Tmax, 1536);

    forward_qkv(feats, &qkv, params->linear_qkv_weight,
                       params->linear_qkv_bias);
    
    Tensor<float> *fsmn_memory = new Tensor<float>(Tmax, 512);
    fsmn_memory->zeros();
    int i;
    int offset0 = 0;
    int offset1 = 0;
    for (i = 0; i < Tmax; i++) {
        memcpy(fsmn_memory->buff + offset0, qkv.buff + 1024 + offset1,
               512 * sizeof(float));
        offset0 += 512;
        offset1 += 1536;
    }
    
    forward_fsmn(fsmn_memory, conv_im2col);
    
    Tensor<float> scores(Tmax, Tmax);
    Tensor<float> attnout(Tmax, 512);
    attnout.zeros();

    int head_step = 512 / 4;
    int q_offset = 0;
    int k_offset = 512;
    int v_offset = 1024;
    int next_column = 1536;

    for (i = 0; i < 4; i++) {
        float *sub_q = qkv.buff + i * head_step + q_offset;
        float *sub_k = qkv.buff + i * head_step + k_offset;
        float *sub_v = qkv.buff + i * head_step + v_offset;
        float *sub_attn = attnout.buff + i * head_step;

        scores.zeros();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, Tmax,
                    head_step, 1, sub_q, next_column, sub_k, next_column, 1,
                    scores.buff, Tmax);


        int j;
        for (j = 0; j < Tmax; j++) {
            int offset = j * Tmax;
            softmax(scores.buff + offset, scores.size[3], scores.size[3]);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Tmax, head_step,
                    Tmax, 1, scores.buff, Tmax, sub_v, next_column, 1, sub_attn,
                    512);
    }

    Tensor<float> *tmp_out = new Tensor<float>(Tmax, 512);
    for (i = 0; i < Tmax; i++) {
        int offset = i * 512;
        memcpy(tmp_out->buff + offset, params->linear_out_bias,
               512 * sizeof(float));
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, Tmax, 512, 512, 1,
                attnout.buff, 512, params->linear_out_weight, 512, 1,
                tmp_out->buff, 512);
    
    tmp_out->add(fsmn_memory);
    delete fsmn_memory;
    delete feats;
    feats = tmp_out;
}



MultiHeadedAttentionSANMDecoder::MultiHeadedAttentionSANMDecoder(SubDecoderParams *params) : params(params) {
    fsmn_cache_ = nullptr;
    
    weights_m.h = 1;
    weights_m.w = 11;
    weights_m.c = 512;
    weights_m.data = params->fsmn_block_weight;
    
}

MultiHeadedAttentionSANMDecoder::~MultiHeadedAttentionSANMDecoder() {
    delete fsmn_cache_;
    free(weights_m.data);
}

void MultiHeadedAttentionSANMDecoder::forward_fsmn(Tensor<float> *fsmn_memory, Tensor<float> *fsmn_out) {
    int Tmax = fsmn_memory->size[2];
    // pad
    float *mem_data = (float *)malloc((Tmax)*512*sizeof(float));
    
    memcpy(mem_data, fsmn_memory->buff, fsmn_memory->buff_size*sizeof(float));
    
    float *fsmn_memory_T = T(mem_data, Tmax, 512);
    free(mem_data);
    
    Matrix conv_input_m;
    conv_input_m.h = 1;
    conv_input_m.w = Tmax;
    conv_input_m.c = 512;
    conv_input_m.data = fsmn_memory_T;
    
    Matrix conv_output_m;
    conv_compute(conv_input_m, conv_output_m, weights_m);
    
    delete fsmn_memory_T;
    
    float *fsmn_block_out = T((float *)conv_output_m.data, conv_output_m.c, conv_output_m.h*conv_output_m.w);
    
    for (int i = 0; i < conv_output_m.c*conv_output_m.h*conv_output_m.w; i++) {
        fsmn_out->buff[i] = fsmn_block_out[i];
    }
    
    free(conv_output_m.data);
    delete fsmn_block_out;
}

void MultiHeadedAttentionSANMDecoder::forward_chunk(Tensor<float> *&feats, int *conv_im2col) {
    // pad
    int Tmax = feats->size[2];
    int ch = feats->size[3];
    Tensor<float> padded_feats(Tmax+10, ch);
    
    if (fsmn_cache_ == nullptr) {
        // pad left
        memset(padded_feats.buff, 0, 10*ch * sizeof(float));
        memcpy(padded_feats.buff+10*ch, feats->buff, feats->buff_size*sizeof(float));
    } else {
        int cache_T = fsmn_cache_->size[2];
        Tensor<float> padded_tmp(cache_T-1+Tmax, ch);

        memcpy(padded_tmp.buff, fsmn_cache_->buff+ch, (cache_T-1)*ch*sizeof(float));
        memcpy(padded_tmp.buff+(cache_T-1)*ch, feats->buff, feats->buff_size*sizeof(float));
        
        memcpy(padded_feats.buff, padded_tmp.buff+(cache_T-11)*ch, (Tmax+10)*ch*sizeof(float));
        
    }
    if (fsmn_cache_ != nullptr) {
        delete fsmn_cache_;
    }
    fsmn_cache_ = new Tensor<float>(Tmax+10, ch);
    fsmn_cache_->reload(&padded_feats);
    
    Tensor<float> fsmn_out(feats);
    forward_fsmn(&padded_feats, &fsmn_out);
    
    feats->add(&fsmn_out);
}


MultiHeadedAttentionCrossAtt::MultiHeadedAttentionCrossAtt(DecSelfAttnParams *params) : params(params) {
    
}

MultiHeadedAttentionCrossAtt::~MultiHeadedAttentionCrossAtt() {
    
}

void MultiHeadedAttentionCrossAtt::linear_forward(Tensor<float> *din, Tensor<float> *dout, float *weight, float *bias) {
    int mm = din->size[2];
    int o_size = dout->size[3];
    int offset = 0;
    int i;

    for (i = 0; i < mm; i++) {
        memcpy(dout->buff + offset, bias, sizeof(float) * o_size);
        offset += o_size;
    }

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, mm, o_size, 512, 1,
                din->buff, 512, weight, 512, 1, dout->buff, o_size);
}

void MultiHeadedAttentionCrossAtt::forward(Tensor<float> *&feats, Tensor<float> *enc) {
    int m1 = feats->size[2];
    int m2 = enc->size[2];
    Tensor<float> q(m1, 512);
    Tensor<float> kv(m2, 1024);

    linear_forward(feats, &q, params->linear_q_weight, params->linear_q_bias);
    linear_forward(enc, &kv, params->linear_kv_weight, params->linear_kv_bias);

    Tensor<float> scores(m1, m2);
    Tensor<float> attnout(m1, 512);
    attnout.zeros();

    int head_step = 512 / 4;
    int k_offset = 0;
    int v_offset = 512;

    int i;

    for (i = 0; i < 4; i++) {
        float *sub_q = q.buff + i * head_step;
        float *sub_k = kv.buff + i * head_step + k_offset;
        float *sub_v = kv.buff + i * head_step + v_offset;
        float *sub_attn = attnout.buff + i * head_step;

        scores.zeros();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m1, m2, head_step,
                    1, sub_q, 512, sub_k, 1024, 1, scores.buff, m2);

        int j;
        for (j = 0; j < m1; j++) {
            int offset = j * m2;
            softmax(scores.buff + offset, scores.size[3], scores.size[3]);
        }

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m1, head_step,
                    m2, 1, scores.buff, m2, sub_v, 1024, 1, sub_attn, 512);
    }

    linear_forward(&attnout, feats, params->linear_out_weight,
                   params->linear_out_bias);
}



}
