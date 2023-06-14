/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  model_params.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef model_params_hpp
#define model_params_hpp

#include <stdio.h>

namespace paraformer_online {

typedef struct {
    float *weight;
    float *bias;
} NormParams;

typedef struct {
    float *fsmn_block_weight;
    float *linear_out_bias;
    float *linear_out_weight;
    float *linear_qkv_bias;
    float *linear_qkv_weight;
} EncSelfAttnParams;

typedef struct {
    float *w1_weight;
    float *w1_bias;
    float *w2_weight;
    float *w2_bias;
} FeedForwardParams;

typedef struct {
    float *w1_bias;
    float *w1_weight;
    float *w2_weight;
    NormParams norm;
} DecoderFeedForwardParams;

typedef struct {
    FeedForwardParams feedforward;
    NormParams norm1;
    NormParams norm2;
    EncSelfAttnParams self_attn;
} SubEncoderParams;

typedef struct {
    SubEncoderParams sub_encoders0;
    SubEncoderParams sub_encoders[49];
    NormParams after_norm;
} EncoderParams;

typedef struct {
    float *linear_kv_bias;
    float *linear_kv_weight;
    float *linear_out_bias;
    float *linear_out_weight;
    float *linear_q_bias;
    float *linear_q_weight;
} DecSelfAttnParams;

typedef struct {
    DecoderFeedForwardParams feedforward;
    NormParams norm1;
    NormParams norm2;
    NormParams norm3;
    float *fsmn_block_weight;
    DecSelfAttnParams src_attn;
} SubDecoderParams;

typedef struct {
    DecoderFeedForwardParams feedforward;
    NormParams norm1;
} SubDecoder3Params;

typedef struct {
    SubDecoderParams sub_decoders[16];
    SubDecoder3Params sub_decoders3;
    NormParams after_norm;
    float *linear_out_bias;
    float *linear_out_weight;
} DecoderParams;

typedef struct {
    float *cif_conv1d_bias;
    float *cif_conv1d_weight;
    float *cif_output_bias;
    float *cif_output_weight;
} PredictorParams;

typedef struct {
    EncoderParams encoder;
    DecoderParams decoder;
    PredictorParams predictor;
} ModelParams;

class ModelParamsHelper {
  private:
    float *params_addr;
    int offset;
    int vocab_size;

    float *get_addr(int num);

  public:
    ModelParamsHelper(const char *path, int vocab_size);
    ~ModelParamsHelper();
    ModelParams params;

    void param_init_decoder(DecoderParams &p_in);
    void param_init_decoderfeedforward(DecoderFeedForwardParams &p_in);
    void param_init_decselfattn(DecSelfAttnParams &p_in);
    void param_init_encoder(EncoderParams &p_in);
    void param_init_encoder_selfattn(EncSelfAttnParams &p_in, int size);
    void param_init_encoder_subencoder(SubEncoderParams &p_in, int size);
    void param_init_feedforward(FeedForwardParams &p_in);
    void param_init_predictor(PredictorParams &p_in);
    void param_init_subdecoder(SubDecoderParams &p_in);
    void param_init_subdecoder3(SubDecoder3Params &p_in);
    void params_init(ModelParams &p_in);
    void param_init_layernorm(NormParams &p_in, int size);
};

}
#endif /* model_params_hpp */
