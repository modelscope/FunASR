/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  model_params.cpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#include "model_params.hpp"
#include "../util.h"
#include "params_config.h"

using namespace paraformer_online;

ModelParamsHelper::ModelParamsHelper(const char *path, int vocab_size)
    : vocab_size(vocab_size)
{
    params_addr = loadparams(path);
    offset = 0;
    params_init(params);
}

ModelParamsHelper::~ModelParamsHelper()
{
    aligned_free(params_addr);
}

float *ModelParamsHelper::get_addr(int num)
{
    float *tmp = params_addr + offset;
    offset += val_align(num, 32);
    return tmp;
}

void ModelParamsHelper::params_init(ModelParams &p_in)
{
    param_init_encoder(p_in.encoder);
    param_init_predictor(p_in.predictor);
    param_init_decoder(p_in.decoder);
}

void ModelParamsHelper::param_init_encoder(EncoderParams &p_in)
{
    param_init_encoder_subencoder(p_in.sub_encoders0, lfr_m*n_mels); 
    int i;
    for (i = 0; i < 49; i++) {
        param_init_encoder_subencoder(p_in.sub_encoders[i], output_size);
    }
    param_init_layernorm(p_in.after_norm, output_size);
}

void ModelParamsHelper::param_init_encoder_subencoder(SubEncoderParams &p_in,
                                                      int size)
{

    param_init_feedforward(p_in.feedforward);
    param_init_layernorm(p_in.norm1, size);
    param_init_layernorm(p_in.norm2, output_size);

    param_init_encoder_selfattn(p_in.self_attn, size);
}

void ModelParamsHelper::param_init_encoder_selfattn(EncSelfAttnParams &p_in,
                                                    int size)
{
    p_in.fsmn_block_weight = get_addr(output_size * kernel_size);
    p_in.linear_out_bias = get_addr(output_size);
    p_in.linear_out_weight = get_addr(output_size * output_size);
    p_in.linear_qkv_bias = get_addr(output_size*3);
    p_in.linear_qkv_weight = get_addr(output_size*3 * size);
}

void ModelParamsHelper::param_init_feedforward(FeedForwardParams &p_in)
{
    p_in.w1_bias = get_addr(linear_units);
    p_in.w1_weight = get_addr(linear_units * output_size);

    p_in.w2_bias = get_addr(output_size);
    p_in.w2_weight = get_addr(output_size * linear_units);
}

void ModelParamsHelper::param_init_decoderfeedforward(
    DecoderFeedForwardParams &p_in)
{
    param_init_layernorm(p_in.norm, linear_units);
    p_in.w1_bias = get_addr(linear_units);
    p_in.w1_weight = get_addr(linear_units * output_size);
    p_in.w2_weight = get_addr(output_size * linear_units);

}

void ModelParamsHelper::param_init_layernorm(NormParams &p_in, int size)
{
    p_in.bias = get_addr(size);
    p_in.weight = get_addr(size);
}

void ModelParamsHelper::param_init_decoder(DecoderParams &p_in)
{
    int i;
    for (i = 0; i < 16; i++) {
        param_init_subdecoder(p_in.sub_decoders[i]);
    }

    param_init_subdecoder3(p_in.sub_decoders3);
    param_init_layernorm(p_in.after_norm, output_size);

    p_in.linear_out_bias = get_addr(token_list);
    p_in.linear_out_weight = get_addr(token_list * output_size);
}

void ModelParamsHelper::param_init_subdecoder(SubDecoderParams &p_in)
{
    param_init_decoderfeedforward(p_in.feedforward);
    param_init_layernorm(p_in.norm1, output_size);
    param_init_layernorm(p_in.norm2, output_size);
    param_init_layernorm(p_in.norm3, output_size);
    p_in.fsmn_block_weight = get_addr(output_size * kernel_size);
    param_init_decselfattn(p_in.src_attn);
}

void ModelParamsHelper::param_init_subdecoder3(SubDecoder3Params &p_in)
{
    param_init_decoderfeedforward(p_in.feedforward);
    param_init_layernorm(p_in.norm1, output_size);
}

void ModelParamsHelper::param_init_decselfattn(DecSelfAttnParams &p_in)
{
    p_in.linear_kv_bias = get_addr(output_size*2);
    p_in.linear_kv_weight = get_addr(output_size*2 * output_size);

    p_in.linear_out_bias = get_addr(output_size);
    p_in.linear_out_weight = get_addr(output_size * output_size);

    p_in.linear_q_bias = get_addr(output_size);
    p_in.linear_q_weight = get_addr(output_size * output_size);
}

void ModelParamsHelper::param_init_predictor(PredictorParams &p_in)
{
    p_in.cif_conv1d_bias = get_addr(predictor_idim);
    p_in.cif_conv1d_weight = get_addr(predictor_idim * predictor_idim * (predictor_l_order + predictor_r_order + 1));

    p_in.cif_output_bias = get_addr(1);
    p_in.cif_output_weight = get_addr(predictor_idim);
}
