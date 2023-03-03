
#ifndef WENETPARAMS_H
#define WENETPARAMS_H
// #pragma pack(1)

#define vocab_size 5538

typedef struct {
    float conv0_weight[512 * 9];
    float conv0_bias[512];

    float conv1_weight[512 * 512 * 9];
    float conv1_bias[512];

    float out0_weight[9728 * 512];
    float out0_bias[512];

} EncEmbedParams;

typedef struct {
    float linear_q_weight[512 * 512];
    float linear_q_bias[512];
    float linear_k_weight[512 * 512];
    float linear_k_bias[512];
    float linear_v_weight[512 * 512];
    float linear_v_bias[512];
    float linear_out_weight[512 * 512];
    float linear_out_bias[512];
} SelfAttnParams;

typedef struct {
    SelfAttnParams linear0;
    float linear_pos_weight[512 * 512];
    float pos_bias_u[512];
    float pos_bias_v[512];

} EncSelfAttnParams;

typedef struct {
    float w1_weight[512 * 2048];
    float w1_bias[2048];
    float w2_weight[2048 * 512];
    float w2_bias[512];
} FeedForwardParams;

typedef struct {
    float weight[512];
    float bias[512];
} NormParams;

typedef struct {
    float pointwise_conv1_weight[1024 * 512];
    float pointwise_conv1_bias[1024];

    float depthwise_conv_weight[512 * 15];
    float depthwise_conv_bias[512];

    float pointwise_conv2_weight[512 * 512];
    float pointwise_conv2_bias[512];
    NormParams norm;
} EncConvParams;

typedef struct {
    EncSelfAttnParams self_attn;
    FeedForwardParams feedforward;
    FeedForwardParams feedforward_macaron;
    EncConvParams conv_module;
    NormParams norm_ff;
    NormParams norm_mha;
    NormParams norm_macaron;
    NormParams norm_conv;
    NormParams norm_final;
    // float concat_weight[1024 * 512];
    // float concat_bias[512];
} SubEncoderParams;

typedef struct {
    EncEmbedParams embed;
    SubEncoderParams sub_encoder[12];
    NormParams after_norm;
} EncoderParams;

typedef struct {
    SelfAttnParams self_attn;
    SelfAttnParams src_attn;
    FeedForwardParams feedward;
    NormParams norm1;
    NormParams norm2;
    NormParams norm3;
    // float concat_weight1[1024 * 512];
    // float concat_bias1[512];
    // float concat_weight2[1024 * 512];
    // float concat_bias2[512];
} SubDecoderParams;

typedef struct {
    float embed_weight[vocab_size * 512];
    SubDecoderParams sub_decoder[6];
    NormParams after_norm;
    float output_weight[vocab_size * 512];
    float output_bias[vocab_size];
} DecoderParams;

typedef struct {
    EncoderParams encoder;
    float ctc_weight[512 * vocab_size];
    float ctc_bias[vocab_size];
    DecoderParams decoder;
} WenetParams;

// #pragma pack()
#endif
