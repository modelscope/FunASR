/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  params_config.h
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef params_config_h
#define params_config_h

// feature
#define sampling_rate 16000
#define n_mels 80
#define frame_length 25
#define frame_shift 10
#define lfr_m 7
#define lfr_n 6

// model
#define encoder_num_blocks 50
#define decoder_num_blocks 16
#define output_size 512
#define attention_heads 4
#define linear_units 2048
#define kernel_size 11
#define encoder_sanm_shift 0
#define decoder_sanm_shift 5

#define predictor_idim 512 // equal to output_size
#define predictor_l_order 1
#define predictor_r_order 1

// vocab
#define token_list 8404

#endif /* params_config_h */
