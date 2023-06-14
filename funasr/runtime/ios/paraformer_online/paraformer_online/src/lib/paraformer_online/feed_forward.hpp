/**
 * Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights
 * Reserved. MIT License  (https://opensource.org/licenses/MIT)
 */

//
//  feed_forward.hpp
//  paraformer_online
//
//  Created by 邱威 on 2023/6/6.
//

#ifndef feed_forward_hpp
#define feed_forward_hpp

#include <stdio.h>
#include "Tensor.h"
#include "model_params.hpp"
#include "layer_norm.hpp"

namespace paraformer_online {

class FeedForward {
  private:
    FeedForwardParams *params;
    void (*activate)(Tensor<float> *din);

  public:
    FeedForward(FeedForwardParams *params, int active_type);
    ~FeedForward();
    void forward(Tensor<float> *din);
};

class FeedForwardDecoder {
private:
    DecoderFeedForwardParams *params;
    void (*activate)(Tensor<float> *din);
    LayerNorm *norm;

public:
    FeedForwardDecoder(DecoderFeedForwardParams *params);
    ~FeedForwardDecoder();
    void forward(Tensor<float> *din);
};

}

#endif /* feed_forward_hpp */
