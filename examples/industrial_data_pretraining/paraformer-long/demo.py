#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="../modelscope_models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                  vad_model="../modelscope_models/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                  punc_model="../modelscope_models/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                  )

res = model(input="../modelscope_models/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav", batch_size_s=300, batch_size_threshold_s=60)
print(res)