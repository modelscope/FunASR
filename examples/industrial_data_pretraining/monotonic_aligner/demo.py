#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="../modelscope_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

res = model(input=("../modelscope_models/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav",
                   "欢迎大家来到魔搭社区进行体验"),
            data_type=("sound", "text"),
            batch_size=2,
            )
print(res)
