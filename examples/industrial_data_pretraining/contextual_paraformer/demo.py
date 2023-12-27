#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="../modelscope_models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404")

res = model(input="../modelscope_models/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404/example/asr_example.wav",
            hotword='达摩院 魔搭')
print(res)