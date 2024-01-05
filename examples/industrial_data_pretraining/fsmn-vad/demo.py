#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="../modelscope_models/speech_fsmn_vad_zh-cn-16k-common-pytorch")

res = model(input="../modelscope_models/speech_fsmn_vad_zh-cn-16k-common-pytorch/example/vad_example.wav")
print(res)