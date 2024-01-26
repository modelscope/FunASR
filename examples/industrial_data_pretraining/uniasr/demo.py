#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel


model = AutoModel(model="iic/speech_UniASR-large_asr_2pass-zh-cn-16k-common-vocab8358-tensorflow1-offline", model_revision="v2.0.4",)


res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
print(res)


''' can not use currently
from funasr import AutoFrontend

frontend = AutoFrontend(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", model_revision="v2.0.4")

fbanks = frontend(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav", batch_size=2)

for batch_idx, fbank_dict in enumerate(fbanks):
    res = model.generate(**fbank_dict)
    print(res)
'''