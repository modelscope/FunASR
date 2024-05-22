#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="iic/speech_conformer_asr_nat-zh-cn-16k-aishell2-vocab5212-pytorch")

res = model.generate(
    input="https://modelscope.oss-cn-beijing.aliyuncs.com/test/audios/asr_example.wav",
    decoding_ctc_weight=0.0,
)
print(res)
