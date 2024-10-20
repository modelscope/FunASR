#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_charctc_kws_phone-xiaoyun_mt",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu'
)

test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

res = model.generate(input=test_wav, cache={},)
print(res)
