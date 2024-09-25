#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="iic/speech_sanm_kws_phone-xiaoyun-commands-online",
    keywords="小云小云",
    output_dir="./outputs/debug",
    device='cpu',
    chunk_size=[4, 8, 4],
    encoder_chunk_look_back=0,
    decoder_chunk_look_back=0,
)

test_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/KWS/pos_testset/kws_xiaoyunxiaoyun.wav"

res = model.generate(input=test_wav, cache={},)
print(res)
