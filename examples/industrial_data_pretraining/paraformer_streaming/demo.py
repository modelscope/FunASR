#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4 #number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1 #number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", model_revision="v2.0.0")
cache = {}
res = model(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
            chunk_size=chunk_size,
            encoder_chunk_look_back=encoder_chunk_look_back,
            decoder_chunk_look_back=decoder_chunk_look_back,
            )
print(res)


import soundfile
import os

speech, sample_rate = soundfile.read(os.path.expanduser('~')+
                                     "/.cache/modelscope/hub/damo/"+
                                     "speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/"+
                                     "example/asr_example.wav")

chunk_stride = chunk_size[1] * 960 # 600ms、480ms

cache = {}

for i in range(int(len((speech)-1)/chunk_stride+1)):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == int(len((speech)-1)/chunk_stride+1)
    res = model(input=speech_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                encoder_chunk_look_back=encoder_chunk_look_back,
                decoder_chunk_look_back=decoder_chunk_look_back,
                )
    print(res)
