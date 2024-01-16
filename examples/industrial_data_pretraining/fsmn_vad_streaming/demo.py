#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
wav_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"

chunk_size = 60000 # ms
model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.2")

res = model.generate(input=wav_file, chunk_size=chunk_size, )
print(res)



import soundfile
import os

wav_file = os.path.join(model.model_path, "example/vad_example.wav")
speech, sample_rate = soundfile.read(wav_file)

chunk_stride = int(chunk_size * sample_rate / 1000)

cache = {}

total_chunk_num = int(len((speech)-1)/chunk_stride+1)
for i in range(total_chunk_num):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(input=speech_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                )
    if len(res[0]["value"]):
        print(res)
