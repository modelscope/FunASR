#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
wav_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"

chunk_size = 60000 # ms
model = AutoModel(model="/Users/zhifu/Downloads/modelscope_models/speech_fsmn_vad_zh-cn-16k-common-streaming", model_revision="v2.0.0")

res = model(input=wav_file,
            chunk_size=chunk_size,
            )
print(res)



import soundfile
import os

wav_file = os.path.join(model.model_path, "example/vad_example.wav")
speech, sample_rate = soundfile.read(wav_file)

chunk_stride = int(chunk_size * 16000 / 1000)

cache = {}

for i in range(int(len((speech)-1)/chunk_stride+1)):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    is_final = i == int(len((speech)-1)/chunk_stride+1) - 1
    res = model(input=speech_chunk,
                cache=cache,
                is_final=is_final,
                chunk_size=chunk_size,
                )
    print(res)
