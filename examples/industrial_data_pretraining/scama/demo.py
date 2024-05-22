#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

chunk_size = [5, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 0  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 0  # number of encoder chunks to lookback for decoder cross-attention

model = AutoModel(
    model="/Users/zhifu/Downloads/modelscope_models/speech_SCAMA_asr-zh-cn-16k-common-vocab8358-streaming"
)
cache = {}
res = model.generate(
    input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
    chunk_size=chunk_size,
    encoder_chunk_look_back=encoder_chunk_look_back,
    decoder_chunk_look_back=decoder_chunk_look_back,
)
print(res)


import soundfile
import os

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)

chunk_stride = chunk_size[1] * 960  # 600ms、480ms

cache = {}
total_chunk_num = int(len((speech) - 1) / chunk_stride + 1)
for i in range(total_chunk_num):
    speech_chunk = speech[i * chunk_stride : (i + 1) * chunk_stride]
    is_final = i == total_chunk_num - 1
    res = model.generate(
        input=speech_chunk,
        cache=cache,
        is_final=is_final,
        chunk_size=chunk_size,
        encoder_chunk_look_back=encoder_chunk_look_back,
        decoder_chunk_look_back=decoder_chunk_look_back,
    )
    print(res)
