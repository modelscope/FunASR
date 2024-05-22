#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os

from funasr import AutoModel

chunk_size = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
encoder_chunk_look_back = 4  # number of chunks to lookback for encoder self-attention
decoder_chunk_look_back = 1  # number of encoder chunks to lookback for decoder cross-attention
model = AutoModel(model="iic/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online")

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
res = model.generate(
    input=wav_file,
    chunk_size=chunk_size,
    encoder_chunk_look_back=encoder_chunk_look_back,
    decoder_chunk_look_back=decoder_chunk_look_back,
)
print(res)


import soundfile


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
