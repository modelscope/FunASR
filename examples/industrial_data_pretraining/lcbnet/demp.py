#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="iic/LCB-NET"
                  )


# example1
#res = model.generate(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav",
                    )
#print(res)


'''
# tensor or numpy as input
# example2
import torchaudio
import os
wav_file = os.path.join(model.model_path, "example/asr_example.wav")
input_tensor, sample_rate = torchaudio.load(wav_file)
input_tensor = input_tensor.mean(0)
res = model.generate(input=[input_tensor], batch_size_s=300, is_final=True)


# example3
import soundfile

wav_file = os.path.join(model.model_path, "example/asr_example.wav")
speech, sample_rate = soundfile.read(wav_file)
res = model.generate(input=[speech], batch_size_s=300, is_final=True)
'''
