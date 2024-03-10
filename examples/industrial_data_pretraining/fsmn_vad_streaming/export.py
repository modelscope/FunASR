#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel
wav_file = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/vad_example.wav"

model = AutoModel(model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch", model_revision="v2.0.4", export_model=True)

res = model.export(input=wav_file, type="onnx", quantize=False)
print(res)
