#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel


model_dir = "iic/SenseVoiceSmall"
model = AutoModel(
    model=model_dir,
    device="cuda:0",
)

res = model.export(type="onnx", quantize=False)