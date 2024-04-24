#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

multilingual_wavs = [
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_zh-CN.mp3",
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_en.mp3",
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_ja.mp3",
    "https://www.modelscope.cn/api/v1/models/iic/speech_whisper-large_lid_multilingual_pytorch/repo?Revision=master&FilePath=examples/example_ko.mp3",
]

inference_pipeline = pipeline(
    task=Tasks.auto_speech_recognition, model="iic/speech_whisper-large_lid_multilingual_pytorch"
)

for wav in multilingual_wavs:
    rec_result = inference_pipeline(input=wav, inference_clip_length=250)
    print(rec_result)
