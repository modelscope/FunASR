#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="/Users/zhifu/Downloads/modelscope_models/SenseVoice",
                  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
				  vad_kwargs={"max_single_segment_time": 30000},
                  )
task = "ASR"
language = None
input_wav = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav"
res = model.generate(task=task, language=language, input=input_wav, batch_size_s=0,)
print(res)
