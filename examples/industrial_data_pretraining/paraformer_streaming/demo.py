#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

# from funasr import AutoModel
#
# model = AutoModel(model="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch", model_revison="v2.0.0")
#
# res = model(input="https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/asr_example_zh.wav")
# print(res)


from funasr import AutoFrontend

frontend = AutoFrontend(model="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online", model_revison="v2.0.0")



import soundfile
speech, sample_rate = soundfile.read("/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/example/asr_example.wav")

chunk_size = [0, 10, 5] #[0, 10, 5] 600ms, [0, 8, 4] 480ms
chunk_stride = chunk_size[1] * 960 # 600ms、480ms
# first chunk, 600ms

cache = {}

for i in range(int(len((speech)-1)/chunk_stride+1)):
    speech_chunk = speech[i*chunk_stride:(i+1)*chunk_stride]
    fbanks = frontend(input=speech_chunk,
                      batch_size=2,
                      cache=cache)


# for batch_idx, fbank_dict in enumerate(fbanks):
# 	res = model(**fbank_dict)
# 	print(res)