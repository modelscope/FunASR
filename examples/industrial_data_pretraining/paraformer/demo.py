#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

res = model(input="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/example/asr_example.wav")
print(res)


from funasr import AutoFrontend

frontend = AutoFrontend(model="/Users/zhifu/Downloads/modelscope_models/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")

fbanks = frontend(input="/Users/zhifu/funasr_github/test_local/wav.scp", batch_size=2)

for batch_idx, fbank_dict in enumerate(fbanks):
	res = model(**fbank_dict)
	print(res)