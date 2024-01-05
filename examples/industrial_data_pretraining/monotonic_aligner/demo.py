#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(model="/Users/zhifu/modelscope_models/speech_timestamp_prediction-v1-16k-offline")

res = model(input=("/Users/zhifu/funasr_github/test_local/wav.scp",
                   "/Users/zhifu/funasr_github/test_local/text.txt"),
            data_type=("sound", "text"),
            batch_size=2,
            )
print(res)
