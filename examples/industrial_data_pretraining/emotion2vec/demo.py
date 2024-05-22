#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

# model="iic/emotion2vec_base"
# model="iic/emotion2vec_base_finetuned"
# model="iic/emotion2vec_plus_seed"
# model="iic/emotion2vec_plus_base"
model = "iic/emotion2vec_plus_large"

model = AutoModel(
    model=model,
    # vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    # vad_model_revision="master",
    # vad_kwargs={"max_single_segment_time": 2000},
)

wav_file = f"{model.model_path}/example/test.wav"

res = model.generate(
    wav_file, output_dir="./outputs", granularity="utterance", extract_embedding=False
)
print(res)
