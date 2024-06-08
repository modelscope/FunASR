#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

from funasr import AutoModel

model = AutoModel(
    model="/nfs/beinian.lzr/workspace/GPT-4o/Exp/exp6/4m-8gpu/exp6_speech2text_0607_linear_ddp",
)

jsonl = (
    "/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData/aishell1_test_speech2text.jsonl"
)

with open(jsonl, "r") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    data_dict = json.loads(line.strip())
    data = data_dict["messages"]

    res = model.generate(
        input=data,
        cache={},
    )

    print(res)
