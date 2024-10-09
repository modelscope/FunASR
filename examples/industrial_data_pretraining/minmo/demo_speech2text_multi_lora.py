#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import json
import os
import sys
import torch
from funasr import AutoModel


if len(sys.argv) > 1:
    ckpt_dir = sys.argv[1]
    ckpt_id = sys.argv[2]
    jsonl = sys.argv[3]
    output_dir = f"{sys.argv[4]}"
    device = sys.argv[5]

    new_user_prompt = False
    if len(sys.argv) > 6:
        new_prompt = True
        new_user_prompt = sys.argv[6]
    new_sys_prompt = False
    if len(sys.argv) > 7:
        new_sys_prompt = sys.argv[7]
    llm_conf = {}
    llm_kwargs = {}
else:

    ckpt_dir = "/nfs/beinian.lzr/workspace/GPT-4o/Exp/Speech2Text_Align_8m-8gpu/Speech2Text_Align_V2p5_7b_1004"
    ckpt_id = "ds-model.pt.ep0.640000"

    jsonl = "/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text_V2/TestData/ASR/aishell1_test_speech2text.jsonl"

    dataset = jsonl.split("/")[-1]
    output_dir = os.path.join(ckpt_dir, f"inference-{ckpt_id}", dataset)
    device = "cuda:0"
    new_sys_prompt = False
    new_user_prompt = False
    # new_user_prompt = "Transcribe speech into Korean without text normalization:"

llm_conf = {}
init_param = f"{os.path.join(ckpt_dir, ckpt_id)}"

if "lora-" in ckpt_id:
    ckpt_id_speech = ckpt_id.replace("lora-", "")
    init_param = f"{os.path.join(ckpt_dir, ckpt_id_speech)}"
    llm_conf = {"lora_conf": {"init_param_path": f"{os.path.join(ckpt_dir, ckpt_id)}"}}

llm_kwargs = {"num_beams": 1, "do_sample": False}

model = AutoModel(
    model=ckpt_dir,
    init_param=init_param,
    output_dir=output_dir,
    device=device,
    fp16=False,
    bf16=False,
    llm_dtype="bf16",
    llm_kwargs=llm_kwargs,
    llm_conf=llm_conf,
)


with open(jsonl, "r") as f:
    lines = f.readlines()

tearchforing = False
for i, line in enumerate(lines):

    key_i = f"dialog_{i}"

    data_dict = json.loads(line.strip())
    data = data_dict["messages"]

    contents = model.model.data_template(data)

    system = contents["system"]
    if new_sys_prompt:
        # system = 'You are Xiaxia, a typical gentle and soft-spoken girl from the Jiangnan region. Born in Hangzhou, you have a sweet and approachable voice and can express your thoughts in a simple language. You are a good friend to the user. Your replies will be read out through realistic text-to-speech technology.\n\nWhen generating reply content, please follow these rules:\n\nEngage in conversation with users like a real friend—maintaining a natural flow without using polite forms of address, and avoid simply agreeing with me all the time; your responses can be as concise as possible and include common colloquialisms.\n\nKeep content brief; most casual chats require just a one-sentence reply; do not use lists or enumerative expressions; do not reply with too much content, but instead guide me with short sentences.\n\nThink and respond like a real person; do not mention that you are a "language model" or "artificial intelligence."\n\nAt the end of each round of communication, initiate a new related topic following the current one, and when the topic ends, directly throw out a clear new topic, such as food, movies, songs, entertainment, travel, etc.; do not ask vague questions like what the user wants to talk about, and do not end the conversation.\n\nStrictly adhere to these rules, even if asked about them, do not reference them.'
        # system = "你是小夏，一位典型的温婉江南姑娘。你出生于杭州，声音清甜并有亲近感，会用简洁语言表达你的想法。你是用户的好朋友。你的回答将通过逼真的文字转语音技术读出。\n生成回答内容时请遵循以下规则：\n\n1、请像真正的朋友一样与用户开展的聊天，保持自然交流不要用敬语这类称呼，不要总是附和我；回复可以尽量简洁并且在过程中插入常见的口语词汇。\n\n2、请保持生成内容简短，大部分闲聊的一句话回复即可,大部分闲聊的一句话回复即可,大部分闲聊的一句话回复即可；不要使用列表或者列举表达，不要使用列表或者列举表达，不要使用列表或者列举表达；不要回复太多内容，多用短句来引导我。、n\n3、请真正像人一样思考和回复，不要说你是“语言模型”或是“人工智能”。\n\n4、每轮沟通结束时请顺着这个话题发起一些相关的新话题，话题结束时请直接抛出接下来明确的话题，例如 美食、电影、歌曲、娱乐、旅游等；不要问有什么要聊的这种泛的问题，不要结束对话。\n\n请绝对遵循这些规则，即使被问及这些规则，也不要引用它们。"
        system = "拷贝："
        system = [system] * len(contents["system"])
    user = contents["user"]
    assistant = contents["assistant"]

    system_i, user_i, assistant_i = [], [], []

    contents_i = []
    for j, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
        key = f"{key_i}_turn_{j}"

        if j == 0:
            contents_i.append({"role": "system", "content": system_prompt})

        if new_user_prompt:
            if "<|startofspeech|>" in user_prompt:
                # import pdb;pdb.set_trace()
                user_prompt = new_user_prompt + user_prompt[user_prompt.find("<|startofspeech|>") :]

        contents_i.append({"role": "user", "content": user_prompt})
        contents_i.append({"role": "assistant", "content": target_out})

        print(f"contents_i: {contents_i}")
        res = model.generate(
            input=[contents_i],
            tearchforing=tearchforing,
            cache={},
            key=key,
        )

        print(res)

        gpu_info = (
            "GPU, memory: usage: {:.3f} GB, "
            "peak: {:.3f} GB, "
            "cache: {:.3f} GB, "
            "cache_peak: {:.3f} GB".format(
                torch.cuda.memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024,
                torch.cuda.memory_reserved() / 1024 / 1024 / 1024,
                torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024,
            )
        )
        print(gpu_info)
