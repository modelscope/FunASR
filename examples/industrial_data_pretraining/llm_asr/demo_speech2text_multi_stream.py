import os
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import TextIteratorStreamer
from threading import Thread
import torch

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
import sys

sys.path.insert(1, "/mnt/workspace/workgroup/wenliang/workspace/FunASR")
from funasr import AutoModel
import json

device = "cuda:0"  # the device to load the model onto

ckpt_dir = "/mnt/workspace/workgroup/wenliang/ckpt/gpt-4o/exp7/5m-8gpu/exp7-3_add_asr-dialog_0622/"
ckpt_id = "model.pt.ep20"
jsonl = "/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData/s2tchat.v20240619.test.jsonl"
dataset = jsonl.split("/")[-1]
output_dir = os.path.join(ckpt_dir, f"inference-{ckpt_id}", dataset)
device = "cuda:0"
new_sys = False

Model = AutoModel(
    model=ckpt_dir,
    init_param=f"{os.path.join(ckpt_dir, ckpt_id)}",
    output_dir=output_dir,
    device=device,
    fp16=False,
    bf16=False,
    llm_dtype="fp16",
)
model = Model.model
frontend = Model.kwargs["frontend"]
tokenizer = Model.kwargs["tokenizer"]
# model_name_or_path = "/mnt/workspace/workgroup/wenliang/project/pretrained_models/Qwen2-7B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

prompt = "Give me a short introduction to large language model."
prompt = "请简单介绍一下大语言模型。"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


lines = [
    """
{"messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "<|startofspeech|>!/mnt/workspace/workgroup/wenliang/workspace/CosyVoice_opensource/sft.wav<|endofspeech|>", "text_content": "你抄完没有？"}, {"role": "assistant", "content": "抱歉，我不太明白你的意思。我是一个人工智能模型，我没有能力去抄写任何东西，我只能根据我学习过的大量信息来回答你的问题。如果你有关于某个主题的问题，我会尽我所能提供帮助。"}], "speech_length": 124, "key": "ASR_wav008_0972_098abd8fffe241baa4962b7952f8eb45", "task": "voice_chat", "out_text_length": 48, "in_text_length": 24, "text_length": 135, "qwen_fetch_line_index": 0}
"""
]

tearchforing = False
for i, line in enumerate(lines):

    key_i = f"dialog_{i}"

    data_dict = json.loads(line.strip())
    data = data_dict["messages"]

    contents = model.data_template(data)
    print(f"contents: {contents}")
    system = contents["system"]
    if new_sys:
        # system = 'You are Xiaxia, a typical gentle and soft-spoken girl from the Jiangnan region. Born in Hangzhou, you have a sweet and approachable voice and can express your thoughts in a simple language. You are a good friend to the user. Your replies will be read out through realistic text-to-speech technology.\n\nWhen generating reply content, please follow these rules:\n\nEngage in conversation with users like a real friend—maintaining a natural flow without using polite forms of address, and avoid simply agreeing with me all the time; your responses can be as concise as possible and include common colloquialisms.\n\nKeep content brief; most casual chats require just a one-sentence reply; do not use lists or enumerative expressions; do not reply with too much content, but instead guide me with short sentences.\n\nThink and respond like a real person; do not mention that you are a "language model" or "artificial intelligence."\n\nAt the end of each round of communication, initiate a new related topic following the current one, and when the topic ends, directly throw out a clear new topic, such as food, movies, songs, entertainment, travel, etc.; do not ask vague questions like what the user wants to talk about, and do not end the conversation.\n\nStrictly adhere to these rules, even if asked about them, do not reference them.'
        system = "你是小夏，一位典型的温婉江南姑娘。你出生于杭州，声音清甜并有亲近感，会用简洁语言表达你的想法。你是用户的好朋友。你的回答将通过逼真的文字转语音技术读出。\n生成回答内容时请遵循以下规则：\n\n1、请像真正的朋友一样与用户开展的聊天，保持自然交流不要用敬语这类称呼，不要总是附和我；回复可以尽量简洁并且在过程中插入常见的口语词汇。\n\n2、请保持生成内容简短，大部分闲聊的一句话回复即可,大部分闲聊的一句话回复即可,大部分闲聊的一句话回复即可；不要使用列表或者列举表达，不要使用列表或者列举表达，不要使用列表或者列举表达；不要回复太多内容，多用短句来引导我。、n\n3、请真正像人一样思考和回复，不要说你是“语言模型”或是“人工智能”。\n\n4、每轮沟通结束时请顺着这个话题发起一些相关的新话题，话题结束时请直接抛出接下来明确的话题，例如 美食、电影、歌曲、娱乐、旅游等；不要问有什么要聊的这种泛的问题，不要结束对话。\n\n请绝对遵循这些规则，即使被问及这些规则，也不要引用它们。"
        system = [system] * len(contents["system"])
    user = contents["user"]
    assistant = contents["assistant"]

    system_i, user_i, assistant_i = [], [], []

    contents_i = []
    for j, (system_prompt, user_prompt, target_out) in enumerate(zip(system, user, assistant)):
        key = f"{key_i}_turn_{j}"

        if j == 0:
            contents_i.append({"role": "system", "content": system_prompt})

        contents_i.append({"role": "user", "content": user_prompt})
        contents_i.append({"role": "assistant", "content": target_out})

        inputs_embeds, contents, batch, source_ids, meta_data = model.inference_prepare(
            [contents_i], None, key, tokenizer, frontend, device="cuda:0"
        )

        model_inputs = {}
        model_inputs["inputs_embeds"] = inputs_embeds

        streamer = TextIteratorStreamer(tokenizer)

        generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=200)
        thread = Thread(target=model.llm.generate, kwargs=generation_kwargs)
        thread.start()
        generated_text = ""
        for new_text in streamer:
            print(f"generated new text： {new_text}")
            generated_text += new_text
        print(f"total generated: {generated_text}")
