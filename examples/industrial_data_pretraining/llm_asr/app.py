# coding=utf-8

import librosa
import base64
import io
import gradio as gr
import re

import numpy as np
import torch
import torchaudio

# from modelscope import HubApi
#
# api = HubApi()
#
# api.login('')

from funasr import AutoModel

# model = "/Users/zhifu/Downloads/modelscope_models/SenseVoiceCTC"
# model = "iic/SenseVoiceCTC"
# model = AutoModel(model=model,
# 				  vad_model="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
# 				  vad_kwargs={"max_single_segment_time": 30000},
# 				  trust_remote_code=True,
# 				  )

import re
import os
import sys

if len(sys.argv) > 1:
    ckpt_dir = sys.argv[1]
    ckpt_id = sys.argv[2]
    jsonl = sys.argv[3]
    output_dir = sys.argv[4]
    device = sys.argv[5]
    new_sys = False
    if len(sys.argv) > 6:
        new_sys = True
else:
    ckpt_dir = "/nfs/beinian.lzr/workspace/GPT-4o/Exp/exp7/5m-8gpu/exp5-1-0619"
    ckpt_id = "model.pt.ep6"
    jsonl = (
        "/nfs/beinian.lzr/workspace/GPT-4o/Data/Speech2Text/TestData/s2tchat.v20240619.test.jsonl"
    )
    dataset = jsonl.split("/")[-1]
    output_dir = os.path.join(ckpt_dir, f"inference-{ckpt_id}", dataset)


model = AutoModel(
    model=ckpt_dir,
    init_param=f"{os.path.join(ckpt_dir, ckpt_id)}",
    output_dir=output_dir,
    device=device,
    fp16=False,
    bf16=False,
    llm_dtype="bf16",
)


def model_inference(input_wav, text_inputs, fs=16000):

    if isinstance(input_wav, tuple):
        fs, input_wav = input_wav
        input_wav = input_wav.astype(np.float32) / np.iinfo(np.int16).max
        if len(input_wav.shape) > 1:
            input_wav = input_wav.mean(-1)
        if fs != 16000:
            print(f"audio_fs: {fs}")
            resampler = torchaudio.transforms.Resample(fs, 16000)
            input_wav_t = torch.from_numpy(input_wav).to(torch.float32)
            input_wav = resampler(input_wav_t[None, :])[0, :].numpy().astype("float32")

    input_wav_byte = input_wav.tobytes()

    contents_i = []
    system_prompt = text_inputs
    user_prompt = f"<|startofspeech|>!!{input_wav_byte}<|endofspeech|>"
    contents_i.append({"role": "system", "content": system_prompt})
    contents_i.append({"role": "user", "content": user_prompt})
    contents_i.append({"role": "assistant", "content": "target_out"})

    res = model.generate(
        input=[contents_i],
        tearchforing=tearchforing,
        cache={},
        key=key,
    )

    print(res)

    return res


audio_examples = [
    [
        "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav",
        "You are a helpful assistant.",
    ],
]

description = """
Upload an audio file or input through a microphone, then type te System Prompt.


"""


def launch():
    with gr.Blocks() as demo:
        gr.Markdown(description)
        with gr.Row():
            with gr.Column():
                audio_inputs = gr.Audio(label="Upload audio or use the microphone")
                text_inputs = gr.Text(label="System Prompt", value="You are a helpful assistant.")

                # with gr.Accordion("Configuration"):
                # 	# task_inputs = gr.Radio(choices=["Speech Recognition", "Rich Text Transcription"],
                # 	# 					   value="Speech Recognition", label="Task")
                # 	language_inputs = gr.Dropdown(choices=["auto", "zh", "en", "yue", "ja", "ko", "nospeech"],
                # 								  value="auto",
                # 								  label="Language")
            gr.Examples(examples=audio_examples, inputs=[audio_inputs, text_inputs])

        fn_button = gr.Button("Start")

        text_outputs = gr.HTML(label="Results")

        fn_button.click(model_inference, inputs=[audio_inputs, text_inputs], outputs=text_outputs)
        # with gr.Accordion("More examples"):
        # 	gr.HTML(centered_table_html)
    demo.launch()


if __name__ == "__main__":
    # iface.launch()
    launch()
