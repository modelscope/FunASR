import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="127.0.0.1", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="grpc server port")
parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    help="model from modelscope",
)
parser.add_argument("--asr_model_revision", type=str, default="master", help="")
parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="model from modelscope",
)
parser.add_argument("--asr_model_online_revision", type=str, default="master", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="model from modelscope",
)
parser.add_argument("--vad_model_revision", type=str, default="master", help="")
parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")
parser.add_argument(
    "--certfile",
    type=str,
    default="../../ssl_key/server.crt",
    required=False,
    help="certfile for ssl",
)

parser.add_argument(
    "--keyfile",
    type=str,
    default="../../ssl_key/server.key",
    required=False,
    help="keyfile for ssl",
)
args = parser.parse_args()


websocket_users = set()

print("model loading")
from funasr import AutoModel

# # asr
# model_asr = AutoModel(
#     model=args.asr_model,
#     model_revision=args.asr_model_revision,
#     ngpu=args.ngpu,
#     ncpu=args.ncpu,
#     device=args.device,
#     disable_pbar=False,
#     disable_log=True,
# )

# vad
model_vad = AutoModel(
    model=args.vad_model,
    model_revision=args.vad_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
    # chunk_size=60,
)


# async def async_asr(websocket, audio_in):
#     if len(audio_in) > 0:
#         # print(len(audio_in))
#         print(type(audio_in))
#         rec_result = model_asr.generate(input=audio_in, **websocket.status_dict_asr)[0]
#         print("offline_asr, ", rec_result)
#
#
#         if len(rec_result["text"]) > 0:
#             # print("offline", rec_result)
#             mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
#             message = json.dumps(
#                 {
#                     "mode": mode,
#                     "text": rec_result["text"],
#                     "wav_name": websocket.wav_name,
#                     "is_final": websocket.is_speaking,
#                 }
#             )
#             await websocket.send(message)

import os

# from install_model_requirements import install_requirements
#
# install_requirements()

# import librosa
# import base64
# import io
# import gradio as gr
# import re

import numpy as np
import torch
import torchaudio
from transformers import TextIteratorStreamer
from threading import Thread
import torch
import time
import traceback

# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)

from funasr import AutoModel

import re

import sys

from modelscope.hub.api import HubApi

api = HubApi()
if "key" in os.environ:
    key = os.environ["key"]
    api.login(key)

from modelscope.hub.snapshot_download import snapshot_download

# os.environ["MODELSCOPE_CACHE"] = "/nfs/zhifu.gzf/modelscope"
# llm_dir = snapshot_download('qwen/Qwen2-7B-Instruct', cache_dir=None, revision='master')
# audio_encoder_dir = snapshot_download('iic/SenseVoice', cache_dir=None, revision='master')

llm_dir = "/cpfs_speech/zhifu.gzf/init_model/qwen/Qwen2-7B-Instruct"
audio_encoder_dir = "/nfs/zhifu.gzf/init_model/SenseVoiceLargeModelscope"

device = "cuda:0"

all_file_paths = [
    "/nfs/zhifu.gzf/init_model/Speech2Text_Align_V0712_modelscope"
    # "FunAudioLLM/Speech2Text_Align_V0712",
    # "FunAudioLLM/Speech2Text_Align_V0718",
    # "FunAudioLLM/Speech2Text_Align_V0628",
]

llm_kwargs = {"num_beams": 1, "do_sample": False}

ckpt_dir = all_file_paths[0]

model_llm = AutoModel(
    model=ckpt_dir,
    device=device,
    fp16=False,
    bf16=False,
    llm_dtype="bf16",
    max_length=1024,
    llm_kwargs=llm_kwargs,
    llm_conf={"init_param_path": llm_dir},
    tokenizer_conf={"init_param_path": llm_dir},
    audio_encoder=audio_encoder_dir,
)

model = model_llm.model
frontend = model_llm.kwargs["frontend"]
tokenizer = model_llm.kwargs["tokenizer"]

model_dict = {"model": model, "frontend": frontend, "tokenizer": tokenizer}


async def model_inference(
    websocket,
    audio_in,
    his_state=None,
    system_prompt="",
    state=None,
    turn_num=5,
    history=None,
    text_usr="",
):
    if his_state is None:
        his_state = model_dict
    model = his_state["model"]
    frontend = his_state["frontend"]
    tokenizer = his_state["tokenizer"]
    # print(f"text_inputs: {text_inputs}")
    # print(f"audio_in: {audio_in}")
    # print(f"websocket.llm_state: {websocket.llm_state}")

    if websocket.llm_state is None:
        websocket.llm_state = {"contents_i": []}
    # print(f"history: {history}")
    # if history is None:
    #     history = []

    # audio_in = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/tmp/1.wav"
    # user_prompt = f"<|startofspeech|>!{audio_in}<|endofspeech|>"
    user_prompt = f"{text_usr}<|startofspeech|>!!<|endofspeech|>"

    contents_i = websocket.llm_state["contents_i"]
    # print(f"contents_i_0: {contents_i}")
    if len(system_prompt) == 0:
        system_prompt = "你是小夏，一位典型的温婉江南姑娘。你出生于杭州，声音清甜并有亲近感，会用简洁语言表达你的想法。你是用户的好朋友。你的回答将通过逼真的文字转语音技术读出。"

    if len(contents_i) < 1:
        contents_i.append({"role": "system", "content": system_prompt})
    contents_i.append({"role": "user", "content": user_prompt, "audio": audio_in})
    contents_i.append({"role": "assistant", "content": "target_out"})
    if len(contents_i) > 2 * turn_num + 1:
        print(
            f"clip dialog pairs from: {len(contents_i)} to: {turn_num}, \ncontents_i_before_clip: {contents_i}"
        )
        contents_i = [{"role": "system", "content": system_prompt}] + contents_i[3:]

    print(f"contents_i: {contents_i}")

    inputs_embeds, contents, batch, source_ids, meta_data = model.inference_prepare(
        [contents_i], None, "test_demo", tokenizer, frontend, device=device
    )
    model_inputs = {}
    model_inputs["inputs_embeds"] = inputs_embeds

    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=1024)
    thread = Thread(target=model.llm.generate, kwargs=generation_kwargs)
    thread.start()
    res = ""
    beg_llm = time.time()
    for new_text in streamer:
        end_llm = time.time()
        print(f"generated new text： {new_text}, time: {end_llm-beg_llm:.2f}")

        if len(new_text) > 0:
            res += new_text.replace("<|im_end|>", "")
            contents_i[-1]["content"] = res
            websocket.llm_state["contents_i"] = contents_i
            # history[-1][1] = res

            mode = "2pass-online"
            message = json.dumps(
                {
                    "mode": mode,
                    "text": new_text,
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            print(f"online: {message}")
            await websocket.send(message)

    mode = "2pass-offline"
    message = json.dumps(
        {
            "mode": mode,
            "text": res,
            "wav_name": websocket.wav_name,
            "is_final": websocket.is_speaking,
        }
    )
    print(f"offline: {message}")
    await websocket.send(message)


print("model loaded! only support one client at the same time now!!!!")


async def ws_reset(websocket):
    print("ws reset now, total num is ", len(websocket_users))

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True
    websocket.status_dict_punc["cache"] = {}

    await websocket.close()


async def clear_websocket():
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path):
    frames = []
    frames_asr = []
    frames_asr_online = []
    global websocket_users
    # await clear_websocket()
    websocket_users.add(websocket)
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}
    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    websocket.llm_state = None
    print("new user connected", flush=True)

    try:
        async for message in websocket:
            if isinstance(message, str):
                messagejson = json.loads(message)

                if "is_speaking" in messagejson:
                    websocket.is_speaking = messagejson["is_speaking"]
                    websocket.status_dict_asr_online["is_final"] = not websocket.is_speaking
                if "chunk_interval" in messagejson:
                    websocket.chunk_interval = messagejson["chunk_interval"]
                if "wav_name" in messagejson:
                    websocket.wav_name = messagejson.get("wav_name")
                if "chunk_size" in messagejson:
                    chunk_size = messagejson["chunk_size"]
                    if isinstance(chunk_size, str):
                        chunk_size = chunk_size.split(",")
                    websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]
                if "encoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson[
                        "encoder_chunk_look_back"
                    ]
                if "decoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson[
                        "decoder_chunk_look_back"
                    ]
                if "hotword" in messagejson:
                    websocket.status_dict_asr["hotword"] = messagejson["hotwords"]
                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"]

            websocket.status_dict_vad["chunk_size"] = int(
                websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
            )
            if len(frames_asr_online) > 0 or len(frames_asr) > 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    frames.append(message)
                    duration_ms = len(message) // 32
                    websocket.vad_pre_idx += duration_ms

                    if speech_start:
                        frames_asr.append(message)
                    # vad online
                    try:
                        speech_start_i, speech_end_i = await async_vad(websocket, message)
                    except:
                        print("error in vad")
                    if speech_start_i != -1:
                        speech_start = True
                        beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                        frames_pre = frames[-beg_bias:]
                        frames_asr = []
                        frames_asr.extend(frames_pre)
                # asr punc offline
                if speech_end_i != -1 or not websocket.is_speaking:
                    # print("vad end point")
                    if websocket.mode == "2pass" or websocket.mode == "offline":
                        audio_in = b"".join(frames_asr)
                        try:
                            # await async_asr(websocket, audio_in)
                            await model_inference(websocket, audio_in)
                        except Exception as e:
                            print(f"{str(e)}, {traceback.format_exc()}")
                    frames_asr = []
                    speech_start = False

                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.status_dict_vad["cache"] = {}
                    else:
                        frames = frames[-20:]
            else:
                print(f"message: {message}")
    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users, flush=True)
        await ws_reset(websocket)
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)


async def async_vad(websocket, audio_in):
    segments_result = model_vad.generate(input=audio_in, **websocket.status_dict_vad)[0]["value"]
    # print(segments_result)

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


if len(args.certfile) > 0:
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)

    # Generate with Lets Encrypt, copied to this location, chown to current user and 400 permissions
    ssl_cert = args.certfile
    ssl_key = args.keyfile

    ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
    start_server = websockets.serve(
        ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
    )
else:
    start_server = websockets.serve(
        ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None
    )
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
