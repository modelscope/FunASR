import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
import nls
from collections import deque
import threading
from datetime import datetime


class NlsTtsSynthesizer:
    def __init__(
        self,
        websocket,
        tts_fifo,
        token,
        appkey,
        url="wss://nls-gateway-cn-beijing.aliyuncs.com/ws/v1",
    ):
        self.websocket = websocket
        self.tts_fifo = tts_fifo
        self.url = url
        self.token = token
        self.appkey = appkey
        self.sdk = None
        self.started = False
        self.count = 0
        self.init_sdk()

    def init_sdk(self):
        # 配置回调函数
        self.sdk = nls.NlsStreamInputTtsSynthesizer(
            url=self.url,
            token=self.token,
            appkey=self.appkey,
            on_data=self.on_data,
            on_sentence_begin=self.on_sentence_begin,
            on_sentence_synthesis=self.on_sentence_synthesis,
            on_sentence_end=self.on_sentence_end,
            on_completed=self.on_completed,
            on_error=self.on_error,
            on_close=self.on_close,
            callback_args=[],
        )

    def on_data(self, data, *args):
        print(f"on_data: {datetime.now()}, len: {len(data)}")
        self.count += len(data)
        self.tts_fifo.append(data)
        # with open('tts_server.pcm', 'ab') as file:
        #    file.write(data)

    def on_sentence_begin(self, message, *args):
        print("on sentence begin =>{}".format(message))

    def on_sentence_synthesis(self, message, *args):
        print("on sentence synthesis =>{}".format(message))

    def on_sentence_end(self, message, *args):
        print("on sentence end =>{}".format(message))

    def on_completed(self, message, *args):
        print("on message data cout: =>{}".format(self.count))
        print("on completed =>{}".format(message))
        self.started = False

    def on_error(self, message, *args):
        print("on_error args=>{}".format(args))

    def on_close(self, *args):
        print("on_close: args=>{}".format(args))

    def start(self):
        self.sdk.startStreamInputTts()
        self.started = True

    def send_text(self, text):
        self.sdk.sendStreamInputTts(text)

    def stop(self):
        self.sdk.stopStreamInputTts()


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10096, required=False, help="grpc server port")
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

appkey = "xxx"
appkey_token = "xxx"
if "appkey" in os.environ:
    appkey = os.environ["appkey"]
    appkey_token = os.environ["appkey_token"]

from modelscope.hub.snapshot_download import snapshot_download

os.environ["MODELSCOPE_CACHE"] = "/mnt/workspace"
llm_dir = snapshot_download("qwen/Qwen2-7B-Instruct", cache_dir=None, revision="master")
audio_encoder_dir = snapshot_download("iic/SenseVoice", cache_dir=None, revision="master")

# llm_dir = "/cpfs_speech/zhifu.gzf/init_model/qwen/Qwen2-7B-Instruct"
# audio_encoder_dir = "/nfs/zhifu.gzf/init_model/SenseVoiceLargeModelscope"

device = "cuda:0"

all_file_paths = [
    # "/nfs/zhifu.gzf/init_model/Speech2Text_Align_V0712_modelscope"
    "FunAudioLLM/Speech2Text_Align_V0712",
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


async def send_to_client(websocket, syntheszier, tts_fifo):
    # Sending tts data to the client
    while True:
        if websocket.open and (syntheszier.started or len(tts_fifo) > 0):
            try:
                if len(tts_fifo) > 0:
                    await websocket.send(tts_fifo.popleft())
                else:
                    await asyncio.sleep(0.01)
            except Exception as e:
                print(f"Error sending data to client: {e}")
        else:
            print("WebSocket connection is not open or syntheszier is not started.")
            break


def tts_sync_thread(coro):
    asyncio.run(coro)


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
    print(f"model_inference: {datetime.now()}")
    fifo_queue = deque()
    synthesizer = NlsTtsSynthesizer(
        websocket=websocket, tts_fifo=fifo_queue, token=appkey_token, appkey=appkey
    )
    synthesizer.start()
    beg0 = time.time()
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

    print(f"contents_i: {len(contents_i)}")

    inputs_embeds, contents, batch, source_ids, meta_data = model.inference_prepare(
        [contents_i], None, "test_demo", tokenizer, frontend, device=device
    )
    print(f"speech_encoder: {datetime.now()}")
    model_inputs = {}
    model_inputs["inputs_embeds"] = inputs_embeds

    streamer = TextIteratorStreamer(tokenizer)

    generation_kwargs = dict(model_inputs, streamer=streamer, max_new_tokens=1024)
    thread = Thread(target=model.llm.generate, kwargs=generation_kwargs)
    thread.start()
    res = ""
    beg_llm = time.time()
    tts_thread = Thread(
        target=tts_sync_thread, args=(send_to_client(websocket, synthesizer, fifo_queue),)
    )
    tts_thread.start()
    for new_text in streamer:
        end_llm = time.time()
        print(
            f"{datetime.now()}, generated new text： {new_text}, time_fr_receive: {end_llm - beg0:.2f}, time_llm_decode: {end_llm - beg_llm:.2f}"
        )
        if len(new_text) > 0:
            new_text = new_text.replace("<|im_end|>", "")
            res += new_text
            synthesizer.send_text(new_text)

            contents_i[-1]["content"] = res
            websocket.llm_state["contents_i"] = contents_i
            # history[-1][1] = res

            mode = "2pass-online"
            message = json.dumps(
                {
                    "mode": mode,
                    "text": new_text,
                    "wav_name": websocket.wav_name,
                    "is_final": False,
                }
            )
            # print(f"online: {message}")
            await websocket.send(message)

    synthesizer.stop()
    # await tts_to_client_task
    tts_thread.join()
    mode = "2pass-offline"
    message = json.dumps(
        {
            "mode": mode,
            "text": res,
            "wav_name": websocket.wav_name,
            "is_final": True,
        }
    )
    # print(f"offline: {message}")
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
    print(f"connected time: {datetime.now()}")

    try:
        async for message in websocket:
            print(f"reci time: {datetime.now()}")
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
                # if speech_end_i != -1 or not websocket.is_speaking:
                if not websocket.is_speaking:
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


if False:  # len(args.certfile) > 0:
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
