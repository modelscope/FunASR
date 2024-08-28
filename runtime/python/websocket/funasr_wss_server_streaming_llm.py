import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
import os
import torch
import torchaudio
from transformers import TextIteratorStreamer
from threading import Thread
import traceback

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


from funasr import AutoModel
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
audio_encoder_dir = "/nfs/yangyexin.yyx/init_model/iic/SenseVoiceModelscope_0712"
device = "cuda:0"
all_file_paths = [
    "/nfs/yangyexin.yyx/init_model/audiolm_v14_20240824_train_encoder_all_20240822_lr1e-4_warmup2350/"
]

llm_kwargs = {"num_beams": 1, "do_sample": False}
unfix_len = 5
max_streaming_res_onetime = 100

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

def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in "iu":
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype("float32")
    if dtype.kind != "f":
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array

async def streaming_transcribe(websocket, audio_in, his_state=None, prompt=None):
    if his_state is None:
        his_state = model_dict
    model = his_state["model"]
    tokenizer = his_state["tokenizer"]

    if websocket.streaming_state is None:
        previous_asr_text = ""
    else:
        previous_asr_text = websocket.streaming_state["previous_asr_text"]
    
    if prompt is None:
        prompt = "Copy:"

    audio_seconds = load_bytes(audio_in).shape[0] / 16000
    print(f"Streaming audio length: {audio_seconds} seconds")
    
    asr_content = []
    system_prompt = "You are a helpful assistant."
    asr_content.append({"role": "system", "content": system_prompt})

    asr_user_prompt = f"{prompt}<|startofspeech|>!!<|endofspeech|><|im_end|>\n<|im_start|>assistant\n{previous_asr_text}"

    asr_content.append({"role": "user", "content": asr_user_prompt, "audio": audio_in})
    asr_content.append({"role": "assistant", "content": "target_out"})

    streaming_asr_time_beg = time.time()
    asr_inputs_embeds, contents, batch, source_ids, meta_data = model.inference_prepare(
        [asr_content], None, "test_demo", tokenizer, frontend, device=device, infer_with_assistant_input=True
    )
    asr_model_inputs = {}
    asr_model_inputs["inputs_embeds"] = asr_inputs_embeds

    print("previous_asr_text:", previous_asr_text)

    streamer = TextIteratorStreamer(tokenizer)
    generation_kwargs = dict(asr_model_inputs, streamer=streamer, max_new_tokens=1024)
    thread = Thread(target=model.llm.generate, kwargs=generation_kwargs)
    thread.start()
    
    onscreen_asr_res = previous_asr_text
    beg_llm = time.time()
    for new_text in streamer:
        end_llm = time.time()
        print(
            f"generated new text： {new_text}, time_llm_decode: {end_llm - beg_llm:.2f}"
        )
        if len(new_text) > 0:
            onscreen_asr_res += new_text.replace("<|im_end|>", "")

            mode = "online"
            message = json.dumps(
                {
                    "mode": mode,
                    "text": onscreen_asr_res,
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)

    streaming_asr_time_end = time.time()
    print(f"Streaming ASR inference time: {streaming_asr_time_end - streaming_asr_time_beg}")

    asr_text_len = len(tokenizer.encode(onscreen_asr_res))

    if asr_text_len > unfix_len and audio_seconds > 1.1:
        if asr_text_len <= max_streaming_res_onetime:
            previous_asr_text = tokenizer.decode(tokenizer.encode(onscreen_asr_res)[:-unfix_len])
        else:
            onscreen_asr_res = previous_asr_text
    else:
        previous_asr_text = ""

    websocket.streaming_state = {}
    websocket.streaming_state["previous_asr_text"] = previous_asr_text
    print("fix asr part:", previous_asr_text)


print("model loaded! only support one client at the same time now!!!!")


async def ws_reset(websocket):
    print("ws reset now, total num is ", len(websocket_users))

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.streaming_state = None
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True

    await websocket.close()


async def clear_websocket():
    for websocket in websocket_users:
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path):
    frames = []
    frames_asr = []
    global websocket_users
    # await clear_websocket()
    websocket_users.add(websocket)
    websocket.status_dict_asr = {}
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}

    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1
    websocket.wav_name = "microphone"
    websocket.mode = "online"
    websocket.streaming_state = None
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
            if len(frames_asr) > 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    frames.append(message)
                    duration_ms = len(message) // 32
                    websocket.vad_pre_idx += duration_ms

                    # asr online
                    websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                    if (
                        (len(frames_asr) % websocket.chunk_interval == 0
                        or websocket.status_dict_asr_online["is_final"]) 
                        and len(frames_asr) != 0
                    ):
                        if websocket.mode == "2pass" or websocket.mode == "online":
                            audio_in = b"".join(frames_asr)
                            try:
                                await streaming_transcribe(websocket, audio_in)
                            except:
                                print(f"error in asr streaming, {websocket.status_dict_asr_online}")
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
                    frames_asr = []
                    speech_start = False
                    websocket.status_dict_asr_online["cache"] = {}
                    websocket.streaming_state = None
                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.status_dict_vad["cache"] = {}
                        websocket.streaming_state = None
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
