import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import argparse
import ssl
from scipy.spatial.distance import cosine
from modelscope.pipelines import pipeline
import torch

def to_python(obj):
    """递归地把 numpy / torch 等类型转成纯 Python，可 JSON 序列化。"""
    try:
        import numpy as np
        import torch
    except Exception:
        np = None
        torch = None

    # numpy 标量
    if np is not None and isinstance(obj, np.generic):
        return obj.item()

    # numpy 数组
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()

    # torch 张量
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()

    # 字典
    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}

    # 列表 / 元组
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]

    # 其他类型（str、int、float、bool、None 等）
    return obj


parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="0.0.0.0", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="grpc server port")
parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    help="model from modelscope",
)
# iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
# damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="model from modelscope",
)
parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="model from modelscope",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")
parser.add_argument(
    "--punc_model",
    type=str,
    default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    help="model from modelscope",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")
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

# asr
# model_asr = AutoModel(
#     model=args.asr_model,
#     #model_revision=args.asr_model_revision,
#     ngpu=args.ngpu,
#     ncpu=args.ncpu,
#     device=args.device,
#     disable_pbar=True,
#     disable_log=True,
# )

# ====== 离线 ASR：使用 paraformer-zh 多功能模型（带时间戳 + 热词 + 可选 VAD/PUNC）======
model_asr = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",

    # 这里顺带把内置的 VAD / PUNC 也挂上，方便模型自己做时间戳和标点
    vad_model="fsmn-vad",
    vad_model_revision="v2.0.4",
    punc_model="ct-punc-c",
    punc_model_revision="v2.0.4",
    # 如果以后想用内置声纹，也可以加：
    spk_model="cam++",
    spk_model_revision="v2.0.2",

    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)


# asr
model_asr_streaming = AutoModel(
    model=args.asr_model_online,
    model_revision=args.asr_model_online_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
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

if args.punc_model != "":
    model_punc = AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )
else:
    model_punc = None

# sv
model_sv = AutoModel(
    model="iic/speech_campplus_sv_zh-cn_16k-common",
    ngpu=args.ngpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)
# model_sv = pipeline(
#                 task='speaker-verification',
#                 model='damo/speech_campplus_sv_zh-cn_16k-common',
#                 model_revision='v1.0.0',
#                 device="cuda:0" if torch.cuda.is_available() else "cpu"
#                 )

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
    print("new user connected", flush=True)

    try:
        async for message in websocket:
            if isinstance(message, str):
                messagejson = json.loads(message)
                print("=============messagejson============",messagejson)
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
                if "hotwords" in messagejson:
                    hotword_data = messagejson["hotwords"]
                    websocket.status_dict_asr["hotword"] = hotword_data
                    websocket.status_dict_asr_online["hotword"] = hotword_data
                    print(f"热词已更新: {hotword_data}")
                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"]


            websocket.status_dict_vad["chunk_size"] = int(
                websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
            )
            if len(frames_asr_online) > 0 or len(frames_asr) >= 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    frames.append(message)
                    duration_ms = len(message) // 32
                    websocket.vad_pre_idx += duration_ms

                    # asr online
                    frames_asr_online.append(message)
                    websocket.status_dict_asr_online["is_final"] = speech_end_i != -1
                    if (
                        len(frames_asr_online) % websocket.chunk_interval == 0
                        or websocket.status_dict_asr_online["is_final"]
                    ):
                        if websocket.mode == "2pass" or websocket.mode == "online":
                            audio_in = b"".join(frames_asr_online)
                            try:
                                await async_asr_online(websocket, audio_in)
                            except:
                                print(f"error in asr streaming, {websocket.status_dict_asr_online}")
                        frames_asr_online = []
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

                    if websocket.mode == "2pass" or websocket.mode == "offline":
                        audio_in = b"".join(frames_asr)
                        try:
                            await async_asr(websocket, audio_in)
                        except:
                            print("error in asr offline")
                    frames_asr = []
                    speech_start = False
                    frames_asr_online = []
                    websocket.status_dict_asr_online["cache"] = {}
                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.status_dict_vad["cache"] = {}
                    else:
                        frames = frames[-20:]

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


async def async_asr(websocket, audio_in):
    if len(audio_in) > 0:
        # 1. ASR 原始结果
        rec_result = model_asr.generate(
            input=audio_in,
            batch_size_s=300,
            **websocket.status_dict_asr
        )[0]

        print("offline_asr, raw:", rec_result)
        print("offline_asr, keys:", rec_result.keys())

        # 先取出原始字段
        text = rec_result.get("text", "")
        timestamp = rec_result.get("timestamp", None)
        sentence_info = rec_result.get("sentence_info", None)

        # 2. 声纹识别（你现在这一段是 OK 的，就不重复贴了）
        spk_name = "unknown"
        best_score = 0.0
        try:
            sv_out = model_sv.generate(
                input=audio_in,
                embedding=True
            )[0]
            embedding = sv_out["spk_embedding"][0].cpu().numpy()

            if len(speaker_db) > 0:
                for name, ref_embedding in speaker_db.items():
                    if ref_embedding is None:
                        continue
                    data_list = json.loads(ref_embedding)
                    arr = np.array(data_list, dtype=np.float32)

                    similarity = 1.0 - cosine(embedding, arr)
                    print("sv similarity with {}: {}".format(name, similarity))

                    if similarity > best_score and similarity > 0.2:
                        best_score = similarity
                        spk_name = name
        except Exception as e:
            print(f"声纹识别失败: {e}")

        # 3. 标点：只更新 text，不覆盖时间戳
        punc_array = None
        if model_punc is not None and len(text) > 0:
            print("offline, before punc", rec_result, "cache", websocket.status_dict_punc)
            punc_result = model_punc.generate(
                input=text,
                **websocket.status_dict_punc
            )[0]
            print("offline, after punc", punc_result)

            if "text" in punc_result and len(punc_result["text"]) > 0:
                text = punc_result["text"]
            if "punc_array" in punc_result:
                punc_array = punc_result["punc_array"]

        # 4. 构造最终 message，并做类型“净化”
        if len(text) > 0:
            print("======offline final text:", text)
            mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode

            message = {
                "mode": mode,
                "spk_name": spk_name,
                "spk_score": float(best_score),  # 显式转 float 一下更保险
                "text": text,
                "wav_name": websocket.wav_name,
                "is_final": bool(websocket.is_speaking),
            }

            # 时间戳等：可能带 numpy 类型，统一丢给 to_python 处理
            if timestamp is not None:
                message["timestamp"] = to_python(timestamp)
            if sentence_info is not None:
                message["sentence_info"] = to_python(sentence_info)
            if punc_array is not None:
                message["punc_array"] = to_python(punc_array)

            # 关键：json.dumps 前做完所有类型转换
            try:
                await websocket.send(json.dumps(message, ensure_ascii=False))
            except Exception as e:
                print("send json failed:", e)
                # 可以再打印一份类型信息排查
                print("message types:", {k: type(v) for k, v in message.items()})
        # end if len(text) > 0

    else:
        # 空音频的情况
        mode = "2pass-offline" if "2pass" in websocket.mode else websocket.mode
        message = {
            "mode": mode,
            "text": "",
            "wav_name": websocket.wav_name,
            "is_final": bool(websocket.is_speaking),
        }
        await websocket.send(json.dumps(message, ensure_ascii=False))



async def async_asr_online(websocket, audio_in):
    if len(audio_in) > 0:
        # print(websocket.status_dict_asr_online.get("is_final", False))
        rec_result = model_asr_streaming.generate(
            input=audio_in, **websocket.status_dict_asr_online
        )[0]
        print("online, ", rec_result)
        if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
            return
            #     websocket.status_dict_asr_online["cache"] = dict()
        if len(rec_result["text"]):
            mode = "2pass-online" if "2pass" in websocket.mode else websocket.mode
            message = json.dumps(
                {
                    "mode": mode,
                    "text": rec_result["text"],
                    "wav_name": websocket.wav_name,
                    "is_final": websocket.is_speaking,
                }
            )
            await websocket.send(message)


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
