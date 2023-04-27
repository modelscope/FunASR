import asyncio
import json
import websockets
import time
from queue import Queue
import threading
import argparse

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging
import tracemalloc
import numpy as np

tracemalloc.start()

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)


websocket_users = set()  #维护客户端列表

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="0.0.0.0",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10095,
                    required=False,
                    help="grpc server port")
parser.add_argument("--asr_model",
                    type=str,
                    default="damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                    help="model from modelscope")
parser.add_argument("--vad_model",
                    type=str,
                    default="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                    help="model from modelscope")

parser.add_argument("--punc_model",
                    type=str,
                    default="damo/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
                    help="model from modelscope")
parser.add_argument("--ngpu",
                    type=int,
                    default=1,
                    help="0 for cpu, 1 for gpu")

args = parser.parse_args()

print("model loading")

def load_bytes(input):
    middle_data = np.frombuffer(input, dtype=np.int16)
    middle_data = np.asarray(middle_data)
    if middle_data.dtype.kind not in 'iu':
        raise TypeError("'middle_data' must be an array of integers")
    dtype = np.dtype('float32')
    if dtype.kind != 'f':
        raise TypeError("'dtype' must be a floating point type")

    i = np.iinfo(middle_data.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    array = np.frombuffer((middle_data.astype(dtype) - offset) / abs_max, dtype=np.float32)
    return array

inference_pipeline_asr_online = pipeline(
    task=Tasks.auto_speech_recognition,
    # model='damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online',
    model='damo/speech_paraformer_asr_nat-zh-cn-16k-common-vocab8404-online',
    model_revision=None)


print("model loaded")



async def ws_serve(websocket, path):
    frames_online = []
    global websocket_users
    websocket.send_msg = Queue()
    websocket_users.add(websocket)
    websocket.param_dict_asr_online = {"cache": dict()}
    websocket.speek_online = Queue()
    ss_online = threading.Thread(target=asr_online, args=(websocket,))
    ss_online.start()
    ss_ws_send = threading.Thread(target=ws_send, args=(websocket,))
    ss_ws_send.start()
    try:
        async for message in websocket:
            message = json.loads(message)
            audio = bytes(message['audio'], 'ISO-8859-1')
            chunk = message["chunk"]
            chunk_num = 500//chunk
            is_speaking = message["is_speaking"]
            websocket.param_dict_asr_online["is_final"] = not is_speaking
            frames_online.append(audio)

            if len(frames_online) % chunk_num == 0 or not is_speaking:
                audio_in = b"".join(frames_online)
                websocket.speek_online.put(audio_in)
                frames_online = []

            # if not websocket.send_msg.empty():
            #     await websocket.send(websocket.send_msg.get())
            #     websocket.send_msg.task_done()

     
    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users)    # 链接断开
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")    # 无效状态
    except Exception as e:
        print("Exception:", e)
 


def ws_send(websocket):  # ASR推理
    global inference_pipeline_asr_online
    global websocket_users
    while websocket in websocket_users:
        if not websocket.speek_online.empty():
            await websocket.send(websocket.send_msg.get())
            websocket.send_msg.task_done()
        time.sleep(0.005)


def asr_online(websocket):  # ASR推理
    global websocket_users
    while websocket in websocket_users:
        if not websocket.send_msg.empty():
            audio_in = websocket.speek_online.get()
            websocket.speek_online.task_done()
            if len(audio_in) > 0:
                # print(len(audio_in))
                audio_in = load_bytes(audio_in)
                # print(audio_in.shape)
                print(websocket.param_dict_asr_online["is_final"])
                rec_result = inference_pipeline_asr_online(audio_in=audio_in,
                                                           param_dict=websocket.param_dict_asr_online)
                if websocket.param_dict_asr_online["is_final"]:
                    websocket.param_dict_asr_online["cache"] = dict()
                
                print(rec_result)
                if "text" in rec_result:
                    if rec_result["text"] != "sil" and rec_result["text"] != "waiting_for_more_voice":
                        message = json.dumps({"mode": "online", "text": rec_result["text"]})
                        websocket.send_msg.put(message)  # 存入发送队列  直接调用send发送不了
        
        time.sleep(0.005)


start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()