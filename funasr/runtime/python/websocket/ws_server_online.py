import asyncio
import json
import websockets
import time
from queue import Queue
import threading
import logging
import tracemalloc
import numpy as np

from parse_args import args
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
from funasr.runtime.python.onnxruntime.funasr_onnx.utils.frontend import load_bytes

tracemalloc.start()

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)


websocket_users = set()


print("model loading")

inference_pipeline_asr_online = pipeline(
    task=Tasks.auto_speech_recognition,
    model=args.asr_model_online,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    model_revision='v1.0.4')

print("model loaded")



async def ws_serve(websocket, path):
    frames_online = []
    global websocket_users
    websocket.send_msg = Queue()
    websocket_users.add(websocket)
    websocket.param_dict_asr_online = {"cache": dict()}
    websocket.speek_online = Queue()

    try:
        async for message in websocket:
            message = json.loads(message)
            is_finished = message["is_finished"]
            if not is_finished:
                audio = bytes(message['audio'], 'ISO-8859-1')

                is_speaking = message["is_speaking"]
                websocket.param_dict_asr_online["is_final"] = not is_speaking
                websocket.wav_name = message.get("wav_name", "demo")
                websocket.param_dict_asr_online["chunk_size"] = message["chunk_size"]
                
                frames_online.append(audio)
                if len(frames_online) % message["chunk_interval"] == 0 or not is_speaking:
                    audio_in = b"".join(frames_online)
                    await async_asr_online(websocket,audio_in)
                    frames_online = []


     
    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users)
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)
 
async def async_asr_online(websocket,audio_in):
            if len(audio_in) > 0:
                audio_in = load_bytes(audio_in)
                rec_result = inference_pipeline_asr_online(audio_in=audio_in,
                                                           param_dict=websocket.param_dict_asr_online)
                if websocket.param_dict_asr_online["is_final"]:
                    websocket.param_dict_asr_online["cache"] = dict()
                if "text" in rec_result:
                    if rec_result["text"] != "sil" and rec_result["text"] != "waiting_for_more_voice":
                        # if len(rec_result["text"])>0:
                        #     rec_result["text"][0]=rec_result["text"][0] #.replace(" ","")
                        message = json.dumps({"mode": "online", "text": rec_result["text"], "wav_name": websocket.wav_name})
                        await websocket.send(message)



start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()