import asyncio
import json
import websockets
import time
import logging
import tracemalloc
import numpy as np
import ssl
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
# asr
inference_pipeline_asr = pipeline(
    task=Tasks.auto_speech_recognition,
    model=args.asr_model,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    model_revision=None)


# vad
inference_pipeline_vad = pipeline(
    task=Tasks.voice_activity_detection,
    model=args.vad_model,
    model_revision=None,
    output_dir=None,
    batch_size=1,
    mode='online',
    ngpu=args.ngpu,
    ncpu=args.ncpu,
)

if args.punc_model != "":
    inference_pipeline_punc = pipeline(
        task=Tasks.punctuation,
        model=args.punc_model,
        model_revision="v1.0.2",
        ngpu=args.ngpu,
        ncpu=args.ncpu,
    )
else:
    inference_pipeline_punc = None

inference_pipeline_asr_online = pipeline(
    task=Tasks.auto_speech_recognition,
    model=args.asr_model_online,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    model_revision='v1.0.4',
    update_model='v1.0.4'
    mode='paraformer_streaming')

print("model loaded! only support one client at the same time now!!!!")

async def ws_reset(websocket):
    print("ws reset now, total num is ",len(websocket_users))
    websocket.param_dict_asr_online = {"cache": dict()}
    websocket.param_dict_vad = {'in_cache': dict(), "is_final": True}
    websocket.param_dict_asr_online["is_final"]=True
    audio_in=b''.join(np.zeros(int(16000),dtype=np.int16))
    inference_pipeline_vad(audio_in=audio_in, param_dict=websocket.param_dict_vad)
    inference_pipeline_asr_online(audio_in=audio_in, param_dict=websocket.param_dict_asr_online)
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
    await clear_websocket()
    websocket_users.add(websocket)
    websocket.param_dict_asr = {}
    websocket.param_dict_asr_online = {"cache": dict()}
    websocket.param_dict_vad = {'in_cache': dict(), "is_final": False}
    websocket.param_dict_punc = {'cache': list()}
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = False
    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    print("new user connected", flush=True)

    try:
        async for message in websocket:
            if isinstance(message, str):
                messagejson = json.loads(message)
        
                if "is_speaking" in messagejson:
                    websocket.is_speaking = messagejson["is_speaking"]
                    websocket.param_dict_asr_online["is_final"] = not websocket.is_speaking
                if "chunk_interval" in messagejson:
                    websocket.chunk_interval = messagejson["chunk_interval"]
                if "wav_name" in messagejson:
                    websocket.wav_name = messagejson.get("wav_name")
                if "chunk_size" in messagejson:
                    websocket.param_dict_asr_online["chunk_size"] = messagejson["chunk_size"]
                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"]
            if len(frames_asr_online) > 0 or len(frames_asr) > 0 or not isinstance(message, str):
                if not isinstance(message, str):
                    frames.append(message)
                    duration_ms = len(message)//32
                    websocket.vad_pre_idx += duration_ms
        
                    # asr online
                    frames_asr_online.append(message)
                    websocket.param_dict_asr_online["is_final"] = speech_end_i
                    if len(frames_asr_online) % websocket.chunk_interval == 0 or websocket.param_dict_asr_online["is_final"]:
                        if websocket.mode == "2pass" or websocket.mode == "online":
                            audio_in = b"".join(frames_asr_online)
                            await async_asr_online(websocket, audio_in)
                        frames_asr_online = []
                    if speech_start:
                        frames_asr.append(message)
                    # vad online
                    speech_start_i, speech_end_i = await async_vad(websocket, message)
                    if speech_start_i:
                        speech_start = True
                        beg_bias = (websocket.vad_pre_idx-speech_start_i)//duration_ms
                        frames_pre = frames[-beg_bias:]
                        frames_asr = []
                        frames_asr.extend(frames_pre)
                # asr punc offline
                if speech_end_i or not websocket.is_speaking:
                    # print("vad end point")
                    if websocket.mode == "2pass" or websocket.mode == "offline":
                        audio_in = b"".join(frames_asr)
                        await async_asr(websocket, audio_in)
                    frames_asr = []
                    speech_start = False
                    # frames_asr_online = []
                    # websocket.param_dict_asr_online = {"cache": dict()}
                    if not websocket.is_speaking:
                        websocket.vad_pre_idx = 0
                        frames = []
                        websocket.param_dict_vad = {'in_cache': dict()}
                    else:
                        frames = frames[-20:]

     
    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users,flush=True)
        await ws_reset(websocket)
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)


async def async_vad(websocket, audio_in):

    segments_result = inference_pipeline_vad(audio_in=audio_in, param_dict=websocket.param_dict_vad)

    speech_start = False
    speech_end = False
    
    if len(segments_result) == 0 or len(segments_result["text"]) > 1:
        return speech_start, speech_end
    if segments_result["text"][0][0] != -1:
        speech_start = segments_result["text"][0][0]
    if segments_result["text"][0][1] != -1:
        speech_end = True
    return speech_start, speech_end


async def async_asr(websocket, audio_in):
            if len(audio_in) > 0:
                # print(len(audio_in))
                audio_in = load_bytes(audio_in)
                
                rec_result = inference_pipeline_asr(audio_in=audio_in,
                                                    param_dict=websocket.param_dict_asr)
                # print(rec_result)
                if inference_pipeline_punc is not None and 'text' in rec_result and len(rec_result["text"])>0:
                    rec_result = inference_pipeline_punc(text_in=rec_result['text'],
                                                         param_dict=websocket.param_dict_punc)
                    # print("offline", rec_result)
                if 'text' in rec_result:
                    message = json.dumps({"mode": "2pass-offline", "text": rec_result["text"], "wav_name": websocket.wav_name})
                    await websocket.send(message)


async def async_asr_online(websocket, audio_in):
    if len(audio_in) > 0:
        audio_in = load_bytes(audio_in)
        # print(websocket.param_dict_asr_online.get("is_final", False))
        rec_result = inference_pipeline_asr_online(audio_in=audio_in,
                                                   param_dict=websocket.param_dict_asr_online)
        # print(rec_result)
        if websocket.mode == "2pass" and websocket.param_dict_asr_online.get("is_final", False):
            return
            #     websocket.param_dict_asr_online["cache"] = dict()
        if "text" in rec_result:
            if rec_result["text"] != "sil" and rec_result["text"] != "waiting_for_more_voice":
                # print("online", rec_result)
                message = json.dumps({"mode": "2pass-online", "text": rec_result["text"], "wav_name": websocket.wav_name})
                await websocket.send(message)

if len(args.certfile)>0:
    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    
    # Generate with Lets Encrypt, copied to this location, chown to current user and 400 permissions
    ssl_cert = args.certfile
    ssl_key = args.keyfile
    
    ssl_context.load_cert_chain(ssl_cert, keyfile=ssl_key)
    start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None,ssl=ssl_context)
else:
    start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()
