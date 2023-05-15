import asyncio
import json
import websockets
import time
from queue import Queue
import threading
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

inference_pipeline_asr_online = pipeline(
	task=Tasks.auto_speech_recognition,
	model=args.asr_model_online,
	ngpu=args.ngpu,
	ncpu=args.ncpu,
	model_revision='v1.0.4')

print("model loaded")



async def ws_serve(websocket, path):
	frames_asr_online = []
	global websocket_users
	websocket_users.add(websocket)
	websocket.param_dict_asr_online = {"cache": dict()}
	websocket.wav_name = "microphone"
	print("new user connected",flush=True)
	try:
		async for message in websocket:
			
			
			if isinstance(message, str):
				messagejson = json.loads(message)
				
				if "is_speaking" in messagejson:
					websocket.is_speaking = messagejson["is_speaking"]
					websocket.param_dict_asr_online["is_final"] = not websocket.is_speaking
				if "chunk_interval" in messagejson:
					websocket.chunk_interval=messagejson["chunk_interval"]
				if "wav_name" in messagejson:
					websocket.wav_name = messagejson.get("wav_name")
				if "chunk_size" in messagejson:
					websocket.param_dict_asr_online["chunk_size"] = messagejson["chunk_size"]
			# if has bytes in buffer or message is bytes
			if len(frames_asr_online) > 0 or not isinstance(message, str):
				if not isinstance(message,str):
					frames_asr_online.append(message)
				if len(frames_asr_online) % websocket.chunk_interval == 0 or not websocket.is_speaking:
					audio_in = b"".join(frames_asr_online)
					# if not websocket.is_speaking:
						#padding 0.5s at end gurantee that asr engine can fire out last word
						# audio_in=audio_in+b''.join(np.zeros(int(16000*0.5),dtype=np.int16))
					await async_asr_online(websocket,audio_in)
					frames_asr_online = []
	
	
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
		if websocket.param_dict_asr_online.get("is_final", False):
			websocket.param_dict_asr_online["cache"] = dict()
		if "text" in rec_result:
			if rec_result["text"] != "sil" and rec_result["text"] != "waiting_for_more_voice":
				message = json.dumps({"mode": "online", "text": rec_result["text"], "wav_name": websocket.wav_name})
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