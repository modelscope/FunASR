import asyncio
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
                    default="",
                    help="model from modelscope")
parser.add_argument("--ngpu",
                    type=int,
                    default=1,
                    help="0 for cpu, 1 for gpu")

args = parser.parse_args()

print("model loading")
 

# vad
inference_pipeline_vad = pipeline(
    task=Tasks.voice_activity_detection,
    model=args.vad_model,
    model_revision=None,
    output_dir=None,
    batch_size=1,
    mode='online',
    ngpu=args.ngpu,
)
param_dict_vad = {'in_cache': dict(), "is_final": False}
  
# asr
param_dict_asr = {}
# param_dict["hotword"] = "小五 小五月"  # 设置热词，用空格隔开
inference_pipeline_asr = pipeline(
    task=Tasks.auto_speech_recognition,
    model=args.asr_model,
    param_dict=param_dict_asr,
    ngpu=args.ngpu,
)
if args.punc_model != "":
    param_dict_punc = {'cache': list()}
    inference_pipeline_punc = pipeline(
        task=Tasks.punctuation,
        model=args.punc_model,
        model_revision=None,
        ngpu=args.ngpu,
    )
else:
    inference_pipeline_punc = None

print("model loaded")



async def ws_serve(websocket, path):
    #speek = Queue()
    frames = []  # 存储所有的帧数据
    buffer = []  # 存储缓存中的帧数据（最多两个片段）
    RECORD_NUM = 0
    global websocket_users
    speech_start, speech_end = False, False
    # 调用asr函数
    websocket.speek = Queue()  #websocket 添加进队列对象 让asr读取语音数据包
    websocket.send_msg = Queue()   #websocket 添加个队列对象  让ws发送消息到客户端
    websocket_users.add(websocket)
    ss = threading.Thread(target=asr, args=(websocket,))
    ss.start()
    
    try:
        async for message in websocket:
            #voices.put(message)
            #print("put")
            #await websocket.send("123")
            buffer.append(message)
            if len(buffer) > 2:
                buffer.pop(0)  # 如果缓存超过两个片段，则删除最早的一个
              
            if speech_start:
                frames.append(message)
                RECORD_NUM += 1
            speech_start_i, speech_end_i = vad(message)
            #print(speech_start_i, speech_end_i)
            if speech_start_i:
                speech_start = speech_start_i
                frames = []
                frames.extend(buffer)  # 把之前2个语音数据快加入
            if speech_end_i or RECORD_NUM > 300:
                speech_start = False
                audio_in = b"".join(frames)
                websocket.speek.put(audio_in)
                frames = []  # 清空所有的帧数据
                buffer = []  # 清空缓存中的帧数据（最多两个片段）
                RECORD_NUM = 0
            if not websocket.send_msg.empty():
                await websocket.send(websocket.send_msg.get())
                websocket.send_msg.task_done()

     
    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users)    # 链接断开
        websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")    # 无效状态
    except Exception as e:
        print("Exception:", e)
 

def asr(websocket):  # ASR推理
        global inference_pipeline2
        global param_dict_punc
        global websocket_users
        while websocket in  websocket_users:
            if not websocket.speek.empty():
                audio_in = websocket.speek.get()
                websocket.speek.task_done()
                if len(audio_in) > 0:
                    rec_result = inference_pipeline_asr(audio_in=audio_in)
                    if inference_pipeline_punc is not None and 'text' in rec_result:
                        rec_result = inference_pipeline_punc(text_in=rec_result['text'], param_dict=param_dict_punc)
                    results = (rec_result["text"] if "text" in rec_result else rec_result)
                    websocket.send_msg.put(results) # 存入发送队列  直接调用send发送不了
               
            time.sleep(0.1)

def vad(data):  # VAD推理
    global vad_pipline, param_dict_vad
    #print(type(data))
    # print(param_dict_vad)
    segments_result = inference_pipeline_vad(audio_in=data, param_dict=param_dict_vad)
    # print(segments_result)
    # print(param_dict_vad)
    speech_start = False
    speech_end = False
    
    if len(segments_result) == 0 or len(segments_result["text"]) > 1:
        return speech_start, speech_end
    if segments_result["text"][0][0] != -1:
        speech_start = True
    if segments_result["text"][0][1] != -1:
        speech_end = True
    return speech_start, speech_end

 
start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()