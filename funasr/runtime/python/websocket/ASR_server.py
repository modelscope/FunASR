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

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)


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
voices = Queue()
speek = Queue()

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
if args.punc_model is not None:
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
    global voices
    try:
        async for message in websocket:
            voices.put(message)
            #print("put")
    except websockets.exceptions.ConnectionClosedError as e:
        print('Connection closed with exception:', e)
    except Exception as e:
        print('Exception occurred:', e)

start_server = websockets.serve(ws_serve, args.host, args.port, subprotocols=["binary"], ping_interval=None)


def vad(data):  # 推理
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

def asr():  # 推理
    global inference_pipeline2
    global speek, param_dict_punc
    while True:
        while not speek.empty():
            audio_in = speek.get()
            speek.task_done()
            if len(audio_in) > 0:
                rec_result = inference_pipeline_asr(audio_in=audio_in)
                if inference_pipeline_punc is not None and 'text' in rec_result:
                    rec_result = inference_pipeline_punc(text_in=rec_result['text'], param_dict=param_dict_punc)
                print(rec_result["text"] if "text" in rec_result else rec_result)
            time.sleep(0.1)
        time.sleep(0.1)    


def main():  # 推理
    frames = []  # 存储所有的帧数据
    buffer = []  # 存储缓存中的帧数据（最多两个片段）
    # silence_count = 0  # 统计连续静音的次数
    # speech_detected = False  # 标记是否检测到语音
    RECORD_NUM = 0
    global voices 
    global speek
    speech_start, speech_end = False, False
    while True:
        while not voices.empty():
            
            data = voices.get()
            #print("队列排队数",voices.qsize())
            voices.task_done()
            buffer.append(data)
            if len(buffer) > 2:
                buffer.pop(0)  # 如果缓存超过两个片段，则删除最早的一个
            
            if speech_start:
                frames.append(data)
                RECORD_NUM += 1
            speech_start_i, speech_end_i = vad(data)
            # print(speech_start_i, speech_end_i)
            if speech_start_i:
                speech_start = speech_start_i
                # if not speech_detected:
                # print("检测到人声...")
                # speech_detected = True  # 标记为检测到语音
                frames = []
                frames.extend(buffer)  # 把之前2个语音数据快加入
                # silence_count = 0  # 重置静音次数
            if speech_end_i or RECORD_NUM > 300:
                # silence_count += 1  # 增加静音次数
                # speech_end = speech_end_i
                speech_start = False
                # if RECORD_NUM > 300: #这里 50 可根据需求改为合适的数据快数量
                # print("说话结束或者超过设置最长时间...")
                audio_in = b"".join(frames)
                #asrt = threading.Thread(target=asr,args=(audio_in,))
                #asrt.start()
                speek.put(audio_in)
                #rec_result = inference_pipeline2(audio_in=audio_in)  # ASR 模型里跑一跑
                frames = []  # 清空所有的帧数据
                buffer = []  # 清空缓存中的帧数据（最多两个片段）
                # silence_count = 0  # 统计连续静音的次数清零
                # speech_detected = False  # 标记是否检测到语音
                RECORD_NUM = 0
            time.sleep(0.01)
        time.sleep(0.01)
            


s = threading.Thread(target=main)
s.start()
s = threading.Thread(target=asr)
s.start()

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()