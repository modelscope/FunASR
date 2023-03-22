# server.py   注意本例仅处理单个clent发送的语音数据，并未对多client连接进行判断和处理
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.logger import get_logger
import logging

logger = get_logger(log_level=logging.CRITICAL)
logger.setLevel(logging.CRITICAL)
import asyncio
import websockets  #区别客户端这里是 websockets库
import time
from queue import Queue
import  threading

print("model loading")
voices = Queue()
speek = Queue()
# 创建一个VAD对象
vad_pipline = pipeline(
    task=Tasks.voice_activity_detection,
    model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    model_revision="v1.2.0",
    output_dir=None,
    batch_size=1,
)
  
# 创建一个ASR对象
param_dict = dict()
param_dict["hotword"] = "小五 小五月"  # 设置热词，用空格隔开
inference_pipeline2 = pipeline(
    task=Tasks.auto_speech_recognition,
    model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    param_dict=param_dict,
)
print("model loaded")



async def echo(websocket, path):
    global voices
    try:
        async for message in websocket:
            voices.put(message)
            #print("put")
    except websockets.exceptions.ConnectionClosedError as e:
        print('Connection closed with exception:', e)
    except Exception as e:
        print('Exception occurred:', e)

start_server = websockets.serve(echo, "localhost", 8899, subprotocols=["binary"],ping_interval=None)


def vad(data):  # 推理
    global vad_pipline
    #print(type(data))
    segments_result = vad_pipline(audio_in=data)
    #print(segments_result)
    if len(segments_result) == 0:
        return False
    else:
        return True

def asr():  # 推理
    global inference_pipeline2
    global speek
    while True:
        while not speek.empty():
            audio_in = speek.get()
            speek.task_done()
            rec_result = inference_pipeline2(audio_in=audio_in)
            print(rec_result)
            time.sleep(0.1)
        time.sleep(0.1)    


def main():  # 推理
    frames = []  # 存储所有的帧数据
    buffer = []  # 存储缓存中的帧数据（最多两个片段）
    silence_count = 0  # 统计连续静音的次数
    speech_detected = False  # 标记是否检测到语音
    RECORD_NUM = 0
    global voices 
    global speek
    while True:
        while not voices.empty():
            
            data = voices.get()
            #print("队列排队数",voices.qsize())
            voices.task_done()
            buffer.append(data)
            if len(buffer) > 2:
                buffer.pop(0)  # 如果缓存超过两个片段，则删除最早的一个
            
            if speech_detected:
                frames.append(data)
                RECORD_NUM += 1    
            
            if  vad(data):
                if not speech_detected:
                    print("检测到人声...")
                    speech_detected = True  # 标记为检测到语音
                    frames = []
                    frames.extend(buffer)  # 把之前2个语音数据快加入
                silence_count = 0  # 重置静音次数
            else:
                silence_count += 1  # 增加静音次数

                if speech_detected and (silence_count > 4 or RECORD_NUM > 50): #这里 50 可根据需求改为合适的数据快数量
                    print("说话结束或者超过设置最长时间...")
                    audio_in = b"".join(frames)
                    #asrt = threading.Thread(target=asr,args=(audio_in,))
                    #asrt.start()
                    speek.put(audio_in)
                    #rec_result = inference_pipeline2(audio_in=audio_in)  # ASR 模型里跑一跑
                    frames = []  # 清空所有的帧数据
                    buffer = []  # 清空缓存中的帧数据（最多两个片段）
                    silence_count = 0  # 统计连续静音的次数清零
                    speech_detected = False  # 标记是否检测到语音
                    RECORD_NUM = 0
            time.sleep(0.01)
        time.sleep(0.01)
            


s = threading.Thread(target=main)
s.start()
s = threading.Thread(target=asr)
s.start()

asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()


 





 

        

