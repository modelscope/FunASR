# -*- encoding: utf-8 -*-
import os
import time
import websockets
import asyncio
# import threading
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--host",
                    type=str,
                    default="localhost",
                    required=False,
                    help="host ip, localhost, 0.0.0.0")
parser.add_argument("--port",
                    type=int,
                    default=10095,
                    required=False,
                    help="grpc server port")
parser.add_argument("--chunk_size",
                    type=str,
                    default="5, 10, 5",
                    help="chunk")
parser.add_argument("--chunk_interval",
                    type=int,
                    default=10,
                    help="chunk")
parser.add_argument("--audio_in",
                    type=str,
                    default=None,
                    help="audio_in")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]

# voices = asyncio.Queue()
from queue import Queue
voices = Queue()

# 其他函数可以通过调用send(data)来发送数据，例如：
async def record_microphone():
    is_finished = False
    import pyaudio
    #print("2")
    global voices 
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60*args.chunk_size[1]/args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    is_speaking = True
    while True:

        data = stream.read(CHUNK)
        data = data.decode('ISO-8859-1')
        message = json.dumps({"chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval, "audio": data, "is_speaking": is_speaking, "is_finished": is_finished})
        
        voices.put(message)
        #print(voices.qsize())

        await asyncio.sleep(0.005)

# 其他函数可以通过调用send(data)来发送数据，例如：
async def record_from_scp():
    import wave
    global voices
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]
    for wav in wavs:
        wav_splits = wav.strip().split()
        wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
        # bytes_f = open(wav_path, "rb")
        # bytes_data = bytes_f.read()
        with wave.open(wav_path, "rb") as wav_file:
            # 获取音频参数
            params = wav_file.getparams()
            # 获取头信息的长度
            # header_length = wav_file.getheaders()[0][1]
            # 读取音频帧数据，跳过头信息
            # wav_file.setpos(header_length)
            frames = wav_file.readframes(wav_file.getnframes())

        # 将音频帧数据转换为字节类型的数据
        audio_bytes = bytes(frames)
        # stride = int(args.chunk_size/1000*16000*2)
        stride = int(60*args.chunk_size[1]/args.chunk_interval/1000*16000*2)
        chunk_num = (len(audio_bytes)-1)//stride + 1
        # print(stride)
        is_speaking = True
        for i in range(chunk_num):
            if i == chunk_num-1:
                is_speaking = False
            beg = i*stride
            data = audio_bytes[beg:beg+stride]
            data = data.decode('ISO-8859-1')
            message = json.dumps({"chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval, "is_speaking": is_speaking, "audio": data, "is_finished": is_finished})
            voices.put(message)
            # print("data_chunk: ", len(data_chunk))
            # print(voices.qsize())
        
            await asyncio.sleep(60*args.chunk_size[1]/args.chunk_interval/1000)

    is_finished = True
    message = json.dumps({"is_finished": is_finished})
    voices.put(message)

async def ws_send():
    global voices
    global websocket
    print("started to sending data!")
    while True:
        while not voices.empty():
            data = voices.get()
            voices.task_done()
            try:
                await websocket.send(data) # 通过ws对象发送数据
            except Exception as e:
                print('Exception occurred:', e)
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.005)



async def message():
    global websocket
    text_print = ""
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            # print(meg, end = '')
            # print("\r")
            text = meg["text"][0]
            text_print += text
            text_print = text_print[-55:]
            os.system('clear')
            print("\r"+text_print)
        except Exception as e:
            print("Exception:", e)


async def print_messge():
    global websocket
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            print(meg)
        except Exception as e:
            print("Exception:", e)


async def ws_client():
    global websocket # 定义一个全局变量ws，用于保存websocket连接对象
    # uri = "ws://11.167.134.197:8899"
    uri = "ws://{}:{}".format(args.host, args.port)
    #ws = await websockets.connect(uri, subprotocols=["binary"]) # 创建一个长连接
    async for websocket in websockets.connect(uri, subprotocols=["binary"], ping_interval=None):
        if args.audio_in is not None:
            task = asyncio.create_task(record_from_scp()) # 创建一个后台任务录音
        else:
            task = asyncio.create_task(record_microphone())  # 创建一个后台任务录音
        task2 = asyncio.create_task(ws_send()) # 创建一个后台任务发送
        task3 = asyncio.create_task(message()) # 创建一个后台接收消息的任务
        await asyncio.gather(task, task2, task3)


asyncio.get_event_loop().run_until_complete(ws_client()) # 启动协程
asyncio.get_event_loop().run_forever()
