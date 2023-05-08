# -*- encoding: utf-8 -*-
import os
import time
import websockets
import asyncio
# import threading
import argparse
import json
import traceback
from multiprocessing import Process
from funasr.fileio.datadir_writer import DatadirWriter

import logging

logging.basicConfig(level=logging.ERROR)

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
parser.add_argument("--send_without_sleep",
                    action="store_true",
                    default=False,
                    help="if audio_in is set, send_without_sleep")
parser.add_argument("--test_thread_num",
                    type=int,
                    default=1,
                    help="test_thread_num")
parser.add_argument("--words_max_print",
                    type=int,
                    default=100,
                    help="chunk")
parser.add_argument("--output_dir",
                    type=str,
                    default=None,
                    help="output_dir")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)
# voices = asyncio.Queue()
from queue import Queue
voices = Queue()

ibest_writer = None
if args.output_dir is not None:
    writer = DatadirWriter(args.output_dir)
    ibest_writer = writer[f"1best_recog"]

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

        await asyncio.sleep(0.005)

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
        wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
        wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
        
        # bytes_f = open(wav_path, "rb")
        # bytes_data = bytes_f.read()
        with wave.open(wav_path, "rb") as wav_file:
            params = wav_file.getparams()
            # header_length = wav_file.getheaders()[0][1]
            # wav_file.setpos(header_length)
            frames = wav_file.readframes(wav_file.getnframes())

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
            message = json.dumps({"chunk_size": args.chunk_size, "chunk_interval": args.chunk_interval, "is_speaking": is_speaking, "audio": data, "is_finished": is_finished, "wav_name": wav_name})
            voices.put(message)
            # print("data_chunk: ", len(data_chunk))
            # print(voices.qsize())
            sleep_duration = 0.001 if args.send_without_sleep else 60*args.chunk_size[1]/args.chunk_interval/1000
            await asyncio.sleep(sleep_duration)

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
                await websocket.send(data)
            except Exception as e:
                print('Exception occurred:', e)
                traceback.print_exc()
                exit(0)
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.005)



async def message(id):
    global websocket
    text_print = ""
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            # print(wav_name)
            text = meg["text"]
            if ibest_writer is not None:
                ibest_writer["text"][wav_name] = text
            
            if meg["mode"] == "online":
                text_print += " {}".format(text)
                text_print = text_print[-args.words_max_print:]
                os.system('clear')
                print("\rpid"+str(id)+": "+text_print)
            elif meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                os.system('clear')
                print("\rpid"+str(id)+": "+text_print)
            else:
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += " {}".format(text)
                    text_print = text_print_2pass_offline + text_print_2pass_online
                else:
                    text_print_2pass_online = " "
                    text_print = text_print_2pass_offline + "{}".format(text)
                    text_print_2pass_offline += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                os.system('clear')
                print("\rpid" + str(id) + ": " + text_print)

        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()
            exit(0)

async def print_messge():
    global websocket
    while True:
        try:
            meg = await websocket.recv()
            meg = json.loads(meg)
            print(meg)
        except Exception as e:
            print("Exception:", e)
            traceback.print_exc()
            exit(0)

async def ws_client(id):
    global websocket
    uri = "ws://{}:{}".format(args.host, args.port)
    async for websocket in websockets.connect(uri, subprotocols=["binary"], ping_interval=None):
        if args.audio_in is not None:
            task = asyncio.create_task(record_from_scp())
        else:
            task = asyncio.create_task(record_microphone())
        task2 = asyncio.create_task(ws_send())
        task3 = asyncio.create_task(message(id))
        await asyncio.gather(task, task2, task3)

def one_thread(id):
   asyncio.get_event_loop().run_until_complete(ws_client(id))
   asyncio.get_event_loop().run_forever()


if __name__ == '__main__':
    process_list = []
    for i in range(args.test_thread_num):   
        p = Process(target=one_thread,args=(i,))
        p.start()
        process_list.append(p)

    for i in process_list:
        p.join()

    print('end')
 

