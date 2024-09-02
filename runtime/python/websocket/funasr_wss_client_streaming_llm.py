# -*- encoding: utf-8 -*-
import os
import time
import websockets, ssl
import asyncio

# import threading
import argparse
import json
import traceback
from multiprocessing import Process

# from funasr.fileio.datadir_writer import DatadirWriter

import logging

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--host", type=str, default="localhost", required=False, help="host ip, localhost, 0.0.0.0"
)
parser.add_argument("--port", type=int, default=10095, required=False, help="grpc server port")
parser.add_argument("--chunk_size", type=str, default="5, 10, 5", help="chunk")
parser.add_argument("--chunk_interval", type=int, default=10, help="chunk")
parser.add_argument("--audio_in", type=str, default=None, help="audio_in")
parser.add_argument("--audio_fs", type=int, default=16000, help="audio_fs")
parser.add_argument("--asr_prompt", type=str, default="Copy:", help="asr prompt")
parser.add_argument("--s2tt_prompt", type=str, default="Translate the following sentence into English:", help="s2tt prompt")

parser.add_argument(
    "--send_without_sleep",
    action="store_true",
    default=True,
    help="if audio_in is set, send_without_sleep",
)
parser.add_argument("--thread_num", type=int, default=1, help="thread_num")
parser.add_argument("--words_max_print", type=int, default=10000, help="chunk")
parser.add_argument("--ssl", type=int, default=1, help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--mode", type=str, default="online", help="offline, online, 2pass")
parser.add_argument("--skip_seconds", type=int, default=0, help="skip how many seconds in audio")


args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)
# voices = asyncio.Queue()
from queue import Queue

voices = Queue()

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'   # 重置颜色
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


async def record_microphone():
    is_finished = False
    import pyaudio

    # print("2")
    global voices
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    chunk_size = 60 * args.chunk_size[1] / args.chunk_interval
    CHUNK = int(RATE / 1000 * chunk_size)

    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    message = json.dumps(
        {
            "mode": args.mode,
            "chunk_size": args.chunk_size,
            "chunk_interval": args.chunk_interval,
            "wav_name": "microphone",
            "is_speaking": True,
            "asr_prompt": args.asr_prompt,
            "s2tt_prompt": args.s2tt_prompt,
        }
    )
    # voices.put(message)
    await websocket.send(message)
    while True:
        data = stream.read(CHUNK)
        message = data
        # voices.put(message)
        await websocket.send(message)
        await asyncio.sleep(0.0005)


async def record_from_scp(chunk_begin, chunk_size):
    global voices
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    sample_rate = args.audio_fs
    wav_format = "pcm"

    if chunk_size > 0:
        wavs = wavs[chunk_begin : chunk_begin + chunk_size]
    for wav in wavs:
        wav_splits = wav.strip().split()

        wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
        wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
        if not len(wav_path.strip()) > 0:
            continue
        if wav_path.endswith(".pcm"):
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
        elif wav_path.endswith(".wav"):
            import wave

            with wave.open(wav_path, "rb") as wav_file:
                params = wav_file.getparams()
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = bytes(frames)
        else:
            wav_format = "others"
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()
        
        # skip seconds in audio_bytes
        if args.skip_seconds > 0:
            audio_bytes = audio_bytes[args.skip_seconds * sample_rate * 2 :]

        stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1
        # print(stride)

        # send first time
        message = json.dumps(
            {
                "mode": args.mode,
                "chunk_size": args.chunk_size,
                "chunk_interval": args.chunk_interval,
                "wav_name": "microphone",
                "is_speaking": True,
                "asr_prompt": args.asr_prompt,
                "s2tt_prompt": args.s2tt_prompt,
            }
        )

        # voices.put(message)
        await websocket.send(message)
        is_speaking = True
        for i in range(chunk_num):

            beg = i * stride
            data = audio_bytes[beg : beg + stride]
            message = data
            # voices.put(message)
            await websocket.send(message)
            if i == chunk_num - 1:
                is_speaking = False
                message = json.dumps({"is_speaking": is_speaking})
                # voices.put(message)
                await websocket.send(message)

            # sleep_duration = 0.00001  # 60 * args.chunk_size[1] / args.chunk_interval / 1000
            sleep_duration = 60 * args.chunk_size[1] / args.chunk_interval / 1000

            await asyncio.sleep(sleep_duration)

    await asyncio.sleep(2)

    await websocket.close()


async def message(id):
    global websocket, voices
    text_print = ""
    prev_asr_text = ""
    prev_s2tt_text = ""
    try:
        while True:
            meg = await websocket.recv()
            meg = json.loads(meg)
            asr_text = meg["asr_text"]
            s2tt_text = meg["s2tt_text"]

            clean_prev_asr_text = prev_asr_text.replace("<em>", "").replace("</em>", "")
            clean_prev_s2tt_text = prev_s2tt_text.replace("<em>", "").replace("</em>", "")
            clean_asr_text = asr_text.replace("<em>", "").replace("</em>", "")
            clean_s2tt_text = s2tt_text.replace("<em>", "").replace("</em>", "")

            if clean_prev_asr_text.startswith(clean_asr_text):
                new_asr_unfix_pos = asr_text.find("<em>")
                asr_text = clean_prev_asr_text[:new_asr_unfix_pos] + "<em>" + clean_prev_asr_text[new_asr_unfix_pos:] + "</em>"

            if clean_prev_s2tt_text.startswith(clean_s2tt_text):
                new_s2tt_unfix_pos = s2tt_text.find("<em>")
                s2tt_text = clean_prev_s2tt_text[:new_s2tt_unfix_pos] + "<em>" + clean_prev_s2tt_text[new_s2tt_unfix_pos:] + "</em>"

            prev_asr_text = asr_text
            prev_s2tt_text = s2tt_text
            print_asr_text = Colors.OKGREEN + asr_text[:asr_text.find("<em>")] + Colors.ENDC + Colors.OKCYAN + asr_text[asr_text.find("<em>") + len("<em>"): -len("</em>")] + Colors.ENDC
            print_s2tt_text = Colors.OKGREEN + s2tt_text[:s2tt_text.find("<em>")] + Colors.ENDC + Colors.OKCYAN + s2tt_text[s2tt_text.find("<em>") + len("<em>"): -len("</em>")] + Colors.ENDC
            text_print = "\n\n" + "ASR: " + print_asr_text + "\n\n" + "S2TT: " + print_s2tt_text
            os.system("clear")
            print("\rpid" + str(id) + ": " + text_print)

    except Exception as e:
        print("Exception:", e)
        # traceback.print_exc()
        # await websocket.close()


async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1
    global websocket, voices

    for i in range(chunk_begin, chunk_begin + chunk_size):
        voices = Queue()
        if args.ssl == 1:
            ssl_context = ssl.SSLContext()
            ssl_context.check_hostname = False
            ssl_context.verify_mode = ssl.CERT_NONE
            uri = "wss://{}:{}".format(args.host, args.port)
        else:
            uri = "ws://{}:{}".format(args.host, args.port)
            ssl_context = None
        print("connect to", uri)
        async with websockets.connect(
            uri, subprotocols=["binary"], ping_interval=None, ssl=ssl_context
        ) as websocket:
            if args.audio_in is not None:
                task = asyncio.create_task(record_from_scp(i, 1))
            else:
                task = asyncio.create_task(record_microphone())
            task3 = asyncio.create_task(message(str(id) + "_" + str(i)))  # processid+fileid
            await asyncio.gather(task, task3)
    exit(0)


def one_thread(id, chunk_begin, chunk_size):
    asyncio.get_event_loop().run_until_complete(ws_client(id, chunk_begin, chunk_size))
    asyncio.get_event_loop().run_forever()


if __name__ == "__main__":
    # for microphone
    if args.audio_in is None:
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print("end")
    else:
        # calculate the number of wavs for each preocess
        if args.audio_in.endswith(".scp"):
            f_scp = open(args.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [args.audio_in]
        for wav in wavs:
            wav_splits = wav.strip().split()
            wav_name = wav_splits[0] if len(wav_splits) > 1 else "demo"
            wav_path = wav_splits[1] if len(wav_splits) > 1 else wav_splits[0]
            audio_type = os.path.splitext(wav_path)[-1].lower()

        total_len = len(wavs)
        if total_len >= args.thread_num:
            chunk_size = int(total_len / args.thread_num)
            remain_wavs = total_len - chunk_size * args.thread_num
        else:
            chunk_size = 1
            remain_wavs = 0

        process_list = []
        chunk_begin = 0
        for i in range(args.thread_num):
            now_chunk_size = chunk_size
            if remain_wavs > 0:
                now_chunk_size = chunk_size + 1
                remain_wavs = remain_wavs - 1
            # process i handle wavs at chunk_begin and size of now_chunk_size
            p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            chunk_begin = chunk_begin + now_chunk_size
            p.start()
            process_list.append(p)

        for i in process_list:
            p.join()

        print("end")


"""
python funasr_wss_client.py --host "127.0.0.1" --port 10095 --audio_in audio_file
"""
