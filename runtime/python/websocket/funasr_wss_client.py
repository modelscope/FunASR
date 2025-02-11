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
parser.add_argument("--encoder_chunk_look_back", type=int, default=4, help="chunk")
parser.add_argument("--decoder_chunk_look_back", type=int, default=0, help="chunk")
parser.add_argument("--chunk_interval", type=int, default=10, help="chunk")
parser.add_argument(
    "--hotword",
    type=str,
    default="",
    help="hotword file path, one hotword perline (e.g.:阿里巴巴 20)",
)
parser.add_argument("--audio_in", type=str, default=None, help="audio_in")
parser.add_argument("--audio_fs", type=int, default=16000, help="audio_fs")
parser.add_argument(
    "--send_without_sleep",
    action="store_true",
    default=True,
    help="if audio_in is set, send_without_sleep",
)
parser.add_argument("--thread_num", type=int, default=1, help="thread_num")
parser.add_argument("--words_max_print", type=int, default=10000, help="chunk")
parser.add_argument("--output_dir", type=str, default=None, help="output_dir")
parser.add_argument("--ssl", type=int, default=1, help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn", type=int, default=1, help="1 for using itn, 0 for not itn")
parser.add_argument("--mode", type=str, default="2pass", help="offline, online, 2pass")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)
# voices = asyncio.Queue()
from queue import Queue

voices = Queue()
offline_msg_done = False

if args.output_dir is not None:
    # if os.path.exists(args.output_dir):
    #     os.remove(args.output_dir)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


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
    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword, encoding="utf-8")
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please checkout format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please checkout format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword

    use_itn = True
    if args.use_itn == 0:
        use_itn = False

    message = json.dumps(
        {
            "mode": args.mode,
            "chunk_size": args.chunk_size,
            "chunk_interval": args.chunk_interval,
            "encoder_chunk_look_back": args.encoder_chunk_look_back,
            "decoder_chunk_look_back": args.decoder_chunk_look_back,
            "wav_name": "microphone",
            "is_speaking": True,
            "hotwords": hotword_msg,
            "itn": use_itn,
        }
    )
    # voices.put(message)
    await websocket.send(message)
    while True:
        data = stream.read(CHUNK)
        message = data
        # voices.put(message)
        await websocket.send(message)
        await asyncio.sleep(0.005)


async def record_from_scp(chunk_begin, chunk_size):
    global voices
    is_finished = False
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    # hotwords
    fst_dict = {}
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            f_scp = open(args.hotword)
            hot_lines = f_scp.readlines()
            for line in hot_lines:
                words = line.strip().split(" ")
                if len(words) < 2:
                    print("Please checkout format of hotwords")
                    continue
                try:
                    fst_dict[" ".join(words[:-1])] = int(words[-1])
                except ValueError:
                    print("Please checkout format of hotwords")
            hotword_msg = json.dumps(fst_dict)
        else:
            hotword_msg = args.hotword
        print(hotword_msg)

    sample_rate = args.audio_fs
    wav_format = "pcm"
    use_itn = True
    if args.use_itn == 0:
        use_itn = False

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

        stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1
        # print(stride)

        # send first time
        message = json.dumps(
            {
                "mode": args.mode,
                "chunk_size": args.chunk_size,
                "chunk_interval": args.chunk_interval,
                "encoder_chunk_look_back": args.encoder_chunk_look_back,
                "decoder_chunk_look_back": args.decoder_chunk_look_back,
                "audio_fs": sample_rate,
                "wav_name": wav_name,
                "wav_format": wav_format,
                "is_speaking": True,
                "hotwords": hotword_msg,
                "itn": use_itn,
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

            sleep_duration = (
                0.001
                if args.mode == "offline"
                else 60 * args.chunk_size[1] / args.chunk_interval / 1000
            )

            await asyncio.sleep(sleep_duration)

    if not args.mode == "offline":
        await asyncio.sleep(2)
    # offline model need to wait for message recved

    if args.mode == "offline":
        global offline_msg_done
        while not offline_msg_done:
            await asyncio.sleep(1)

    await websocket.close()


async def message(id):
    global websocket, voices, offline_msg_done
    text_print = ""
    text_print_2pass_online = ""
    text_print_2pass_offline = ""
    if args.output_dir is not None:
        ibest_writer = open(
            os.path.join(args.output_dir, "text.{}".format(id)), "a", encoding="utf-8"
        )
    else:
        ibest_writer = None
    try:
        while True:

            meg = await websocket.recv()
            meg = json.loads(meg)
            wav_name = meg.get("wav_name", "demo")
            text = meg["text"]
            timestamp = ""
            offline_msg_done = meg.get("is_final", False)
            if "timestamp" in meg:
                timestamp = meg["timestamp"]

            if ibest_writer is not None:
                if timestamp != "":
                    text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp)
                else:
                    text_write_line = "{}\t{}\n".format(wav_name, text)
                ibest_writer.write(text_write_line)

            if "mode" not in meg:
                continue
            if meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print :]
                os.system("clear")
                print("\rpid" + str(id) + ": " + text_print)
            elif meg["mode"] == "offline":
                if timestamp != "":
                    text_print += "{} timestamp: {}".format(text, timestamp)
                else:
                    text_print += "{}".format(text)

                # text_print = text_print[-args.words_max_print:]
                # os.system('clear')
                print("\rpid" + str(id) + ": " + wav_name + ": " + text_print)
                offline_msg_done = True
            else:
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += "{}".format(text)
                    text_print = text_print_2pass_offline + text_print_2pass_online
                else:
                    text_print_2pass_online = ""
                    text_print = text_print_2pass_offline + "{}".format(text)
                    text_print_2pass_offline += "{}".format(text)
                text_print = text_print[-args.words_max_print :]
                os.system("clear")
                print("\rpid" + str(id) + ": " + text_print)
                # offline_msg_done=True

    except Exception as e:
        print("Exception:", e)
        # traceback.print_exc()
        # await websocket.close()


async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1
    global websocket, voices, offline_msg_done

    for i in range(chunk_begin, chunk_begin + chunk_size):
        offline_msg_done = False
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

        for p in process_list:
            p.join()

        print("end")
