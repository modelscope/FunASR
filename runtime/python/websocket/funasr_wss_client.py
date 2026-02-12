# -*- encoding: utf-8 -*-
import os
import time
import websockets, ssl
import asyncio

import argparse
import json
import traceback
from multiprocessing import Process

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
parser.add_argument(
    "--audio_in",
    type=str,
    default=None,
    help="音频输入路径；不传则使用麦克风（需安装 PyAudio）",
)
parser.add_argument("--audio_fs", type=int, default=16000, help="audio_fs")

# ✅ 修复语义：默认 False；传入参数则不 sleep（用于压测）
parser.add_argument(
    "--send_without_sleep",
    action="store_true",
    default=False,
    help="若设置：发送音频不按实时节奏 sleep（用于压测）",
)

parser.add_argument("--thread_num", type=int, default=1, help="thread_num")
parser.add_argument("--words_max_print", type=int, default=10000, help="chunk")
parser.add_argument("--output_dir", type=str, default=None, help="output_dir")
parser.add_argument("--ssl", type=int, default=1, help="1 for ssl connect, 0 for no ssl")
parser.add_argument("--use_itn", type=int, default=1, help="1 for using itn, 0 for not itn")
parser.add_argument("--mode", type=str, default="2pass", help="offline, online, 2pass")

# ✅ 验收日志输出目录（每个 meeting 单独写，避免多进程抢文件）
parser.add_argument("--log_dir", type=str, default="./asr_logs", help="验收日志输出目录")
parser.add_argument("--log_flush_every", type=int, default=1, help="events.jsonl 每写N行flush一次（默认1更安全）")

args = parser.parse_args()
args.chunk_size = [int(x) for x in args.chunk_size.split(",")]
print(args)

from queue import Queue
from datetime import datetime

voices = Queue()
offline_msg_done = False

# === 延迟统计相关：对每个 wav_name 记录首包/末包发送时间 & 是否已经打印过延迟 ===
latency_first_audio_time = {}      # {wav_name: t_first_chunk_send}
latency_last_audio_time = {}       # {wav_name: t_last_chunk_send}
latency_first_text_printed = {}    # {wav_name: bool}


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


class MeetingWriter:
    """
    每个进程/meeting 单独写：
      - events.jsonl：收到的每条服务端消息（在线/离线/2pass）
      - meta.json：本次运行参数（方便复现）
    """
    def __init__(self, log_dir: str, meeting_id: str, flush_every: int = 1):
        self.meeting_id = str(meeting_id)
        self.base = os.path.join(log_dir, f"meeting_{self.meeting_id}")
        os.makedirs(self.base, exist_ok=True)

        self.fp_events = open(os.path.join(self.base, "events.jsonl"), "a", encoding="utf-8")
        self.flush_every = max(1, int(flush_every))
        self._cnt = 0

        meta_path = os.path.join(self.base, "meta.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w", encoding="utf-8") as f:
                meta = {
                    "created_at": _iso(time.time()),
                    "meeting_id": self.meeting_id,
                    "args": vars(args),
                }
                f.write(json.dumps(meta, ensure_ascii=False, indent=2))

    def write_event(self, obj: dict):
        self.fp_events.write(json.dumps(obj, ensure_ascii=False) + "\n")
        self._cnt += 1
        if self._cnt % self.flush_every == 0:
            self.fp_events.flush()

    def close(self):
        try:
            self.fp_events.flush()
            self.fp_events.close()
        except Exception:
            pass


if args.output_dir is not None:
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


async def record_microphone():
    """从麦克风实时录音发送到服务端（一般单路测试使用）"""
    try:
        import pyaudio
    except ImportError as e:
        raise ImportError(
            "缺少 PyAudio，麦克风推流前请先运行 `pip install pyaudio`"
        ) from e

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
            hotword_msg = json.dumps(fst_dict, ensure_ascii=False)
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
        },
        ensure_ascii=False,
    )
    await websocket.send(message)
    while True:
        data = stream.read(CHUNK)
        await websocket.send(data)
        await asyncio.sleep(0.01)


async def record_from_scp(chunk_begin, chunk_size):
    """从 wav/scp 文件读取音频分片发送，用于压测和延迟测试"""
    global voices, latency_first_audio_time, latency_last_audio_time
    if args.audio_in.endswith(".scp"):
        f_scp = open(args.audio_in)
        wavs = f_scp.readlines()
    else:
        wavs = [args.audio_in]

    # hotwords
    hotword_msg = ""
    if args.hotword.strip() != "":
        if os.path.exists(args.hotword):
            with open(args.hotword, encoding="utf-8") as f_scp:
                hot_lines = f_scp.readlines()

            hot_list = []
            for line in hot_lines:
                words = line.strip().split()
                if not words:
                    continue
                # Python AutoModel: 用逗号分隔多个热词
                hot_list.append(words[0])

            hotword_msg = ",".join(hot_list)
        else:
            hotword_msg = args.hotword

    print("hotword", hotword_msg)

    sample_rate = args.audio_fs
    wav_format = "pcm"
    use_itn = True
    if args.use_itn == 0:
        use_itn = False

    if chunk_size > 0:
        wavs = wavs[chunk_begin: chunk_begin + chunk_size]

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
                sample_rate = wav_file.getframerate()
                frames = wav_file.readframes(wav_file.getnframes())
                audio_bytes = bytes(frames)
        else:
            wav_format = "others"
            with open(wav_path, "rb") as f:
                audio_bytes = f.read()

        stride = int(60 * args.chunk_size[1] / args.chunk_interval / 1000 * sample_rate * 2)
        chunk_num = (len(audio_bytes) - 1) // stride + 1

        # send first control message
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
            },
            ensure_ascii=False,
        )

        await websocket.send(message)
        is_speaking = True

        # 初始化该 wav 的统计状态
        latency_first_audio_time[wav_name] = None
        latency_last_audio_time[wav_name] = None
        latency_first_text_printed[wav_name] = False

        for i in range(chunk_num):
            beg = i * stride
            data = audio_bytes[beg: beg + stride]

            now_ts = time.time()
            if latency_first_audio_time[wav_name] is None:
                latency_first_audio_time[wav_name] = now_ts
            latency_last_audio_time[wav_name] = now_ts

            await websocket.send(data)

            if i == chunk_num - 1:
                is_speaking = False
                await websocket.send(json.dumps({"is_speaking": is_speaking}, ensure_ascii=False))

            # ✅ sleep策略：默认按实时节奏；若开启 send_without_sleep 则几乎不 sleep（压测）
            if args.send_without_sleep:
                sleep_duration = 0.001
            else:
                sleep_duration = (
                    0.001
                    if args.mode == "offline"
                    else 60 * args.chunk_size[1] / args.chunk_interval / 1000
                )
            await asyncio.sleep(sleep_duration)

    if not args.mode == "offline":
        await asyncio.sleep(2)

    if args.mode == "offline":
        global offline_msg_done
        while not offline_msg_done:
            await asyncio.sleep(1)

    await asyncio.sleep(10)

    await websocket.close()


async def message(id, writer: MeetingWriter):
    """接收服务端识别结果 + 打印实时文本 + 打印延迟 + 写验收日志(events.jsonl)"""
    import websockets
    global websocket, voices, offline_msg_done
    global latency_first_audio_time, latency_last_audio_time, latency_first_text_printed

    multi_mode = args.thread_num > 1  # 多路并发时，打印风格更简洁
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
            text = meg.get("text", "")
            mode = meg.get("mode", "")
            spk_name = meg.get("spk_name", "")
            spk_score = meg.get("spk_score", None)
            now_ts = time.time()

            # === 延迟统计：仅在首条 online/2pass-online 文本时计算并打印一次 ===
            latency_last_ms = None
            latency_first_ms = None
            if text and mode in ("online", "2pass-online"):
                if not latency_first_text_printed.get(wav_name, False):
                    t_last = latency_last_audio_time.get(wav_name, None)
                    t_first = latency_first_audio_time.get(wav_name, None)
                    latency_last_ms = (now_ts - t_last) * 1000.0 if t_last is not None else None
                    latency_first_ms = (now_ts - t_first) * 1000.0 if t_first is not None else None

                    latency_first_text_printed[wav_name] = True

                    if multi_mode:
                        parts = [f"[MEETING {id}][LATENCY] wav={wav_name}, mode={mode}"]
                        if latency_last_ms is not None:
                            parts.append(f"from_last_chunk={latency_last_ms:.1f} ms")
                        if latency_first_ms is not None:
                            parts.append(f"from_first_chunk={latency_first_ms:.1f} ms")
                        print(" ".join(parts))
                    else:
                        print(
                            f"[LATENCY] wav={wav_name}, mode={mode}, "
                            f"from_last_chunk={(latency_last_ms or 0):.1f} ms, "
                            f"from_first_chunk={(latency_first_ms or 0):.1f} ms"
                        )

            timestamp = meg.get("timestamp", "")
            offline_msg_done = meg.get("is_final", False)

            # ✅ 验收友好：每条消息落 events.jsonl（便于后处理）
            event = {
                "ts": _iso(now_ts),
                "recv_ts": now_ts,
                "meeting_id": str(id),
                "wav_name": wav_name,
                "mode": mode,
                "is_final": bool(meg.get("is_final", False)),
                "text": text,
                "spk_name": spk_name,
                "spk_score": spk_score,
                "latency_first_ms": latency_first_ms,
                "latency_last_ms": latency_last_ms,
                "server_timestamp": meg.get("timestamp", None),
                "sentence_info": meg.get("sentence_info", None),
                "punc_array": meg.get("punc_array", None),
            }
            if writer is not None:
                writer.write_event(event)

            # 保存到 output_dir（保留你原来的逻辑）
            if ibest_writer is not None and text:
                if timestamp != "":
                    text_write_line = "{}\t{}\t{}\n".format(wav_name, text, timestamp)
                else:
                    text_write_line = "{}\t{}\n".format(wav_name, text)
                ibest_writer.write(text_write_line)

            if "mode" not in meg:
                continue

            # ===== 多路并发输出风格：只打印精简行 =====
            if multi_mode:
                if mode in ("offline", "2pass-offline") and text:
                    spk_name2 = meg.get("spk_name", "unknown")
                    spk_score2 = meg.get("spk_score", 0.0)
                    print(
                        f"[MEETING {id}][FINAL][{wav_name}] "
                        f"spk={spk_name2}({float(spk_score2):.3f}) text=\"{text}\""
                    )
                    if timestamp:
                        print(f"[MEETING {id}][TIMESTAMP][{wav_name}] {timestamp}")
                continue

            # ===== 单路模式输出：保留原滚动体验 =====
            if meg["mode"] == "online":
                text_print += "{}".format(text)
                text_print = text_print[-args.words_max_print:]
                print("pid" + str(id) + ": " + text_print)

            elif meg["mode"] == "offline":
                if timestamp != "":
                    text_print += "{} timestamp: {}".format(text, timestamp)
                else:
                    text_print += "{}".format(text)

                spk_info = ""
                if spk_name:
                    if spk_score is not None:
                        spk_info = f" [spk={spk_name} score={float(spk_score):.3f}]"
                    else:
                        spk_info = f" [spk={spk_name}]"

                print("pid" + str(id) + ": " + wav_name + ": " + text_print + spk_info)
                offline_msg_done = True

            else:
                # 2pass 模式
                if meg["mode"] == "2pass-online":
                    text_print_2pass_online += "{}".format(text)
                    text_print = text_print_2pass_offline + text_print_2pass_online
                else:
                    text_print_2pass_online = ""
                    text_print = text_print_2pass_offline + "{}".format(text)
                    text_print_2pass_offline += "{}".format(text)

                    if spk_name:
                        if spk_score is not None:
                            text_print += f" [spk={spk_name} score={float(spk_score):.3f}]"
                        else:
                            text_print += f" [spk={spk_name}]"

                text_print = text_print[-args.words_max_print:]
                print("pid" + str(id) + ": " + text_print)

    except websockets.exceptions.ConnectionClosedOK:
        print(f"[MEETING {id}] connection closed normally")
    except Exception as e:
        print(f"[MEETING {id}] Exception:", e)
    finally:
        try:
            if ibest_writer is not None:
                ibest_writer.flush()
                ibest_writer.close()
        except Exception:
            pass


async def ws_client(id, chunk_begin, chunk_size):
    if args.audio_in is None:
        chunk_begin = 0
        chunk_size = 1
    global websocket, voices, offline_msg_done

    for i in range(chunk_begin, chunk_begin + chunk_size):
        offline_msg_done = False
        voices = Queue()

        if args.ssl == 1:
            ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
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
            meeting_tag = f"{id}_{i}"
            writer = MeetingWriter(args.log_dir, meeting_id=meeting_tag, flush_every=args.log_flush_every)
            try:
                if args.audio_in is not None:
                    task = asyncio.create_task(record_from_scp(i, 1))
                else:
                    task = asyncio.create_task(record_microphone())
                task3 = asyncio.create_task(message(str(id) + "_" + str(i), writer))  # processid+fileid
                await asyncio.gather(task, task3)
            finally:
                writer.close()

    return


def one_thread(id, chunk_begin, chunk_size):
    # ✅ 子进程里用 asyncio.run 更稳
    asyncio.run(ws_client(id, chunk_begin, chunk_size))


if __name__ == "__main__":
    # for microphone
    if args.audio_in is None:
        p = Process(target=one_thread, args=(0, 0, 0))
        p.start()
        p.join()
        print("end")
    else:
        # calculate the number of wavs for each process
        if args.audio_in.endswith(".scp"):
            f_scp = open(args.audio_in)
            wavs = f_scp.readlines()
        else:
            wavs = [args.audio_in]

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

            p = Process(target=one_thread, args=(i, chunk_begin, now_chunk_size))
            chunk_begin = chunk_begin + now_chunk_size
            p.start()
            process_list.append(p)

        for p in process_list:
            p.join()

        print("end")
