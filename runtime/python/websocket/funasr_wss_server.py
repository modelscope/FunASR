import asyncio
import json
import websockets
import time
import numpy as np
import argparse
import ssl
import os
import wave
import functools
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial.distance import cosine

import torch  # 保留不影响


def to_python(obj):
    """递归地把 numpy / torch 等类型转成纯 Python，可 JSON 序列化。"""
    try:
        import numpy as np  # noqa
        import torch  # noqa
    except Exception:
        np = None
        torch = None

    if np is not None and isinstance(obj, np.generic):
        return obj.item()
    if np is not None and isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch is not None and isinstance(obj, torch.Tensor):
        return obj.cpu().tolist()

    if isinstance(obj, dict):
        return {k: to_python(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python(v) for v in obj]

    return obj


parser = argparse.ArgumentParser()
parser.add_argument("--host", type=str, default="0.0.0.0", required=False, help="host ip")
parser.add_argument("--port", type=int, default=10095, required=False, help="grpc server port")

parser.add_argument(
    "--asr_model",
    type=str,
    default="iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
    help="model from modelscope",
)
parser.add_argument("--asr_model_revision", type=str, default="v2.0.4", help="")

parser.add_argument(
    "--asr_model_online",
    type=str,
    default="iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
    help="model from modelscope",
)
parser.add_argument("--asr_model_online_revision", type=str, default="v2.0.4", help="")

parser.add_argument(
    "--vad_model",
    type=str,
    default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
    help="model from modelscope",
)
parser.add_argument("--vad_model_revision", type=str, default="v2.0.4", help="")

parser.add_argument(
    "--punc_model",
    type=str,
    default="iic/punc_ct-transformer_zh-cn-common-vad_realtime-vocab272727",
    help="model from modelscope",
)
parser.add_argument("--punc_model_revision", type=str, default="v2.0.4", help="")

parser.add_argument("--ngpu", type=int, default=1, help="0 for cpu, 1 for gpu")
parser.add_argument("--device", type=str, default="cuda", help="cuda, cpu")
parser.add_argument("--ncpu", type=int, default=4, help="cpu cores")

parser.add_argument(
    "--certfile",
    type=str,
    default="../../ssl_key/server.crt",
    required=False,
    help="certfile for ssl",
)
parser.add_argument(
    "--keyfile",
    type=str,
    default="../../ssl_key/server.key",
    required=False,
    help="keyfile for ssl",
)

# ====== 保存 2pass 离线阶段送入 ASR 的音频片段（排查 VAD 切分）======
parser.add_argument(
    "--save_offline_segments",
    action="store_true",
    help="Save each offline (2pass) audio segment sent to offline ASR as wav for debugging VAD split.",
)
parser.add_argument(
    "--save_offline_segments_dir",
    type=str,
    default="./offline_segments",
    help="Directory to save offline wav segments when --save_offline_segments is enabled.",
)

# ====== 并发控制：核心新增 ======
parser.add_argument(
    "--worker_threads",
    type=int,
    default=max(4, (os.cpu_count() or 4)),
    help="ThreadPoolExecutor max_workers. Used to offload blocking inference so event loop won't be blocked.",
)
parser.add_argument("--concurrent_vad", type=int, default=4, help="Max concurrent VAD generate() calls.")
parser.add_argument("--concurrent_asr_online", type=int, default=4, help="Max concurrent streaming ASR generate() calls.")
parser.add_argument("--concurrent_asr_offline", type=int, default=2, help="Max concurrent offline ASR generate() calls.")
parser.add_argument("--concurrent_punc", type=int, default=1, help="Max concurrent punctuation generate() calls.")
parser.add_argument("--concurrent_sv", type=int, default=1, help="Max concurrent speaker verification generate() calls.")
parser.add_argument(
    "--speaker_db_reload_sec",
    type=int,
    default=5,
    help="Reload speaker_db.json at most once every N seconds (avoid frequent disk IO).",
)

args = parser.parse_args()

websocket_users = set()
SPEAKER_DB_PATH = os.path.join(os.path.dirname(__file__), "speaker_db.json")


def _ensure_dir(p: str):
    try:
        os.makedirs(p, exist_ok=True)
    except Exception:
        pass


def _pcm_duration_ms(pcm_bytes: bytes, fs: int, ch: int = 1, sampwidth: int = 2) -> int:
    """根据 fs/ch/sampwidth 计算 PCM 时长，避免写死 16k -> 32 bytes/ms。"""
    if not pcm_bytes:
        return 0
    bytes_per_ms = (fs * ch * sampwidth) / 1000.0
    if bytes_per_ms <= 0:
        return 0
    return int(len(pcm_bytes) / bytes_per_ms)


def _safe_int(v, default):
    try:
        return int(v)
    except Exception:
        return default


# ========= speaker db：加缓存，避免每段都读盘 =========
_SPEAKER_DB_CACHE = {}
_SPEAKER_DB_CACHE_TS = 0.0


def _load_speaker_db_sync():
    if not os.path.exists(SPEAKER_DB_PATH):
        return {}
    try:
        with open(SPEAKER_DB_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_speaker_db_cached(now_ts: float, reload_sec: int):
    global _SPEAKER_DB_CACHE, _SPEAKER_DB_CACHE_TS
    if (now_ts - _SPEAKER_DB_CACHE_TS) >= max(1, int(reload_sec)):
        _SPEAKER_DB_CACHE = _load_speaker_db_sync()
        _SPEAKER_DB_CACHE_TS = now_ts
    return _SPEAKER_DB_CACHE or {}


def _save_wav_sync(out_path: str, audio_bytes: bytes, fs: int, ch: int, sampwidth: int):
    with wave.open(out_path, "wb") as wf:
        wf.setnchannels(ch)
        wf.setsampwidth(sampwidth)
        wf.setframerate(fs)
        wf.writeframes(audio_bytes)


def save_offline_wav_segment_sync(websocket, audio_bytes: bytes, reason: str = "offline"):
    """
    保存离线阶段送入 ASR 的音频片段，方便人工试听排查 VAD 切分是否正确。
    约定：audio_bytes 为 单声道 PCM16 little-endian（默认 16k）。
    （注意：这是同步函数，外层会放线程池执行）
    """
    if not getattr(websocket, "save_offline_segments", False):
        return
    if "2pass" not in (getattr(websocket, "mode", "") or ""):
        return
    if not audio_bytes:
        return

    fs = int(getattr(websocket, "audio_fs", 16000) or 16000)
    ch = 1
    sampwidth = 2  # int16

    # int16 对齐
    if len(audio_bytes) % 2 == 1:
        audio_bytes = audio_bytes[:-1]
        if not audio_bytes:
            return

    seg_idx = int(getattr(websocket, "offline_seg_idx", 0))
    websocket.offline_seg_idx = seg_idx + 1

    duration_ms = _pcm_duration_ms(audio_bytes, fs=fs, ch=ch, sampwidth=sampwidth)

    base_dir = getattr(websocket, "offline_save_dir", args.save_offline_segments_dir)
    _ensure_dir(base_dir)

    wav_name = (getattr(websocket, "wav_name", "microphone") or "microphone").replace("/", "_")
    ts = int(time.time() * 1000)
    fname = f"{wav_name}_{ts}_seg{seg_idx:04d}_{reason}_{duration_ms}ms.wav"
    out_path = os.path.join(base_dir, fname)

    try:
        _save_wav_sync(out_path, audio_bytes, fs=fs, ch=ch, sampwidth=sampwidth)
        print(f"[SAVE_OFFLINE_SEG] {out_path} ({duration_ms} ms, {len(audio_bytes)} bytes)")
    except Exception as e:
        print(f"[SAVE_OFFLINE_SEG] failed: {e}")


print("model loading")
from funasr import AutoModel  # noqa

# ====== 离线 ASR ======
model_asr = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)

# streaming asr
model_asr_streaming = AutoModel(
    model=args.asr_model_online,
    model_revision=args.asr_model_online_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)

# vad
model_vad = AutoModel(
    model=args.vad_model,
    model_revision=args.vad_model_revision,
    ngpu=args.ngpu,
    ncpu=args.ncpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)

# punc
if args.punc_model != "":
    model_punc = AutoModel(
        model=args.punc_model,
        model_revision=args.punc_model_revision,
        ngpu=args.ngpu,
        ncpu=args.ncpu,
        device=args.device,
        disable_pbar=True,
        disable_log=True,
    )
else:
    model_punc = None

# sv
model_sv = AutoModel(
    model="iic/speech_campplus_sv_zh-cn_16k-common",
    ngpu=args.ngpu,
    device=args.device,
    disable_pbar=True,
    disable_log=True,
)

print("model loaded! (now supports multi-client with non-blocking inference)")


# ====== 线程池 + 并发阈值（核心）======
EXECUTOR = ThreadPoolExecutor(max_workers=int(args.worker_threads))

SEM_VAD = asyncio.Semaphore(max(1, int(args.concurrent_vad)))
SEM_ASR_ONLINE = asyncio.Semaphore(max(1, int(args.concurrent_asr_online)))
SEM_ASR_OFFLINE = asyncio.Semaphore(max(1, int(args.concurrent_asr_offline)))
SEM_PUNC = asyncio.Semaphore(max(1, int(args.concurrent_punc)))
SEM_SV = asyncio.Semaphore(max(1, int(args.concurrent_sv)))
SEM_WAV = asyncio.Semaphore(max(1, 4))  # 保存 wav 一般不需要太大


async def run_blocking(fn, *a, sem: asyncio.Semaphore | None = None, **kw):
    """
    把阻塞函数丢线程池执行，避免卡 event loop。
    sem 用于限流（避免 GPU / 模型被打爆）。
    """
    loop = asyncio.get_running_loop()
    call = functools.partial(fn, *a, **kw)
    if sem is None:
        return await loop.run_in_executor(EXECUTOR, call)
    async with sem:
        return await loop.run_in_executor(EXECUTOR, call)


def _generate_sync(model, audio_or_text, status_dict):
    # 注意：status_dict 里包含 cache，会被 generate 更新
    return model.generate(input=audio_or_text, **status_dict)


async def ws_reset(websocket):
    print("ws reset now, total num is ", len(websocket_users))

    websocket.status_dict_asr_online["cache"] = {}
    websocket.status_dict_asr_online["is_final"] = True
    websocket.status_dict_vad["cache"] = {}
    websocket.status_dict_vad["is_final"] = True
    websocket.status_dict_punc["cache"] = {}

    await websocket.close()


async def clear_websocket():
    for websocket in list(websocket_users):
        await ws_reset(websocket)
    websocket_users.clear()


async def ws_serve(websocket, path=None):
    # websockets 新版本不会传 path，这里做兼容
    if path is None:
        path = getattr(websocket, "path", None)
    frames = []
    frames_asr = []
    frames_asr_online = []
    global websocket_users
    websocket_users.add(websocket)

    websocket.status_dict_asr = {}  # hotword 等
    websocket.status_dict_asr_online = {"cache": {}, "is_final": False}
    websocket.status_dict_vad = {"cache": {}, "is_final": False}
    websocket.status_dict_punc = {"cache": {}}

    websocket.chunk_interval = 10
    websocket.vad_pre_idx = 0
    speech_start = False
    speech_end_i = -1

    websocket.wav_name = "microphone"
    websocket.mode = "2pass"
    websocket.is_speaking = True  # ✅ 默认初始化，避免 AttributeError

    # 保存离线片段
    websocket.audio_fs = 16000
    websocket.offline_seg_idx = 0
    websocket.save_offline_segments = bool(args.save_offline_segments)
    websocket.offline_save_dir = args.save_offline_segments_dir
    if websocket.save_offline_segments:
        _ensure_dir(websocket.offline_save_dir)
        print(f"[SAVE_OFFLINE_SEG] enabled, dir={websocket.offline_save_dir}")

    print("new user connected", flush=True)

    try:
        async for message in websocket:
            # ========== 1) 先处理“文本配置消息” ==========
            if isinstance(message, str):
                try:
                    messagejson = json.loads(message)
                except Exception as e:
                    print("bad json message:", e, message[:200])
                    continue

                print("=============messagejson============", messagejson)

                if "is_speaking" in messagejson:
                    websocket.is_speaking = bool(messagejson["is_speaking"])
                    websocket.status_dict_asr_online["is_final"] = (not websocket.is_speaking)

                if "chunk_interval" in messagejson:
                    websocket.chunk_interval = _safe_int(
                        messagejson["chunk_interval"], websocket.chunk_interval
                    )

                if "wav_name" in messagejson:
                    websocket.wav_name = messagejson.get("wav_name") or websocket.wav_name

                if "chunk_size" in messagejson:
                    chunk_size = messagejson["chunk_size"]
                    if isinstance(chunk_size, str):
                        chunk_size = [x.strip() for x in chunk_size.split(",") if x.strip()]
                    websocket.status_dict_asr_online["chunk_size"] = [int(x) for x in chunk_size]

                if "encoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online["encoder_chunk_look_back"] = messagejson[
                        "encoder_chunk_look_back"
                    ]

                if "decoder_chunk_look_back" in messagejson:
                    websocket.status_dict_asr_online["decoder_chunk_look_back"] = messagejson[
                        "decoder_chunk_look_back"
                    ]

                if "hotwords" in messagejson:
                    hotword_data = messagejson["hotwords"]
                    websocket.status_dict_asr["hotword"] = hotword_data
                    websocket.status_dict_asr_online["hotword"] = hotword_data
                    print(f"热词已更新: {hotword_data}")

                if "mode" in messagejson:
                    websocket.mode = messagejson["mode"] or websocket.mode

                if "audio_fs" in messagejson:
                    websocket.audio_fs = _safe_int(messagejson["audio_fs"], 16000)

                continue

            # ========== 2) 处理“二进制音频消息” ==========
            if "chunk_size" not in websocket.status_dict_asr_online:
                print("[WARN] chunk_size not set yet, skip audio frame (send config first).")
                continue

            try:
                websocket.status_dict_vad["chunk_size"] = int(
                    websocket.status_dict_asr_online["chunk_size"][1] * 60 / websocket.chunk_interval
                )
            except Exception as e:
                print("[WARN] set vad chunk_size failed:", e)
                continue

            pcm = message
            frames.append(pcm)

            duration_ms = _pcm_duration_ms(pcm, fs=websocket.audio_fs, ch=1, sampwidth=2)
            websocket.vad_pre_idx += duration_ms

            # online asr
            frames_asr_online.append(pcm)
            websocket.status_dict_asr_online["is_final"] = (speech_end_i != -1)

            if (len(frames_asr_online) % websocket.chunk_interval == 0) or websocket.status_dict_asr_online["is_final"]:
                if websocket.mode in ("2pass", "online"):
                    audio_in = b"".join(frames_asr_online)
                    try:
                        await async_asr_online(websocket, audio_in)
                    except Exception:
                        print(f"error in asr streaming, {websocket.status_dict_asr_online}")
                frames_asr_online = []

            if speech_start:
                frames_asr.append(pcm)

            # vad online
            try:
                speech_start_i, speech_end_i = await async_vad(websocket, pcm)
            except Exception as e:
                print("error in vad:", e)
                speech_start_i, speech_end_i = -1, -1

            if speech_start_i != -1:
                speech_start = True
                if duration_ms > 0:
                    beg_bias = (websocket.vad_pre_idx - speech_start_i) // duration_ms
                else:
                    beg_bias = 0
                frames_pre = frames[-beg_bias:] if beg_bias > 0 else []
                frames_asr = []
                frames_asr.extend(frames_pre)

            # ========== 3) 2pass：离线阶段触发点 ==========
            if (speech_end_i != -1) or (not websocket.is_speaking):
                if websocket.mode in ("2pass", "offline"):
                    audio_in = b"".join(frames_asr)
                    reason = "vad_end" if speech_end_i != -1 else "not_speaking"

                    # 保存 wav：放线程池，避免磁盘 IO 卡 loop
                    if websocket.save_offline_segments and audio_in:
                        try:
                            await run_blocking(
                                save_offline_wav_segment_sync,
                                websocket,
                                audio_in,
                                reason,
                                sem=SEM_WAV,
                            )
                        except Exception as e:
                            print("[SAVE_OFFLINE_SEG] async failed:", e)

                    try:
                        await async_asr(websocket, audio_in)
                    except Exception as e:
                        print("error in asr offline:", e)

                frames_asr = []
                speech_start = False
                frames_asr_online = []
                websocket.status_dict_asr_online["cache"] = {}

                if not websocket.is_speaking:
                    websocket.vad_pre_idx = 0
                    frames = []
                    websocket.status_dict_vad["cache"] = {}
                    speech_end_i = -1
                else:
                    frames = frames[-20:]

    except websockets.ConnectionClosed:
        print("ConnectionClosed...", websocket_users, flush=True)
        await ws_reset(websocket)
        if websocket in websocket_users:
            websocket_users.remove(websocket)
    except websockets.InvalidState:
        print("InvalidState...")
    except Exception as e:
        print("Exception:", e)
        try:
            await ws_reset(websocket)
        except Exception:
            pass
        if websocket in websocket_users:
            websocket_users.remove(websocket)


# ===================== 推理：全部改为“线程池 + 限流” =====================

async def async_vad(websocket, audio_in: bytes):
    # model_vad.generate 是阻塞的，必须 offload
    out = await run_blocking(_generate_sync, model_vad, audio_in, websocket.status_dict_vad, sem=SEM_VAD)
    segments_result = out[0].get("value", [])

    speech_start = -1
    speech_end = -1

    if len(segments_result) == 0 or len(segments_result) > 1:
        return speech_start, speech_end
    if segments_result[0][0] != -1:
        speech_start = segments_result[0][0]
    if segments_result[0][1] != -1:
        speech_end = segments_result[0][1]
    return speech_start, speech_end


def _sv_and_match_sync(audio_in: bytes, reload_sec: int):
    """
    同步执行：SV embedding + speaker_db 匹配
    返回 (spk_name, best_score)
    """
    spk_name = "unknown"
    best_score = 0.0

    sv_out = model_sv.generate(input=audio_in, embedding=True)[0]
    embedding = sv_out["spk_embedding"][0].cpu().numpy()

    now_ts = time.time()
    local_speaker_db = get_speaker_db_cached(now_ts, reload_sec=reload_sec)
    if local_speaker_db:
        for name, ref_embedding in local_speaker_db.items():
            if ref_embedding is None:
                continue
            arr = np.array(ref_embedding, dtype=np.float32)
            similarity = 1.0 - cosine(embedding, arr)
            print("sv similarity with {}: {}".format(name, similarity))
            if similarity > best_score and similarity > 0.2:
                best_score = similarity
                spk_name = name

    return spk_name, float(best_score)


async def async_asr(websocket, audio_in: bytes):
    mode = "2pass-offline" if "2pass" in (websocket.mode or "") else websocket.mode

    if len(audio_in) <= 0:
        message = {
            "mode": mode,
            "text": "",
            "wav_name": websocket.wav_name,
            "is_final": True,
        }
        await websocket.send(json.dumps(message, ensure_ascii=False))
        return

    # 1) ASR（阻塞，线程池执行）
    rec_result_list = await run_blocking(
        _generate_sync,
        model_asr,
        audio_in,
        websocket.status_dict_asr,
        sem=SEM_ASR_OFFLINE,
    )
    rec_result = rec_result_list[0]

    print("offline_asr, raw:", rec_result)
    print("offline_asr, keys:", rec_result.keys())

    text = rec_result.get("text", "")
    timestamp = rec_result.get("timestamp", None)
    sentence_info = rec_result.get("sentence_info", None)

    # 2) 声纹识别（阻塞，线程池执行）
    spk_name = "unknown"
    best_score = 0.0
    try:
        spk_name, best_score = await run_blocking(
            _sv_and_match_sync,
            audio_in,
            int(args.speaker_db_reload_sec),
            sem=SEM_SV,
        )
    except Exception as e:
        print(f"声纹识别失败: {e}")

    # 3) 标点（阻塞，线程池执行）
    punc_array = None
    if model_punc is not None and len(text) > 0:
        try:
            # punc 只对文本处理
            punc_out = await run_blocking(
                _generate_sync,
                model_punc,
                text,
                websocket.status_dict_punc,
                sem=SEM_PUNC,
            )
            punc_result = punc_out[0]
            print("offline, after punc", punc_result)

            if "text" in punc_result and punc_result["text"]:
                text = punc_result["text"]
            if "punc_array" in punc_result:
                punc_array = punc_result["punc_array"]
        except Exception as e:
            print("punc failed:", e)

    # 4) 构造最终 message
    if len(text) > 0:
        print("======offline final text:", text)
        message = {
            "mode": mode,
            "spk_name": spk_name,
            "spk_score": float(best_score),
            "text": text,
            "wav_name": websocket.wav_name,
            "is_final": True,
        }
        if timestamp is not None:
            message["timestamp"] = to_python(timestamp)
        if sentence_info is not None:
            message["sentence_info"] = to_python(sentence_info)
        if punc_array is not None:
            message["punc_array"] = to_python(punc_array)

        try:
            await websocket.send(json.dumps(message, ensure_ascii=False))
        except Exception as e:
            print("send json failed:", e)
            print("message types:", {k: type(v) for k, v in message.items()})
    else:
        message = {
            "mode": mode,
            "spk_name": spk_name,
            "spk_score": float(best_score),
            "text": "",
            "wav_name": websocket.wav_name,
            "is_final": True,
        }
        await websocket.send(json.dumps(message, ensure_ascii=False))


async def async_asr_online(websocket, audio_in: bytes):
    if len(audio_in) <= 0:
        return

    # streaming generate 也是阻塞：线程池执行
    rec_out = await run_blocking(
        _generate_sync,
        model_asr_streaming,
        audio_in,
        websocket.status_dict_asr_online,
        sem=SEM_ASR_ONLINE,
    )
    rec_result = rec_out[0]
    print("online, ", rec_result)

    # 2pass：online 只要 partial，不发 final（final 交给 offline）
    if websocket.mode == "2pass" and websocket.status_dict_asr_online.get("is_final", False):
        return

    if rec_result.get("text"):
        mode = "2pass-online" if "2pass" in (websocket.mode or "") else websocket.mode
        message = {
            "mode": mode,
            "text": rec_result["text"],
            "wav_name": websocket.wav_name,
            "is_final": bool(
                websocket.status_dict_asr_online.get("is_final", False) or (not websocket.is_speaking)
            ),
        }
        await websocket.send(json.dumps(message, ensure_ascii=False))


# ===================== 启动服务 =====================

async def main():
    if len(args.certfile) > 0:
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(args.certfile, keyfile=args.keyfile)
        server = await websockets.serve(
            ws_serve,
            args.host,
            args.port,
            subprotocols=["binary"],
            ping_interval=None,
            ssl=ssl_context,
        )
    else:
        server = await websockets.serve(
            ws_serve,
            args.host,
            args.port,
            subprotocols=["binary"],
            ping_interval=None,
        )

    print(f"WS server started at ws(s)://{args.host}:{args.port}")
    await server.wait_closed()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        try:
            EXECUTOR.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
