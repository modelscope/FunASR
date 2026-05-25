#!/usr/bin/env python3
"""Fun-ASR-Nano vLLM Inference Server.

Unified server with three interfaces:
- HTTP REST: POST /asr (file upload)
- WebSocket: ws://host:port/ws (streaming audio)
- OpenAI API: POST /v1/audio/transcriptions (Whisper-compatible)

All endpoints share the same vLLM engine + dynamic VAD + SPK + timestamps.

Usage:
    CUDA_VISIBLE_DEVICES=0 python serve_vllm.py --port 8000
    CUDA_VISIBLE_DEVICES=0 python serve_vllm.py --port 8000 --model FunAudioLLM/Fun-ASR-Nano-2512
"""

import asyncio
import argparse
import io
import json
import logging
import os
import re
import time
import tempfile

import numpy as np
import soundfile as sf
import torch
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, UploadFile, Form, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse
    import uvicorn
except ImportError:
    raise ImportError("pip install fastapi uvicorn python-multipart")

from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD
from funasr import AutoModel


# ============================================================
# Global state
# ============================================================
_engine = None
_vad_model = None
_spk_model = None
_args = None


def load_engine(args):
    global _engine, _vad_model, _spk_model, _args
    _args = args
    if _engine is None:
        logger.info(f"Loading vLLM engine: {args.model}")
        _engine = FunASRNanoVLLM.from_pretrained(
            model=args.model, hub=args.hub, device=args.device, dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        logger.info("Loading VAD: fsmn-vad")
        _vad_model = AutoModel(model="fsmn-vad", device=args.device, disable_update=True)
        logger.info("Loading SPK: eres2netv2")
        _spk_model = AutoModel(model="iic/speech_eres2netv2_sv_zh-cn_16k-common", device=args.device, disable_update=True)
        logger.info("All models ready!")


def process_audio(audio_data, sr=16000, language=None, hotwords=None, 
                  use_vad=True, use_spk=False, use_timestamp=True):
    """Core processing: VAD segment → vLLM ASR → timestamps → SPK."""
    if sr != 16000:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    if audio_data.ndim > 1:
        audio_data = audio_data[:, 0]
    audio_data = audio_data.astype(np.float32)

    # VAD segmentation
    if use_vad and len(audio_data) > sr * 1:
        vad_res = _vad_model.generate(input=audio_data, fs=sr)
        segments = vad_res[0]["value"]
    else:
        segments = [[0, int(len(audio_data) * 1000 / sr)]]

    if not segments:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    # Extract segment audio
    seg_audios = []
    seg_times = []
    for seg in segments:
        s0 = int(seg[0] * sr / 1000)
        s1 = int(seg[1] * sr / 1000)
        seg_audio = audio_data[s0:s1]
        if len(seg_audio) > sr * 0.3:
            seg_audios.append(seg_audio)
            seg_times.append((seg[0], seg[1]))

    if not seg_audios:
        return {"text": "", "segments": [], "duration": len(audio_data) / sr}

    # vLLM batch ASR
    gen_kwargs = {"max_new_tokens": 500}
    if language:
        gen_kwargs["language"] = language
    if hotwords:
        gen_kwargs["hotwords"] = hotwords

    results = _engine.generate(inputs=seg_audios, **gen_kwargs)

    # Build segments with timestamps
    output_segments = []
    full_text_parts = []

    for i, (r, (start_ms, end_ms)) in enumerate(zip(results, seg_times)):
        seg_info = {
            "text": r["text"],
            "start": start_ms / 1000,
            "end": end_ms / 1000,
        }
        if use_timestamp and "timestamps" in r:
            # Offset timestamps by segment start
            offset = start_ms / 1000
            seg_info["words"] = [
                {"word": ts["token"], "start": ts["start_time"] + offset, "end": ts["end_time"] + offset}
                for ts in r["timestamps"]
            ]
        output_segments.append(seg_info)
        full_text_parts.append(r["text"])

    # SPK diarization
    if use_spk and _spk_model is not None:
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend

        vad_segs = [[st, et, audio_data[int(st*sr):int(et*sr)]] 
                    for st, et in [(s["start"], s["end"]) for s in output_segments]]
        chunks = sv_chunk(vad_segs)
        if chunks:
            speech_list = [ch[2] for ch in chunks]
            spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
            embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
            cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
            labels = cluster(embs.cpu(), oracle_num=None)
            if not isinstance(labels, np.ndarray):
                labels = np.array(labels)
            all_sorted = sorted(chunks, key=lambda x: x[0])
            sv_output = postprocess(all_sorted, None, labels, embs.cpu())
            sentences = [{"text": s["text"], "start": int(s["start"]*1000), "end": int(s["end"]*1000)} 
                        for s in output_segments]
            distribute_spk(sentences, sv_output)
            for i, s in enumerate(sentences):
                output_segments[i]["speaker"] = f"SPK{s.get('spk', 0)}"

    return {
        "text": " ".join(full_text_parts),
        "segments": output_segments,
        "duration": len(audio_data) / sr,
    }


# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(title="Fun-ASR-Nano vLLM Server", version="1.0")


@app.on_event("startup")
async def startup():
    load_engine(_args)


# --- HTTP REST: POST /asr ---
@app.post("/asr")
async def asr_endpoint(
    file: UploadFile = File(...),
    language: str = Form(default=None),
    hotwords: str = Form(default=""),
    spk: bool = Form(default=False),
    timestamp: bool = Form(default=True),
):
    """ASR with file upload. Returns text + segments + timestamps + speaker."""
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))

    hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None

    t0 = time.perf_counter()
    result = process_audio(audio_data, sr=sr, language=language, 
                          hotwords=hw_list, use_spk=spk, use_timestamp=timestamp)
    t1 = time.perf_counter()

    result["processing_time"] = round(t1 - t0, 3)
    result["rtf"] = round((t1 - t0) / result["duration"], 4) if result["duration"] > 0 else 0
    return JSONResponse(content=result)


# --- OpenAI API: POST /v1/audio/transcriptions ---
@app.post("/v1/audio/transcriptions")
async def openai_transcriptions(
    file: UploadFile = File(...),
    model: str = Form(default="fun-asr-nano"),
    language: str = Form(default=None),
    response_format: str = Form(default="json"),
    timestamp_granularities: str = Form(default="word"),
    spk: bool = Form(default=False),
):
    """OpenAI Whisper-compatible transcription API (extended with spk support)."""
    content = await file.read()
    audio_data, sr = sf.read(io.BytesIO(content))

    use_ts = "word" in timestamp_granularities or "segment" in timestamp_granularities
    result = process_audio(audio_data, sr=sr, language=language, use_spk=spk, use_timestamp=use_ts)

    if response_format == "text":
        return JSONResponse(content=result["text"])
    elif response_format == "verbose_json":
        return JSONResponse(content={
            "task": "transcribe",
            "language": language or "zh",
            "duration": result["duration"],
            "text": result["text"],
            "segments": [
                {
                    "id": i,
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "words": seg.get("words", []),
                }
                for i, seg in enumerate(result["segments"])
            ],
        })
    else:
        return JSONResponse(content={"text": result["text"]})


# --- WebSocket: ws://host:port/ws ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Streaming WebSocket ASR with dynamic VAD + SPK."""
    await websocket.accept()
    logger.info(f"WebSocket connected: {websocket.client}")

    vad = DynamicStreamingVAD(_vad_model)
    audio_buffer = np.array([], dtype=np.float32)
    locked_sentences = []
    language = None
    hotwords = None
    use_spk = False
    is_active = False

    try:
        while True:
            message = await websocket.receive()

            if "text" in message:
                cmd = message["text"].strip()
                if cmd.upper() == "START":
                    vad.reset()
                    audio_buffer = np.array([], dtype=np.float32)
                    locked_sentences = []
                    is_active = True
                    await websocket.send_json({"event": "started"})
                elif cmd.upper().startswith("LANGUAGE:"):
                    language = cmd[9:].strip() or None
                    await websocket.send_json({"event": "language_set", "language": language})
                elif cmd.upper().startswith("HOTWORDS:"):
                    hotwords = [w.strip() for w in cmd[9:].split(",") if w.strip()]
                    await websocket.send_json({"event": "hotwords_set", "hotwords": hotwords})
                elif cmd.upper().startswith("SPK:"):
                    use_spk = cmd[4:].strip().lower() in ("true", "1", "on", "yes")
                    await websocket.send_json({"event": "spk_set", "spk": use_spk})
                elif cmd.upper() == "STOP":
                    if is_active and len(audio_buffer) > 0:
                        # Final: process remaining audio
                        final_segs = vad.finalize()
                        for seg in final_segs:
                            seg_audio = audio_buffer[int(seg[0]*16):int(seg[1]*16)]
                            if len(seg_audio) > 8000:
                                gen_kw = {"max_new_tokens": 500}
                                if language: gen_kw["language"] = language
                                if hotwords: gen_kw["hotwords"] = hotwords
                                res = _engine.generate(inputs=[seg_audio], **gen_kw)
                                if res[0]["text"].strip():
                                    locked_sentences.append({
                                        "text": res[0]["text"], "start": seg[0], "end": seg[1]
                                    })

                        # Handle ongoing speech
                        if vad.is_speaking:
                            end_ms = int(len(audio_buffer) * 1000 / 16000)
                            start_ms = int(vad.current_speech_start) if hasattr(vad, 'current_speech_start') and vad.current_speech_start else 0
                            seg_audio = audio_buffer[int(start_ms*16):]
                            if len(seg_audio) > 8000:
                                gen_kw = {"max_new_tokens": 500}
                                if language: gen_kw["language"] = language
                                if hotwords: gen_kw["hotwords"] = hotwords
                                res = _engine.generate(inputs=[seg_audio], **gen_kw)
                                if res[0]["text"].strip():
                                    locked_sentences.append({
                                        "text": res[0]["text"], "start": start_ms, "end": end_ms
                                    })

                        # SPK: run full clustering on all sentences (only if enabled)
                        if use_spk and locked_sentences and _spk_model is not None:
                            try:
                                from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
                                from funasr.models.campplus.cluster_backend import ClusterBackend
                                vad_segs = [[s["start"]/1000, s["end"]/1000, 
                                            audio_buffer[int(s["start"]*16):int(s["end"]*16)]]
                                           for s in locked_sentences]
                                chunks = sv_chunk(vad_segs)
                                if chunks:
                                    speech_list = [ch[2] for ch in chunks]
                                    spk_res = _spk_model.generate(input=speech_list, cache={}, is_final=True)
                                    import torch as _torch
                                    embs = _torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
                                    cluster = ClusterBackend(merge_thr=0.78).to(_args.device)
                                    labels = cluster(embs.cpu(), oracle_num=None)
                                    if not isinstance(labels, np.ndarray):
                                        labels = np.array(labels)
                                    all_sorted = sorted(chunks, key=lambda x: x[0])
                                    sv_output = postprocess(all_sorted, None, labels, embs.cpu())
                                    spk_sents = [{"text": s["text"], "start": int(s["start"]), "end": int(s["end"])}
                                                for s in locked_sentences]
                                    distribute_spk(spk_sents, sv_output)
                                    for i, ss in enumerate(spk_sents):
                                        locked_sentences[i]["spk"] = ss.get("spk", 0)
                            except Exception as e:
                                logger.warning(f"SPK failed: {e}")

                        await websocket.send_json({
                            "sentences": locked_sentences,
                            "is_final": True,
                            "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                        })
                        is_active = False
                    await websocket.send_json({"event": "stopped"})

            elif "bytes" in message and is_active:
                pcm = np.frombuffer(message["bytes"], dtype=np.int16).astype(np.float32) / 32768.0
                audio_buffer = np.concatenate([audio_buffer, pcm])

                # Feed VAD
                new_confirmed = vad.feed(torch.from_numpy(pcm).float())
                for seg in new_confirmed:
                    seg_audio = audio_buffer[int(seg[0]*16):int(seg[1]*16)]
                    if len(seg_audio) > 8000:
                        gen_kw = {"max_new_tokens": 500}
                        if language: gen_kw["language"] = language
                        if hotwords: gen_kw["hotwords"] = hotwords
                        res = _engine.generate(inputs=[seg_audio], **gen_kw)
                        if res[0]["text"].strip():
                            locked_sentences.append({
                                "text": res[0]["text"], "start": seg[0], "end": seg[1]
                            })

                # Send partial update
                await websocket.send_json({
                    "sentences": locked_sentences,
                    "is_final": False,
                    "duration_ms": int(len(audio_buffer) * 1000 / 16000),
                })

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano vLLM Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", type=str, default="ms")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    _args = parser.parse_args()

    load_engine(_args)
    uvicorn.run(app, host=_args.host, port=_args.port)
