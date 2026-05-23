#!/usr/bin/env python3
"""
Fun-ASR-Nano Real-time WebSocket Server.

Key fix: Only decode CURRENT VAD segment (not entire cumulative audio).
Confirmed segments get their text locked and never re-decoded.
This prevents model collapse on long recordings.
"""

import asyncio
import json
import logging
import os
import time
import argparse
import numpy as np
import torch
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

try:
    import websockets
except ImportError:
    os.system("pip install websockets --break-system-packages -q")
    import websockets


class StreamingVAD:
    """Streaming VAD: [start,-1]=speech start, [-1,end]=speech end."""

    def __init__(self, vad_model):
        self.model = vad_model
        self.cache = {}
        self.confirmed_segments = []
        self.current_speech_start = None

    def feed(self, audio_chunk_tensor, is_final=False):
        res = self.model.generate(
            input=[audio_chunk_tensor], cache=self.cache,
            is_final=is_final, chunk_size=200,
        )
        signals = res[0].get("value", [])
        new_confirmed = []
        for sig in signals:
            if sig[0] >= 0 and sig[1] == -1:
                self.current_speech_start = sig[0]
            elif sig[0] == -1 and sig[1] >= 0:
                start = self.current_speech_start if self.current_speech_start is not None else 0
                self.confirmed_segments.append([start, sig[1]])
                new_confirmed.append([start, sig[1]])
                self.current_speech_start = None
            elif sig[0] >= 0 and sig[1] >= 0:
                self.confirmed_segments.append(sig)
                new_confirmed.append(sig)
                self.current_speech_start = None
        return new_confirmed

    def reset(self):
        self.cache = {}
        self.confirmed_segments = []
        self.current_speech_start = None


class RealtimeASRSession:
    def __init__(self, asr_model, asr_kwargs, tokenizer, vad, sample_rate=16000, chunk_ms=720):
        self.asr_model = asr_model
        self.asr_kwargs = asr_kwargs
        self.tokenizer = tokenizer
        self.vad = vad
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)

        self.audio_buffer = np.array([], dtype=np.float32)
        self.vad_fed_samples = 0
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_decode_samples = 0
        self.locked_sentences = []  # text locked for confirmed segments
        self.is_active = False

    def add_audio(self, pcm_bytes):
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])

        # Feed new audio to streaming VAD
        new_audio = self.audio_buffer[self.vad_fed_samples:]
        if len(new_audio) > 0:
            new_confirmed = self.vad.feed(torch.from_numpy(new_audio).float(), is_final=False)
            self.vad_fed_samples = len(self.audio_buffer)

            # Lock text for newly confirmed segments
            for seg in new_confirmed:
                seg_text = self._decode_segment(seg)
                self.locked_sentences.append({"text": seg_text, "start_ms": seg[0], "end_ms": seg[1]})
                self.prev_text = ""  # reset for new segment
                logger.info(f"  Locked: [{seg[0]}-{seg[1]}ms] \"{seg_text[:30]}\"")

    def should_decode(self):
        return (len(self.audio_buffer) - self.last_decode_samples) >= self.chunk_samples

    @torch.no_grad()
    def decode(self, is_final=False):
        if len(self.audio_buffer) < self.chunk_samples:
            return self._build_response(is_final)

        if is_final:
            # Feed remaining to VAD
            remaining = self.audio_buffer[self.vad_fed_samples:]
            if len(remaining) > 0:
                new_confirmed = self.vad.feed(torch.from_numpy(remaining).float(), is_final=True)
                self.vad_fed_samples = len(self.audio_buffer)
                for seg in new_confirmed:
                    seg_text = self._decode_segment(seg)
                    self.locked_sentences.append({"text": seg_text, "start_ms": seg[0], "end_ms": seg[1]})

            # If still in speech, close and decode it
            if self.vad.current_speech_start is not None:
                end_ms = int(len(self.audio_buffer) * 1000 / self.sample_rate)
                seg = [self.vad.current_speech_start, end_ms]
                seg_text = self._decode_segment(seg)
                self.locked_sentences.append({"text": seg_text, "start_ms": seg[0], "end_ms": seg[1]})
                self.vad.current_speech_start = None

            return self._build_response(is_final)

        # Decode ONLY current ongoing segment (not full buffer!)
        if self.vad.current_speech_start is not None:
            seg_start_sample = int(self.vad.current_speech_start * self.sample_rate / 1000)
            seg_audio = self.audio_buffer[seg_start_sample:]
        else:
            # Not in speech - nothing to decode
            self.last_decode_samples = len(self.audio_buffer)
            self.last_partial_text = ""
            return self._build_response(is_final)

        if len(seg_audio) < self.chunk_samples // 2:
            return self._build_response(is_final)

        audio_tensor = torch.from_numpy(seg_audio).float()
        try:
            res = self.asr_model.inference(
                [audio_tensor], prev_text=self.prev_text, **self.asr_kwargs
            )
            text = res[0][0]["text"]
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return self._build_response(is_final)

        self.last_decode_samples = len(self.audio_buffer)
        self.last_partial_text = text

        # Update prev_text (rollback 5 tokens)
        encoded = self.tokenizer.encode(text)
        if len(encoded) > 5:
            self.prev_text = self.tokenizer.decode(encoded[:-5]).replace("�", "")
        else:
            self.prev_text = ""

        return self._build_response(is_final)

    @torch.no_grad()
    def _decode_segment(self, seg):
        """Decode a specific completed VAD segment."""
        start_sample = int(seg[0] * self.sample_rate / 1000)
        end_sample = min(int(seg[1] * self.sample_rate / 1000), len(self.audio_buffer))
        seg_audio = self.audio_buffer[start_sample:end_sample]
        if len(seg_audio) < 1600:
            return ""
        audio_tensor = torch.from_numpy(seg_audio).float()
        try:
            res = self.asr_model.inference([audio_tensor], prev_text="", **self.asr_kwargs)
            return res[0][0]["text"]
        except Exception as e:
            logger.error(f"Segment decode error: {e}")
            return ""

    def _build_response(self, is_final):
        duration_ms = int(len(self.audio_buffer) * 1000 / self.sample_rate)
        sentences = list(self.locked_sentences)
        partial = self.last_partial_text
        partial_start = self.vad.current_speech_start or duration_ms

        if is_final:
            return {"sentences": sentences, "partial": "", "partial_start_ms": 0,
                    "duration_ms": duration_ms, "is_final": True}

        return {"sentences": sentences, "partial": partial,
                "partial_start_ms": partial_start,
                "duration_ms": duration_ms, "is_final": False}

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.vad_fed_samples = 0
        self.vad.reset()
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_decode_samples = 0
        self.locked_sentences = []


_asr_model = None
_asr_kwargs = None
_tokenizer = None
_vad_model = None


def load_models(args):
    global _asr_model, _asr_kwargs, _tokenizer, _vad_model
    if _asr_model is None:
        from funasr import AutoModel
        logger.info(f"Loading ASR: {args.model}")
        model = AutoModel(
            model=args.model, trust_remote_code=True,
            remote_code=os.path.join(os.path.dirname(__file__), "model.py"),
            device=args.device, hub=args.hub, disable_update=True,
        )
        _asr_model = model.model
        _asr_kwargs = model.kwargs
        _tokenizer = model.kwargs["tokenizer"]

        logger.info("Loading VAD: fsmn-vad (streaming)")
        _vad_model = AutoModel(model="fsmn-vad", device=args.device, disable_update=True)
        logger.info("All models ready!")
    return _asr_model, _asr_kwargs, _tokenizer, _vad_model


async def handle_client(websocket, args):
    asr_model, asr_kwargs, tokenizer, vad_automodel = load_models(args)
    vad = StreamingVAD(vad_automodel)
    session = RealtimeASRSession(asr_model, asr_kwargs, tokenizer, vad)
    logger.info(f"Client connected: {websocket.remote_address}")

    decode_interval = args.decode_interval
    last_decode_time = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                cmd = message.strip().upper()
                if cmd == "START":
                    session.reset()
                    session.is_active = True
                    await websocket.send(json.dumps({"event": "started"}))
                    logger.info("Session started")
                elif cmd == "STOP":
                    if session.is_active and len(session.audio_buffer) > 0:
                        result = session.decode(is_final=True)
                        await websocket.send(json.dumps(result))
                        n = len(result["sentences"])
                        logger.info(f"Final: {n} sentences")
                        session.is_active = False
                    await websocket.send(json.dumps({"event": "stopped"}))
            elif isinstance(message, bytes) and session.is_active:
                session.add_audio(message)

                now = time.time()
                if now - last_decode_time >= decode_interval and session.should_decode():
                    result = session.decode(is_final=False)
                    await websocket.send(json.dumps(result))
                    last_decode_time = now

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()


async def main(args):
    load_models(args)
    logger.info(f"Server on ws://0.0.0.0:{args.port}")
    async with websockets.serve(
        lambda ws: handle_client(ws, args), "0.0.0.0", args.port,
        max_size=10*1024*1024,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=10095)
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", type=str, default="ms")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--decode-interval", type=float, default=0.7)
    args = parser.parse_args()
    asyncio.run(main(args))
