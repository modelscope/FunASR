#!/usr/bin/env python3
"""Fun-ASR-Nano Streaming WebSocket Server.

Features:
- Streaming VAD segmentation (fsmn-vad)
- Per-segment ASR decoding (Fun-ASR-Nano via vLLM or AutoModel)
- Speaker diarization (eres2netv2 + ClusterBackend)
- Hotword customization
- Hallucination detection & prevention
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
import regex
import websockets

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def detect_and_fix_hallucination(text, max_ngram_length=12, max_occurrences=3):
    """Detect repeated patterns (hallucination) and truncate to keep one occurrence."""
    if not text or len(text) < max_ngram_length * 2:
        return text, False

    cleaned = regex.sub(r'\p{P}+', '', text)

    word_pattern = rf'(?<!\S)(?!\d+$)(\w+)(?:\s+\1){{{max_occurrences - 1},}}(?!\S)'
    match = regex.search(word_pattern, cleaned, regex.IGNORECASE)
    if match:
        repeated = match.group(1)
        pos = text.find(repeated)
        if pos >= 0:
            end_pos = text.find(repeated, pos + len(repeated))
            if end_pos >= 0:
                return text[:end_pos], True
        return text[:len(text)//2], True

    for length in range(1, max_ngram_length):
        pattern = rf'(?<!\d)(\S{{{length}}})\1{{{max_occurrences - 1},}}(?!\d)'
        combined = rf'(?=.*\D){pattern}'
        match = regex.search(combined, cleaned)
        if match:
            repeated = match.group(1)
            pos = text.find(repeated)
            if pos >= 0:
                end_pos = text.find(repeated, pos + len(repeated))
                if end_pos >= 0:
                    return text[:end_pos], True
            return text[:len(text)//2], True

    return text, False


def _clean_asr_text(text):
    """Remove timestamp tags and artifacts from vLLM output."""
    import re
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[Ｏ\[\]&＆|｜]', '', text)
    text = re.sub(r'/sil|endofbreak|FFFF', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


from funasr.models.fsmn_vad_streaming.model import FsmnVADStreaming as DynamicStreamingVAD


class HybridSpeakerTracker:
    """Speaker diarization: streaming ClusterBackend + final re-clustering."""

    def __init__(self, spk_model, device, threshold=0.6):
        self.spk_model = spk_model
        self.device = device
        self.threshold = threshold
        self.speaker_centers = []
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend
        self.sv_chunk = sv_chunk
        self.postprocess = postprocess
        self.distribute_spk = distribute_spk
        self.cluster_backend = ClusterBackend(merge_thr=0.78).to(device)
        self.all_chunks = []
        self.all_embeddings = []
        self.display_map = {}
        self.next_display_id = 0

    @torch.no_grad()
    def assign_streaming(self, audio_samples, seg_start_s, seg_end_s, sentence):
        """Assign speaker ID during streaming using ClusterBackend."""
        vad_seg = [[seg_start_s, seg_end_s, audio_samples]]
        chunks = self.sv_chunk(vad_seg)
        if not chunks:
            sentence["spk"] = self.next_display_id
            self.next_display_id += 1
            return

        self.all_chunks.extend(chunks)
        speech_list = [ch[2] for ch in chunks]
        spk_res = self.spk_model.generate(input=speech_list, cache={}, is_final=True)
        embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
        self.all_embeddings.append(embs)

        all_embs = torch.cat(self.all_embeddings, dim=0)
        labels = self.cluster_backend(all_embs.cpu(), oracle_num=None)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        all_sorted = sorted(self.all_chunks, key=lambda x: x[0])
        sv_output = self.postprocess(all_sorted, None, labels, all_embs.cpu())
        temp = [{"start": int(seg_start_s*1000), "end": int(seg_end_s*1000), "text": sentence["text"]}]
        self.distribute_spk(temp, sv_output)
        raw_spk = temp[0].get("spk", 0)

        if raw_spk not in self.display_map:
            self.display_map[raw_spk] = self.next_display_id
            self.next_display_id += 1
        sentence["spk"] = self.display_map[raw_spk]

    @torch.no_grad()
    def finalize(self, sentences, min_split_s=3.0):
        """Final re-clustering for accurate speaker assignment."""
        if not self.all_embeddings or not sentences:
            return sentences

        all_embs = torch.cat(self.all_embeddings, dim=0)
        labels = self.cluster_backend(all_embs.cpu(), oracle_num=None)
        if not isinstance(labels, np.ndarray):
            labels = np.array(labels)

        all_sorted = sorted(self.all_chunks, key=lambda x: x[0])
        sv_output = self.postprocess(all_sorted, None, labels, all_embs.cpu())

        for s in sentences:
            s.pop("spk", None)
        self.distribute_spk(sentences, sv_output)

        id_map = {}
        next_id = 0
        for s in sentences:
            raw = s.get("spk", 0)
            if raw not in id_map:
                id_map[raw] = next_id
                next_id += 1
            s["spk"] = id_map[raw]

        final_sentences = []
        for s in sentences:
            sub = self._try_split(s, sv_output, id_map, min_split_s)
            final_sentences.extend(sub)

        return final_sentences

    def _try_split(self, sentence, sv_output, id_map, min_split_s):
        """Split a sentence if multiple speakers detected within its time range."""
        sent_start = sentence["start"] / 1000.0
        sent_end = sentence["end"] / 1000.0
        text = sentence["text"]

        overlapping = []
        for sv_start, sv_end, sv_spk in sv_output:
            o_start = max(sent_start, sv_start)
            o_end = min(sent_end, sv_end)
            if o_end > o_start:
                mapped_spk = id_map.get(int(sv_spk), int(sv_spk))
                overlapping.append([o_start, o_end, mapped_spk])

        if len(overlapping) <= 1:
            return [sentence]

        filtered = [overlapping[0]]
        for i in range(1, len(overlapping)):
            cur = overlapping[i]
            prev = filtered[-1]
            if cur[2] == prev[2]:
                filtered[-1] = [prev[0], cur[1], prev[2]]
            elif (cur[1] - cur[0]) < min_split_s:
                filtered[-1] = [prev[0], cur[1], prev[2]]
            else:
                filtered.append(cur)

        merged = [filtered[0]]
        for i in range(1, len(filtered)):
            if (merged[-1][1] - merged[-1][0]) < min_split_s:
                merged[-1] = [merged[-1][0], filtered[i][1], filtered[i][2]]
            else:
                merged.append(filtered[i])
        if len(merged) > 1 and (merged[-1][1] - merged[-1][0]) < min_split_s:
            merged[-2] = [merged[-2][0], merged[-1][1], merged[-2][2]]
            merged.pop()

        if len(merged) <= 1:
            return [sentence]

        total_dur = sum(m[1] - m[0] for m in merged)
        sub_sentences = []
        char_pos = 0
        for i, (m_start, m_end, m_spk) in enumerate(merged):
            if i == len(merged) - 1:
                sub_text = text[char_pos:]
            else:
                n_chars = max(1, int(len(text) * (m_end - m_start) / total_dur))
                sub_text = text[char_pos:char_pos + n_chars]
                char_pos += n_chars
            if sub_text.strip():
                sub_sentences.append({"text": sub_text.strip(), "start": int(m_start*1000), "end": int(m_end*1000), "spk": m_spk})

        return sub_sentences if sub_sentences else [sentence]

    def reset(self):
        self.speaker_centers = []
        self.all_chunks = []
        self.all_embeddings = []
        self.display_map = {}
        self.next_display_id = 0


class RealtimeASRSession:
    """Manages a single streaming ASR session."""

    def __init__(self, vllm_engine, asr_kwargs, vad, spk_tracker=None, sample_rate=16000, chunk_ms=960):
        self.vllm_engine = vllm_engine
        self.asr_kwargs = asr_kwargs
        self.vad = vad
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.first_chunk_samples = int(sample_rate * 480 / 1000)
        self.first_decode_done = False

        self._audio_buffer = np.array([], dtype=np.float32)
        self.audio_chunks = []
        self._audio_buffer_dirty = False
        self.vad_fed_samples = 0
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_decode_samples = 0
        self.locked_sentences = []
        self.prev_seg_text = ""
        self.spk_tracker = spk_tracker
        self.use_context = True
        self.is_active = False

    @property
    def audio_buffer(self):
        if getattr(self, '_audio_buffer_dirty', False):
            if self.audio_chunks:
                self._audio_buffer = np.concatenate([self._audio_buffer, *self.audio_chunks])
                self.audio_chunks = []
            self._audio_buffer_dirty = False
        return self._audio_buffer

    @audio_buffer.setter
    def audio_buffer(self, value):
        self._audio_buffer = value
        self.audio_chunks = []
        self._audio_buffer_dirty = False

    def add_audio(self, pcm_bytes):
        if len(pcm_bytes) % 2 != 0:
            pcm_bytes = pcm_bytes[:len(pcm_bytes) - (len(pcm_bytes) % 2)]
        if not pcm_bytes:
            return
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        
        self.audio_chunks.append(audio_float)
        self._audio_buffer_dirty = True

        new_confirmed = self.vad.feed(torch.from_numpy(audio_float).float(), is_final=False)
        self.vad_fed_samples += len(audio_float)

        for seg in new_confirmed:
            seg_text = self._decode_segment(seg)
            self.prev_text = ""
            if not seg_text.strip():
                continue
            self.locked_sentences.append({"text": seg_text, "start": int(seg[0]), "end": int(seg[1])})
            if self.spk_tracker:
                s0 = int(seg[0] * self.sample_rate / 1000)
                s1 = min(int(seg[1] * self.sample_rate / 1000), len(self.audio_buffer))
                self.spk_tracker.assign_streaming(self.audio_buffer[s0:s1], seg[0]/1000, seg[1]/1000, self.locked_sentences[-1])
            logger.info(f"Locked: [{seg[0]}-{seg[1]}ms] \"{seg_text[:40]}\"")

    def should_decode(self):
        threshold = self.first_chunk_samples if not self.first_decode_done else self.chunk_samples
        return (len(self.audio_buffer) - self.last_decode_samples) >= threshold

    @torch.no_grad()
    def decode(self, is_final=False):
        if len(self.audio_buffer) < self.chunk_samples:
            return self._build_response(is_final)

        if is_final:
            remaining = self.audio_buffer[self.vad_fed_samples:]
            if len(remaining) > 0:
                new_confirmed = self.vad.feed(torch.from_numpy(remaining).float(), is_final=True)
                self.vad_fed_samples = len(self.audio_buffer)
                for seg in new_confirmed:
                    seg_text = self._decode_segment(seg)
                    if not seg_text.strip():
                        continue
                    self.locked_sentences.append({"text": seg_text, "start": int(seg[0]), "end": int(seg[1])})

            if self.vad.current_speech_start is not None:
                end_ms = int(len(self.audio_buffer) * 1000 / self.sample_rate)
                seg = [self.vad.current_speech_start, end_ms]
                seg_text = self._decode_segment(seg)
                if seg_text.strip():
                    self.locked_sentences.append({"text": seg_text, "start": int(seg[0]), "end": int(seg[1])})
                self.vad.current_speech_start = None

            if self.spk_tracker and self.locked_sentences:
                self.locked_sentences = self.spk_tracker.finalize(self.locked_sentences)
            return self._build_response(is_final)

        if self.vad.current_speech_start is not None:
            seg_start_sample = int(self.vad.current_speech_start * self.sample_rate / 1000)
            seg_audio = self.audio_buffer[seg_start_sample:]
        else:
            self.last_decode_samples = len(self.audio_buffer)
            self.last_partial_text = ""
            return self._build_response(is_final)

        if len(seg_audio) < self.chunk_samples // 2:
            return self._build_response(is_final)

        audio_tensor = torch.from_numpy(seg_audio).float()
        try:
            if hasattr(self.vllm_engine, 'generate') and not hasattr(self.vllm_engine, '_engine'):
                # Standard AutoModel: returns list of dicts with 'text'
                results = self.vllm_engine.generate(
                    input=audio_tensor,
                    hotword=self.asr_kwargs.get("hotwords"),
                    language=self.asr_kwargs.get("language"),
                )
                # AutoModel results format is typically: [{'text': '...'}]
                if isinstance(results, list) and len(results) > 0:
                    text = results[0].get('text', '')
                else:
                    text = str(results) if results is not None else ""
            else:
                # AutoModelVLLM
                results = self.vllm_engine.generate(
                    inputs=[audio_tensor],
                    hotwords=self.asr_kwargs.get("hotwords"),
                    language=self.asr_kwargs.get("language"),
                    max_new_tokens=200,
                )
                text = results[0]["text"] if (results and len(results) > 0) else ""
            
            text = _clean_asr_text(text)
        except Exception as e:
            logger.error(f"ASR error: {e}", exc_info=True)
            return self._build_response(is_final)

        text, hallucinated = detect_and_fix_hallucination(text)
        if hallucinated:
            self.prev_text = ""

        self.last_decode_samples = len(self.audio_buffer)
        self.last_partial_text = text
        if text.strip() and not self.first_decode_done:
            self.first_decode_done = True

        if hasattr(self.vllm_engine, '_engine'):
            tokenizer = self.vllm_engine._engine.tokenizer
        else:
            tokenizer = getattr(self.vllm_engine, 'kwargs', {}).get("tokenizer")

        if tokenizer is not None:
            try:
                encoded = tokenizer.encode(text)
                if len(encoded) > 5:
                    try:
                        self.prev_text = tokenizer.decode(encoded[:-5], skip_special_tokens=True)
                    except TypeError:
                        self.prev_text = tokenizer.decode(encoded[:-5])
                else:
                    self.prev_text = ""
            except Exception as e:
                logger.warning(f"Failed to encode/decode text with tokenizer: {e}")
                self.prev_text = ""
        else:
            self.prev_text = ""

        return self._build_response(is_final)

    @torch.no_grad()
    def _decode_segment(self, seg):
        """Decode a completed VAD segment via vLLM or AutoModel."""
        start_sample = int(seg[0] * self.sample_rate / 1000)
        end_sample = min(int(seg[1] * self.sample_rate / 1000), len(self.audio_buffer))
        seg_audio = self.audio_buffer[start_sample:end_sample]
        if len(seg_audio) < 1600:
            return ""
        audio_tensor = torch.from_numpy(seg_audio).float()
        try:
            if hasattr(self.vllm_engine, 'generate') and not hasattr(self.vllm_engine, '_engine'):
                # Standard AutoModel
                results = self.vllm_engine.generate(
                    input=audio_tensor,
                    hotword=self.asr_kwargs.get("hotwords"),
                    language=self.asr_kwargs.get("language"),
                )
                text = results[0].get('text', '') if (isinstance(results, list) and len(results) > 0) else str(results)
            else:
                # AutoModelVLLM
                results = self.vllm_engine.generate(
                    inputs=[audio_tensor],
                    hotwords=self.asr_kwargs.get("hotwords"),
                    language=self.asr_kwargs.get("language"),
                    max_new_tokens=512,
                )
                text = results[0]["text"] if (results and len(results) > 0) else ""

            text = _clean_asr_text(text)
            self.prev_seg_text = text
            return text
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
        self.first_decode_done = False
        self.vad.reset()
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_decode_samples = 0
        self.locked_sentences = []
        if self.spk_tracker:
            self.spk_tracker.reset()


_vllm_engine = None
_asr_kwargs = None
_vad_model = None
_spk_model = None


def load_models(args):
    global _vllm_engine, _asr_kwargs, _vad_model, _spk_model
    if _vllm_engine is None:
        from funasr import AutoModel
        
        if args.device == "cpu":
            logger.info(f"CPU detected: Loading ASR (AutoModel) instead of vLLM")
            _vllm_engine = AutoModel(
                model=args.model, hub=args.hub, device=args.device,
                trust_remote_code=True, disable_update=True
            )
        else:
            from funasr.auto.auto_model_vllm import AutoModelVLLM
            logger.info(f"Loading ASR (vLLM): {args.model}")
            _vllm_engine = AutoModelVLLM(
                model=args.model, hub=args.hub, device=args.device,
                dtype=getattr(args, 'dtype', 'bf16'),
                tensor_parallel_size=getattr(args, 'tensor_parallel_size', 1),
                gpu_memory_utilization=getattr(args, 'gpu_memory_utilization', 0.8),
                max_model_len=getattr(args, 'max_model_len', 2048),
            )

        _asr_kwargs = {}
        hw_file = getattr(args, 'hotword_file', '热词列表')
        if hw_file and os.path.isfile(hw_file):
            with open(hw_file, "r", encoding="utf-8") as hf:
                hotwords = [line.strip() for line in hf if line.strip()]
            _asr_kwargs["hotwords"] = hotwords
            logger.info(f"Loaded {len(hotwords)} hotwords from '{hw_file}'")

        if getattr(args, 'language', None):
            _asr_kwargs["language"] = args.language
            logger.info(f"Language: {args.language}")

        logger.info("Loading VAD: fsmn-vad (streaming)")
        _vad_model = AutoModel(model="fsmn-vad", device=args.device, disable_update=True)

        logger.info("Loading SPK: eres2netv2")
        _spk_model = AutoModel(model="iic/speech_eres2netv2_sv_zh-cn_16k-common", device=args.device, disable_update=True)

        logger.info("All models ready!")
    return _vllm_engine, _asr_kwargs, _vad_model, _spk_model


async def handle_client(websocket, args):
    vllm_engine, asr_kwargs, vad_model, spk_model = load_models(args)
    vad = DynamicStreamingVAD(vad_model)
    spk_tracker = HybridSpeakerTracker(spk_model, args.device)
    session = RealtimeASRSession(vllm_engine, asr_kwargs, vad, spk_tracker=spk_tracker)
    logger.info(f"Client connected: {websocket.remote_address}")

    decode_interval = args.decode_interval
    last_decode_time = 0

    try:
        async for message in websocket:
            if isinstance(message, str):
                cmd = message.strip()
                if cmd.upper() == "START":
                    session.reset()
                    session.is_active = True
                    await websocket.send(json.dumps({"event": "started"}))
                    logger.info("Session started")
                elif cmd.upper().startswith("HOTWORDS:"):
                    hw_str = cmd[9:]
                    hotwords = [w.strip() for w in hw_str.split(",") if w.strip()]
                    session.asr_kwargs = dict(session.asr_kwargs)
                    session.asr_kwargs["hotwords"] = hotwords
                    await websocket.send(json.dumps({"event": "hotwords_set", "hotwords": hotwords}))
                    logger.info(f"Hotwords set: {len(hotwords)} words")
                elif cmd.upper().startswith("LANGUAGE:"):
                    lang = cmd[9:].strip()
                    session.asr_kwargs = dict(session.asr_kwargs)
                    session.asr_kwargs["language"] = lang if lang else None
                    await websocket.send(json.dumps({"event": "language_set", "language": lang}))
                    logger.info(f"Language set: {lang}")
                elif cmd.upper() == "STOP":
                    if session.is_active and len(session.audio_buffer) > 0:
                        result = session.decode(is_final=True)
                        await websocket.send(json.dumps(result))
                        logger.info(f"Final: {len(result['sentences'])} sentences")
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
        logger.error(f"Error: {e}", exc_info=True)


async def main(args):
    load_models(args)
    logger.info(f"Server on ws://0.0.0.0:{args.port}")
    async with websockets.serve(
        lambda ws: handle_client(ws, args), "0.0.0.0", args.port,
        max_size=10*1024*1024,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano Streaming WebSocket Server")
    parser.add_argument("--port", type=int, default=10095)
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", type=str, default="ms", choices=["ms", "hf"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-context", action="store_true", default=True)
    parser.add_argument("--no-context", dest="use_context", action="store_false")
    parser.add_argument("--decode-interval", type=float, default=0.48)
    parser.add_argument("--hotword-file", type=str, default="热词列表")
    parser.add_argument("--language", type=str, default=None, help="Language hint (e.g. 中文, English, 日本語)")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    args = parser.parse_args()
    asyncio.run(main(args))
