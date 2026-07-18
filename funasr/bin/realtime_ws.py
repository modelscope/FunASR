#!/usr/bin/env python3
"""Fun-ASR-Nano Streaming WebSocket Server.

Features:
- Streaming VAD segmentation (fsmn-vad)
- Client-driven utterance endpoints (COMMIT, without server VAD)
- Per-segment ASR decoding (Fun-ASR-Nano via vLLM)
- Speaker diarization (eres2netv2 + ClusterBackend)
- Hotword customization
- Hallucination detection & prevention
"""

import asyncio
from collections import deque
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
    if regex.search(word_pattern, cleaned, regex.IGNORECASE):
        match = regex.search(word_pattern, cleaned, regex.IGNORECASE)
        repeated = match.group(1)
        pos = text.find(repeated)
        if pos >= 0:
            end_pos = text.find(repeated, pos + len(repeated))
            if end_pos >= 0:
                return text[:end_pos + len(repeated)], True
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
                    return text[:end_pos + len(repeated)], True
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


from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD


class ClientEndpointVAD:
    """Track an utterance start without loading or running a VAD model."""

    def __init__(self):
        self.current_speech_start = None

    def start_utterance(self, start_ms):
        if self.current_speech_start is None:
            self.current_speech_start = start_ms

    def reset(self):
        self.current_speech_start = None


class HybridSpeakerTracker:
    """Speaker diarization: streaming ClusterBackend + final re-clustering."""

    def __init__(
        self,
        spk_model,
        device,
        threshold=0.6,
        max_history_chunks=128,
        max_speakers=15,
    ):
        if max_history_chunks <= 0:
            raise ValueError("max_history_chunks must be positive")
        if max_speakers <= 0:
            raise ValueError("max_speakers must be positive")
        self.spk_model = spk_model
        self.device = device
        self.threshold = threshold
        self.max_speakers = max_speakers
        self.speaker_centers = []
        self.speaker_center_updates = []
        from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk
        from funasr.models.campplus.cluster_backend import ClusterBackend
        self.sv_chunk = sv_chunk
        self.postprocess = postprocess
        self.distribute_spk = distribute_spk
        self.cluster_backend = ClusterBackend(merge_thr=0.78).to(device)
        self.all_chunks = deque(maxlen=max_history_chunks)
        self.all_embeddings = deque(maxlen=max_history_chunks)
        self.last_speaker_id = 0

    @torch.no_grad()
    def assign_streaming(self, audio_samples, seg_start_s, seg_end_s, sentence):
        """Assign speaker ID during streaming using ClusterBackend."""
        vad_seg = [[seg_start_s, seg_end_s, audio_samples]]
        chunks = self.sv_chunk(vad_seg)
        if not chunks:
            sentence["spk"] = self.last_speaker_id
            return

        speech_list = [ch[2] for ch in chunks]
        spk_res = self.spk_model.generate(input=speech_list, cache={}, is_final=True)
        embeddings = torch.cat([r["spk_embedding"] for r in spk_res], dim=0).detach().cpu()
        for chunk, embedding in zip(chunks, embeddings):
            # Speaker post-processing only needs timestamps after embedding extraction.
            # Do not retain NumPy views into the session audio buffer.
            self.all_chunks.append((float(chunk[0]), float(chunk[1])))
            self.all_embeddings.append(embedding.clone())

        sv_output = self._cluster_recent(update_centers=True)
        temp = [{"start": int(seg_start_s*1000), "end": int(seg_end_s*1000), "text": sentence["text"]}]
        self.distribute_spk(temp, sv_output)
        sentence["spk"] = temp[0].get("spk", self.last_speaker_id)
        self.last_speaker_id = sentence["spk"]

    def _cluster_recent(self, update_centers):
        all_embeddings = torch.stack(list(self.all_embeddings), dim=0)
        labels = self.cluster_backend(all_embeddings, oracle_num=None)
        if not isinstance(labels, np.ndarray):
            labels = np.asarray(labels)

        chunks = [[start, end, None] for start, end in self.all_chunks]
        sv_output, cluster_centers = self.postprocess(
            chunks,
            None,
            labels,
            all_embeddings,
            return_spk_center=True,
        )
        stable_ids = self._map_cluster_centers(cluster_centers, update=update_centers)
        return [[start, end, stable_ids[int(spk)]] for start, end, spk in sv_output]

    def _map_cluster_centers(self, cluster_centers, update):
        centers = torch.as_tensor(cluster_centers, dtype=torch.float32).cpu()
        centers = torch.nn.functional.normalize(centers, dim=1)
        stable_ids = []
        used_ids = set()

        for center in centers:
            best_id = None
            best_similarity = float("-inf")
            created = False
            if self.speaker_centers:
                similarities = torch.stack(
                    [torch.dot(center, known_center) for known_center in self.speaker_centers]
                )
                for candidate in torch.argsort(similarities, descending=True).tolist():
                    if candidate not in used_ids:
                        best_id = candidate
                        best_similarity = float(similarities[candidate])
                        break

            matched = best_id is not None and best_similarity >= self.threshold
            if not matched and update and len(self.speaker_centers) < self.max_speakers:
                best_id = len(self.speaker_centers)
                self.speaker_centers.append(center.clone())
                self.speaker_center_updates.append(1)
                matched = True
                created = True
            elif best_id is None:
                # There can be more active clusters than the configured identity cap.
                if not self.speaker_centers:
                    best_id = 0
                else:
                    similarities = torch.stack(
                        [torch.dot(center, known_center) for known_center in self.speaker_centers]
                    )
                    best_id = int(torch.argmax(similarities))

            if update and matched and not created:
                count = self.speaker_center_updates[best_id]
                weight = 1.0 / min(count + 1, 20)
                updated = (1.0 - weight) * self.speaker_centers[best_id] + weight * center
                self.speaker_centers[best_id] = torch.nn.functional.normalize(updated, dim=0)
                self.speaker_center_updates[best_id] = count + 1

            stable_ids.append(best_id)
            used_ids.add(best_id)

        return stable_ids

    @torch.no_grad()
    def finalize(self, sentences, min_split_s=3.0):
        """Final re-clustering for accurate speaker assignment."""
        if not self.all_embeddings or not sentences:
            return sentences

        sv_output = self._cluster_recent(update_centers=False)
        history_start_ms = int(min(start for start, _ in self.all_chunks) * 1000)
        final_sentences = []
        for s in sentences:
            if s["end"] <= history_start_ms:
                final_sentences.append(s)
                continue
            current = dict(s)
            if s["start"] < history_start_ms:
                duration_ms = s["end"] - s["start"]
                boundary_ratio = (history_start_ms - s["start"]) / duration_ms
                split_index = min(
                    len(s["text"]) - 1,
                    max(1, round(len(s["text"]) * boundary_ratio)),
                )
                prefix_text = s["text"][:split_index].strip()
                suffix_text = s["text"][split_index:].strip()
                if not prefix_text or not suffix_text:
                    final_sentences.append(s)
                    continue

                prefix = dict(s)
                prefix.update(text=prefix_text, end=history_start_ms)
                final_sentences.append(prefix)
                current.update(
                    text=suffix_text,
                    start=history_start_ms,
                )
            self.distribute_spk([current], sv_output)
            final_sentences.extend(self._try_split(current, sv_output, {}, min_split_s))

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
        self.speaker_center_updates = []
        self.all_chunks.clear()
        self.all_embeddings.clear()
        self.last_speaker_id = 0

    def session_stats(self):
        return {
            "speaker_history_chunks": len(self.all_chunks),
            "speaker_history_embeddings": len(self.all_embeddings),
            "speaker_history_limit": self.all_chunks.maxlen,
            "speaker_centers": len(self.speaker_centers),
            "speaker_center_limit": self.max_speakers,
            "last_speaker_id": self.last_speaker_id,
        }


class RealtimeASRSession:
    """Manages a single streaming ASR session."""

    def __init__(
        self,
        vllm_engine,
        asr_kwargs,
        vad,
        spk_tracker=None,
        sample_rate=16000,
        chunk_ms=960,
        partial_window_sec=15.0,
        audio_lookback_sec=5.0,
        endpoint_mode="server",
    ):
        if endpoint_mode not in {"server", "client"}:
            raise ValueError(f"Unsupported endpoint mode: {endpoint_mode}")
        self.vllm_engine = vllm_engine
        self.asr_kwargs = asr_kwargs
        self.vad = vad
        self.endpoint_mode = endpoint_mode
        self.sample_rate = sample_rate
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.first_chunk_samples = int(sample_rate * 480 / 1000)
        # Bound the interim (partial) re-decode window. While a speech segment has
        # not yet hit a VAD pause it keeps growing, and the partial path re-encodes
        # it from the start on every chunk -> O(L^2) total re-encoding for a
        # length-L segment. Under concurrency that saturates the GPU and long-segment
        # requests time out. Capping the partial window to the most recent
        # `partial_window_sec` seconds makes interim re-decoding ~O(L) per segment
        # without changing the final result (completed segments are always decoded
        # in full by _decode_segment / the is_final path). Set <=0 to disable.
        self.partial_window_samples = int(sample_rate * partial_window_sec) if partial_window_sec and partial_window_sec > 0 else 0
        self.audio_lookback_samples = max(0, int(sample_rate * audio_lookback_sec))
        self.first_decode_done = False

        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_buffer_start_sample = 0
        self.total_samples = 0
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_partial_start_ms = 0
        self.last_decode_samples = 0
        self.locked_sentences = []
        self.prev_seg_text = ""
        self.spk_tracker = spk_tracker
        self.use_context = True
        self.is_active = False

    def add_audio(self, pcm_bytes):
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        audio_float = audio_int16.astype(np.float32) / 32768.0
        chunk_start_sample = self.total_samples
        self.audio_buffer = np.concatenate([self.audio_buffer, audio_float])
        self.total_samples += len(audio_float)

        if len(audio_float) > 0:
            if self.endpoint_mode == "client":
                start_ms = int(chunk_start_sample * 1000 / self.sample_rate)
                self.vad.start_utterance(start_ms)
                new_confirmed = []
            else:
                new_confirmed = self.vad.feed(
                    torch.from_numpy(audio_float).float(), is_final=False
                )

            for seg in new_confirmed:
                seg_text = self._decode_segment(seg)
                self.prev_text = ""
                if not seg_text.strip():
                    continue
                self.locked_sentences.append({"text": seg_text, "start": int(seg[0]), "end": int(seg[1])})
                if self.spk_tracker:
                    s0 = int(seg[0] * self.sample_rate / 1000)
                    s1 = min(int(seg[1] * self.sample_rate / 1000), self.total_samples)
                    segment_audio = self._slice_audio(s0, s1).copy()
                    self.spk_tracker.assign_streaming(segment_audio, seg[0]/1000, seg[1]/1000, self.locked_sentences[-1])
                logger.info(f"Locked: [{seg[0]}-{seg[1]}ms] \"{seg_text[:40]}\"")

        self._compact_audio_buffer()

    def _slice_audio(self, start_sample, end_sample):
        local_start = max(0, start_sample - self.audio_buffer_start_sample)
        local_end = min(len(self.audio_buffer), end_sample - self.audio_buffer_start_sample)
        if local_end <= local_start:
            return np.array([], dtype=np.float32)
        return self.audio_buffer[local_start:local_end]

    def _compact_audio_buffer(self):
        if self.vad.current_speech_start is not None:
            keep_from = int(self.vad.current_speech_start * self.sample_rate / 1000)
        else:
            keep_from = self.total_samples - self.audio_lookback_samples
        keep_from = min(self.total_samples, max(self.audio_buffer_start_sample, keep_from))
        drop_samples = keep_from - self.audio_buffer_start_sample
        if drop_samples > 0:
            # A copy is required here; a slice would keep the discarded backing
            # array alive and recreate the long-session memory leak.
            self.audio_buffer = self.audio_buffer[drop_samples:].copy()
            self.audio_buffer_start_sample = keep_from

    def _release_audio_buffer(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_buffer_start_sample = self.total_samples

    def should_decode(self):
        threshold = self.first_chunk_samples if not self.first_decode_done else self.chunk_samples
        return (self.total_samples - self.last_decode_samples) >= threshold

    @torch.no_grad()
    def decode(self, is_final=False):
        if is_final and self.endpoint_mode == "client":
            return self.commit()

        if self.endpoint_mode == "server" and self.total_samples < self.chunk_samples:
            if is_final:
                self._release_audio_buffer()
            return self._build_response(is_final)

        if self.endpoint_mode == "client":
            if self.vad.current_speech_start is None:
                return self._build_response(is_final=False)
            utterance_start_sample = int(
                self.vad.current_speech_start * self.sample_rate / 1000
            )
            if self.total_samples - utterance_start_sample < self.first_chunk_samples:
                return self._build_response(is_final=False)

        if is_final:
            if self.vad.current_speech_start is not None:
                end_ms = int(self.total_samples * 1000 / self.sample_rate)
                seg = [self.vad.current_speech_start, end_ms]
                seg_text = self._decode_segment(seg)
                if seg_text.strip():
                    self.locked_sentences.append({"text": seg_text, "start": int(seg[0]), "end": int(seg[1])})
                self.vad.current_speech_start = None

            if self.spk_tracker and self.locked_sentences:
                self.locked_sentences = self.spk_tracker.finalize(self.locked_sentences)
            self._release_audio_buffer()
            return self._build_response(is_final)

        if self.vad.current_speech_start is not None:
            seg_audio, partial_start_ms = self.get_partial_decode_audio()
        else:
            self.last_decode_samples = self.total_samples
            self.last_partial_text = ""
            return self._build_response(is_final)

        if len(seg_audio) < self.chunk_samples // 2:
            return self._build_response(is_final)

        audio_tensor = torch.from_numpy(seg_audio).float()
        try:
            results = self.vllm_engine.generate(
                inputs=[audio_tensor],
                hotwords=self.asr_kwargs.get("hotwords"),
                language=self.asr_kwargs.get("language"),
                max_new_tokens=200,
            )
            text = results[0]["text"] if results else ""
            text = _clean_asr_text(text)
        except Exception as e:
            logger.error(f"ASR error: {e}")
            return self._build_response(is_final)

        text, hallucinated = detect_and_fix_hallucination(text)
        if hallucinated:
            self.prev_text = ""

        self.last_decode_samples = self.total_samples
        self.last_partial_text = text
        self.last_partial_start_ms = partial_start_ms
        if text.strip() and not self.first_decode_done:
            self.first_decode_done = True

        tokenizer = self.vllm_engine._engine.tokenizer
        encoded = tokenizer.encode(text)
        if len(encoded) > 5:
            self.prev_text = tokenizer.decode(encoded[:-5], skip_special_tokens=True)
        else:
            self.prev_text = ""

        return self._build_response(is_final)

    @torch.no_grad()
    def commit(self):
        """Finalize one client-delimited utterance while keeping the session open."""
        if self.endpoint_mode != "client":
            raise RuntimeError("COMMIT requires endpoint_mode='client'")

        if self.vad.current_speech_start is not None:
            end_ms = int(self.total_samples * 1000 / self.sample_rate)
            seg = [self.vad.current_speech_start, end_ms]
            seg_text = self._decode_segment(seg)
            if seg_text.strip():
                sentence = {
                    "text": seg_text,
                    "start": int(seg[0]),
                    "end": int(seg[1]),
                }
                if self.spk_tracker:
                    s0 = int(seg[0] * self.sample_rate / 1000)
                    segment_audio = self._slice_audio(s0, self.total_samples).copy()
                    self.spk_tracker.assign_streaming(
                        segment_audio,
                        seg[0] / 1000,
                        seg[1] / 1000,
                        sentence,
                    )
                self.locked_sentences.append(sentence)
            self.vad.current_speech_start = None

        if self.spk_tracker and self.locked_sentences:
            self.locked_sentences = self.spk_tracker.finalize(self.locked_sentences)

        response = self._build_response(is_final=True)
        self._reset_utterance_state()
        return response

    def _reset_utterance_state(self):
        self._release_audio_buffer()
        self.first_decode_done = False
        self.vad.reset()
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_partial_start_ms = 0
        self.last_decode_samples = self.total_samples
        self.locked_sentences = []
        self.prev_seg_text = ""

    def get_partial_decode_audio(self):
        """Return the bounded audio window used for unstable partial decoding."""
        seg_start_sample = int(self.vad.current_speech_start * self.sample_rate / 1000)
        decode_start_sample = seg_start_sample

        if self.partial_window_samples:
            min_start = self.total_samples - self.partial_window_samples
            if min_start > decode_start_sample:
                decode_start_sample = min_start

        decode_start_sample = max(self.audio_buffer_start_sample, decode_start_sample)
        start_ms = int(decode_start_sample * 1000 / self.sample_rate)
        return self._slice_audio(decode_start_sample, self.total_samples), start_ms

    @torch.no_grad()
    def _decode_segment(self, seg):
        """Decode a completed VAD segment via vLLM."""
        start_sample = int(seg[0] * self.sample_rate / 1000)
        end_sample = min(int(seg[1] * self.sample_rate / 1000), self.total_samples)
        seg_audio = self._slice_audio(start_sample, end_sample)
        if len(seg_audio) < 1600:
            return ""
        audio_tensor = torch.from_numpy(seg_audio).float()
        try:
            results = self.vllm_engine.generate(
                inputs=[audio_tensor],
                hotwords=self.asr_kwargs.get("hotwords"),
                language=self.asr_kwargs.get("language"),
                max_new_tokens=512,
            )
            text = results[0]["text"] if results else ""
            text = _clean_asr_text(text)
            self.prev_seg_text = text
            return text
        except Exception as e:
            logger.error(f"Segment decode error: {e}")
            return ""

    def _build_response(self, is_final):
        duration_ms = int(self.total_samples * 1000 / self.sample_rate)
        sentences = list(self.locked_sentences)
        partial = self.last_partial_text
        if partial:
            partial_start = self.last_partial_start_ms
        elif self.vad.current_speech_start is not None:
            partial_start = self.vad.current_speech_start
        else:
            partial_start = duration_ms

        if is_final:
            return {"sentences": sentences, "partial": "", "partial_start_ms": 0,
                    "duration_ms": duration_ms, "is_final": True}
        return {"sentences": sentences, "partial": partial,
                "partial_start_ms": partial_start,
                "duration_ms": duration_ms, "is_final": False}

    def session_stats(self):
        stats = {
            "duration_ms": int(self.total_samples * 1000 / self.sample_rate),
            "total_samples": self.total_samples,
            "audio_buffer_samples": len(self.audio_buffer),
            "audio_buffer_start_ms": int(
                self.audio_buffer_start_sample * 1000 / self.sample_rate
            ),
            "locked_sentences": len(self.locked_sentences),
            "partial_active": self.vad.current_speech_start is not None,
            "last_decode_ms": int(self.last_decode_samples * 1000 / self.sample_rate),
        }
        if self.spk_tracker:
            stats.update(self.spk_tracker.session_stats())
        return stats

    def reset(self):
        self.audio_buffer = np.array([], dtype=np.float32)
        self.audio_buffer_start_sample = 0
        self.total_samples = 0
        self.first_decode_done = False
        self.vad.reset()
        self.prev_text = ""
        self.last_partial_text = ""
        self.last_partial_start_ms = 0
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

        if getattr(args, "endpoint_mode", "server") == "server":
            logger.info("Loading VAD: fsmn-vad (streaming)")
            _vad_model = AutoModel(
                model="fsmn-vad", device=args.device, disable_update=True
            )
        else:
            _vad_model = None
            logger.info("Server VAD disabled; client COMMIT controls utterance endpoints")

        if getattr(args, "enable_spk", False):
            logger.info(f"Loading SPK: {args.spk_model}")
            _spk_model = AutoModel(model=args.spk_model, device=args.device, disable_update=True)
        else:
            _spk_model = None
            logger.info("SPK disabled; use --enable-spk to include speaker diarization")

        logger.info("All models ready!")
    return _vllm_engine, _asr_kwargs, _vad_model, _spk_model


def create_speaker_tracker(spk_model, args):
    if not getattr(args, "enable_spk", False) or spk_model is None:
        return None
    return HybridSpeakerTracker(spk_model, args.device)


async def run_session_work(args, operation, *operation_args, **operation_kwargs):
    """Run blocking session work off-loop without concurrent shared-model access."""
    lock = getattr(args, "_session_work_lock", None)
    if lock is None:
        lock = asyncio.Lock()
        args._session_work_lock = lock

    async with lock:
        return await asyncio.to_thread(operation, *operation_args, **operation_kwargs)


def log_session_stats(session):
    stats = session.session_stats()
    logger.info(
        "Session stats: duration_ms=%d audio_buffer_samples=%d "
        "audio_buffer_start_ms=%d locked_sentences=%d partial_active=%s "
        "speaker_history_chunks=%s speaker_history_embeddings=%s "
        "speaker_centers=%s",
        stats["duration_ms"],
        stats["audio_buffer_samples"],
        stats["audio_buffer_start_ms"],
        stats["locked_sentences"],
        stats["partial_active"],
        stats.get("speaker_history_chunks", "-"),
        stats.get("speaker_history_embeddings", "-"),
        stats.get("speaker_centers", "-"),
    )
    return stats


async def handle_client(websocket, args):
    vllm_engine, asr_kwargs, vad_model, spk_model = load_models(args)
    endpoint_mode = getattr(args, "endpoint_mode", "server")
    if endpoint_mode == "client":
        vad = ClientEndpointVAD()
    else:
        vad = DynamicStreamingVAD(vad_model)
    spk_tracker = create_speaker_tracker(spk_model, args)
    session = RealtimeASRSession(
        vllm_engine,
        asr_kwargs,
        vad,
        spk_tracker=spk_tracker,
        partial_window_sec=getattr(args, 'partial_window_sec', 15.0),
        endpoint_mode=endpoint_mode,
    )
    logger.info(f"Client connected: {websocket.remote_address}")

    decode_interval = args.decode_interval
    last_decode_time = 0
    stats_interval = getattr(args, "log_session_stats_interval", 0.0)
    last_stats_time = time.time()

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
                elif cmd.upper() == "COMMIT":
                    if endpoint_mode != "client":
                        await websocket.send(
                            json.dumps(
                                {
                                    "event": "error",
                                    "error": "COMMIT requires --endpoint-mode client",
                                }
                            )
                        )
                    elif not session.is_active:
                        await websocket.send(
                            json.dumps(
                                {
                                    "event": "error",
                                    "error": "Session is not active; send START first",
                                }
                            )
                        )
                    else:
                        commit_started = time.perf_counter()
                        result = await run_session_work(args, session.commit)
                        await websocket.send(json.dumps(result))
                        last_decode_time = time.time()
                        elapsed_ms = (time.perf_counter() - commit_started) * 1000
                        logger.info(
                            "Commit final: %d sentences in %.1fms",
                            len(result.get("sentences", [])),
                            elapsed_ms,
                        )
                elif cmd.upper() == "STOP":
                    if endpoint_mode == "client":
                        has_pending_audio = (
                            session.total_samples > session.audio_buffer_start_sample
                        )
                    else:
                        has_pending_audio = session.total_samples > 0
                    if session.is_active and has_pending_audio:
                        if endpoint_mode == "client":
                            result = await run_session_work(args, session.commit)
                        else:
                            result = await run_session_work(
                                args, session.decode, is_final=True
                            )
                        await websocket.send(json.dumps(result))
                        logger.info(
                            "Final: %d sentences", len(result.get("sentences", []))
                        )
                    session.is_active = False
                    await websocket.send(json.dumps({"event": "stopped"}))
            elif isinstance(message, bytes) and session.is_active:
                await run_session_work(args, session.add_audio, message)
                now = time.time()
                if (
                    stats_interval
                    and stats_interval > 0
                    and now - last_stats_time >= stats_interval
                ):
                    log_session_stats(session)
                    last_stats_time = now
                if now - last_decode_time >= decode_interval and session.should_decode():
                    result = await run_session_work(args, session.decode, is_final=False)
                    await websocket.send(json.dumps(result))
                    last_decode_time = now

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)


def _positive_or_none(value):
    return None if value <= 0 else value


def build_websocket_serve_kwargs(args):
    return {
        "max_size": args.ws_max_size,
        "ping_interval": _positive_or_none(args.ws_ping_interval),
        "ping_timeout": _positive_or_none(args.ws_ping_timeout),
        "close_timeout": args.ws_close_timeout,
    }


async def main(args):
    load_models(args)
    logger.info(f"Server on ws://0.0.0.0:{args.port}")
    serve_kwargs = build_websocket_serve_kwargs(args)
    logger.info(
        "WebSocket options: "
        f"max_size={serve_kwargs['max_size']}, "
        f"ping_interval={serve_kwargs['ping_interval']}, "
        f"ping_timeout={serve_kwargs['ping_timeout']}, "
        f"close_timeout={serve_kwargs['close_timeout']}"
    )
    async with websockets.serve(
        lambda ws: handle_client(ws, args), "0.0.0.0", args.port, **serve_kwargs,
    ):
        await asyncio.Future()


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Fun-ASR-Nano Streaming WebSocket Server")
    parser.add_argument("--port", type=int, default=10095)
    parser.add_argument("--model", type=str, default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", type=str, default="ms", choices=["ms", "hf"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--use-context", action="store_true", default=True)
    parser.add_argument("--no-context", dest="use_context", action="store_false")
    parser.add_argument("--decode-interval", type=float, default=0.48)
    parser.add_argument(
        "--endpoint-mode",
        choices=["server", "client"],
        default="server",
        help=(
            "Use server VAD endpoints (default), or accept client COMMIT messages "
            "without loading the VAD model."
        ),
    )
    parser.add_argument("--partial-window-sec", type=float, default=15.0,
                        help="Cap the interim partial re-decode window to the most recent N seconds. "
                             "A long ongoing speech segment is otherwise re-encoded from its start on "
                             "every chunk (O(L^2)), which saturates the GPU under concurrency and times "
                             "out long-segment requests. Lower it (e.g. 8-10) for high-concurrency "
                             "self-hosting; <=0 disables (legacy behaviour). Final transcripts are unaffected.")
    parser.add_argument("--enable-spk", action="store_true", help="Enable streaming speaker diarization.")
    parser.add_argument("--spk-model", type=str, default="iic/speech_eres2netv2_sv_zh-cn_16k-common")
    parser.add_argument("--hotword-file", type=str, default="热词列表")
    parser.add_argument("--language", type=str, default=None, help="Language hint (e.g. 中文, English, 日本語)")
    parser.add_argument(
        "--dtype",
        type=lambda value: "fp32" if value == "float32" else value,
        default="bf16",
        choices=["bf16", "fp16", "fp32", "float32"],
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--ws-ping-interval", type=float, default=20.0,
                        help="WebSocket ping interval in seconds; <=0 disables keepalive pings.")
    parser.add_argument("--ws-ping-timeout", type=float, default=20.0,
                        help="WebSocket ping timeout in seconds; <=0 disables ping timeout.")
    parser.add_argument("--ws-close-timeout", type=float, default=10.0,
                        help="WebSocket close handshake timeout in seconds.")
    parser.add_argument("--ws-max-size", type=int, default=10 * 1024 * 1024,
                        help="Maximum incoming WebSocket message size in bytes.")
    parser.add_argument("--log-session-stats-interval", type=float, default=0.0,
                        help="Log bounded long-session state every N seconds; <=0 disables.")
    return parser


def cli_main():
    args = build_arg_parser().parse_args()
    asyncio.run(main(args))


if __name__ == "__main__":
    cli_main()
