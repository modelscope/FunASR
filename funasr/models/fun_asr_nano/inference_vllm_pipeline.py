#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano vLLM Pipeline: VAD + ASR(vLLM) + Speaker Diarization.

Replicates AutoModel's inference_with_vad pipeline but uses vLLM for
the LLM decoding step, enabling batch processing of all VAD segments
in a single generate() call.

Usage:
    from funasr.models.fun_asr_nano.inference_vllm_pipeline import FunASRNanoVLLMPipeline

    model = FunASRNanoVLLMPipeline(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model="fsmn-vad",
        spk_model="cam++",
        tensor_parallel_size=2,
    )
    results = model.generate("long_meeting.wav", language="中文")
"""

import logging
import os
import re
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _clean_text(text: str) -> str:
    """Remove tags, fillers, and garbage from output."""
    text = re.sub(r"<[^>]*>|</[^>]*>", "", text)
    text = re.sub(r"(>.{2,8}?)\1{3,}", "", text)
    text = re.sub(r"\[breath\]|\[noise\]|/sil|endofbreak|FFFF", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("�", "").lstrip(">")
    return text.strip()


class FunASRNanoVLLMPipeline:
    """VAD + ASR(vLLM) + Speaker pipeline.

    Pipeline:
        1. VAD: segment long audio into speech regions (torch)
        2. ASR: batch ALL segments through vLLM in single generate() call
        3. Speaker: extract embeddings per segment, cluster (torch)
        4. Combine: merge text + timestamps + speaker labels

    Args:
        model: Fun-ASR-Nano model name or path.
        vad_model: VAD model name (e.g. "fsmn-vad"). None to disable.
        vad_kwargs: VAD config (e.g. {"max_single_segment_time": 30000}).
        spk_model: Speaker model name (e.g. "cam++"). None to disable.
        hub: "ms" or "hf".
        device: Device for audio encoder + VAD + speaker.
        dtype: Compute dtype for ASR.
        tensor_parallel_size: GPUs for vLLM.
        gpu_memory_utilization: GPU memory fraction for vLLM.
        max_model_len: Maximum sequence length for vLLM.
    """

    def __init__(
        self,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        vad_model: str = None,
        vad_kwargs: dict = None,
        spk_model: str = None,
        spk_kwargs: dict = None,
        hub: str = "ms",
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        **kwargs,
    ):
        from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

        # ASR engine (vLLM)
        self.asr_engine = FunASRNanoVLLM.from_pretrained(
            model=model, hub=hub, device=device, dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, enforce_eager=enforce_eager,
            **kwargs,
        )

        # VAD model (torch)
        self.vad_model = None
        if vad_model is not None:
            from funasr import AutoModel
            vad_kw = vad_kwargs or {}
            self.vad_model = AutoModel(
                model=vad_model, device=device, disable_update=True, **vad_kw
            )

        # Speaker model (torch)
        self.spk_model = None
        self.cb_model = None
        if spk_model is not None:
            from funasr import AutoModel
            from funasr.models.campplus.cluster_backend import ClusterBackend
            spk_kw = spk_kwargs or {}
            self.spk_model = AutoModel(
                model=spk_model, device=device, disable_update=True, **spk_kw
            )
            cb_kwargs = spk_kw.get("cb_kwargs", {})
            self.cb_model = ClusterBackend(**cb_kwargs).to(device)

        self.device = device
        self.sample_rate = 16000

    def generate(
        self,
        input: Union[str, List[str]],
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
        max_new_tokens: int = 512,
        batch_size_s: int = 300,
        return_spk_res: bool = True,
        **kwargs,
    ) -> List[dict]:
        """Run the full pipeline: VAD → ASR(vLLM) → Speaker.

        Args:
            input: Audio file path(s).
            hotwords: Hotwords for ASR.
            language: Language hint.
            itn: Inverse text normalization.
            max_new_tokens: Max tokens per segment.
            batch_size_s: Max batch duration in seconds (for memory control).
            return_spk_res: Whether to return speaker info.

        Returns:
            List of dicts: [{"key", "text", "timestamp", "sentence_info"}]
        """
        if isinstance(input, str):
            input = [input]

        results_all = []
        for audio_path in input:
            result = self._process_one(
                audio_path, hotwords=hotwords, language=language,
                itn=itn, max_new_tokens=max_new_tokens,
                batch_size_s=batch_size_s, return_spk_res=return_spk_res,
                **kwargs,
            )
            results_all.append(result)
        return results_all

    def _process_one(self, audio_path, **kwargs):
        """Process a single audio file through the full pipeline."""
        from funasr.utils.load_utils import load_audio_text_image_video
        from funasr.utils.vad_utils import slice_padding_audio_samples

        key = os.path.splitext(os.path.basename(audio_path))[0]

        # Load audio
        audio_data = load_audio_text_image_video(audio_path, fs=self.sample_rate)
        if isinstance(audio_data, torch.Tensor):
            audio_np = audio_data.numpy()
        else:
            audio_np = np.array(audio_data)
        speech_length = len(audio_np)

        # Step 1: VAD
        if self.vad_model is not None:
            vad_res = self.vad_model.generate(input=audio_path, cache={}, is_final=True)
            vad_segments = vad_res[0]["value"]  # [[start_ms, end_ms], ...]
        else:
            vad_segments = [[0, int(speech_length / self.sample_rate * 1000)]]

        if not vad_segments:
            return {"key": key, "text": "", "timestamp": []}

        n_segments = len(vad_segments)
        logger.info(f"VAD: {n_segments} segments for {key}")

        # Step 2: Slice audio by VAD segments and encode
        segment_audios = []
        for seg in vad_segments:
            start_sample = int(seg[0] * self.sample_rate / 1000)
            end_sample = int(seg[1] * self.sample_rate / 1000)
            end_sample = min(end_sample, speech_length)
            segment_audios.append(audio_np[start_sample:end_sample])

        # Step 3: Batch ASR via vLLM
        # Encode all segments and build prompts
        from vllm import SamplingParams
        try:
            from vllm.inputs import EmbedsPrompt
        except ImportError:
            from vllm.inputs.data import EmbedsPrompt

        prompts = []
        for seg_audio in segment_audios:
            seg_tensor = torch.from_numpy(seg_audio).float()
            adaptor_out, adaptor_out_lens, _, _ = self.asr_engine._encode_audio(seg_tensor)
            input_embeds = self.asr_engine._build_input_embeds(
                adaptor_out, adaptor_out_lens,
                hotwords=kwargs.get("hotwords"),
                language=kwargs.get("language"),
                itn=kwargs.get("itn", True),
            )
            prompts.append(EmbedsPrompt(prompt_embeds=input_embeds.float()))

        params = SamplingParams(
            max_tokens=kwargs.get("max_new_tokens", 512),
            temperature=0.0,
            repetition_penalty=kwargs.get("repetition_penalty", 1.0),
            skip_special_tokens=True,
        )

        # Single batch generate for ALL segments
        t0 = time.perf_counter()
        outputs = self.asr_engine.vllm_engine.generate(prompts, params, use_tqdm=False)
        t1 = time.perf_counter()
        logger.info(f"vLLM batch ASR: {n_segments} segments in {t1-t0:.3f}s")

        # Decode results
        asr_results = []
        for output in outputs:
            text = output.outputs[0].text
            if not text and output.outputs[0].token_ids:
                text = self.asr_engine.tokenizer.decode(
                    list(output.outputs[0].token_ids), skip_special_tokens=True
                )
            text = _clean_text(text)
            asr_results.append(text)

        # Step 4: Speaker embeddings (if spk_model configured)
        spk_embeddings = None
        if self.spk_model is not None and kwargs.get("return_spk_res", True):
            from funasr.models.campplus.utils import sv_chunk, postprocess, distribute_spk

            all_segments = []
            all_spk_embs = []
            for i, seg_audio in enumerate(segment_audios):
                vad_seg = [
                    [vad_segments[i][0] / 1000.0, vad_segments[i][1] / 1000.0, seg_audio]
                ]
                chunks = sv_chunk(vad_seg)
                all_segments.extend(chunks)
                speech_chunks = [c[2] for c in chunks]
                spk_res = self.spk_model.generate(input=speech_chunks, cache={}, is_final=True)
                embs = torch.cat([r["spk_embedding"] for r in spk_res], dim=0)
                all_spk_embs.append(embs)

            if all_spk_embs:
                spk_embeddings = torch.cat(all_spk_embs, dim=0)

        # Step 5: Combine results
        # Merge text with timestamps
        full_text = ""
        all_timestamps = []
        for i, (seg, text) in enumerate(zip(vad_segments, asr_results)):
            if text:
                if full_text:
                    full_text += " "
                full_text += text
                # Simple word-level timestamp from VAD boundaries
                all_timestamps.append([int(seg[0]), int(seg[1])])

        result = {"key": key, "text": full_text}

        # Add timestamps if available from CTC
        if self.asr_engine.ctc_decoder is not None:
            try:
                detailed_timestamps = self._compute_all_timestamps(
                    segment_audios, vad_segments, asr_results
                )
                if detailed_timestamps:
                    result["timestamp"] = detailed_timestamps
            except Exception as e:
                logger.debug(f"Timestamp computation failed: {e}")
                result["timestamp"] = all_timestamps
        else:
            result["timestamp"] = all_timestamps

        # Add speaker info
        if spk_embeddings is not None and self.cb_model is not None:
            from funasr.models.campplus.utils import postprocess, distribute_spk

            all_segments_sorted = sorted(all_segments, key=lambda x: x[0])
            labels = self.cb_model(
                spk_embeddings.cpu(),
                oracle_num=kwargs.get("preset_spk_num", None),
            )
            sv_output = postprocess(all_segments_sorted, None, labels, spk_embeddings.cpu())

            # Build sentence_info
            sentence_list = []
            for i, (seg, text) in enumerate(zip(vad_segments, asr_results)):
                if text:
                    sentence_list.append({
                        "start": seg[0],
                        "end": seg[1],
                        "text": text,
                        "timestamp": [[int(seg[0]), int(seg[1])]],
                    })
            distribute_spk(sentence_list, sv_output)
            result["sentence_info"] = sentence_list

        return result

    def _compute_all_timestamps(self, segment_audios, vad_segments, asr_results):
        """Compute CTC timestamps for all segments with VAD offsets."""
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        all_timestamps = []
        for seg_audio, vad_seg, text in zip(segment_audios, vad_segments, asr_results):
            if not text:
                continue
            try:
                seg_tensor = torch.from_numpy(seg_audio).float()
                from funasr.utils.load_utils import extract_fbank
                speech, speech_lengths = extract_fbank(
                    seg_tensor, data_type="sound",
                    frontend=self.asr_engine.frontend, is_final=True
                )
                speech = speech.to(self.device, dtype=torch.float32)
                speech_lengths = speech_lengths.to(self.device)

                with torch.no_grad():
                    enc_out, enc_lens = self.asr_engine.audio_encoder(speech, speech_lengths)
                    dec_out, dec_lens = self.asr_engine.ctc_decoder(enc_out, enc_lens)
                    ctc_logits = self.asr_engine.ctc.log_softmax(dec_out)

                x = ctc_logits[0, :enc_lens[0].item(), :]
                target_ids = torch.tensor(
                    self.asr_engine.ctc_tokenizer.encode(text), dtype=torch.int64
                )
                if len(target_ids) == 0:
                    continue

                timestamps = forced_align(x, target_ids, self.asr_engine.blank_id)
                vad_offset_ms = int(vad_seg[0])
                for ts in timestamps:
                    ts["token"] = self.asr_engine.ctc_tokenizer.decode([ts["token"]])
                    ts["start_time"] = ts["start_time"] * 6 * 10 / 1000 + vad_offset_ms / 1000
                    ts["end_time"] = ts["end_time"] * 6 * 10 / 1000 + vad_offset_ms / 1000
                all_timestamps.extend(timestamps)
            except Exception as e:
                logger.debug(f"Timestamp failed for segment: {e}")
                all_timestamps.append({
                    "start_time": vad_seg[0] / 1000,
                    "end_time": vad_seg[1] / 1000,
                    "token": text,
                })
        return all_timestamps

    @classmethod
    def from_pretrained(cls, model="FunAudioLLM/Fun-ASR-Nano-2512", **kwargs):
        """Convenience constructor."""
        return cls(model=model, **kwargs)
