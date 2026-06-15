#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano Streaming vLLM Inference Engine.

Design:
    - Audio split into 720ms chunks (cumulative re-encoding)
    - ALL chunks batched into single vLLM generate call for correctness
    - Fixed/Unfixed: last 8 chars are unfixed (may change on next chunk)
    - Output stabilizes as more audio accumulates (~3s+)

Note: vLLM processes all chunks in one batch for throughput.
For real-time streaming, use the torch-based inference in demo2.py.
"""

import logging
import os
import re
from typing import Generator, List, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

_CJK_RE = re.compile(r"[一-鿿]")


def _clean_text(text: str) -> str:
    """Remove tags, repetitive garbage, filler tokens, and invalid chars."""
    text = re.sub(r'<[^>]*>|</[^>]*>', '', text)
    text = re.sub(r'(>.{2,8}?){3,}', '', text)
    text = re.sub(r'\[breath\]|\[noise\]|/sil|endofbreak|FFFF', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('�', '').lstrip('>')
    return text.strip()


def _is_meaningful(text: str) -> bool:
    """Check if text has real ASR content."""
    return len(_CJK_RE.findall(text)) >= 2


class FunASRNanoStreamingVLLM:
    """Streaming ASR with vLLM backend (batch-all-chunks approach).

    Processes audio in 720ms chunks. All chunks are encoded and batched
    into a single vLLM generate() call for correct and efficient inference.
    Results are returned per-chunk with fixed/unfixed regions.

    Args:
        model_dir: Path to Fun-ASR-Nano model directory.
        device: Device for audio encoder/adaptor.
        dtype: Compute dtype ("bf16", "fp16", "fp32").
        tensor_parallel_size: GPUs for vLLM tensor parallelism.
        gpu_memory_utilization: GPU memory fraction for KV cache.
        max_model_len: Maximum sequence length.
        chunk_ms: Chunk duration in ms (default 720).
        rollback_chars: Characters to rollback per chunk (default 8).
    """

    def __init__(self, model_dir, device="cuda:0", dtype="bf16",
                 tensor_parallel_size=1, gpu_memory_utilization=0.8,
                 max_model_len=2048, enforce_eager=False,
                 chunk_ms=720, rollback_chars=8, **kwargs):
        from vllm import LLM
        from funasr.models.fun_asr_nano.inference_vllm import prepare_vllm_model_dir

        self.device = device
        self.dtype = dtype
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self.model_dir = model_dir
        self.chunk_ms = chunk_ms
        self.rollback_chars = rollback_chars

        vllm_model_dir = prepare_vllm_model_dir(model_dir)
        self._load_audio_components(model_dir)

        vllm_kwargs = kwargs.get("vllm_kwargs", {})
        self.vllm_engine = LLM(
            enable_prompt_embeds=True, model=vllm_model_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, enforce_eager=enforce_eager,
            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "auto"}.get(dtype, dtype),
            trust_remote_code=True, **vllm_kwargs,
        )
        self.tokenizer = self.vllm_engine.get_tokenizer()
        self._load_embedding_layer(model_dir)
        self.sample_rate = self.frontend.fs
        self.chunk_samples = int(self.sample_rate * self.chunk_ms / 1000)

    def _load_audio_components(self, model_dir):
        from omegaconf import OmegaConf
        from funasr.register import tables
        config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
        self._config = OmegaConf.to_container(config, resolve=True)

        frontend_class = tables.frontend_classes.get(config["frontend"])
        frontend_conf = OmegaConf.to_container(config.get("frontend_conf", {}), resolve=True)
        self.frontend = frontend_class(**frontend_conf)
        self.frontend.eval()

        encoder_conf = OmegaConf.to_container(config.get("audio_encoder_conf", {}), resolve=True)
        if encoder_conf.get("hub") == "ms":
            from funasr import AutoModel as FAM
            enc_m = FAM(model=config["audio_encoder"], model_revision="master", disable_update=True)
            self.audio_encoder_output_size = getattr(enc_m.model, "encoder_output_size", -1)
            self.audio_encoder = enc_m.model.model.encoder if hasattr(enc_m.model, "model") else enc_m.model.encoder
        else:
            encoder_class = tables.encoder_classes.get(config["audio_encoder"])
            self.audio_encoder = encoder_class(input_size=self.frontend.output_size(), **encoder_conf)
            self.audio_encoder_output_size = self.audio_encoder.output_size()
        self.audio_encoder.eval()
        for p in self.audio_encoder.parameters(): p.requires_grad = False

        adaptor_conf = OmegaConf.to_container(config.get("audio_adaptor_conf", {}), resolve=True)
        adaptor_class = tables.adaptor_classes.get(config["audio_adaptor"])
        if self.audio_encoder_output_size > 0:
            adaptor_conf["encoder_dim"] = self.audio_encoder_output_size
        self.audio_adaptor = adaptor_class(**adaptor_conf)
        self.audio_adaptor.eval()
        for p in self.audio_adaptor.parameters(): p.requires_grad = False

        model_pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_pt):
            ckpt = torch.load(model_pt, map_location="cpu")
            sd = ckpt.get("state_dict", ckpt)
            enc_s = {k[len("audio_encoder."):]: v for k, v in sd.items() if k.startswith("audio_encoder.")}
            if enc_s: self.audio_encoder.load_state_dict(enc_s, strict=False)
            adp_s = {k[len("audio_adaptor."):]: v for k, v in sd.items() if k.startswith("audio_adaptor.")}
            if adp_s: self.audio_adaptor.load_state_dict(adp_s, strict=False)

        self.audio_encoder = self.audio_encoder.to(self.device, dtype=torch.float32)
        self.audio_adaptor = self.audio_adaptor.to(self.device, dtype=self.torch_dtype)

    def _load_embedding_layer(self, model_dir):
        ckpt = torch.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        for key in sd:
            if "embed_tokens.weight" in key and key.startswith("llm."):
                self.embed_tokens = nn.Embedding.from_pretrained(sd[key], freeze=True)
                self.embed_tokens = self.embed_tokens.to(self.device, dtype=self.torch_dtype)
                return
        raise RuntimeError("Could not find LLM embedding weights")

    @torch.no_grad()
    def _encode_audio(self, audio_samples):
        from funasr.utils.load_utils import extract_fbank
        speech, speech_lengths = extract_fbank(
            audio_samples, data_type="sound", frontend=self.frontend, is_final=True)
        speech = speech.to(self.device, dtype=torch.float32)
        speech_lengths = speech_lengths.to(self.device)
        enc_out, enc_lens = self.audio_encoder(speech, speech_lengths)
        adp_out, adp_lens = self.audio_adaptor(enc_out.to(dtype=self.torch_dtype), enc_lens)
        return adp_out, adp_lens

    def _build_prompt_text(self, hotwords=None, language=None, itn=True):
        hotwords = hotwords or []
        if hotwords:
            prompt = "请结合上下文信息，更加准确地完成语音转写任务。如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            prompt += f"热词列表：[{', '.join(hotwords)}]\n"
        else:
            prompt = ""
        prompt += f"语音转写成{language}" if language else "语音转写"
        if not itn: prompt += "，不进行文本规整"
        return prompt + "："

    @torch.no_grad()
    def _build_embeds(self, audio_embeds, audio_embed_lens, prev_text="", hotwords=None, language=None, itn=True):
        """Build input embeddings. prev_text is appended as assistant prefix for continuation."""
        prompt = self._build_prompt_text(hotwords, language, itn)
        prefix_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt}<|startofspeech|>"
        suffix_text = "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if prev_text:
            suffix_text += prev_text

        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        prefix_emb = self.embed_tokens(torch.tensor(prefix_ids, dtype=torch.long, device=self.device))
        suffix_emb = self.embed_tokens(torch.tensor(suffix_ids, dtype=torch.long, device=self.device))
        audio_emb = audio_embeds[0, :audio_embed_lens[0].item(), :]
        return torch.cat([prefix_emb, audio_emb, suffix_emb], dim=0)

    def streaming_generate(self, audio_input, chunk_ms=None, rollback_chars=None,
                           hotwords=None, language=None, itn=True,
                           max_new_tokens=200, temperature=0.0, **kwargs):
        """Streaming ASR: process all chunks and yield results per chunk.

        All chunks are batched into a single vLLM generate() call for
        correct results. Yields incrementally improving transcriptions.

        Args:
            audio_input: File path, numpy array, or tensor (16kHz).
            chunk_ms: Chunk size in ms (default 720).
            rollback_chars: Chars to rollback (default 8).
            hotwords: Hotword list.
            language: Language hint (e.g. "中文").
            itn: Inverse text normalization.
            max_new_tokens: Max tokens per chunk generation.
            temperature: Sampling temperature (0 = greedy).

        Yields:
            {"text": full_text, "fixed_text": confirmed_text,
             "is_final": bool, "chunk_idx": int, "audio_duration_ms": float}
        """
        from vllm import SamplingParams
        try:
            from vllm.inputs import EmbedsPrompt
        except ImportError:
            from vllm.inputs.data import EmbedsPrompt
        from funasr.utils.load_utils import load_audio_text_image_video

        chunk_ms = chunk_ms or self.chunk_ms
        rollback_chars = rollback_chars or self.rollback_chars

        if isinstance(audio_input, str):
            audio_data = load_audio_text_image_video(audio_input, fs=self.sample_rate)
        elif isinstance(audio_input, np.ndarray):
            audio_data = torch.from_numpy(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            audio_data = audio_input.float()
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_input)}")
        if audio_data.dim() > 1:
            audio_data = audio_data.squeeze()

        total_samples = audio_data.shape[0]
        chunk_samples = int(self.sample_rate * chunk_ms / 1000)
        num_chunks = (total_samples + chunk_samples - 1) // chunk_samples

        from funasr.models.fun_asr_nano.vllm_utils import resolve_repetition_penalty

        # Prompt-embeds mode has no token IDs to penalize; see #2948.
        params = SamplingParams(
            max_tokens=max_new_tokens, temperature=temperature,
            repetition_penalty=resolve_repetition_penalty(
                kwargs.get("repetition_penalty", 1.0)
            ),
            skip_special_tokens=True)

        # Two-stage approach for long audio:
        # Stage 1: batch first N chunks fresh (no prev_text) to find stable output
        # Stage 2: batch remaining chunks WITH prev_text from stable output
        stage1_count = min(10, num_chunks)  # ~7.2s should be enough to stabilize

        # Stage 1: encode and batch first chunks
        prompts_s1 = []
        chunk_infos_s1 = []
        for i in range(stage1_count):
            end_sample = min((i + 1) * chunk_samples, total_samples)
            adaptor_out, adaptor_out_lens = self._encode_audio(audio_data[:end_sample])
            embeds = self._build_embeds(adaptor_out, adaptor_out_lens, prev_text="",
                                        hotwords=hotwords, language=language, itn=itn)
            prompts_s1.append(EmbedsPrompt(prompt_embeds=embeds.float()))
            chunk_infos_s1.append({
                "chunk_idx": i + 1,
                "is_final": end_sample >= total_samples,
                "audio_duration_ms": end_sample * 1000 / self.sample_rate,
            })

        outputs_s1 = self.vllm_engine.generate(prompts_s1, params, use_tqdm=False)

        # Find best stable output from stage 1
        best_text = ""
        results_s1 = []
        for output in outputs_s1:
            text = output.outputs[0].text
            if not text and output.outputs[0].token_ids:
                text = self.tokenizer.decode(list(output.outputs[0].token_ids), skip_special_tokens=True)
            text = _clean_text(text)
            results_s1.append(text)
            if _is_meaningful(text) and len(text) > len(best_text):
                best_text = text

        # Yield stage 1 results
        for i, (text, info) in enumerate(zip(results_s1, chunk_infos_s1)):
            if info["is_final"]:
                fixed_text = text
            elif _is_meaningful(text) and len(text) > rollback_chars:
                fixed_text = text[:-rollback_chars]
            else:
                fixed_text = ""
            yield {"text": text, "fixed_text": fixed_text, **info}

        # Stage 2: if more chunks remain, use prev_text from stable output
        if stage1_count < num_chunks:
            prev_text = best_text[:-rollback_chars] if len(best_text) > rollback_chars else best_text

            prompts_s2 = []
            chunk_infos_s2 = []
            for i in range(stage1_count, num_chunks):
                end_sample = min((i + 1) * chunk_samples, total_samples)
                adaptor_out, adaptor_out_lens = self._encode_audio(audio_data[:end_sample])
                embeds = self._build_embeds(adaptor_out, adaptor_out_lens, prev_text=prev_text,
                                            hotwords=hotwords, language=language, itn=itn)
                prompts_s2.append(EmbedsPrompt(prompt_embeds=embeds.float()))
                chunk_infos_s2.append({
                    "chunk_idx": i + 1,
                    "is_final": end_sample >= total_samples,
                    "audio_duration_ms": end_sample * 1000 / self.sample_rate,
                })

            outputs_s2 = self.vllm_engine.generate(prompts_s2, params, use_tqdm=False)

            for output, info in zip(outputs_s2, chunk_infos_s2):
                text = output.outputs[0].text
                if not text and output.outputs[0].token_ids:
                    text = self.tokenizer.decode(list(output.outputs[0].token_ids), skip_special_tokens=True)
                text = _clean_text(text)
                full_text = prev_text + text

                if info["is_final"]:
                    fixed_text = full_text
                elif _is_meaningful(full_text) and len(full_text) > rollback_chars:
                    fixed_text = full_text[:-rollback_chars]
                else:
                    fixed_text = prev_text

                yield {"text": full_text, "fixed_text": fixed_text, **info}

    def generate(self, audio_input, **kwargs):
        """Run streaming and return all chunk results."""
        return list(self.streaming_generate(audio_input, **kwargs))

    @classmethod
    def from_pretrained(cls, model="FunAudioLLM/Fun-ASR-Nano-2512", hub="ms",
                        device="cuda:0", dtype="bf16", tensor_parallel_size=1,
                        gpu_memory_utilization=0.8, max_model_len=2048,
                        chunk_ms=720, rollback_chars=8, **kwargs):
        """Load from hub or local path."""
        if os.path.isdir(model):
            model_dir = model
        else:
            if hub in ("ms", "modelscope"):
                from modelscope.hub.snapshot_download import snapshot_download
                model_dir = snapshot_download(model, revision=kwargs.pop("revision", "master"))
            else:
                from huggingface_hub import snapshot_download
                model_dir = snapshot_download(model)
        return cls(model_dir=model_dir, device=device, dtype=dtype,
                   tensor_parallel_size=tensor_parallel_size,
                   gpu_memory_utilization=gpu_memory_utilization,
                   max_model_len=max_model_len, chunk_ms=chunk_ms,
                   rollback_chars=rollback_chars, **kwargs)
