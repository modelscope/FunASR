#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Fun-ASR-Nano vLLM inference engine.

Uses vLLM for high-throughput LLM decoding while keeping the audio encoder
and adaptor in PyTorch. Supports batch inference and tensor-parallel for
multi-GPU acceleration.

Usage:
    from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

    engine = FunASRNanoVLLM.from_pretrained(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        tensor_parallel_size=2,
    )
    results = engine.generate(["audio1.wav", "audio2.wav"])
"""

import glob
import json
import logging
import os
import re
import shutil
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def prepare_vllm_model_dir(model_dir: str, output_dir: str = None) -> str:
    """Extract LLM weights from Fun-ASR-Nano model.pt and save in HuggingFace format.

    Fun-ASR-Nano stores all weights (audio encoder + adaptor + LLM) in a single
    model.pt file. vLLM needs the LLM weights in standard HuggingFace format.
    This function extracts LLM weights and saves them alongside the config/tokenizer
    files from the Qwen3-0.6B subdirectory.

    Args:
        model_dir: Path to the Fun-ASR-Nano model directory.
        output_dir: Where to save the extracted LLM. Defaults to model_dir/Qwen3-0.6B-vllm.

    Returns:
        Path to the directory containing the vLLM-ready LLM model.
    """
    if output_dir is None:
        output_dir = os.path.join(model_dir, "Qwen3-0.6B-vllm")

    # Check if already prepared
    safetensors_files = glob.glob(os.path.join(output_dir, "*.safetensors"))
    bin_files = glob.glob(os.path.join(output_dir, "model*.bin"))
    if safetensors_files or bin_files:
        logger.info(f"vLLM model already prepared at {output_dir}")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Copy config and tokenizer from Qwen3-0.6B
    qwen_dir = os.path.join(model_dir, "Qwen3-0.6B")
    if not os.path.isdir(qwen_dir):
        raise FileNotFoundError(f"Qwen3-0.6B config directory not found at {qwen_dir}")

    for fname in os.listdir(qwen_dir):
        src = os.path.join(qwen_dir, fname)
        dst = os.path.join(output_dir, fname)
        if os.path.isfile(src) and not os.path.exists(dst):
            shutil.copy2(src, dst)

    # Load model.pt and extract LLM weights
    model_pt = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(
            f"model.pt not found at {model_pt}. Make sure the model is fully downloaded."
        )

    logger.info(f"Loading model.pt from {model_pt}...")
    checkpoint = torch.load(model_pt, map_location="cpu")
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Extract LLM weights (prefixed with "llm.")
    llm_state = {}
    for key, value in state_dict.items():
        if key.startswith("llm."):
            new_key = key[len("llm."):]
            llm_state[new_key] = value

    if not llm_state:
        raise RuntimeError("No LLM weights found in model.pt (expected prefix 'llm.')")

    logger.info(f"Extracted {len(llm_state)} LLM weight tensors")

    # Save in safetensors format (preferred by vLLM)
    try:
        from safetensors.torch import save_file

        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(llm_state, save_path)
        logger.info(f"Saved LLM weights to {save_path}")

        # Create model index
        index = {
            "metadata": {"total_size": sum(v.numel() * v.element_size() for v in llm_state.values())},
            "weight_map": {k: "model.safetensors" for k in llm_state.keys()},
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    except ImportError:
        save_path = os.path.join(output_dir, "model.bin")
        torch.save(llm_state, save_path)
        logger.info(f"Saved LLM weights to {save_path} (install safetensors for faster loading)")

    return output_dir


class FunASRNanoVLLM:
    """Fun-ASR-Nano with vLLM backend for high-throughput inference.

    Architecture:
        Audio -> WavFrontend -> SenseVoiceEncoder -> AudioAdaptor -> audio embeddings
        Text tokens -> LLM embedding layer -> text embeddings
        Combined embeddings -> vLLM (Qwen3-0.6B) -> generated text

    The audio encoder and adaptor run in PyTorch on a single GPU,
    while vLLM handles the LLM inference with optional tensor parallelism.

    Args:
        model_dir: Path to the Fun-ASR-Nano model directory.
        device: Device for audio encoder/adaptor (e.g. "cuda:0").
        dtype: Dtype for audio processing ("bf16", "fp16", "fp32").
        tensor_parallel_size: Number of GPUs for vLLM tensor parallelism.
        gpu_memory_utilization: Fraction of GPU memory for vLLM KV cache.
        max_model_len: Maximum sequence length for vLLM.
        enforce_eager: Disable CUDA graph for debugging.

    Example:
        >>> engine = FunASRNanoVLLM(
        ...     model_dir="/path/to/Fun-ASR-Nano-2512",
        ...     tensor_parallel_size=2,
        ... )
        >>> results = engine.generate(["audio1.wav", "audio2.wav"])
        >>> for r in results:
        ...     print(r["text"])
    """

    def __init__(
        self,
        model_dir: str,
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 2048,
        enforce_eager: bool = False,
        **kwargs,
    ):
        from vllm import LLM, SamplingParams
        from vllm.inputs.data import EmbedsPrompt

        self.device = device
        self.dtype = dtype
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self.model_dir = model_dir

        # Step 1: Prepare LLM weights for vLLM (extract from model.pt if needed)
        vllm_model_dir = prepare_vllm_model_dir(model_dir)

        # Step 2: Load audio components (encoder + adaptor + frontend)
        self._load_audio_components(model_dir, **kwargs)

        # Step 3: Initialize vLLM engine
        logger.info(f"Initializing vLLM with model: {vllm_model_dir}")
        logger.info(f"  tensor_parallel_size={tensor_parallel_size}")
        logger.info(f"  gpu_memory_utilization={gpu_memory_utilization}")

        vllm_kwargs = kwargs.get("vllm_kwargs", {})
        self.vllm_engine = LLM(
            enable_prompt_embeds=True,
            model=vllm_model_dir,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "auto"}.get(dtype, dtype),
            trust_remote_code=True,
            **vllm_kwargs,
        )

        # Step 4: Get tokenizer and LLM embedding layer
        self.tokenizer = self.vllm_engine.get_tokenizer()
        self._load_embedding_layer(model_dir)

    def _load_audio_components(self, model_dir: str, **kwargs):
        """Load audio encoder, adaptor, frontend, and CTC from checkpoint."""
        from omegaconf import OmegaConf
        from funasr.register import tables

        config_path = os.path.join(model_dir, "config.yaml")
        config = OmegaConf.load(config_path)
        self._config = OmegaConf.to_container(config, resolve=True)

        # --- Frontend ---
        frontend_class = tables.frontend_classes.get(config["frontend"])
        frontend_conf = OmegaConf.to_container(config.get("frontend_conf", {}), resolve=True)
        cmvn_file = frontend_conf.get("cmvn_file")
        if cmvn_file and not os.path.isabs(cmvn_file):
            frontend_conf["cmvn_file"] = os.path.join(model_dir, cmvn_file)
        self.frontend = frontend_class(**frontend_conf)
        self.frontend.eval()

        # --- Audio Encoder ---
        encoder_conf = OmegaConf.to_container(config.get("audio_encoder_conf", {}), resolve=True)
        hub = encoder_conf.get("hub", None)
        if hub == "ms":
            from funasr import AutoModel as FunAutoModel

            enc_model = FunAutoModel(
                model=config["audio_encoder"], model_revision="master", disable_update=True
            )
            self.audio_encoder_output_size = (
                enc_model.model.encoder_output_size
                if hasattr(enc_model.model, "encoder_output_size")
                else -1
            )
            self.audio_encoder = (
                enc_model.model.model.encoder
                if hasattr(enc_model.model, "model")
                else enc_model.model.encoder
            )
        else:
            encoder_class = tables.encoder_classes.get(config["audio_encoder"])
            input_size = self.frontend.output_size()
            self.audio_encoder = encoder_class(input_size=input_size, **encoder_conf)
            self.audio_encoder_output_size = self.audio_encoder.output_size()

        self.audio_encoder.eval()
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

        # --- Audio Adaptor ---
        adaptor_conf = OmegaConf.to_container(config.get("audio_adaptor_conf", {}), resolve=True)
        adaptor_class = tables.adaptor_classes.get(config["audio_adaptor"])
        if self.audio_encoder_output_size > 0:
            adaptor_conf["encoder_dim"] = self.audio_encoder_output_size
        self.audio_adaptor = adaptor_class(**adaptor_conf)
        self.audio_adaptor.eval()
        for p in self.audio_adaptor.parameters():
            p.requires_grad = False
        self.use_low_frame_rate = adaptor_conf.get("use_low_frame_rate", False)

        # --- CTC Decoder (optional, for timestamps) ---
        self.ctc_decoder = None
        self.ctc = None
        self.ctc_tokenizer = None
        self.blank_id = None

        ctc_decoder_name = self._config.get("ctc_decoder", None)
        if ctc_decoder_name:
            ctc_decoder_class = tables.adaptor_classes.get(ctc_decoder_name)
            ctc_decoder_conf = self._config.get("ctc_decoder_conf", {})
            if self.audio_encoder_output_size > 0:
                ctc_decoder_conf["encoder_dim"] = self.audio_encoder_output_size
            self.ctc_decoder = ctc_decoder_class(**ctc_decoder_conf)
            self.ctc_decoder.eval()
            for p in self.ctc_decoder.parameters():
                p.requires_grad = False

            from funasr.models.fun_asr_nano.ctc import CTC

            ctc_conf = self._config.get("ctc_conf", {})
            ctc_vocab_size = self._config.get("ctc_vocab_size", 60515)
            self.blank_id = ctc_conf.get("blank_id", ctc_vocab_size - 1)
            self.ctc = CTC(
                odim=ctc_vocab_size,
                encoder_output_size=self.audio_encoder_output_size,
                blank_id=self.blank_id,
                **ctc_conf,
            )

            # CTC tokenizer
            ds_conf = self._config.get("dataset_conf", {})
            ctc_tokenizer_name = ds_conf.get("ctc_tokenizer", None)
            ctc_tokenizer_conf = ds_conf.get("ctc_tokenizer_conf", {})
            if ctc_tokenizer_name:
                ctc_tokenizer_class = tables.tokenizer_classes.get(ctc_tokenizer_name)
                vocab_path = ctc_tokenizer_conf.get("vocab_path")
                if vocab_path is None or not os.path.isabs(vocab_path):
                    multilingual_path = os.path.join(model_dir, "multilingual.tiktoken")
                    if os.path.exists(multilingual_path):
                        ctc_tokenizer_conf["vocab_path"] = multilingual_path
                    elif vocab_path and not os.path.isabs(vocab_path):
                        ctc_tokenizer_conf["vocab_path"] = os.path.join(model_dir, vocab_path)
                self.ctc_tokenizer = ctc_tokenizer_class(**ctc_tokenizer_conf)

        # --- Load weights from model.pt ---
        model_pt = os.path.join(model_dir, "model.pt")
        if os.path.exists(model_pt):
            logger.info(f"Loading audio component weights from {model_pt}")
            checkpoint = torch.load(model_pt, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)

            # Audio encoder
            enc_state = {
                k[len("audio_encoder."):]: v
                for k, v in state_dict.items()
                if k.startswith("audio_encoder.")
            }
            if enc_state:
                self.audio_encoder.load_state_dict(enc_state, strict=False)
                logger.info(f"  Loaded audio_encoder: {len(enc_state)} params")

            # Audio adaptor
            adp_state = {
                k[len("audio_adaptor."):]: v
                for k, v in state_dict.items()
                if k.startswith("audio_adaptor.")
            }
            if adp_state:
                self.audio_adaptor.load_state_dict(adp_state, strict=False)
                logger.info(f"  Loaded audio_adaptor: {len(adp_state)} params")

            # CTC decoder
            if self.ctc_decoder is not None:
                ctc_dec_state = {
                    k[len("ctc_decoder."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("ctc_decoder.")
                }
                if ctc_dec_state:
                    self.ctc_decoder.load_state_dict(ctc_dec_state, strict=False)
                ctc_state = {
                    k[len("ctc."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("ctc.") and not k.startswith("ctc_decoder.")
                }
                if ctc_state:
                    self.ctc.load_state_dict(ctc_state, strict=False)

        # Move to device
        self.audio_encoder = self.audio_encoder.to(self.device, dtype=torch.float32)
        self.audio_adaptor = self.audio_adaptor.to(self.device, dtype=self.torch_dtype)
        if self.ctc_decoder is not None:
            self.ctc_decoder = self.ctc_decoder.to(self.device, dtype=torch.float32)
            self.ctc = self.ctc.to(self.device, dtype=torch.float32)

    def _load_embedding_layer(self, model_dir: str):
        """Load the LLM embedding layer for text token embedding computation."""
        model_pt = os.path.join(model_dir, "model.pt")
        checkpoint = torch.load(model_pt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)

        # Look for embedding weights
        embed_key = None
        for key in state_dict.keys():
            if "embed_tokens.weight" in key and key.startswith("llm."):
                embed_key = key
                break

        if embed_key is None:
            raise RuntimeError("Could not find LLM embedding weights in model.pt")

        embed_weight = state_dict[embed_key]
        self.embed_tokens = nn.Embedding.from_pretrained(embed_weight, freeze=True)
        self.embed_tokens = self.embed_tokens.to(self.device, dtype=self.torch_dtype)
        logger.info(f"Loaded embedding layer: {embed_weight.shape}")

    @torch.no_grad()
    def _encode_audio(self, audio_input: Union[str, torch.Tensor, np.ndarray]):
        """Encode audio through frontend -> encoder -> adaptor.

        Returns:
            adaptor_out: (1, T', D_llm) audio embeddings for LLM input
            adaptor_out_lens: (1,) lengths
            encoder_out: (1, T, D_enc) encoder output for CTC
            encoder_out_lens: (1,) encoder output lengths
        """
        from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank

        if isinstance(audio_input, str):
            data_src = load_audio_text_image_video(audio_input, fs=self.frontend.fs)
        elif isinstance(audio_input, np.ndarray):
            data_src = torch.from_numpy(audio_input).float()
        elif isinstance(audio_input, torch.Tensor):
            data_src = audio_input.float()
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        speech, speech_lengths = extract_fbank(
            data_src, data_type="sound", frontend=self.frontend, is_final=True
        )
        speech = speech.to(self.device, dtype=torch.float32)
        speech_lengths = speech_lengths.to(self.device)

        encoder_out, encoder_out_lens = self.audio_encoder(speech, speech_lengths)
        encoder_out_for_adaptor = encoder_out.to(dtype=self.torch_dtype)
        adaptor_out, adaptor_out_lens = self.audio_adaptor(encoder_out_for_adaptor, encoder_out_lens)

        # Apply low frame rate: compute effective token count from fbank length
        # Matches PyTorch model.py data_load_speech formula exactly
        if self.use_low_frame_rate:
            for i in range(adaptor_out.shape[0]):
                fbank_len = speech_lengths[i].item()
                olens = 1 + (fbank_len - 3 + 2 * 1) // 2
                olens = 1 + (olens - 3 + 2 * 1) // 2
                fake_token_len = (olens - 1) // 2 + 1
                adaptor_out_lens[i] = fake_token_len

        return adaptor_out, adaptor_out_lens, encoder_out, encoder_out_lens

    def _build_prompt_text(
        self,
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
    ) -> str:
        """Build the ASR prompt string."""
        hotwords = hotwords or []
        if len(hotwords) > 0:
            hotwords_str = ", ".join(hotwords)
            prompt = (
                "请结合上下文信息，更加准确地完成语音转写任务。"
                "如果没有相关信息，我们会留空。\n\n\n**上下文信息：**\n\n\n"
            )
            prompt += f"热词列表：[{hotwords_str}]\n"
        else:
            prompt = ""
        if language is None:
            prompt += "语音转写"
        else:
            prompt += f"语音转写成{language}"
        if not itn:
            prompt += "，不进行文本规整"
        return prompt + "："

    @torch.no_grad()
    def _build_input_embeds(
        self,
        audio_embeds: torch.Tensor,
        audio_embed_lens: torch.Tensor,
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
        system_prompt: str = "You are a helpful assistant.",
    ) -> torch.Tensor:
        """Build the full input embedding sequence with audio inserted.

        Returns:
            Tensor of shape (seq_len, D_llm)
        """
        prompt = self._build_prompt_text(hotwords, language, itn)

        # ChatML format with speech markers and thinking prefix
        prefix_text = (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|startofspeech|>"
        )
        suffix_text = "<|endofspeech|><|im_end|>\n<|im_start|>assistant\n"

        # Tokenize
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)

        # Embed text tokens
        prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=self.device)
        suffix_tensor = torch.tensor(suffix_ids, dtype=torch.long, device=self.device)
        prefix_embeds = self.embed_tokens(prefix_tensor)
        suffix_embeds = self.embed_tokens(suffix_tensor)

        # Audio embeddings
        audio_len = audio_embed_lens[0].item()
        audio_emb = audio_embeds[0, :audio_len, :]

        # Concat: [prefix_text_emb | audio_emb | suffix_text_emb]
        inputs_embeds = torch.cat([prefix_embeds, audio_emb, suffix_embeds], dim=0)
        return inputs_embeds

    def generate(
        self,
        inputs: Union[str, List[str], np.ndarray, torch.Tensor, List],
        hotwords: List[str] = None,
        language: str = None,
        itn: bool = True,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> List[dict]:
        """Run batch ASR inference using vLLM.

        Args:
            inputs: Audio input(s). Accepts:
                - str: single file path
                - List[str]: batch of file paths
                - np.ndarray / torch.Tensor: raw audio samples (16kHz)
            hotwords: Keywords to boost recognition accuracy.
            language: Language hint (e.g. "中文", "英文", "日文").
            itn: Apply inverse text normalization (default True).
            max_new_tokens: Maximum tokens to generate per sample.
            temperature: Sampling temperature (0 = greedy decoding).
            top_p: Nucleus sampling parameter.
            top_k: Top-k sampling (-1 = disabled).
            repetition_penalty: Repetition penalty factor.

        Returns:
            List of result dicts: [{"key": str, "text": str, "timestamps": [...]}]
        """
        from vllm import SamplingParams
        from vllm.inputs.data import EmbedsPrompt

        if isinstance(inputs, (str, np.ndarray, torch.Tensor)):
            inputs = [inputs]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            repetition_penalty=repetition_penalty,
            skip_special_tokens=True,
        )

        # Batch encode audio and build embedding prompts
        prompts = []
        encoder_outputs = []

        t0 = time.perf_counter()

        # Pre-compute text embeddings (shared across batch)
        prompt_text = self._build_prompt_text(hotwords, language, itn)
        prefix_text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{prompt_text}"
        suffix_text = "<|im_end|>\n<|im_start|>assistant\n"
        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        prefix_emb = self.embed_tokens(torch.tensor(prefix_ids, dtype=torch.long, device=self.device))
        suffix_emb = self.embed_tokens(torch.tensor(suffix_ids, dtype=torch.long, device=self.device))

        # Batch encode audio (groups of 8 for memory efficiency)
        batch_size_enc = 16
        all_adaptor_outs = []
        all_adaptor_lens = []
        for i in range(0, len(inputs), batch_size_enc):
            batch_inputs = inputs[i:i+batch_size_enc]
            # Load and extract fbank for batch
            from funasr.utils.load_utils import load_audio_text_image_video, extract_fbank
            audio_tensors = []
            for audio_input in batch_inputs:
                if isinstance(audio_input, str):
                    data_src = load_audio_text_image_video(audio_input, fs=self.frontend.fs)
                elif isinstance(audio_input, np.ndarray):
                    data_src = torch.from_numpy(audio_input).float()
                elif isinstance(audio_input, torch.Tensor):
                    data_src = audio_input.float()
                else:
                    raise ValueError(f"Unsupported audio input type: {type(audio_input)}")
                audio_tensors.append(data_src)

            speech, speech_lengths = extract_fbank(
                audio_tensors, data_type="sound", frontend=self.frontend, is_final=True
            )
            speech = speech.to(self.device, dtype=torch.float32)
            speech_lengths = speech_lengths.to(self.device)

            with torch.no_grad():
                enc_out, enc_lens = self.audio_encoder(speech, speech_lengths)
                adp_out, adp_lens = self.audio_adaptor(enc_out.to(dtype=self.torch_dtype), enc_lens)

            # Apply low frame rate token length correction
            if self.use_low_frame_rate:
                for j in range(len(batch_inputs)):
                    fbank_len = speech_lengths[j].item()
                    olens = 1 + (fbank_len - 3 + 2 * 1) // 2
                    olens = 1 + (olens - 3 + 2 * 1) // 2
                    adp_lens[j] = (olens - 1) // 2 + 1

            for j in range(len(batch_inputs)):
                all_adaptor_outs.append(adp_out[j, :adp_lens[j], :])
                all_adaptor_lens.append(adp_lens[j])
                encoder_outputs.append((enc_out[j:j+1, :enc_lens[j], :], enc_lens[j:j+1]))

        # Build prompts
        for audio_emb in all_adaptor_outs:
            input_embeds = torch.cat([prefix_emb, audio_emb, suffix_emb], dim=0)
            prompts.append(EmbedsPrompt(prompt_embeds=input_embeds.float()))

        t1 = time.perf_counter()
        logger.info(f"Audio encoding: {len(inputs)} samples in {t1 - t0:.3f}s")

        # vLLM batch generation
        outputs = self.vllm_engine.generate(prompts, sampling_params, use_tqdm=len(inputs) > 1)

        t2 = time.perf_counter()
        logger.info(f"vLLM generation: {t2 - t1:.3f}s")

        # Process results
        results = []
        for i, output in enumerate(outputs):
            token_ids = list(output.outputs[0].token_ids)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            # Clean vLLM artifacts: remove garbage prefix/tags
            text = re.sub(r'<[^>]*>', '', text)
            text = re.sub(r'\[[^\]]*\]', '', text)
            text = re.sub(r'endofpatch|/sil|FFFF|</strong>', '', text)
            # Strip non-CJK/non-alnum prefix garbage
            text = re.sub(r'^[^\w一-鿿]+', '', text)
            text_clean = re.sub(r"\s+", " ", text).strip()

            key = (
                os.path.splitext(os.path.basename(inputs[i]))[0]
                if isinstance(inputs[i], str)
                else f"sample_{i}"
            )
            result = {"key": key, "text": text_clean}

            # Timestamps via CTC forced alignment
            if self.ctc_decoder is not None and self.ctc_tokenizer is not None:
                try:
                    timestamps = self._compute_timestamps(
                        encoder_outputs[i][0], encoder_outputs[i][1], text_clean
                    )
                    if timestamps:
                        result["timestamps"] = timestamps
                except Exception as e:
                    logger.debug(f"Timestamp computation failed for {key}: {e}")

            results.append(result)

        return results

    @torch.no_grad()
    def _compute_timestamps(self, encoder_out, encoder_out_lens, text):
        """CTC forced alignment for character-level timestamps."""
        from funasr.models.fun_asr_nano.tools.utils import forced_align

        decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, encoder_out_lens)
        ctc_logits = self.ctc.log_softmax(decoder_out)
        x = ctc_logits[0, : encoder_out_lens[0].item(), :]

        target_ids = torch.tensor(self.ctc_tokenizer.encode(text), dtype=torch.int64)
        if len(target_ids) == 0:
            return []

        timestamps = forced_align(x, target_ids, self.blank_id)
        for ts in timestamps:
            ts["token"] = self.ctc_tokenizer.decode([ts["token"]])
            ts["start_time"] = ts["start_time"] * 6 * 10 / 1000
            ts["end_time"] = ts["end_time"] * 6 * 10 / 1000
        return timestamps

    @classmethod
    def from_pretrained(
        cls,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        hub: str = "ms",
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 2048,
        **kwargs,
    ) -> "FunASRNanoVLLM":
        """Load model from hub or local path.

        Args:
            model: Model name or local directory path.
            hub: "ms" (ModelScope) or "hf" (HuggingFace).
            device: Device for audio encoder/adaptor.
            dtype: Compute dtype ("bf16", "fp16", "fp32").
            tensor_parallel_size: GPUs for vLLM tensor parallel.
            gpu_memory_utilization: GPU memory fraction for vLLM.
            max_model_len: Maximum sequence length.

        Returns:
            Initialized FunASRNanoVLLM engine.
        """
        if os.path.isdir(model):
            model_dir = model
        else:
            if hub in ("ms", "modelscope"):
                from modelscope.hub.snapshot_download import snapshot_download

                model_dir = snapshot_download(model, revision=kwargs.pop("revision", "master"))
            elif hub in ("hf", "huggingface"):
                from huggingface_hub import snapshot_download

                model_dir = snapshot_download(model)
            else:
                raise ValueError(f"Unsupported hub: {hub}. Use 'ms' or 'hf'.")

        logger.info(f"Model directory: {model_dir}")
        return cls(
            model_dir=model_dir,
            device=device,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            **kwargs,
        )
