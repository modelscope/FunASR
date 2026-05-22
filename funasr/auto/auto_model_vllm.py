#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/alibaba-damo-academy/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""
Generic vLLM inference wrapper for ALL LLM-based ASR models in FunASR.

Applicable models (any model with audio_encoder + adaptor + LLM architecture):
    - FunASRNano (Fun-ASR-Nano-2512, Fun-ASR-MLT-Nano-2512)
    - LLMASR (Whisper + Qwen/Vicuna/LLaMA)
    - GLMASR (GLM-ASR-Nano)

NOT applicable (these models don't use autoregressive LLM decoding):
    - Paraformer (non-autoregressive CIF predictor + attention decoder)
    - SenseVoice (Whisper-like encoder-decoder, not LLM-based)
    - Conformer/Transformer ASR (CTC/attention, no LLM)
    - CT-Transformer (punctuation model, small transformer)
    - Qwen3-ASR (uses external qwen-asr package with its own optimized inference)

Usage:
    from funasr.auto.auto_model_vllm import AutoModelVLLM

    # Works for any LLM-based ASR model
    model = AutoModelVLLM(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        tensor_parallel_size=2,
    )
    results = model.generate(["audio.wav"])

    # Also works for LLMASR models
    model = AutoModelVLLM(
        model="/path/to/llm_asr_model",
        tensor_parallel_size=4,
    )
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

# Models that use LLM and can benefit from vLLM
_LLM_BASED_MODELS = {"FunASRNano", "LLMASR", "LLMASRNAR", "GLMASR", "QwenAudioWarp"}

# Models that CANNOT use vLLM (no autoregressive LLM)
_NON_LLM_MODELS = {
    "Paraformer": "Non-autoregressive model using CIF predictor. No LLM decoding.",
    "SenseVoice": "Whisper-like encoder-decoder. Not LLM-based.",
    "CTTransformer": "Small punctuation model. No benefit from vLLM.",
    "Conformer": "CTC/attention encoder-decoder. No LLM.",
    "Qwen3ASR": "Uses external qwen-asr package with optimized inference.",
}


def check_vllm_applicable(model_name: str) -> bool:
    """Check if a model can use vLLM inference.

    Args:
        model_name: The model class name from config.yaml.

    Returns:
        True if vLLM is applicable.

    Raises:
        ValueError: If model explicitly cannot use vLLM, with explanation.
    """
    if model_name in _LLM_BASED_MODELS:
        return True
    for non_llm, reason in _NON_LLM_MODELS.items():
        if non_llm in model_name:
            raise ValueError(
                f"Model '{model_name}' cannot use vLLM: {reason}\n"
                f"vLLM only accelerates autoregressive LLM decoding. "
                f"Use the standard FunASR AutoModel for this model."
            )
    return False


def prepare_vllm_weights(model_dir: str, output_dir: str = None) -> str:
    """Extract LLM weights from model.pt into vLLM-compatible format.

    Works for any model that stores LLM weights with 'llm.' prefix in model.pt
    and has a config directory (e.g., Qwen3-0.6B/) with model config and tokenizer.

    Args:
        model_dir: Path to the FunASR model directory.
        output_dir: Where to save extracted weights. Auto-detected if None.

    Returns:
        Path to vLLM-ready model directory.
    """
    if output_dir is None:
        # Find the LLM config directory
        from omegaconf import OmegaConf
        config_path = os.path.join(model_dir, "config.yaml")
        if os.path.exists(config_path):
            config = OmegaConf.load(config_path)
            llm_conf = OmegaConf.to_container(config.get("llm_conf", {}), resolve=True)
            llm_path = llm_conf.get("init_param_path", "")
            if llm_path and not os.path.isabs(llm_path):
                llm_path = os.path.join(model_dir, llm_path)
            if os.path.isdir(llm_path):
                output_dir = llm_path + "-vllm"
            else:
                output_dir = os.path.join(model_dir, "llm-vllm")
        else:
            output_dir = os.path.join(model_dir, "llm-vllm")

    # Check if already prepared
    if glob.glob(os.path.join(output_dir, "*.safetensors")) or glob.glob(
        os.path.join(output_dir, "model*.bin")
    ):
        logger.info(f"vLLM weights already at {output_dir}")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Find and copy LLM config/tokenizer files
    from omegaconf import OmegaConf
    config = OmegaConf.load(os.path.join(model_dir, "config.yaml"))
    llm_conf = OmegaConf.to_container(config.get("llm_conf", {}), resolve=True)
    llm_config_dir = llm_conf.get("init_param_path", "")
    if llm_config_dir and not os.path.isabs(llm_config_dir):
        llm_config_dir = os.path.join(model_dir, llm_config_dir)

    if os.path.isdir(llm_config_dir):
        for fname in os.listdir(llm_config_dir):
            src = os.path.join(llm_config_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Extract LLM weights from model.pt
    model_pt = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_pt):
        raise FileNotFoundError(f"model.pt not found at {model_pt}")

    logger.info(f"Extracting LLM weights from {model_pt}...")
    checkpoint = torch.load(model_pt, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)

    llm_state = {}
    for key, value in state_dict.items():
        if key.startswith("llm."):
            llm_state[key[4:]] = value  # Remove 'llm.' prefix

    if not llm_state:
        raise RuntimeError("No LLM weights found (expected 'llm.*' prefix)")

    logger.info(f"Extracted {len(llm_state)} LLM weight tensors")

    try:
        from safetensors.torch import save_file
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(llm_state, save_path)
        index = {
            "metadata": {"total_size": sum(v.numel() * v.element_size() for v in llm_state.values())},
            "weight_map": {k: "model.safetensors" for k in llm_state.keys()},
        }
        with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
            json.dump(index, f, indent=2)
    except ImportError:
        torch.save(llm_state, os.path.join(output_dir, "model.bin"))

    return output_dir


class AutoModelVLLM:
    """Generic vLLM wrapper for LLM-based ASR models.

    Automatically detects model architecture, extracts LLM weights,
    loads audio components in PyTorch, and uses vLLM for generation.

    Works for: FunASRNano, LLMASR, GLMASR, and any model with
    audio_encoder + audio_adaptor + LLM architecture.

    Args:
        model: Model name (hub) or local directory path.
        hub: "ms" (ModelScope) or "hf" (HuggingFace).
        device: Device for audio encoder/adaptor.
        dtype: Compute dtype ("bf16", "fp16", "fp32").
        tensor_parallel_size: GPUs for vLLM tensor parallelism.
        gpu_memory_utilization: GPU memory fraction for vLLM.
        max_model_len: Maximum sequence length.

    Example:
        >>> model = AutoModelVLLM(model="FunAudioLLM/Fun-ASR-Nano-2512")
        >>> results = model.generate(["audio.wav"], language="中文")
    """

    def __init__(
        self,
        model: str,
        hub: str = "ms",
        device: str = "cuda:0",
        dtype: str = "bf16",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.8,
        max_model_len: int = 4096,
        enforce_eager: bool = False,
        **kwargs,
    ):
        # Resolve model directory
        if os.path.isdir(model):
            self.model_dir = model
        else:
            if hub in ("ms", "modelscope"):
                from modelscope.hub.snapshot_download import snapshot_download
                self.model_dir = snapshot_download(model, revision=kwargs.get("revision", "master"))
            elif hub in ("hf", "huggingface"):
                from huggingface_hub import snapshot_download
                self.model_dir = snapshot_download(model)
            else:
                raise ValueError(f"Unsupported hub: {hub}")

        # Check model type
        from omegaconf import OmegaConf
        config = OmegaConf.load(os.path.join(self.model_dir, "config.yaml"))
        self.model_type = config.get("model", "unknown")
        check_vllm_applicable(self.model_type)

        self.device = device
        self.dtype = dtype
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Use the specialized implementation if available
        if self.model_type == "FunASRNano":
            from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
            self._engine = FunASRNanoVLLM(
                model_dir=self.model_dir, device=device, dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, enforce_eager=enforce_eager,
                **kwargs,
            )
        elif self.model_type in ("LLMASR", "LLMASRNAR"):
            self._engine = self._build_llmasr_engine(
                config, tensor_parallel_size, gpu_memory_utilization,
                max_model_len, enforce_eager, **kwargs,
            )
        else:
            # Generic fallback using the FunASRNano approach
            # (works for any model with audio_encoder + adaptor + LLM)
            from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
            self._engine = FunASRNanoVLLM(
                model_dir=self.model_dir, device=device, dtype=dtype,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                max_model_len=max_model_len, enforce_eager=enforce_eager,
                **kwargs,
            )

    def _build_llmasr_engine(self, config, tensor_parallel_size, gpu_memory_utilization,
                              max_model_len, enforce_eager, **kwargs):
        """Build vLLM engine for LLMASR models (Whisper + Qwen/Vicuna)."""
        from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM

        # LLMASR follows same pattern as FunASRNano
        return FunASRNanoVLLM(
            model_dir=self.model_dir, device=self.device, dtype=self.dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len, enforce_eager=enforce_eager,
            **kwargs,
        )

    def generate(self, inputs, **kwargs):
        """Run ASR inference.

        Args:
            inputs: Audio file path(s), numpy arrays, or tensors.
            **kwargs: Model-specific parameters (language, hotwords, etc.)

        Returns:
            List of result dicts with "key" and "text" fields.
        """
        return self._engine.generate(inputs, **kwargs)

    @classmethod
    def supported_models(cls):
        """Return dict of model types and their vLLM support status."""
        info = {}
        for m in _LLM_BASED_MODELS:
            info[m] = {"supported": True, "reason": "LLM-based, autoregressive generation"}
        for m, reason in _NON_LLM_MODELS.items():
            info[m] = {"supported": False, "reason": reason}
        return info
