#!/usr/bin/env python3
"""GLM-ASR vLLM inference engine.

Architecture: audio_tower (Whisper-like) + multi_modal_projector + language_model (Llama)
Strategy: audio_tower + projector in PyTorch, language_model in vLLM via EmbedsPrompt.
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


def prepare_glmasr_vllm_dir(model_dir: str) -> str:
    """Extract language_model weights into vLLM-compatible Llama format."""
    output_dir = os.path.join(model_dir, "language_model_vllm")

    if glob.glob(os.path.join(output_dir, "*.safetensors")):
        logger.info(f"vLLM LM weights already at {output_dir}")
        return output_dir

    os.makedirs(output_dir, exist_ok=True)

    # Load safetensors and extract language_model weights
    from safetensors import safe_open
    from safetensors.torch import save_file

    st_files = sorted(glob.glob(os.path.join(model_dir, "*.safetensors")))
    lm_state = {}
    for st_file in st_files:
        with safe_open(st_file, framework="pt") as f:
            for key in f.keys():
                if key.startswith("language_model."):
                    new_key = key[len("language_model."):]
                    lm_state[new_key] = f.get_tensor(key)

    if not lm_state:
        raise RuntimeError("No language_model weights found")

    logger.info(f"Extracted {len(lm_state)} LM tensors")
    save_file(lm_state, os.path.join(output_dir, "model.safetensors"))

    # Create config for Llama model
    with open(os.path.join(model_dir, "config.json")) as f:
        full_config = json.load(f)

    text_config = full_config["text_config"]
    text_config["architectures"] = ["LlamaForCausalLM"]
    text_config["model_type"] = "llama"
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(text_config, f, indent=2)

    # Copy tokenizer files
    for fname in os.listdir(model_dir):
        if "tokenizer" in fname or fname == "generation_config.json":
            src = os.path.join(model_dir, fname)
            dst = os.path.join(output_dir, fname)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Create safetensors index
    index = {
        "metadata": {"total_size": sum(v.numel() * v.element_size() for v in lm_state.values())},
        "weight_map": {k: "model.safetensors" for k in lm_state.keys()},
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)

    logger.info(f"Saved vLLM LM to {output_dir}")
    return output_dir


class GLMASRVLLMEngine:
    """GLM-ASR with vLLM backend.

    Audio tower + projector run in PyTorch, language model decoded by vLLM.
    """

    def __init__(self, model_dir, device="cuda:0", dtype="bf16",
                 tensor_parallel_size=1, gpu_memory_utilization=0.5,
                 max_model_len=4096, **kwargs):
        from vllm import LLM
        from transformers import AutoProcessor, AutoConfig, AutoModel as HFAutoModel

        self.device = device
        self.torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        self.model_dir = model_dir

        # Load full model for audio encoding
        logger.info(f"Loading GLM-ASR model from {model_dir}")
        self.full_model = HFAutoModel.from_pretrained(
            model_dir, dtype=self.torch_dtype, device_map=device, trust_remote_code=True
        )
        self.full_model.eval()
        for p in self.full_model.parameters():
            p.requires_grad = False

        # Get embed_tokens for text embedding
        self.embed_tokens = self.full_model.language_model.get_input_embeddings()

        # Load processor for audio preprocessing
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)

        # Load config
        config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        self.audio_token_id = config.audio_token_id

        # Note: full_model kept for audio encoding, vLLM loads separate LM copy

        # Prepare and load vLLM engine
        vllm_dir = prepare_glmasr_vllm_dir(model_dir)
        logger.info(f"Initializing vLLM with LM from {vllm_dir}")
        self.vllm_engine = LLM(
            model=vllm_dir,
            enable_prompt_embeds=True,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype={"bf16": "bfloat16", "fp16": "float16", "fp32": "auto"}.get(dtype, dtype),
            trust_remote_code=True,
        )
        self.tokenizer = self.vllm_engine.get_tokenizer()
        logger.info("GLM-ASR vLLM engine ready")

    @torch.no_grad()
    def _encode_audio(self, audio_input):
        """Encode audio through the full model's get_audio_features."""
        import librosa

        if isinstance(audio_input, str):
            audio, sr = librosa.load(audio_input, sr=16000)
        elif isinstance(audio_input, np.ndarray):
            audio = audio_input.astype(np.float32)
        elif isinstance(audio_input, torch.Tensor):
            audio = audio_input.numpy().astype(np.float32)
        else:
            raise ValueError(f"Unsupported audio type: {type(audio_input)}")

        # Get mel features
        inputs = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device, dtype=self.torch_dtype)

        # Create mask (all valid for single audio)
        feat_len = input_features.shape[-1]
        input_features_mask = torch.ones(1, feat_len, dtype=torch.long, device=self.device)

        # Use full model's audio processing pipeline
        audio_outputs = self.full_model.get_audio_features(
            input_features, input_features_mask, return_dict=True
        )
        audio_embeds = audio_outputs.pooler_output  # (N_valid_tokens, hidden_size)
        return audio_embeds.unsqueeze(0)  # (1, T, hidden_size)

    def _build_prompt_embeds(self, audio_embeds, prompt="转录以下音频内容"):
        """Build full embedding: [prefix] + [audio] + [suffix]."""
        # Chat template
        prefix_text = f"<|user|>\n<|begin_of_audio|>"
        suffix_text = f"<|end_of_audio|><|user|>\n{prompt}<|assistant|>\n"

        prefix_ids = self.tokenizer.encode(prefix_text, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)

        prefix_emb = self.embed_tokens(torch.tensor(prefix_ids, dtype=torch.long, device=self.device))
        suffix_emb = self.embed_tokens(torch.tensor(suffix_ids, dtype=torch.long, device=self.device))

        # audio_embeds: (1, T, D) -> (T, D)
        audio_emb = audio_embeds[0] if audio_embeds.dim() == 3 else audio_embeds

        return torch.cat([prefix_emb, audio_emb, suffix_emb], dim=0)

    def generate(self, inputs, prompt="转录以下音频内容", max_new_tokens=500, **kwargs):
        """Run batch ASR inference.

        Args:
            inputs: Audio file path(s), numpy arrays, or tensors.
            prompt: Instruction prompt.
            max_new_tokens: Max tokens to generate.

        Returns:
            List of {"key": str, "text": str}
        """
        from vllm import SamplingParams
        from vllm.inputs.data import EmbedsPrompt

        if isinstance(inputs, (str, np.ndarray, torch.Tensor)):
            inputs = [inputs]

        sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=0,
            skip_special_tokens=True,
        )

        # Encode all audio and build prompts
        prompts = []
        t0 = time.perf_counter()
        for audio_input in inputs:
            audio_embeds = self._encode_audio(audio_input)
            full_embeds = self._build_prompt_embeds(audio_embeds, prompt=prompt)
            prompts.append(EmbedsPrompt(prompt_embeds=full_embeds.float()))
        t1 = time.perf_counter()
        logger.info(f"Audio encoding: {len(inputs)} samples in {t1-t0:.3f}s")

        # vLLM batch generate
        outputs = self.vllm_engine.generate(prompts, sampling_params, use_tqdm=False)
        t2 = time.perf_counter()
        logger.info(f"vLLM generation: {t2-t1:.3f}s")

        results = []
        for i, output in enumerate(outputs):
            token_ids = list(output.outputs[0].token_ids)
            text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
            text = re.sub(r'\s+', ' ', text).strip()
            key = os.path.splitext(os.path.basename(inputs[i]))[0] if isinstance(inputs[i], str) else f"sample_{i}"
            results.append({"key": key, "text": text})

        return results

    @classmethod
    def from_pretrained(cls, model="zai-org/GLM-ASR-Nano-2512", hub="ms",
                        device="cuda:0", dtype="bf16", **kwargs):
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
        return cls(model_dir=model_dir, device=device, dtype=dtype, **kwargs)
