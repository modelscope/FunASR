import logging
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from funasr.register import tables


@tables.register("model_classes", "Qwen3ASR")
class Qwen3ASR(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        model_path = kwargs.get("model_path", "Qwen/Qwen3-ASR-1.7B")
        device = kwargs.get("device", "cuda:0")
        dtype = kwargs.get("dtype", "bf16")
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        max_inference_batch_size = kwargs.get("max_inference_batch_size", 32)
        forced_aligner = kwargs.get("forced_aligner", None)
        forced_aligner_kwargs = kwargs.get("forced_aligner_kwargs", None)

        self._dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._device = device
        self.model_path = model_path

        self._placeholder = nn.Parameter(torch.empty(0))

        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "qwen-asr package is required. Install with: pip install qwen-asr"
            )

        torch_dtype = self._dtype_map.get(dtype, torch.bfloat16)
        fa_kwargs = None
        if forced_aligner:
            fa_kwargs = forced_aligner_kwargs or {}
            fa_kwargs.setdefault("dtype", torch_dtype)
            fa_kwargs.setdefault("device_map", device)

        self.qwen3_asr_model = Qwen3ASRModel.from_pretrained(
            model_path,
            dtype=torch_dtype,
            device_map=device,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=fa_kwargs,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
        )
        logging.info(f"Qwen3ASR model loaded from {model_path}")

    def forward(self, **kwargs):
        raise NotImplementedError("Qwen3ASR only supports inference mode")

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        meta_data = {}
        time1 = time.perf_counter()

        language = kwargs.get("language", None)
        return_time_stamps = kwargs.get("return_time_stamps", False)
        context = kwargs.get("context", "")

        if isinstance(data_in, (list, tuple)):
            audio_inputs = list(data_in)
        elif isinstance(data_in, str):
            audio_inputs = [data_in]
        elif isinstance(data_in, torch.Tensor):
            audio_np = data_in.cpu().numpy().astype(np.float32)
            if audio_np.ndim == 1:
                audio_inputs = [(audio_np, 16000)]
            else:
                audio_inputs = [(audio_np[i], 16000) for i in range(audio_np.shape[0])]
        elif isinstance(data_in, np.ndarray):
            if data_in.ndim == 1:
                audio_inputs = [(data_in.astype(np.float32), 16000)]
            else:
                audio_inputs = [(data_in[i].astype(np.float32), 16000) for i in range(data_in.shape[0])]
        else:
            audio_inputs = [data_in]

        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"

        results = self.qwen3_asr_model.transcribe(
            audio=audio_inputs,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )

        time3 = time.perf_counter()
        meta_data["batch_data_time"] = time3 - time2

        output = []
        for i, r in enumerate(results):
            k = key[i] if key and i < len(key) else f"sample_{i}"
            result_dict = {"key": k, "text": r.text}
            if r.language:
                result_dict["language"] = r.language
            if return_time_stamps and r.time_stamps is not None:
                result_dict["timestamp"] = [
                    [ts.start_time, ts.end_time, ts.text]
                    for ts in r.time_stamps.items
                ]
            output.append(result_dict)

        return output, meta_data
