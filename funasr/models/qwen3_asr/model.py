import logging
import time
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from funasr.register import tables
from funasr.utils.load_utils import load_audio_text_image_video


@tables.register("model_classes", "Qwen3ASR")
class Qwen3ASR(nn.Module):

    def __init__(
        self,
        model: str = "Qwen/Qwen3-ASR-1.7B",
        model_conf: dict = None,
        **kwargs,
    ):
        super().__init__()
        model_conf = model_conf or {}
        self.model_path = model
        self.qwen3_asr_model = None
        self._device = model_conf.get("device", kwargs.get("device", "cuda:0"))
        self._dtype = model_conf.get("dtype", kwargs.get("dtype", "bf16"))
        self._max_new_tokens = model_conf.get("max_new_tokens", 512)
        self._max_inference_batch_size = model_conf.get("max_inference_batch_size", 32)
        self._forced_aligner = model_conf.get("forced_aligner", None)
        self._forced_aligner_kwargs = model_conf.get("forced_aligner_kwargs", None)
        self._dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._init_model()

    def _init_model(self):
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError:
            raise ImportError(
                "qwen-asr package is required for Qwen3ASR model. "
                "Install with: pip install qwen-asr"
            )

        dtype = self._dtype_map.get(self._dtype, torch.bfloat16)
        fa_kwargs = None
        if self._forced_aligner:
            fa_kwargs = self._forced_aligner_kwargs or {}
            fa_kwargs.setdefault("dtype", dtype)
            fa_kwargs.setdefault("device_map", self._device)

        self.qwen3_asr_model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=self._device,
            forced_aligner=self._forced_aligner,
            forced_aligner_kwargs=fa_kwargs,
            max_inference_batch_size=self._max_inference_batch_size,
            max_new_tokens=self._max_new_tokens,
        )
        logging.info(f"Qwen3ASR model loaded from {self.model_path}")

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
        language = kwargs.get("language", None)
        return_time_stamps = kwargs.get("return_time_stamps", False)
        context = kwargs.get("context", "")
        fs = kwargs.get("fs", 16000)

        meta_data = {}
        time1 = time.perf_counter()

        if isinstance(data_in, (list, tuple)):
            audio_list = data_in
        elif isinstance(data_in, str):
            audio_list = [data_in]
        elif isinstance(data_in, torch.Tensor):
            audio_np = data_in.cpu().numpy()
            if audio_np.ndim == 1:
                audio_list = [(audio_np, fs)]
            else:
                audio_list = [(audio_np[i], fs) for i in range(audio_np.shape[0])]
        elif isinstance(data_in, np.ndarray):
            if data_in.ndim == 1:
                audio_list = [(data_in, fs)]
            else:
                audio_list = [(data_in[i], fs) for i in range(data_in.shape[0])]
        else:
            audio_list = [data_in]

        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"

        results = self.qwen3_asr_model.transcribe(
            audio=audio_list,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
        )

        time3 = time.perf_counter()
        meta_data["inference"] = f"{time3 - time2:0.3f}"

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
