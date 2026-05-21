import logging
import os
import time

import torch
import torch.nn as nn

from funasr.register import tables


@tables.register("model_classes", "GLMASR")
@tables.register("model_classes", "zai-org/GLM-ASR-Nano-2512")
@tables.register("model_classes", "ZhipuAI/GLM-ASR-Nano-2512")
class GLMASR(nn.Module):

    def __init__(self, **kwargs):
        """Initialize GLMASR.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        super().__init__()
        model_path = kwargs.get("model_path", kwargs.get("model", "zai-org/GLM-ASR-Nano-2512"))
        device = kwargs.get("device", "cuda:0")
        dtype = kwargs.get("dtype", "bf16")
        hub = kwargs.get("hub", "ms")
        self._max_new_tokens = kwargs.get("max_new_tokens", 512)

        self._dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._device = device
        self._torch_dtype = self._dtype_map.get(dtype, torch.bfloat16)
        self._placeholder = nn.Parameter(torch.empty(0))

        model_path = self._resolve_model_path(model_path, hub, kwargs)
        self.model_path = model_path

        from transformers import AutoModel as HFAutoModel
        from transformers import AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.glm_model = HFAutoModel.from_pretrained(
            model_path,
            dtype=self._torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.glm_model.eval()
        logging.info(f"GLM-ASR model loaded from {model_path}")

    def _resolve_model_path(self, model_path, hub, kwargs):
        """Internal: resolve model path.
        
            Args:
                model_path: TODO.
                hub: TODO.
                kwargs: Additional keyword arguments.
            """
        if os.path.exists(model_path):
            return model_path

        if hub in ("ms", "modelscope"):
            try:
                from modelscope.hub.snapshot_download import snapshot_download
                model_revision = kwargs.get("model_revision", "master")
                local_path = snapshot_download(model_path, revision=model_revision)
                logging.info(f"Downloaded from ModelScope: {model_path} -> {local_path}")
                return local_path
            except Exception as e:
                logging.warning(f"ModelScope download failed: {e}, falling back to HuggingFace path")

        return model_path

    def forward(self, **kwargs):
        """Forward pass for training.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        raise NotImplementedError("GLMASR only supports inference mode")

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):
        """Run inference on input data.
        
            Args:
                data_in: Input data (audio samples, file paths, or text).
                data_lengths: Lengths of each input sample in the batch.
                key: Sample identifiers.
                tokenizer: Tokenizer instance for text encoding/decoding.
                frontend: Audio frontend for feature extraction.
                **kwargs: Additional keyword arguments.
            """
        meta_data = {}
        time1 = time.perf_counter()

        prompt = kwargs.get("prompt", "Please transcribe this audio into text")

        if isinstance(data_in, (list, tuple)):
            audio_list = list(data_in)
        elif isinstance(data_in, str):
            audio_list = [data_in]
        else:
            audio_list = [data_in]

        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"

        output = []
        for i, audio_input in enumerate(audio_list):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "url": audio_input},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self._device, dtype=self._torch_dtype)

            with torch.inference_mode():
                generated = self.glm_model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                )

            text = self.processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True,
            )[0].strip()

            k = key[i] if key and i < len(key) else f"sample_{i}"
            output.append({"key": k, "text": text})

        time3 = time.perf_counter()
        meta_data["batch_data_time"] = time3 - time2

        return output, meta_data
