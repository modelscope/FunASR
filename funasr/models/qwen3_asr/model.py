import logging
import os
import time
from importlib.metadata import PackageNotFoundError
from importlib.metadata import requires as _package_requires
from importlib.metadata import version as _package_version
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from funasr.register import tables


def _qwen_asr_transformers_specifier():
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        return None

    requirements = _package_requires("qwen-asr") or []
    for requirement_text in requirements:
        try:
            requirement = Requirement(requirement_text)
        except InvalidRequirement:
            continue
        if requirement.name.lower() == "transformers":
            return requirement.specifier
    return None


def _qwen_asr_install_command(qwen_asr_version, transformers_specifier=None):
    if transformers_specifier:
        transformers_requirement = f"transformers{transformers_specifier}"
        return (
            f'pip install -U "qwen-asr=={qwen_asr_version}" '
            f'"{transformers_requirement}" accelerate'
        )
    return 'pip install -U "qwen-asr" transformers accelerate'


def _check_qwen3_asr_dependencies():
    try:
        qwen_asr_version = _package_version("qwen-asr")
    except PackageNotFoundError as e:
        raise ImportError(
            'qwen-asr package is required for Qwen3-ASR. Install with: pip install -U "qwen-asr"'
        ) from e

    try:
        transformers_version = _package_version("transformers")
    except PackageNotFoundError as e:
        raise ImportError(
            "transformers is required by qwen-asr. "
            f"Install with: {_qwen_asr_install_command(qwen_asr_version)}"
        ) from e

    transformers_specifier = _qwen_asr_transformers_specifier()
    if transformers_specifier:
        try:
            from packaging.version import InvalidVersion, Version

            Version(transformers_version)
            is_compatible = transformers_version in transformers_specifier
        except InvalidVersion:
            is_compatible = True

        if not is_compatible:
            install_command = _qwen_asr_install_command(qwen_asr_version, transformers_specifier)
            raise ImportError(
                "Qwen3-ASR dependency mismatch: "
                f"qwen-asr=={qwen_asr_version} requires transformers{transformers_specifier}, "
                f"but the active environment has transformers=={transformers_version}. "
                "This can trigger qwen_asr errors such as "
                "`AttributeError: 'Qwen3ASRConfig' object has no attribute 'thinker_config'`. "
                f"Run: {install_command}"
            )


# qwen-asr's validate_language() only accepts canonical full names ("Chinese", "English",
# ...), but FunASR documents short/ISO codes ("zh", "en", "auto") as valid language hints.
# Defined at module level so the lookup table is built once, not on every inference call.
_ISO_LANG_ALIASES = {
    "zh": "Chinese", "zh-cn": "Chinese", "zho": "Chinese", "cmn": "Chinese",
    "en": "English", "yue": "Cantonese", "ar": "Arabic", "de": "German",
    "fr": "French", "es": "Spanish", "pt": "Portuguese", "id": "Indonesian",
    "it": "Italian", "ko": "Korean", "ru": "Russian", "th": "Thai",
    "vi": "Vietnamese", "ja": "Japanese", "tr": "Turkish", "hi": "Hindi",
    "ms": "Malay", "nl": "Dutch", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "pl": "Polish", "cs": "Czech", "fil": "Filipino",
    "fa": "Persian", "el": "Greek", "ro": "Romanian", "hu": "Hungarian",
    "mk": "Macedonian",
}


@tables.register("model_classes", "Qwen3ASR")
@tables.register("model_classes", "Qwen/Qwen3-ASR-1.7B")
@tables.register("model_classes", "Qwen/Qwen3-ASR-0.6B")
class Qwen3ASR(nn.Module):
    """Qwen3-ASR: Large Language Model based ASR supporting 52 languages.

    Wraps the qwen-asr package's Qwen3ASRModel for use within FunASR's AutoModel interface.
    Supports auto language detection, contextual recognition, and optional forced alignment
    for character-level timestamps.

    Requirements:
        pip install -U "qwen-asr==0.0.6" "transformers==4.57.6" accelerate

    Models:
        - Qwen/Qwen3-ASR-0.6B (lighter, ~4GB GPU memory)
        - Qwen/Qwen3-ASR-1.7B (more accurate, ~8GB GPU memory)
    """

    def __init__(self, **kwargs):
        """Initialize Qwen3ASR.
        
            Args:
                **kwargs: Additional keyword arguments.
            """
        super().__init__()
        model_path = kwargs.get("model_path", kwargs.get("model", "Qwen/Qwen3-ASR-1.7B"))
        device = kwargs.get("device", "cuda:0")
        dtype = kwargs.get("dtype", "bf16")
        hub = kwargs.get("hub", "ms")
        max_new_tokens = kwargs.get("max_new_tokens", 512)
        max_inference_batch_size = kwargs.get("max_inference_batch_size", 32)
        forced_aligner = kwargs.get("forced_aligner", None)
        forced_aligner_kwargs = kwargs.get("forced_aligner_kwargs", None)

        self._dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self._device = device
        self._placeholder = nn.Parameter(torch.empty(0))

        model_path = self._resolve_model_path(model_path, hub, kwargs)
        self.model_path = model_path

        _check_qwen3_asr_dependencies()
        try:
            from qwen_asr import Qwen3ASRModel
        except ImportError as e:
            # Only catch if the package itself is missing, not if its dependencies are broken
            if "qwen_asr" in str(e):
                raise ImportError(
                    'qwen-asr package is required. Install with: pip install -U "qwen-asr"'
                ) from e
            raise e

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

    def _resolve_model_path(self, model_path, hub, kwargs):
        """Resolve model path: use local if exists, otherwise download from hub.

        Args:
            model_path (str): Model name or local path.
            hub (str): "ms" for ModelScope, "hf" for HuggingFace.
            kwargs (dict): Additional options (model_revision, etc.)

        Returns:
            str: Resolved local path to model files.
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
        """Run Qwen3-ASR speech recognition.

        Args:
            data_in: Audio input. Accepts:
                - list of file paths/URLs
                - list of (numpy_array, sample_rate) tuples
                - single numpy array or torch Tensor
            data_lengths: Not used.
            key (list): Sample identifiers.
            tokenizer: Not used (Qwen3-ASR has internal tokenizer).
            frontend: Not used (Qwen3-ASR has internal audio processing).
            **kwargs: Runtime parameters:
                - language (str): Language hint (e.g. "Chinese", "English") or None for auto-detect.
                - return_time_stamps (bool): Return character-level timestamps (requires forced_aligner).
                - output_timestamp (bool): Same as return_time_stamps (for pipeline compatibility).
                - context (str): Context prompt for contextual recognition.

        Returns:
            tuple: (results, meta_data) where results is list of dicts:
                - "key" (str): Sample ID
                - "text" (str): Recognized text (with punctuation)
                - "language" (str): Detected language (if available)
                - "timestamp" (list): [[start_ms, end_ms], ...] (if timestamps enabled)
        """
        meta_data = {}
        time1 = time.perf_counter()

        language = kwargs.get("language", None)
        # Normalize FunASR's documented short/ISO codes (and "auto") to qwen-asr full names
        # so a hint like language="zh" doesn't raise "Unsupported language: Zh".
        if language is not None:
            _lk = str(language).strip().lower()
            if _lk in ("auto", "none", ""):
                language = None
            else:
                language = _ISO_LANG_ALIASES.get(_lk, language)
        return_time_stamps = kwargs.get("return_time_stamps", False) or kwargs.get("output_timestamp", False)
        context = kwargs.get("context", "")

        if isinstance(data_in, (list, tuple)):
            audio_inputs = []
            for item in data_in:
                if isinstance(item, np.ndarray):
                    audio_inputs.append((item.astype(np.float32), 16000))
                elif isinstance(item, torch.Tensor):
                    audio_inputs.append((item.cpu().numpy().astype(np.float32), 16000))
                else:
                    audio_inputs.append(item)
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

        # If timestamps requested but forced_aligner not configured, fall back to no timestamps
        if return_time_stamps and self.qwen3_asr_model.forced_aligner is None:
            logging.warning("return_time_stamps requires forced_aligner. Skipping timestamps. "
                           "Initialize with forced_aligner='Qwen/Qwen3-ForcedAligner-0.6B' to enable.")
            return_time_stamps = False

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
                    [int(ts.start_time), int(ts.end_time)]
                    for ts in r.time_stamps.items
                ]
            output.append(result_dict)

        return output, meta_data
