"""FunASR adapter for OpenMOSS MOSS-Transcribe-Diarize.

The model and its reference implementation are maintained by the OpenMOSS
Team under Apache-2.0:
https://github.com/OpenMOSS/MOSS-Transcribe-Diarize
"""

import copy
import io
import mimetypes
import os
import re
import time

import numpy as np
import torch
import torch.nn as nn

from funasr.register import tables


DEFAULT_MODEL = "OpenMOSS-Team/MOSS-Transcribe-Diarize"
DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)
_NUMBER = r"(?:\d+(?:\.\d+)?|\.\d+)"
_SEGMENT_START = re.compile(r"\[(%s)\]\[(S\d+)\]" % _NUMBER)
_TIMESTAMP = re.compile(r"\[(%s)\]" % _NUMBER)


def _parse_transcript(transcript):
    """Parse the model's ``[start][Sxx]text[end]`` output."""
    segments = []
    cursor = 0
    while True:
        start_match = _SEGMENT_START.search(transcript, cursor)
        if start_match is None:
            break

        text_start = start_match.end()
        candidate_cursor = text_start
        end_match = None
        while True:
            candidate = _TIMESTAMP.search(transcript, candidate_cursor)
            if candidate is None:
                break
            tail = candidate.end()
            next_start = _SEGMENT_START.match(transcript, tail)
            if next_start is None:
                whitespace_tail = tail
                while (
                    whitespace_tail < len(transcript)
                    and transcript[whitespace_tail].isspace()
                ):
                    whitespace_tail += 1
                next_start = _SEGMENT_START.match(transcript, whitespace_tail)
                at_end = whitespace_tail == len(transcript)
            else:
                at_end = False
            if next_start is not None or at_end:
                end_match = candidate
                break
            candidate_cursor = candidate.end()

        if end_match is None:
            cursor = start_match.end()
            continue

        start = float(start_match.group(1))
        end = float(end_match.group(1))
        if end >= start:
            text = transcript[text_start : end_match.start()].strip()
            if text:
                segments.append((start, end, start_match.group(2), text))
        cursor = end_match.end()
    return segments


def _join_texts(texts):
    if not texts:
        return ""
    joined = texts[0]
    for text in texts[1:]:
        previous = joined[-1] if joined else ""
        current = text[0] if text else ""
        cjk = "\u3400" <= previous <= "\u9fff" and "\u3400" <= current <= "\u9fff"
        joined += ("" if cjk else " ") + text
    return joined


def _result_from_transcript(key, transcript):
    segments = _parse_transcript(transcript)
    if not segments:
        return {
            "key": key,
            "text": transcript.strip(),
            "raw_text": transcript,
            "timestamp": [],
            "sentence_info": [],
        }

    sentence_info = []
    timestamps = []
    texts = []
    for start_s, end_s, speaker, text in segments:
        timestamp = [int(round(start_s * 1000)), int(round(end_s * 1000))]
        timestamps.append(timestamp)
        texts.append(text)
        sentence_info.append(
            {
                "start": timestamp[0],
                "end": timestamp[1],
                "text": text,
                "sentence": text,
                "spk": speaker,
                "timestamp": [timestamp],
            }
        )
    return {
        "key": key,
        "text": _join_texts(texts),
        "raw_text": transcript,
        "timestamp": timestamps,
        "sentence_info": sentence_info,
    }


@tables.register("model_classes", "MOSS-Transcribe-Diarize")
@tables.register("model_classes", DEFAULT_MODEL)
class MossTranscribeDiarize(nn.Module):
    """End-to-end ASR, diarization, timestamps, and acoustic-event inference."""

    def __init__(self, **kwargs):
        super().__init__()
        if kwargs.get("vad_model") is not None or kwargs.get("spk_model") is not None:
            raise ValueError(
                "MOSS-Transcribe-Diarize performs long-form segmentation and speaker "
                "diarization end to end; omit vad_model and spk_model to preserve global "
                "speaker identity."
            )

        self.backend = kwargs.get("backend", "hf").lower()
        if self.backend not in {"hf", "vllm"}:
            raise ValueError("backend must be 'hf' or 'vllm'")
        self.model_path = (
            kwargs.get("model_path") or kwargs.get("model") or DEFAULT_MODEL
        )
        self.device_name = kwargs.get("device", "cuda:0")
        self.dtype_name = kwargs.get("dtype", "bf16")
        self.max_new_tokens = int(kwargs.get("max_new_tokens", 5120))
        self.max_length = int(kwargs.get("max_length", 131072))
        self._placeholder = nn.Parameter(torch.empty(0), requires_grad=False)

        self.vllm_base_url = kwargs.get("vllm_base_url")
        self.vllm_model = kwargs.get("vllm_model", DEFAULT_MODEL)
        self.vllm_api_key = kwargs.get("vllm_api_key", "EMPTY")
        self.vllm_timeout = float(kwargs.get("vllm_timeout", 600.0))
        self.http_session = kwargs.get("http_session")

        self.hf_model = None
        self.processor = None
        if self.backend == "vllm":
            if not self.vllm_base_url:
                raise ValueError("vllm_base_url is required when backend='vllm'")
        else:
            self._load_hf_backend(kwargs)

    def _load_hf_backend(self, kwargs):
        try:
            import transformers
            from packaging.version import Version
            from transformers import AutoModelForCausalLM, AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "MOSS-Transcribe-Diarize requires Transformers 5.6 or newer. "
                "Install it in an isolated environment with `pip install 'transformers>=5.6,<6'`."
            ) from exc

        if Version(transformers.__version__) < Version("5.6.0"):
            raise ImportError(
                "MOSS-Transcribe-Diarize requires Transformers 5.6 or newer; found %s. "
                "Use an isolated environment so other FunASR model constraints remain unchanged."
                % transformers.__version__
            )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            revision=kwargs.get("model_revision"),
        )
        attention = kwargs.get("attn_implementation", "sdpa")
        attempts = [attention]
        if attention == "flash_attention_2":
            attempts.append("sdpa")
        if "eager" not in attempts:
            attempts.append("eager")
        last_error = None
        for implementation in attempts:
            try:
                self.hf_model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    revision=kwargs.get("model_revision"),
                    dtype="auto",
                    attn_implementation=implementation,
                )
                break
            except RuntimeError as exc:
                last_error = exc
        if self.hf_model is None:
            raise last_error

        dtype = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
            "fp32": torch.float32,
            "float32": torch.float32,
        }.get(self.dtype_name, torch.bfloat16)
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            device = torch.device("cpu")
            dtype = torch.float32
        self.hf_model.to(dtype=dtype).to(device).eval()
        self._hf_device = device
        self._hf_dtype = dtype

    def forward(self, **kwargs):
        raise NotImplementedError("MOSS-Transcribe-Diarize supports inference only")

    def inference(self, data_in, data_lengths=None, key=None, **kwargs):
        del data_lengths
        started = time.perf_counter()
        inputs = list(data_in) if isinstance(data_in, (list, tuple)) else [data_in]
        keys = key or ["sample_%d" % index for index in range(len(inputs))]
        audio_duration = sum(self._audio_duration(audio) for audio in inputs)
        results = []
        for index, audio in enumerate(inputs):
            if self.backend == "vllm":
                transcript = self._transcribe_vllm(audio, **kwargs)
            else:
                transcript = self._transcribe_hf(audio, **kwargs)
            results.append(_result_from_transcript(keys[index], transcript))
        return results, {
            "batch_data_time": max(audio_duration, 1e-9),
            "forward": time.perf_counter() - started,
        }

    def _transcribe_hf(self, audio, **kwargs):
        from transformers.audio_utils import load_audio

        sampling_rate = self.processor.feature_extractor.sampling_rate
        waveform = (
            audio
            if isinstance(audio, np.ndarray)
            else load_audio(audio, sampling_rate=sampling_rate)
        )
        prompt = kwargs.get("prompt", DEFAULT_PROMPT)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": waveform},
                    {"type": "text", "text": prompt.strip() or DEFAULT_PROMPT},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audio_kwargs = (
            {"device": str(self._hf_device)} if self._hf_device.type == "cuda" else {}
        )
        inputs = self.processor(
            text=text,
            audio=[waveform],
            max_length=int(kwargs.get("max_length", self.max_length)),
            audio_kwargs=audio_kwargs,
            return_tensors="pt",
        ).to(self._hf_device)
        prompt_len = int(inputs["attention_mask"][0].sum().item())
        generation_config = copy.deepcopy(self.hf_model.generation_config)
        generation_config.max_new_tokens = int(
            kwargs.get("max_new_tokens", self.max_new_tokens)
        )
        generation_config.do_sample = bool(kwargs.get("do_sample", False))
        if generation_config.do_sample:
            generation_config.temperature = float(kwargs.get("temperature", 0.8))

        with torch.inference_mode(), (
            torch.amp.autocast("cuda", dtype=self._hf_dtype)
            if self._hf_device.type == "cuda"
            and self._hf_dtype in (torch.float16, torch.bfloat16)
            else torch.no_grad()
        ):
            outputs = self.hf_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                input_features=inputs["input_features"],
                audio_feature_lengths=inputs["audio_feature_lengths"],
                audio_chunk_mapping=inputs["audio_chunk_mapping"],
                generation_config=generation_config,
            )
        generated = outputs[0][prompt_len:]
        return self.processor.tokenizer.decode(
            generated, skip_special_tokens=True
        ).strip()

    def _transcriptions_url(self):
        base = self.vllm_base_url.rstrip("/")
        if base.endswith("/v1/audio/transcriptions"):
            return base
        if not base.endswith("/v1"):
            base += "/v1"
        return base + "/audio/transcriptions"

    def _transcribe_vllm(self, audio, **kwargs):
        import requests

        session = self.http_session or requests.Session()
        filename, payload, content_type = self._multipart_audio(audio)
        data = {
            "model": self.vllm_model,
            "response_format": "json",
            "temperature": str(kwargs.get("temperature", 0)),
        }
        prompt = kwargs.get("prompt")
        if prompt:
            data["prompt"] = prompt
        max_new_tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        if max_new_tokens is not None:
            data["max_completion_tokens"] = str(max_new_tokens)
        headers = {}
        if self.vllm_api_key and self.vllm_api_key != "EMPTY":
            headers["Authorization"] = "Bearer " + self.vllm_api_key
        response = session.post(
            self._transcriptions_url(),
            data=data,
            files={"file": (filename, payload, content_type)},
            headers=headers,
            timeout=self.vllm_timeout,
        )
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise RuntimeError(
                "vLLM transcription response did not contain a text field"
            )
        return result["text"]

    @staticmethod
    def _multipart_audio(audio):
        if isinstance(audio, np.ndarray):
            import soundfile as sf

            payload = io.BytesIO()
            sf.write(payload, audio, 16000, format="WAV")
            payload.seek(0)
            return "audio.wav", payload, "audio/wav"
        if isinstance(audio, bytes):
            return "audio.wav", io.BytesIO(audio), "audio/wav"
        if isinstance(audio, os.PathLike):
            audio = os.fspath(audio)
        if isinstance(audio, str) and os.path.isfile(audio):
            with open(audio, "rb") as source:
                payload = io.BytesIO(source.read())
            content_type = mimetypes.guess_type(audio)[0] or "application/octet-stream"
            return os.path.basename(audio), payload, content_type
        raise TypeError(
            "vLLM backend accepts a local audio path, bytes, or a 16 kHz numpy array"
        )

    @staticmethod
    def _audio_duration(audio):
        if isinstance(audio, np.ndarray):
            return float(audio.size) / 16000.0
        try:
            import soundfile as sf

            if isinstance(audio, bytes):
                return float(sf.info(io.BytesIO(audio)).duration)
            if isinstance(audio, os.PathLike):
                audio = os.fspath(audio)
            if isinstance(audio, str) and os.path.isfile(audio):
                return float(sf.info(audio).duration)
        except (RuntimeError, TypeError, ValueError):
            pass
        return 0.0
