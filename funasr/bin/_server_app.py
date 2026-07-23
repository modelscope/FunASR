"""FunASR Server — unified vLLM-based inference service.

Provides OpenAI-compatible API (/v1/audio/transcriptions) and REST API (/asr).
Uses vLLM for Fun-ASR-Nano (GPU) or falls back to AutoModel for non-LLM models (SenseVoice/Paraformer).
"""

import io
import os
import re
import time
import logging
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "funasr-server requires additional packages. Install with: pip install vllm fastapi uvicorn python-multipart"
    )

logger = logging.getLogger("funasr.server")


_LANGUAGE_TAG_RE = re.compile(r"<\|(zh|en|yue|ja|ko)\|>")


def extract_language_from_asr_text(text):
    """Extract a SenseVoice language code before special tokens are removed."""
    if not isinstance(text, str):
        return None
    match = _LANGUAGE_TAG_RE.search(text)
    return match.group(1) if match else None


def resolve_transcription_language(requested_language, result):
    """Prefer the caller's language hint, then backend detection, else unknown."""
    if requested_language and requested_language.strip().lower() != "auto":
        return requested_language
    detected_language = result.get("language")
    if isinstance(detected_language, str) and detected_language:
        return detected_language
    return "unknown"


def _split_text_for_openai_segments(text: str, max_chars: int = 80):
    """Split unsegmented ASR text into readable OpenAI-compatible cues."""
    text = text.strip()
    if not text:
        return []

    words = text.split()
    if not words:
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    parts = []
    current = []
    current_len = 0
    for word in words:
        next_len = len(word) if not current else current_len + 1 + len(word)
        if current and next_len > max_chars:
            parts.append(" ".join(current))
            current = []
            current_len = 0

        current.append(word)
        current_len = len(word) if current_len == 0 else current_len + 1 + len(word)
        if word[-1:] in ".!?;:" and current_len >= max_chars // 2:
            parts.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        parts.append(" ".join(current))

    if any(len(part) > max_chars for part in parts):
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    return parts


def build_openai_fallback_segments(text: str, duration: float, max_chars: int = 80):
    """Build coarse timestamped segments when a backend returns text only."""
    parts = _split_text_for_openai_segments(text, max_chars=max_chars)
    if not parts:
        return []
    if len(parts) == 1 or duration <= 0:
        return [{"start": 0.0, "end": max(float(duration), 0.0), "text": parts[0]}]

    total_chars = sum(len(part) for part in parts)
    if total_chars <= 0:
        return [{"start": 0.0, "end": float(duration), "text": text.strip()}]

    segments = []
    consumed = 0
    previous_end = 0.0
    for i, part in enumerate(parts):
        consumed += len(part)
        end = float(duration) if i == len(parts) - 1 else float(duration) * consumed / total_chars
        end = max(end, previous_end)
        segments.append({"start": round(previous_end, 3), "end": round(end, 3), "text": part})
        previous_end = end

    return segments


def prepare_audio_for_inference(audio_data, sr, target_sr=16000):
    """Return mono float32 audio at target_sr for ASR inference."""
    audio_data = np.asarray(audio_data)
    if audio_data.ndim > 1:
        channel_axis = -1 if audio_data.shape[-1] <= audio_data.shape[0] else 0
        audio_data = audio_data.mean(axis=channel_axis)

    if sr != target_sr:
        import librosa
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    return audio_data.astype(np.float32), sr

def create_app(device: str = "cuda", preload_model: str = "auto", model_path: str = None, hub: str = "ms") -> FastAPI:
    if preload_model == "auto":
        preload_model = "fun-asr-nano" if device.startswith("cuda") else "sensevoice"

    app = FastAPI(title="FunASR Server", version="1.3.6")
    app.state.device = device
    app.state.engine = None
    app.state.vad_model = None
    app.state.fallback_models = {}
    app.state.model_path = model_path
    app.state.hub = hub

    # Non-LLM model configs (use AutoModel, no vLLM)
    FALLBACK_CONFIGS = {
        "sensevoice": {
            "model": "iic/SenseVoiceSmall",
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
        "paraformer": {
            "model": "paraformer-zh",
            "vad_model": "fsmn-vad",
            "punc_model": "ct-punc",
        },
    }

    def _load_vllm_engine():
        """Load Fun-ASR-Nano vLLM engine. Falls back to AutoModel if vLLM unavailable."""
        if app.state.engine is not None or "fun-asr-nano" in app.state.fallback_models:
            return
        try:
            from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
            from funasr import AutoModel as _AutoModel

            logger.info("Loading Fun-ASR-Nano vLLM engine...")
            t0 = time.time()
            # Use custom model_path if provided, otherwise default. In both
            # cases, honor the server-level hub selection.
            vllm_model = app.state.model_path if app.state.model_path else "FunAudioLLM/Fun-ASR-Nano-2512"
            vllm_hub = app.state.hub
            engine = FunASRNanoVLLM.from_pretrained(
                model=vllm_model,
                hub=vllm_hub,
                device=device,
                dtype="bf16",
                max_model_len=4096,
                gpu_memory_utilization=0.5,
            )
            logger.info(f"vLLM engine ready in {time.time()-t0:.1f}s")

            logger.info("Loading VAD model...")
            vad_model = _AutoModel(model="fsmn-vad", device=device, disable_update=True)
            app.state.engine = engine
            app.state.vad_model = vad_model
            app.state.use_vllm = True
            logger.info("VAD ready.")
        except Exception as e:
            logger.warning(f"vLLM failed ({e}), falling back to AutoModel for fun-asr-nano")
            app.state.use_vllm = False
            from funasr import AutoModel
            cfg = {
                "model": app.state.model_path if app.state.model_path else "FunAudioLLM/Fun-ASR-Nano-2512",
                "hub": app.state.hub,
                "trust_remote_code": True,
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "device": device,
                "disable_update": True,
            }
            app.state.fallback_models["fun-asr-nano"] = AutoModel(**cfg)
            logger.info(f"Fallback AutoModel loaded for fun-asr-nano with model={cfg['model']}, hub={cfg['hub']}.")

    def _load_fallback(name: str):
        """Load non-LLM model via AutoModel."""
        if name in app.state.fallback_models:
            return app.state.fallback_models[name]
        if name not in FALLBACK_CONFIGS and not app.state.model_path:
            return None
        from funasr import AutoModel
        cfg = FALLBACK_CONFIGS.get(name, {}).copy()
        # Override with custom model_path and hub if provided
        if app.state.model_path:
            cfg["model"] = app.state.model_path
            cfg["hub"] = app.state.hub
        elif app.state.hub:
            cfg["hub"] = app.state.hub
        cfg["device"] = device
        cfg["disable_update"] = True
        logger.info(f"Loading fallback model '{name}' with model={cfg['model']}, hub={cfg['hub']}...")
        model = AutoModel(**cfg)
        app.state.fallback_models[name] = model
        return model

    def _process_vllm(audio_data, sr, language=None, hotwords=None, use_spk=False):
        """Process audio with vLLM engine (Fun-ASR-Nano)."""
        audio_data, sr = prepare_audio_for_inference(audio_data, sr)

        # VAD
        vad_res = app.state.vad_model.generate(input=audio_data, fs=sr)
        segments = vad_res[0]["value"] if vad_res and vad_res[0].get("value") else [[0, int(len(audio_data)*1000/sr)]]

        seg_audios = []
        seg_times = []
        for seg in segments:
            s0 = int(seg[0] * sr / 1000)
            s1 = int(seg[1] * sr / 1000)
            seg_audio = audio_data[s0:s1]
            if len(seg_audio) > sr * 0.3:
                seg_audios.append(seg_audio)
                seg_times.append((seg[0], seg[1]))

        if not seg_audios:
            return {"text": "", "segments": [], "duration": len(audio_data)/sr}

        # repetition_penalty is left at the neutral 1.0: the Fun-ASR-Nano vLLM
        # engine runs in prompt-embeds mode, where any other value crashes the
        # CUDA kernel (see issue #2948 and fun_asr_nano.vllm_utils).
        gen_kwargs = {"max_new_tokens": 500, "repetition_penalty": 1.0}
        if language:
            gen_kwargs["language"] = language
        if hotwords:
            gen_kwargs["hotwords"] = hotwords

        results = app.state.engine.generate(inputs=seg_audios, **gen_kwargs)

        output_segments = []
        full_text_parts = []
        for r, (start_ms, end_ms) in zip(results, seg_times):
            text = r["text"]
            seg_info = {"text": text, "start": start_ms/1000, "end": end_ms/1000}
            if "timestamps" in r:
                offset = start_ms / 1000
                seg_info["words"] = [
                    {"word": ts["token"], "start": ts["start_time"]+offset, "end": ts["end_time"]+offset}
                    for ts in r["timestamps"]
                ]
            output_segments.append(seg_info)
            full_text_parts.append(text)

        return {
            "text": "".join(full_text_parts),
            "segments": output_segments,
            "duration": len(audio_data) / sr,
        }

    def _process_fallback(model_name, audio_path, language=None):
        """Process with non-LLM model (SenseVoice/Paraformer)."""
        model = _load_fallback(model_name)
        try:
            duration = float(sf.info(audio_path).duration)
        except Exception:
            duration = 0.0
        kwargs = {"input": audio_path, "batch_size": 1}
        if language:
            kwargs["language"] = language
        result = model.generate(**kwargs)
        raw_text = result[0]["text"]
        detected_language = extract_language_from_asr_text(raw_text)
        text = re.sub(r'<\|[^|]*\|>', '', raw_text).strip()
        segments = []
        if "sentence_info" in result[0]:
            for s in result[0]["sentence_info"]:
                segments.append({
                    "start": s.get("start", 0)/1000,
                    "end": s.get("end", 0)/1000,
                    "text": re.sub(r'<\|[^|]*\|>', '', s.get("text", "")).strip(),
                    "speaker": s.get("spk"),
                })
        if not segments and text:
            segments = build_openai_fallback_segments(text, duration)
        return {
            "text": text,
            "segments": segments,
            "duration": duration,
            "language": detected_language,
        }

    # Pre-load
    if app.state.model_path:
        # When custom model_path is provided, use it as the model name for loading
        logger.info(f"Loading custom model: {app.state.model_path} (hub: {app.state.hub})")
        _load_fallback("custom")
    elif preload_model == "fun-asr-nano":
        _load_vllm_engine()
    else:
        _load_fallback(preload_model)

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(default="fun-asr-nano"),
        language: Optional[str] = Form(default=None),
        response_format: Optional[str] = Form(default="json"),
        spk: bool = Form(default=False),
    ):
        content = await file.read()
        t0 = time.perf_counter()

        if model == "fun-asr-nano":
            _load_vllm_engine()
            if app.state.use_vllm:
                audio_data, sr = sf.read(io.BytesIO(content))
                result = _process_vllm(audio_data, sr, language=language, use_spk=spk)
            else:
                suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(content)
                    tmp_path = tmp.name
                try:
                    result = _process_fallback("fun-asr-nano", tmp_path, language=language)
                finally:
                    os.unlink(tmp_path)
        elif model in FALLBACK_CONFIGS or model == "custom":
            suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                result = _process_fallback(model, tmp_path, language=language)
            finally:
                os.unlink(tmp_path)
        else:
            available = ["fun-asr-nano", "custom"] + list(FALLBACK_CONFIGS.keys())
            raise HTTPException(400, f"Unknown model '{model}'. Available: {', '.join(available)}")

        t1 = time.perf_counter()

        if response_format == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": resolve_transcription_language(language, result),
                "duration": result.get("duration", 0),
                "text": result["text"],
                "segments": [
                    {"id": i, "start": s["start"], "end": s["end"], "text": s["text"], "words": s.get("words", [])}
                    for i, s in enumerate(result["segments"])
                ],
            })
        elif response_format == "text":
            return JSONResponse(result["text"])
        else:
            return JSONResponse({"text": result["text"]})

    @app.post("/asr")
    async def asr_endpoint(
        file: UploadFile = File(...),
        language: Optional[str] = Form(default=None),
        hotwords: str = Form(default=""),
        spk: bool = Form(default=False),
    ):
        """Full-featured ASR endpoint with timestamps and speaker diarization."""
        content = await file.read()
        _load_vllm_engine()
        hw_list = [w.strip() for w in hotwords.split(",") if w.strip()] if hotwords else None

        t0 = time.perf_counter()
        if app.state.use_vllm:
            audio_data, sr = sf.read(io.BytesIO(content))
            result = _process_vllm(audio_data, sr, language=language, hotwords=hw_list, use_spk=spk)
        else:
            suffix = ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                result = _process_fallback("fun-asr-nano", tmp_path, language=language)
            finally:
                os.unlink(tmp_path)
        t1 = time.perf_counter()

        result["processing_time"] = round(t1 - t0, 3)
        result["rtf"] = round((t1 - t0) / result["duration"], 4) if result.get("duration", 0) > 0 else 0
        return JSONResponse(result)

    @app.get("/v1/models")
    async def list_models():
        all_models = ["fun-asr-nano"] + list(FALLBACK_CONFIGS.keys())
        if app.state.model_path:
            all_models.append("custom")
        return JSONResponse({"object": "list", "data": [{"id": n, "object": "model"} for n in all_models]})

    @app.get("/health")
    async def health():
        loaded = []
        if app.state.engine is not None:
            loaded.append("fun-asr-nano (vLLM)")
        loaded.extend(app.state.fallback_models.keys())
        return {"status": "ok", "device": device, "models_loaded": loaded}

    return app
