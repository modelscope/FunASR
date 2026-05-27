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


def create_app(device: str = "cuda", preload_model: str = "auto") -> FastAPI:
    if preload_model == "auto":
        preload_model = "fun-asr-nano" if device.startswith("cuda") else "sensevoice"

    app = FastAPI(title="FunASR Server", version="1.3.6")
    app.state.device = device
    app.state.engine = None
    app.state.vad_model = None
    app.state.fallback_models = {}

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
        if app.state.engine is not None:
            return
        try:
            from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
            from funasr import AutoModel as _AutoModel

            logger.info("Loading Fun-ASR-Nano vLLM engine...")
            t0 = time.time()
            app.state.engine = FunASRNanoVLLM.from_pretrained(
                model="FunAudioLLM/Fun-ASR-Nano-2512",
                hub="hf",
                device=device,
                dtype="bf16",
                max_model_len=4096,
                gpu_memory_utilization=0.5,
            )
            logger.info(f"vLLM engine ready in {time.time()-t0:.1f}s")
            app.state.use_vllm = True

            logger.info("Loading VAD model...")
            app.state.vad_model = _AutoModel(model="fsmn-vad", device=device, disable_update=True)
            logger.info("VAD ready.")
        except Exception as e:
            logger.warning(f"vLLM failed ({e}), falling back to AutoModel for fun-asr-nano")
            app.state.use_vllm = False
            from funasr import AutoModel
            cfg = {
                "model": "FunAudioLLM/Fun-ASR-Nano-2512",
                "hub": "hf",
                "trust_remote_code": True,
                "vad_model": "fsmn-vad",
                "vad_kwargs": {"max_single_segment_time": 30000},
                "device": device,
                "disable_update": True,
            }
            app.state.fallback_models["fun-asr-nano"] = AutoModel(**cfg)
            logger.info("Fallback AutoModel loaded for fun-asr-nano.")

    def _load_fallback(name: str):
        """Load non-LLM model via AutoModel."""
        if name in app.state.fallback_models:
            return app.state.fallback_models[name]
        if name not in FALLBACK_CONFIGS:
            return None
        from funasr import AutoModel
        cfg = FALLBACK_CONFIGS[name].copy()
        cfg["device"] = device
        cfg["disable_update"] = True
        logger.info(f"Loading fallback model '{name}'...")
        model = AutoModel(**cfg)
        app.state.fallback_models[name] = model
        return model

    def _process_vllm(audio_data, sr, language=None, hotwords=None, use_spk=False):
        """Process audio with vLLM engine (Fun-ASR-Nano)."""
        if sr != 16000:
            import librosa
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
            sr = 16000
        if audio_data.ndim > 1:
            audio_data = audio_data[:, 0]
        audio_data = audio_data.astype(np.float32)

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

        # vLLM generate with repetition_penalty
        gen_kwargs = {"max_new_tokens": 500, "repetition_penalty": 1.3}
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
        kwargs = {"input": audio_path, "batch_size": 1}
        if language:
            kwargs["language"] = language
        result = model.generate(**kwargs)
        text = re.sub(r'<\|[^|]*\|>', '', result[0]["text"]).strip()
        segments = []
        if "sentence_info" in result[0]:
            for s in result[0]["sentence_info"]:
                segments.append({
                    "start": s.get("start", 0)/1000,
                    "end": s.get("end", 0)/1000,
                    "text": re.sub(r'<\|[^|]*\|>', '', s.get("text", "")).strip(),
                    "speaker": s.get("spk"),
                })
        return {"text": text, "segments": segments}

    # Pre-load
    if preload_model == "fun-asr-nano":
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
        elif model in FALLBACK_CONFIGS:
            suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            try:
                result = _process_fallback(model, tmp_path, language=language)
            finally:
                os.unlink(tmp_path)
        else:
            raise HTTPException(400, f"Unknown model '{model}'. Available: fun-asr-nano, {', '.join(FALLBACK_CONFIGS.keys())}")

        t1 = time.perf_counter()

        if response_format == "verbose_json":
            return JSONResponse({
                "task": "transcribe",
                "language": language or "zh",
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
        return JSONResponse({"object": "list", "data": [{"id": n, "object": "model"} for n in all_models]})

    @app.get("/health")
    async def health():
        loaded = []
        if app.state.engine is not None:
            loaded.append("fun-asr-nano (vLLM)")
        loaded.extend(app.state.fallback_models.keys())
        return {"status": "ok", "device": device, "models_loaded": loaded}

    return app
