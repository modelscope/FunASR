"""Internal: FastAPI app for funasr-server command."""

import tempfile
import time
import os
import re
import logging
from typing import Optional

try:
    from fastapi import FastAPI, UploadFile, File, Form, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:
    raise ImportError(
        "funasr-server requires fastapi. Install with: pip install fastapi uvicorn python-multipart"
    )

logger = logging.getLogger("funasr.server")

MODEL_CONFIGS = {
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
    "paraformer-en": {
        "model": "paraformer-en",
        "vad_model": "fsmn-vad",
    },
    "fun-asr-nano": {
        "model": "FunAudioLLM/Fun-ASR-Nano-2512",
        "hub": "hf",
        "trust_remote_code": True,
        "vad_model": "fsmn-vad",
        "vad_kwargs": {"max_single_segment_time": 30000},
    },
}


def create_app(device: str = "cuda", preload_model: str = "auto") -> FastAPI:
    if preload_model == "auto":
        preload_model = "fun-asr-nano" if device.startswith("cuda") else "sensevoice"
    app = FastAPI(title="FunASR Server", version="1.3.2")
    app.state.models = {}
    app.state.device = device

    def _load_model(name: str):
        if name in app.state.models:
            return app.state.models[name]
        if name not in MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_CONFIGS.keys())}")
        from funasr import AutoModel
        cfg = MODEL_CONFIGS[name].copy()
        cfg["device"] = app.state.device
        cfg["disable_update"] = True
        logger.info(f"Loading '{name}' on {device}...")
        t0 = time.time()
        model = AutoModel(**cfg)
        logger.info(f"'{name}' ready in {time.time()-t0:.1f}s")
        app.state.models[name] = model
        return model

    # Pre-load
    _load_model(preload_model)

    @app.post("/v1/audio/transcriptions")
    async def transcribe(
        file: UploadFile = File(...),
        model: str = Form(default="sensevoice"),
        language: Optional[str] = Form(default=None),
        response_format: Optional[str] = Form(default="json"),
    ):
        if model not in MODEL_CONFIGS:
            raise HTTPException(400, f"Unknown model '{model}'. Available: {list(MODEL_CONFIGS.keys())}")
        suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        try:
            asr = _load_model(model)
            kwargs = {"input": tmp_path, "batch_size": 1}
            if language:
                kwargs["language"] = language
            result = asr.generate(**kwargs)
            text = re.sub(r'<\|[^|]*\|>', '', result[0]["text"]).strip()
            if response_format == "verbose_json":
                segments = []
                if "sentence_info" in result[0]:
                    for s in result[0]["sentence_info"]:
                        segments.append({"start": s.get("start",0)/1000, "end": s.get("end",0)/1000, "text": re.sub(r'<\|[^|]*\|>','',s.get("text","")).strip(), "speaker": s.get("spk")})
                return JSONResponse({"text": text, "segments": segments, "model": model})
            return JSONResponse({"text": text})
        except Exception as e:
            raise HTTPException(500, str(e))
        finally:
            os.unlink(tmp_path)

    @app.get("/v1/models")
    async def list_models():
        return JSONResponse({"object": "list", "data": [{"id": n, "object": "model", "owned_by": "funasr"} for n in MODEL_CONFIGS]})

    @app.get("/health")
    async def health():
        return {"status": "ok", "device": app.state.device, "models_loaded": list(app.state.models.keys())}

    return app
