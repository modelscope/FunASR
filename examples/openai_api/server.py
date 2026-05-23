"""
FunASR OpenAI-Compatible API Server

Drop-in replacement for OpenAI's /v1/audio/transcriptions endpoint.
Works with any agent framework that supports OpenAI audio API.

Usage:
    python server.py --model sensevoice --device cuda --port 8000

Then use with any OpenAI-compatible client:
    curl http://localhost:8000/v1/audio/transcriptions \
      -F file=@audio.wav -F model=sensevoice
"""

import argparse
import tempfile
import time
import os
import re
import logging
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="FunASR OpenAI-Compatible API", version="1.0.0")

MODEL_REGISTRY = {}
DEVICE = "cpu"

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


def load_model(model_name: str):
    """Load a model and store in registry."""
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    if model_name not in MODEL_CONFIGS:
        available = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")

    from funasr import AutoModel

    cfg = MODEL_CONFIGS[model_name].copy()
    cfg["device"] = DEVICE
    cfg["disable_update"] = True

    logger.info(f"Loading model '{model_name}' on {DEVICE}...")
    t0 = time.time()
    model = AutoModel(**cfg)
    elapsed = time.time() - t0
    logger.info(f"Model '{model_name}' loaded in {elapsed:.1f}s")

    MODEL_REGISTRY[model_name] = model
    return model


def clean_text(text: str) -> str:
    """Remove SenseVoice special tags from output."""
    return re.sub(r'<\|[^|]*\|>', '', text).strip()


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="sensevoice"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
):
    """
    OpenAI-compatible audio transcription endpoint.
    
    Accepts the same parameters as OpenAI's /v1/audio/transcriptions:
    - file: Audio file (wav, mp3, flac, m4a, ogg, webm)
    - model: Model to use (sensevoice, paraformer, fun-asr-nano)
    - language: Optional language hint
    - response_format: json or verbose_json
    """
    # Validate model
    if model not in MODEL_CONFIGS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found. Available: {list(MODEL_CONFIGS.keys())}"
        )

    # Save uploaded file
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        asr_model = load_model(model)
        t0 = time.time()

        generate_kwargs = {"input": tmp_path, "batch_size": 1}
        if language:
            generate_kwargs["language"] = language

        result = asr_model.generate(**generate_kwargs)
        elapsed = time.time() - t0

        text = clean_text(result[0]["text"])

        if response_format == "verbose_json":
            segments = []
            if "sentence_info" in result[0]:
                for seg in result[0]["sentence_info"]:
                    segments.append({
                        "start": seg.get("start", 0) / 1000.0,
                        "end": seg.get("end", 0) / 1000.0,
                        "text": clean_text(seg.get("text", "")),
                        "speaker": seg.get("spk", None),
                    })
            return JSONResponse({
                "text": text,
                "segments": segments,
                "language": language or "auto",
                "duration": round(elapsed, 3),
                "model": model,
            })
        else:
            return JSONResponse({"text": text})

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    models = []
    for name in MODEL_CONFIGS:
        models.append({
            "id": name,
            "object": "model",
            "created": 1700000000,
            "owned_by": "funasr",
            "ready": name in MODEL_REGISTRY,
        })
    return JSONResponse({"object": "list", "data": models})


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "device": DEVICE,
        "models_loaded": list(MODEL_REGISTRY.keys()),
        "models_available": list(MODEL_CONFIGS.keys()),
    }


def main():
    parser = argparse.ArgumentParser(description="FunASR OpenAI-Compatible API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--device", default="cuda", help="Device: cuda, cpu, mps")
    parser.add_argument("--model", default="sensevoice", help="Pre-load model at startup")
    args = parser.parse_args()

    global DEVICE
    DEVICE = args.device

    # Pre-load default model
    load_model(args.model)

    logger.info(f"FunASR API server starting on http://{args.host}:{args.port}")
    logger.info(f"  Device: {DEVICE}")
    logger.info(f"  Models: {list(MODEL_CONFIGS.keys())}")
    logger.info(f"  Docs:   http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
