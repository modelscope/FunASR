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
from typing import Optional

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

app = FastAPI(title="FunASR OpenAI-Compatible API")

MODEL_REGISTRY = {}


def get_model(model_name: str):
    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]

    from funasr import AutoModel

    configs = {
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
        "fun-asr-nano": {
            "model": "FunAudioLLM/Fun-ASR-Nano-2512",
            "hub": "hf",
            "trust_remote_code": True,
            "vad_model": "fsmn-vad",
            "vad_kwargs": {"max_single_segment_time": 30000},
        },
    }

    if model_name not in configs:
        model_name = "sensevoice"

    cfg = configs[model_name]
    cfg["device"] = app.state.device
    cfg["disable_update"] = True
    model = AutoModel(**cfg)
    MODEL_REGISTRY[model_name] = model
    return model


@app.post("/v1/audio/transcriptions")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(default="sensevoice"),
    language: Optional[str] = Form(default=None),
    response_format: Optional[str] = Form(default="json"),
    timestamp_granularities: Optional[str] = Form(default=None),
):
    """OpenAI-compatible audio transcription endpoint."""
    suffix = os.path.splitext(file.filename)[1] if file.filename else ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        asr_model = get_model(model)
        t0 = time.time()

        kwargs = {"input": tmp_path, "batch_size": 1}
        if language:
            kwargs["language"] = language

        result = asr_model.generate(**kwargs)
        elapsed = time.time() - t0

        text = result[0]["text"]

        # Strip SenseVoice tags if present
        import re
        text = re.sub(r'<\|[^|]*\|>', '', text).strip()

        if response_format == "verbose_json":
            segments = []
            if "sentence_info" in result[0]:
                for seg in result[0]["sentence_info"]:
                    segments.append({
                        "start": seg.get("start", 0) / 1000.0,
                        "end": seg.get("end", 0) / 1000.0,
                        "text": seg.get("text", ""),
                        "speaker": seg.get("spk", None),
                    })
            return JSONResponse({
                "text": text,
                "segments": segments,
                "language": language or "auto",
                "duration": elapsed,
            })
        else:
            return JSONResponse({"text": text})

    finally:
        os.unlink(tmp_path)


@app.get("/v1/models")
async def list_models():
    """List available models."""
    return JSONResponse({
        "data": [
            {"id": "sensevoice", "object": "model", "owned_by": "funasr"},
            {"id": "paraformer", "object": "model", "owned_by": "funasr"},
            {"id": "fun-asr-nano", "object": "model", "owned_by": "funasr"},
        ]
    })


@app.get("/health")
async def health():
    return {"status": "ok", "models_loaded": list(MODEL_REGISTRY.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model", default="sensevoice", help="Pre-load model")
    args = parser.parse_args()

    app.state.device = args.device
    get_model(args.model)
    print(f"FunASR API server ready on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
