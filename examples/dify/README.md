# FunASR + Dify Integration Guide

Use FunASR as a local speech-to-text backend for [Dify](https://github.com/langgenius/dify) via the OpenAI-compatible API.

## Why FunASR with Dify?

- **Privacy**: All audio stays local, no cloud API calls
- **Speed**: 170x realtime (vs ~1x for cloud APIs)
- **Free**: No per-minute charges, no API keys needed
- **Features**: Speaker diarization, emotion detection, 50+ languages

## Setup

### 1. Start FunASR Server

```bash
pip install torch torchaudio
pip install funasr vllm fastapi uvicorn python-multipart
funasr-server --device cuda --port 8000
```

### 2. Configure in Dify

In Dify Settings -> Model Provider:

1. Select **OpenAI-API-Compatible** provider
2. Set:
   - **API Base URL**: `http://your-server:8000/v1`
   - **API Key**: `unused` (any value works)
   - **Model Name**: `fun-asr-nano`

### 3. Enable Speech-to-Text

In your Dify app settings:
1. Go to Features -> Speech to Text
2. Enable it
3. Select the OpenAI-compatible provider you configured

## Supported Models

| Model Name | Best For | Speed |
|-----------|----------|-------|
| `fun-asr-nano` | zh/en/ja + Chinese dialects/accents | 340x realtime (vLLM) |
| `sensevoice` | Ultra-fast, 5 languages | 170x realtime |
| `paraformer` | Chinese (classic, stable) | 120x realtime |

## Links

- [FunASR GitHub](https://github.com/modelscope/FunASR)
- [Dify GitHub](https://github.com/langgenius/dify)
- [FunASR Website](https://www.funasr.com)
