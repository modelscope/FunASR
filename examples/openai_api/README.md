# FunASR OpenAI-Compatible API Server

Drop-in replacement for OpenAI's `/v1/audio/transcriptions` endpoint. Works with **any agent framework** that supports OpenAI audio API.

## Quick Start

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

## Usage with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

result = client.audio.transcriptions.create(
    model="sensevoice",  # or "paraformer", "fun-asr-nano"
    file=open("meeting.wav", "rb"),
)
print(result.text)
```

## Usage with curl

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=sensevoice
```

## Available Models

| Model | Speed | Languages | Features |
|-------|-------|-----------|----------|
| `sensevoice` | 170x realtime | 5 | Emotion detection |
| `paraformer` | 120x realtime | zh/en | Punctuation, streaming |
| `fun-asr-nano` | 17x realtime | 31 | LLM-based, timestamps |

## Agent Framework Integration

Works with: LangChain, LlamaIndex, AutoGen, CrewAI, Semantic Kernel, or any framework supporting OpenAI audio API.

```python
# LangChain example
from langchain_openai import ChatOpenAI
# Just point your audio transcription to localhost:8000
```
