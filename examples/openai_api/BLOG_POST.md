# Give Your AI Agent Ears: FunASR as a Drop-in Speech Backend

**TL;DR**: One Python file turns FunASR into an OpenAI-compatible `/v1/audio/transcriptions` endpoint. Any agent framework (LangChain, AutoGen, CrewAI) can use it with zero code changes — just change the base URL.

---

## The Problem

Every voice-enabled AI agent needs speech-to-text. Most developers default to:
- **OpenAI Whisper API** — costs money per minute, data leaves your network
- **Local Whisper** — slow (13x realtime), no speaker diarization built-in
- **Google/Azure STT** — vendor lock-in, complex auth

What if you could get **170x realtime speed**, **50+ languages**, **speaker diarization**, and **emotion detection** — all self-hosted, MIT licensed, and compatible with existing OpenAI SDK code?

## The Solution: FunASR + OpenAI-Compatible Server

```bash
pip install funasr fastapi uvicorn python-multipart
python server.py --model sensevoice --device cuda --port 8000
```

That's it. You now have a local API that speaks the same language as OpenAI.

## Use with Any Agent Framework

### OpenAI SDK
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")
result = client.audio.transcriptions.create(
    model="sensevoice",
    file=open("user_voice.wav", "rb"),
)
print(result.text)
```

### LangChain
```python
# Just override the base_url in your audio chain
transcription = openai_client.audio.transcriptions.create(
    model="sensevoice", file=audio_file
)
# Feed to your agent as context
agent.invoke({"input": transcription.text})
```

### MCP (Claude, Cursor, Windsurf)
```json
{
    "mcpServers": {
        "funasr": {
            "command": "python",
            "args": ["funasr_mcp.py"]
        }
    }
}
```

Now your AI assistant can transcribe any audio file by just asking.

## Why FunASR Over Whisper?

| | FunASR (SenseVoice) | Whisper large-v3 |
|---|---|---|
| Speed | **170x** realtime | 13x realtime |
| Architecture | Non-autoregressive (parallel) | Autoregressive (sequential) |
| Speaker ID | Built-in | Needs pyannote + HF token |
| Emotion | Detects happy/sad/angry | No |
| CPU viable | 17x realtime on CPU | Impractical |
| Cost | Free (MIT) | $0.006/min (API) |

## Available Models

| Model | Best For | Speed |
|-------|----------|-------|
| `sensevoice` | General purpose, emotion | 170x GPU / 17x CPU |
| `paraformer` | Chinese production | 120x GPU / 15x CPU |
| `fun-asr-nano` | 31 languages, LLM-based | 17x GPU |

## Get Started

```bash
git clone https://github.com/modelscope/FunASR
cd FunASR/examples/openai_api
pip install funasr fastapi uvicorn python-multipart
python server.py --device cuda
```

Then point your agent's audio transcription to `http://localhost:8000/v1`.

---

**Links:**
- GitHub: https://github.com/modelscope/FunASR
- Benchmark: https://modelscope.github.io/FunASR/benchmark.html
- Live Demo: https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano-GPU-Debug
- PyPI: `pip install funasr`
