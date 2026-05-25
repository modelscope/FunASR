# Give Your AI Agent Ears: FunASR as a Drop-in Speech Backend

**TL;DR**: `funasr-server` turns FunASR into an OpenAI-compatible `/v1/audio/transcriptions` endpoint. Agent frameworks such as LangChain, AutoGen, CrewAI, Dify, and MCP-based assistants can use it by changing the base URL.

---

## The Problem

Every voice-enabled AI agent needs speech-to-text. Most developers default to:

- **OpenAI Whisper API** - convenient, but paid per minute and sends audio to a hosted service
- **Local Whisper** - self-hosted, but slower and does not include speaker diarization by default
- **Google/Azure STT** - mature, but adds vendor lock-in and service-specific authentication

What if you could get **170x realtime speed**, **50+ languages**, **speaker diarization**, **emotion detection**, and **private deployment** while keeping OpenAI SDK compatibility?

## The Solution: FunASR + OpenAI-Compatible Server

```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda --port 8000
```

That is it. You now have a local speech API at `http://localhost:8000/v1`.

## Verify It in 60 Seconds

In another terminal, use the bundled smoke test:

```bash
git clone https://github.com/modelscope/FunASR
cd FunASR/examples/openai_api
bash smoke_test.sh
# Cross-platform alternative:
python smoke_test.py
```

Or run the equivalent commands manually:

```bash
curl -L https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav -o sample.wav
curl http://localhost:8000/v1/audio/transcriptions \
  -F file=@sample.wav \
  -F model=sensevoice \
  -F response_format=verbose_json
```

The response includes `text`; with `verbose_json`, supported models can also return segment-level metadata.

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
# Just override the base_url in your audio chain.
transcription = openai_client.audio.transcriptions.create(
    model="sensevoice",
    file=audio_file,
)
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

Now your AI assistant can transcribe local audio files while keeping the audio inside your environment.

## Why FunASR Over Whisper?

| | FunASR (SenseVoice) | Whisper large-v3 |
|---|---|---|
| Speed | **170x** realtime | 13x realtime |
| Architecture | Non-autoregressive (parallel) | Autoregressive (sequential) |
| Speaker ID | Built-in | Needs pyannote + HF token |
| Emotion | Detects happy/sad/angry | No |
| CPU viable | 17x realtime on CPU | Impractical |
| Cost | Free (MIT) | $0.006/min (API) |
| Deployment | Self-hosted API server | Local model or hosted API |

## Available Models

| Model | Best For | Speed |
|-------|----------|-------|
| `sensevoice` | General purpose, emotion | 170x GPU / 17x CPU |
| `paraformer` | Chinese production | 120x GPU / 15x CPU |
| `paraformer-en` | English production | 120x GPU / 15x CPU |
| `fun-asr-nano` | 31 languages, LLM-based | 17x GPU |

## Get Started

```bash
pip install funasr fastapi uvicorn python-multipart
funasr-server --model sensevoice --device cuda
```

Then point your agent's audio transcription client to `http://localhost:8000/v1`.

---

**Links:**

- GitHub: https://github.com/modelscope/FunASR
- OpenAI API example: https://github.com/modelscope/FunASR/tree/main/examples/openai_api
- Agent integration: https://modelscope.github.io/FunASR/agent.html
- Benchmark: https://modelscope.github.io/FunASR/benchmark.html
- Live demo: https://huggingface.co/spaces/FunAudioLLM/Fun-ASR-Nano-GPU-Debug
- PyPI: `pip install funasr`
