# Command-Line Interface

FunASR provides an agent-friendly CLI for speech recognition from the terminal. Designed for AI agents (Claude Code, Codex, Cursor), shell scripts, and automation pipelines.

## Installation

```bash
pip install funasr
```

## Basic Usage

```bash
# Transcribe audio (simplest)
funasr audio.wav

# Specify model
funasr audio.wav --model paraformer

# JSON output (structured, parseable)
funasr audio.wav --output-format json

# SRT subtitles
funasr audio.wav --output-format srt --output-dir ./subs
```

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | sensevoice | Model: sensevoice, paraformer, paraformer-en, fun-asr-nano |
| `--language` | `-l` | auto | Language: zh, en, ja, ko, yue, auto |
| `--device` | | auto | Device: cuda:0, cpu |
| `--output-format` | `-f` | text | Output: text, json, srt, tsv |
| `--output-dir` | `-o` | stdout | Write output files to directory |
| `--timestamps` | | off | Include word-level timestamps |
| `--spk` | | off | Enable speaker diarization |
| `--hotwords` | | none | Comma-separated hotwords |
| `--verbose` | `-v` | off | Show loading/timing info on stderr |

## Output Formats

### text (default)
Plain transcription text, one result per file. Best for piping:
```bash
funasr audio.wav | wc -w
```

### json
Structured output for programmatic use:
```json
{
  "text": "欢迎大家来体验达摩院推出的语音识别模型",
  "segments": [
    {"start": 0, "end": 5540, "text": "欢迎大家来体验达摩院推出的语音识别模型"}
  ],
  "file": "audio.wav",
  "model": "sensevoice",
  "language": "auto",
  "duration_s": 0.29
}
```

### srt
SubRip subtitle format:
```
1
00:00:00,000 --> 00:00:05,540
欢迎大家来体验达摩院推出的语音识别模型
```

### tsv
Tab-separated values (start/end in seconds):
```
start	end	text
0.000	5.540	欢迎大家来体验达摩院推出的语音识别模型
```

## Advanced Examples

```bash
# Speaker diarization + JSON
funasr meeting.wav --spk --timestamps -f json

# Batch transcribe all WAV files
funasr *.wav --output-format srt --output-dir ./output

# Chinese with hotwords
funasr audio.wav --model paraformer --language zh --hotwords "FunASR,达摩院"

# Pipe to jq for processing
funasr audio.wav -f json | jq '.text'

# Use with AI agents
result=$(funasr audio.wav -f json)
echo "$result" | jq -r '.text'
```

## Models

| Model | Languages | Speed | Best for |
|-------|-----------|-------|----------|
| sensevoice | 50+ | ~70ms/10s | General use, multilingual |
| paraformer | zh + mixed | ~60ms/10s | Chinese production (with punctuation) |
| paraformer-en | en | ~60ms/10s | English |
| fun-asr-nano | 31 | varies | Encoder+LLM, complex audio |

## Legacy CLI

The original Hydra-based CLI is available as `funasr-hydra`:
```bash
funasr-hydra ++model=paraformer-zh ++input=audio.wav
```
