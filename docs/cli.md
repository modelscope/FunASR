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

`srt` and `tsv` outputs request sentence-level timestamps. In FunASR 1.3.18
and newer, the default `sensevoice` CLI path also loads punctuation for subtitle
generation, so subtitle files are split into sentence cues instead of one
full-text block when the model returns `sentence_info`.

## Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `--model` | `-m` | sensevoice | Model: sensevoice, paraformer, paraformer-en, fun-asr-nano |
| `--hub` | `-H` | ms | Model hub: ms (ModelScope) or hf (Hugging Face) |
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
00:00:00,000 --> 00:00:01,200
第一句。

2
00:00:01,200 --> 00:00:02,600
第二句。
```

If a model does not return sentence-level timestamps, the CLI falls back to one
valid cue spanning the known timestamp or audio duration.

### tsv
Tab-separated values (start/end in seconds):
```
start	end	text
0.000	1.200	第一句。
1.200	2.600	第二句。
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

# Load models from Hugging Face instead of ModelScope
funasr audio.wav --hub hf --model fun-asr-nano

# Use with AI agents
result=$(funasr audio.wav -f json)
echo "$result" | jq -r '.text'
```

## Models

| Model | Languages | Speed | Best for |
|-------|-----------|-------|----------|
| sensevoice | zh/en/ja/ko/yue | ~70ms/10s | CPU-friendly ASR, emotion/audio events |
| paraformer | zh + mixed | ~60ms/10s | Chinese production (with punctuation) |
| paraformer-en | en | ~60ms/10s | English |
| fun-asr-nano | zh/en/ja + Chinese dialects/accents | varies | Encoder+LLM, complex audio |

Language coverage is checkpoint-specific. For example, the separate
Fun-ASR-MLT-Nano checkpoint covers 31 languages, while the default CLI
`fun-asr-nano` choice targets Chinese, English, Japanese, and Chinese dialects
or accents.

## Legacy CLI

The original Hydra-based CLI is available as `funasr-hydra`:
```bash
funasr-hydra ++model=paraformer-zh ++input=audio.wav
```
