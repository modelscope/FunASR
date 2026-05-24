# FunASR Subtitle Generator

Generate SRT/VTT subtitles from audio/video files.

## Usage

```bash
# Basic (auto-detect language)
python generate_subtitle.py video.mp4

# With speaker labels
python generate_subtitle.py meeting.wav --spk

# VTT format
python generate_subtitle.py podcast.mp3 --format vtt

# Use specific model
python generate_subtitle.py audio.wav --model paraformer-zh

# CPU mode
python generate_subtitle.py audio.wav --device cpu
```

## Output Example (SRT)

```
1
00:00:00,420 --> 00:00:03,800
[Speaker 0] Let's discuss the Q3 plan.

2
00:00:04,200 --> 00:00:07,100
[Speaker 1] Sounds good. I have three points.
```

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--format` | srt | Output format: srt or vtt |
| `--model` | SenseVoiceSmall | ASR model |
| `--device` | cuda | Device: cuda or cpu |
| `--spk` | off | Add speaker labels |
| `--lang` | auto | Language hint |
| `-o` | input.srt | Output path |

## Install

```bash
pip install funasr
```
