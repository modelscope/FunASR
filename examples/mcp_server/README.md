# FunASR MCP Server

[Model Context Protocol](https://modelcontextprotocol.io/) server for FunASR.
Enables AI assistants (Claude, Cursor, Windsurf, etc.) to transcribe audio.

## Setup

```bash
pip install funasr
```

Add to your MCP config (e.g. `~/.claude.json` or `claude_desktop_config.json`):

```json
{
    "mcpServers": {
        "funasr": {
            "command": "python",
            "args": ["/path/to/funasr_mcp.py"],
            "env": {
                "FUNASR_DEVICE": "cuda"
            }
        }
    }
}
```

## Available Tools

### `transcribe_audio`

Transcribe speech audio to text.

**Input:**
- `audio_path` (required): Path to audio file (wav, mp3, flac, etc.)
- `language` (optional): Language hint, auto-detected by default

**Output:** Transcribed text with optional speaker labels and timestamps.

## Example

Once configured, you can ask your AI assistant:

> "Transcribe the audio file at /path/to/meeting.wav"

The assistant will use the `transcribe_audio` tool and return the transcription with speaker labels.

## Features

- 50+ languages with auto-detection
- Speaker diarization (who said what)
- Timestamps per segment
- 170x realtime on GPU, 17x on CPU
