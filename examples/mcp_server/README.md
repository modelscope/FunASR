# FunASR MCP Server

[Model Context Protocol](https://modelcontextprotocol.io/) server that gives AI assistants local audio transcription with SenseVoiceSmall by default.

## Setup

### 1. Install dependencies

```bash
pip install funasr
```

### 2. Optional: run with Docker

The Dockerfile starts the MCP server over stdio and is suitable for MCP directory
checks that initialize the server and call `tools/list`.

```bash
docker build -t funasr-mcp examples/mcp_server
docker run --rm -i \
  -e FUNASR_DEVICE=cpu \
  -v /path/to/audio:/audio:ro \
  funasr-mcp
```

When submitting this server to MCP directories such as Glama, use this folder as
the Docker build context so the container entrypoint runs `funasr_mcp.py`.
The repository root `glama.json` declares GitHub maintainer ownership for Glama,
while the `glama.json` file in this directory declares the container command and
metadata for directory scanners.

### Official MCP Registry checklist

The Dockerfile includes the OCI ownership label expected by the official MCP
Registry:

```dockerfile
LABEL io.modelcontextprotocol.server.name="io.github.modelscope/funasr-mcp"
```

Before publishing, push a public OCI image (for example to GHCR) and create a
matching `server.json` whose `name` is `io.github.modelscope/funasr-mcp` and
whose package identifier points at that image tag. The Registry verifies that
the Docker/OCI label and `server.json` name match.

### Glama submission checklist

Use these values when adding the server at <https://glama.ai/mcp/servers>:

| Field | Value |
|------|-------|
| Repository URL | <https://github.com/modelscope/FunASR> |
| Docker build context | `examples/mcp_server` |
| Dockerfile path | `examples/mcp_server/Dockerfile` |
| Server command | `python /app/funasr_mcp.py` |
| Expected MCP tool | `transcribe_audio` |

After Glama finishes evaluation, verify that the score badge endpoint returns
success before adding it to directory PRs:

```markdown
[![modelscope/FunASR MCP server](https://glama.ai/mcp/servers/modelscope/FunASR/badges/score.svg)](https://glama.ai/mcp/servers/modelscope/FunASR)
```

If the badge endpoint still returns 404, keep the badge out of external
directory submissions until the Glama listing is live.

### Directory listings

The FunASR MCP server is listed on mcp.so:

- <https://mcp.so/server/mcp-server-funasr/radial-hks>

### 3. Configure your AI tool

**Claude Code** (`~/.claude.json`):
```json
{
    "mcpServers": {
        "funasr": {
            "command": "python",
            "args": ["/path/to/examples/mcp_server/funasr_mcp.py"],
            "env": {"FUNASR_DEVICE": "cuda"}
        }
    }
}
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{
    "mcpServers": {
        "funasr": {
            "command": "python",
            "args": ["/path/to/funasr_mcp.py"],
            "env": {"FUNASR_DEVICE": "cpu"}
        }
    }
}
```

**Cursor** (Settings → MCP Servers → Add):
- Command: `python /path/to/funasr_mcp.py`
- Environment: `FUNASR_DEVICE=cuda`

## Tools

### `transcribe_audio`

Transcribe a speech audio file to text.

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| `audio_path` | string | Yes | Path to audio file (wav, mp3, flac, m4a, ogg) |
| `language` | string | No | `auto`, `zh`, `yue`, `en`, `ja`, or `ko` (default: `auto`) |

**Returns:** Transcribed text with per-segment timestamps when the model returns them.

## Example Usage

Once configured, ask your AI assistant:

- "Transcribe the meeting recording at ~/Downloads/meeting.wav"
- "What was said in this audio file? /path/to/interview.mp3"
- "Convert this voice memo to text: ~/voice_note.m4a"

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FUNASR_DEVICE` | `cpu` | Device: `cuda`, `cpu`, or `mps` |
| `FUNASR_MODEL` | `iic/SenseVoiceSmall` | Model name or local model path passed to `AutoModel` |

## Features

- **Five-language transcription** — Mandarin, Cantonese, English, Japanese, and Korean
- **Automatic detection or explicit hints** — `auto`, `zh`, `yue`, `en`, `ja`, and `ko`
- **VAD segmentation** — splits longer audio before recognition
- **Optional segment timestamps** — included only when the configured model returns them
- **Configurable local inference** — choose the model and CPU, CUDA, or MPS with environment variables
- **No API key needed** — fully local inference
- MIT licensed, privacy-friendly (audio never leaves your machine)

## Verified Compatibility

| Tool | Status |
|------|--------|
| Claude Code | ✅ Tested |
| Claude Desktop | ✅ Compatible |
| Cursor | ✅ Compatible |
| Windsurf | ✅ Compatible |
| Any MCP client | ✅ Standard protocol |
