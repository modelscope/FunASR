"""
FunASR MCP Server

Model Context Protocol server that exposes FunASR speech recognition
as a tool for AI assistants (Claude, Cursor, etc).

Usage:
    python funasr_mcp.py

Add to claude_desktop_config.json:
{
    "mcpServers": {
        "funasr": {
            "command": "python",
            "args": ["path/to/funasr_mcp.py"]
        }
    }
}
"""

import json
import sys
import os
import tempfile
import base64

# MCP protocol over stdio
def send_response(id, result):
    msg = {"jsonrpc": "2.0", "id": id, "result": result}
    out = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(out)}\r\n\r\n{out}")
    sys.stdout.flush()


def send_notification(method, params=None):
    msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    out = json.dumps(msg)
    sys.stdout.write(f"Content-Length: {len(out)}\r\n\r\n{out}")
    sys.stdout.flush()


_model = None


def get_model():
    global _model
    if _model is None:
        from funasr import AutoModel
        device = os.environ.get("FUNASR_DEVICE", "cpu")
        _model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
    return _model


def transcribe(audio_path: str, language: str = "auto") -> dict:
    """Transcribe an audio file to text."""
    import re
    model = get_model()
    result = model.generate(input=audio_path, batch_size=1)
    text = result[0]["text"]
    text = re.sub(r'<\|[^|]*\|>', '', text).strip()

    response = {"text": text}
    if "sentence_info" in result[0]:
        response["segments"] = [
            {
                "text": seg.get("text", ""),
                "start": seg.get("start", 0) / 1000.0,
                "end": seg.get("end", 0) / 1000.0,
                "speaker": seg.get("spk", None),
            }
            for seg in result[0]["sentence_info"]
        ]
    return response


def handle_request(request):
    method = request.get("method")
    id = request.get("id")
    params = request.get("params", {})

    if method == "initialize":
        send_response(id, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "funasr", "version": "1.3.2"},
        })
    elif method == "tools/list":
        send_response(id, {
            "tools": [
                {
                    "name": "transcribe_audio",
                    "description": "Transcribe speech audio to text. Supports 50+ languages, auto-detection, speaker diarization. Input: file path to audio.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "Path to audio file (wav, mp3, flac, etc)"
                            },
                            "language": {
                                "type": "string",
                                "description": "Language hint (optional, auto-detected by default)",
                                "default": "auto"
                            }
                        },
                        "required": ["audio_path"]
                    }
                }
            ]
        })
    elif method == "tools/call":
        tool_name = params.get("name")
        args = params.get("arguments", {})

        if tool_name == "transcribe_audio":
            audio_path = args.get("audio_path", "")
            language = args.get("language", "auto")

            if not os.path.exists(audio_path):
                send_response(id, {
                    "content": [{"type": "text", "text": f"Error: file not found: {audio_path}"}],
                    "isError": True
                })
                return

            result = transcribe(audio_path, language)
            text_output = f"Transcription: {result['text']}"
            if "segments" in result:
                text_output += "\n\nSegments:"
                for seg in result["segments"]:
                    spk = f" [Speaker {seg['speaker']}]" if seg.get('speaker') is not None else ""
                    text_output += f"\n  [{seg['start']:.1f}s - {seg['end']:.1f}s]{spk} {seg['text']}"

            send_response(id, {
                "content": [{"type": "text", "text": text_output}]
            })
        else:
            send_response(id, {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True
            })
    elif method == "notifications/initialized":
        pass  # Client confirmed initialization
    else:
        if id is not None:
            send_response(id, {})


def main():
    """Run MCP server over stdio."""
    import re
    buffer = ""
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        buffer += line
        if "\r\n\r\n" in buffer:
            header, body_start = buffer.split("\r\n\r\n", 1)
            match = re.search(r"Content-Length: (\d+)", header)
            if match:
                length = int(match.group(1))
                while len(body_start) < length:
                    body_start += sys.stdin.read(length - len(body_start))
                request = json.loads(body_start[:length])
                buffer = body_start[length:]
                handle_request(request)
            else:
                buffer = ""


if __name__ == "__main__":
    main()
