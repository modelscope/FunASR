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
import os
import re
import sys
from importlib.util import find_spec
from pathlib import Path


DEFAULT_MODEL = "iic/SenseVoiceSmall"
SUPPORTED_LANGUAGES = ("auto", "zh", "yue", "en", "ja", "ko")


def get_server_version():
    package_dirs = []
    package_spec = find_spec("funasr")
    if package_spec is not None and package_spec.submodule_search_locations is not None:
        package_dirs.extend(package_spec.submodule_search_locations)
    package_dirs.append(Path(__file__).resolve().parent.parent.parent / "funasr")

    for package_dir in package_dirs:
        version_file = Path(package_dir) / "version.txt"
        try:
            version = version_file.read_text().strip()
        except OSError:
            continue
        if version:
            return version
    return "unknown"


# MCP protocol over stdio
def send_response(id, result):
    msg = {"jsonrpc": "2.0", "id": id, "result": result}
    out = json.dumps(msg)
    sys.stdout.write(f"{out}\n")
    sys.stdout.flush()


def send_tool_error(id, message):
    send_response(
        id,
        {
            "content": [{"type": "text", "text": f"Error: {message}"}],
            "isError": True,
        },
    )


_model = None


def get_model():
    global _model
    if _model is None:
        from funasr import AutoModel

        model_name = os.environ.get("FUNASR_MODEL") or DEFAULT_MODEL
        device = os.environ.get("FUNASR_DEVICE") or "cpu"
        _model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            device=device,
            disable_update=True,
        )
    return _model


def transcribe(audio_path: str, language: str = "auto") -> dict:
    """Transcribe an audio file to text."""
    model = get_model()
    result = model.generate(input=audio_path, batch_size=1, language=language)
    text = result[0]["text"]
    text = re.sub(r"<\|[^|]*\|>", "", text).strip()

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
            "serverInfo": {"name": "funasr", "version": get_server_version()},
        })
    elif method == "tools/list":
        send_response(id, {
            "tools": [
                {
                    "name": "transcribe_audio",
                    "description": "Transcribe local speech audio with SenseVoiceSmall. Supports automatic or explicit Mandarin, Cantonese, English, Japanese, and Korean recognition with VAD segmentation.",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "audio_path": {
                                "type": "string",
                                "description": "Path to audio file (wav, mp3, flac, etc)"
                            },
                            "language": {
                                "type": "string",
                                "description": "Language hint: auto, zh, yue, en, ja, or ko",
                                "enum": list(SUPPORTED_LANGUAGES),
                                "default": "auto",
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
            audio_path = args.get("audio_path")
            language = args.get("language", "auto")

            if not isinstance(audio_path, str) or not audio_path.strip():
                send_tool_error(id, "audio_path is required")
                return

            if language not in SUPPORTED_LANGUAGES:
                supported = ", ".join(SUPPORTED_LANGUAGES)
                send_tool_error(
                    id,
                    f"unsupported language '{language}'; choose one of: {supported}",
                )
                return

            audio_path = os.path.expanduser(audio_path)
            if not os.path.isfile(audio_path):
                send_tool_error(id, f"file not found: {audio_path}")
                return

            try:
                result = transcribe(audio_path, language)
            except Exception as error:
                send_tool_error(id, f"transcription failed: {error}")
                return

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
            send_tool_error(id, f"unknown tool: {tool_name}")
    elif method == "notifications/initialized":
        pass  # Client confirmed initialization
    else:
        if id is not None:
            send_response(id, {})


def main():
    """Run MCP server over stdio."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        handle_request(json.loads(line))


if __name__ == "__main__":
    main()
