#!/usr/bin/env python3
"""Cross-platform smoke test for the FunASR OpenAI-compatible API."""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import uuid

SAMPLE_URL = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/BAC009S0764W0121.wav"


def request_json(url: str, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def download_if_needed(path: Path, sample_url: str, timeout: float) -> None:
    if path.exists():
        return
    print(f"Downloading sample audio to {path}")
    with urllib.request.urlopen(sample_url, timeout=timeout) as response:
        path.write_bytes(response.read())


def multipart_body(audio_path: Path, model: str, response_format: str) -> tuple[bytes, str]:
    boundary = f"----funasr-smoke-{uuid.uuid4().hex}"
    content_type = mimetypes.guess_type(str(audio_path))[0] or "application/octet-stream"
    parts: list[bytes] = []

    def add_text(name: str, value: str) -> None:
        parts.append(
            (
                f"--{boundary}\r\n"
                f"Content-Disposition: form-data; name=\"{name}\"\r\n\r\n"
                f"{value}\r\n"
            ).encode("utf-8")
        )

    parts.append(
        (
            f"--{boundary}\r\n"
            f"Content-Disposition: form-data; name=\"file\"; filename=\"{audio_path.name}\"\r\n"
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    parts.append(audio_path.read_bytes())
    parts.append(b"\r\n")
    add_text("model", model)
    add_text("response_format", response_format)
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), boundary


def transcribe(base_url: str, audio_path: Path, model: str, response_format: str, timeout: float) -> dict:
    body, boundary = multipart_body(audio_path, model, response_format)
    request = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions",
        data=body,
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def print_json(title: str, payload: dict) -> None:
    print(f"\n== {title} ==")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="Smoke test the FunASR OpenAI-compatible API")
    parser.add_argument("audio_path", nargs="?", default="sample.wav", help="Audio file to transcribe; downloads a sample if missing")
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8000"), help="FunASR API base URL")
    parser.add_argument("--model", default=os.getenv("MODEL", "sensevoice"), help="Model alias to use")
    parser.add_argument("--response-format", default=os.getenv("RESPONSE_FORMAT", "verbose_json"), choices=["json", "verbose_json"], help="Transcription response format")
    parser.add_argument("--sample-url", default=os.getenv("SAMPLE_URL", SAMPLE_URL), help="Sample audio URL used when audio_path does not exist")
    parser.add_argument("--timeout", type=float, default=float(os.getenv("TIMEOUT", "300")), help="HTTP timeout in seconds")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    audio_path = Path(args.audio_path)

    try:
        download_if_needed(audio_path, args.sample_url, args.timeout)
        print_json("health", request_json(f"{base_url}/health", args.timeout))
        print_json("models", request_json(f"{base_url}/v1/models", args.timeout))
        print(f"\nTranscribing {audio_path} with model={args.model}, response_format={args.response_format}")
        print_json("transcription", transcribe(base_url, audio_path, args.model, args.response_format, args.timeout))
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"HTTP {error.code} from {error.url}: {detail}", file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Smoke test failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
