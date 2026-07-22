import importlib.util
import json
import os
import sys
import threading
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "server" / "funasr_gguf_server.py"


def load_server_module():
    spec = importlib.util.spec_from_file_location("funasr_gguf_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    sys.modules["funasr_gguf_server"] = module
    spec.loader.exec_module(module)
    return module


def test_build_command_adds_model_audio_vad_backend_and_extra_args(tmp_path):
    server = load_server_module()
    cfg = server.ServerConfig(
        binary="/opt/funasr/llama-funasr-sensevoice",
        model="/models/sensevoice.gguf",
        vad="/models/fsmn-vad.gguf",
        backend="cuda",
        extra_args=["--keep-tags"],
        work_dir=str(tmp_path),
    )

    command = server.build_command(cfg, "/tmp/request.wav")

    assert command == [
        "/opt/funasr/llama-funasr-sensevoice",
        "-m",
        "/models/sensevoice.gguf",
        "-a",
        "/tmp/request.wav",
        "--vad",
        "/models/fsmn-vad.gguf",
        "--backend",
        "cuda",
        "--keep-tags",
    ]


def test_transcription_endpoint_runs_binary_and_returns_openai_json(tmp_path):
    server = load_server_module()
    fake_binary = tmp_path / "fake_funasr.py"
    captured_args = tmp_path / "args.json"
    fake_binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(captured_args)!r}).write_text(json.dumps(sys.argv[1:]))\n"
        "print('hello from gguf')\n",
        encoding="utf-8",
    )
    fake_binary.chmod(0o755)

    cfg = server.ServerConfig(
        binary=str(fake_binary),
        model=str(tmp_path / "sensevoice.gguf"),
        vad=str(tmp_path / "fsmn-vad.gguf"),
        backend=None,
        extra_args=["--ids"],
        work_dir=str(tmp_path),
    )
    httpd = server.create_server("127.0.0.1", 0, cfg)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        boundary = "----funasr-test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="file"; filename="sample.wav"\r\n'
            "Content-Type: audio/wav\r\n\r\n"
            "RIFFfake-audio\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        request = urllib.request.Request(
            f"http://127.0.0.1:{httpd.server_port}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            method="POST",
        )

        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))

        assert response.status == 200
        assert payload == {"text": "hello from gguf"}
        args = json.loads(captured_args.read_text(encoding="utf-8"))
        assert args[:4] == ["-m", str(tmp_path / "sensevoice.gguf"), "-a", args[3]]
        assert os.path.exists(args[3]) is False
        assert args[4:] == ["--vad", str(tmp_path / "fsmn-vad.gguf"), "--ids"]
    finally:
        httpd.shutdown()
        httpd.server_close()
