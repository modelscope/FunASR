import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    ROOT
    / "examples"
    / "industrial_data_pretraining"
    / "qwen3_asr"
    / "transcribe_vllm_offline.py"
)


def load_example():
    spec = importlib.util.spec_from_file_location("qwen3_vllm_offline", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeResult:
    def __init__(self, text, language="Chinese"):
        self.text = text
        self.language = language


class FakeModel:
    def __init__(self):
        self.calls = []

    def transcribe(self, *, audio, language):
        self.calls.append((audio, language))
        return [FakeResult(f"chunk-{len(self.calls)}")]


class Qwen3AsrVllmOfflineExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.example = load_example()

    def test_transcribe_chunks_preserves_offsets_and_language(self):
        self.assertTrue(hasattr(self.example, "transcribe_chunks"))
        model = FakeModel()
        chunks = [([0.1] * 16000, 0.0), ([0.2] * 8000, 3.25)]

        segments = self.example.transcribe_chunks(
            model, chunks, sample_rate=16000, language="Chinese"
        )

        self.assertEqual(
            segments,
            [
                {
                    "start_ms": 0,
                    "end_ms": 1000,
                    "text": "chunk-1",
                    "language": "Chinese",
                },
                {
                    "start_ms": 3250,
                    "end_ms": 3750,
                    "text": "chunk-2",
                    "language": "Chinese",
                },
            ],
        )
        self.assertTrue(all(language == "Chinese" for _, language in model.calls))

    def test_transcribe_chunks_rejects_result_count_mismatch(self):
        class EmptyModel:
            def transcribe(self, **kwargs):
                return []

        with self.assertRaisesRegex(RuntimeError, "one result"):
            self.example.transcribe_chunks(
                EmptyModel(), [([0.1], 0.0)], sample_rate=16000, language=None
            )

    def test_help_does_not_require_gpu_dependencies(self):
        self.assertTrue(hasattr(self.example, "build_parser"))
        parser = self.example.build_parser()
        args = parser.parse_args(["input.mp3"])

        self.assertEqual(args.audio, Path("input.mp3"))
        self.assertEqual(args.chunk_seconds, 180.0)
        self.assertEqual(args.max_inference_batch_size, 4)


@pytest.fixture
def moss_run(tmp_path, monkeypatch):
    example = load_example()
    audio = tmp_path / "meeting.wav"
    import wave

    with wave.open(str(audio), "wb") as source:
        source.setnchannels(1)
        source.setsampwidth(2)
        source.setframerate(16000)
        source.writeframes(b"\x00\x00" * 1600)
    args = example.build_parser().parse_args([str(audio)])
    args.engine = "moss"
    args.vllm_base_url = "http://moss.test:8898/v1"
    args.served_model = "meeting-model"
    args.request_timeout = 37.5
    args.max_completion_tokens = 1234
    response = MagicMock()
    response.json.return_value = {
        "text": "hello again",
        "segments": [
            {"start": 0.01, "end": 0.04, "text": "hello", "speaker": "S01"},
            {"start": 0.05, "end": 0.09, "text": "again", "speaker": "S02"},
        ],
    }
    monkeypatch.setenv("MOSS_VLLM_API_KEY", "test-only-key")
    monkeypatch.setitem(sys.modules, "qwen_asr", None)
    monkeypatch.setitem(sys.modules, "qwen_asr.inference.utils", None)
    with patch(
        "funasr.auto.auto_model.download_model",
        side_effect=AssertionError("HTTP client must not download weights"),
    ), patch("requests.Session.post", return_value=response) as post, patch.object(
        example,
        "_convert_to_mono_wav",
        side_effect=AssertionError("No Qwen3 splitter/conversion"),
    ):
        yield example, args, response, post


def test_moss_uses_actual_adapter_once_and_preserves_speakers(moss_run):
    example, args, response, post = moss_run
    output, payload = example.run(args)
    assert output == args.audio.with_suffix(".moss-vllm.json")
    assert json.loads(output.read_text()) == payload
    assert payload["text"] == payload["raw_text"] == "hello again"
    assert payload["timestamp"] == [[10, 40], [50, 90]]
    assert [(x["spk"], x["start"], x["end"]) for x in payload["sentence_info"]] == [
        ("S01", 10, 40),
        ("S02", 50, 90),
    ]
    assert not args.audio.with_suffix(".qwen3-vllm.json").exists()
    post.assert_called_once()
    call = post.call_args
    assert call.args[0] == "http://moss.test:8898/v1/audio/transcriptions"
    assert call.kwargs["data"] == {
        "model": "meeting-model",
        "response_format": "diarized_json",
        "temperature": "0",
        "max_completion_tokens": "1234",
    }
    assert call.kwargs["timeout"] == 37.5
    assert call.kwargs["headers"] == {"Authorization": "Bearer test-only-key"}
    filename, content, mime = call.kwargs["files"]["file"]
    assert filename == "meeting.wav" and mime in {"audio/wav", "audio/x-wav"}
    assert content.getvalue() == args.audio.read_bytes()
    assert "test-only-key" not in output.read_text()
    response.raise_for_status.assert_called_once()


@pytest.mark.parametrize(
    "body",
    [
        [],
        {"text": "missing segments"},
        {
            "text": "broken",
            "segments": [{"start": 2, "end": 1, "text": "broken", "speaker": "S01"}],
        },
        {"text": "broken", "segments": [{"start": 0, "end": 1, "text": "broken"}]},
    ],
)
def test_moss_malformed_response_does_not_write_success(moss_run, body):
    example, args, response, post = moss_run
    response.json.return_value = body
    args.output = args.audio.parent / "existing.json"
    args.output.write_text("old result")
    with pytest.raises(RuntimeError):
        example.run(args)
    assert args.output.read_text() == "old result"
    post.assert_called_once()


def test_moss_http_failure_does_not_create_output(moss_run):
    import requests

    example, args, response, post = moss_run
    response.raise_for_status.side_effect = requests.HTTPError("503")
    with pytest.raises(requests.HTTPError):
        example.run(args)
    assert not args.audio.with_suffix(".moss-vllm.json").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("vllm_base_url", None),
        ("request_timeout", 0),
        ("request_timeout", float("nan")),
        ("request_timeout", float("inf")),
        ("max_completion_tokens", 0),
        ("served_model", ""),
        ("language", "Chinese"),
    ],
)
def test_moss_rejects_invalid_configuration_before_request(moss_run, field, value):
    example, args, _, post = moss_run
    setattr(args, field, value)
    with pytest.raises(ValueError):
        example.run(args)
    post.assert_not_called()


@pytest.mark.parametrize("alias", ["direct", "symlink", "hardlink"])
def test_moss_rejects_input_as_output_before_request(moss_run, alias):
    example, args, _, post = moss_run
    original = args.audio.read_bytes()
    args.output = args.audio
    if alias != "direct":
        args.output = args.audio.parent / "output.json"
        if alias == "symlink":
            args.output.symlink_to(args.audio)
        else:
            args.output.hardlink_to(args.audio)
    with pytest.raises(ValueError, match="output"):
        example.run(args)
    assert args.audio.read_bytes() == original
    post.assert_not_called()


def test_moss_http_options_require_explicit_engine(moss_run):
    example, args, _, post = moss_run
    args.engine = "qwen3"
    with pytest.raises(ValueError, match="--engine moss"):
        example.run(args)
    post.assert_not_called()


def test_moss_main_displays_anonymous_speakers(moss_run, monkeypatch, capsys):
    example, args, _, _ = moss_run
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT_PATH),
            str(args.audio),
            "--engine",
            "moss",
            "--vllm-base-url",
            args.vllm_base_url,
        ],
    )
    example.main()
    out = capsys.readouterr().out
    assert "S01" in out and "S02" in out and "2 segments" in out
    assert "test-only-key" not in out


def test_parser_exposes_explicit_moss_mode_without_changing_qwen_defaults():
    parser = load_example().build_parser()
    original = parser.parse_args(["input.wav"])
    assert original.engine == "qwen3"
    assert original.model == "Qwen/Qwen3-ASR-1.7B"
    moss = parser.parse_args(
        [
            "input.wav",
            "--engine",
            "moss",
            "--vllm-base-url",
            "http://localhost:8898/v1",
            "--served-model",
            "demo",
            "--request-timeout",
            "123",
            "--max-completion-tokens",
            "4321",
        ]
    )
    assert (
        moss.engine,
        moss.served_model,
        moss.request_timeout,
        moss.max_completion_tokens,
    ) == ("moss", "demo", 123, 4321)


def test_default_qwen_run_keeps_native_configuration_and_output(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import numpy as np
    import soundfile as sf

    example = load_example()
    audio = tmp_path / "source.wav"
    sf.write(audio, np.zeros(1600), 16000)
    args = example.build_parser().parse_args([str(audio)])
    llm = MagicMock(return_value=FakeModel())
    splitter = MagicMock(return_value=[(np.zeros(1600), 0.0)])
    monkeypatch.setitem(
        sys.modules, "qwen_asr", SimpleNamespace(Qwen3ASRModel=SimpleNamespace(LLM=llm))
    )
    monkeypatch.setitem(
        sys.modules,
        "qwen_asr.inference.utils",
        SimpleNamespace(split_audio_into_chunks=splitter),
    )
    monkeypatch.setattr(
        example,
        "_convert_to_mono_wav",
        lambda src, dest: dest.write_bytes(src.read_bytes()),
    )
    output, payload = example.run(args)
    assert output == audio.with_suffix(".qwen3-vllm.json")
    assert payload == {
        "text": "chunk-1",
        "language": "Chinese",
        "segments": [
            {"start_ms": 0, "end_ms": 100, "text": "chunk-1", "language": "Chinese"}
        ],
    }
    llm.assert_called_once_with(
        model="Qwen/Qwen3-ASR-1.7B",
        gpu_memory_utilization=0.8,
        max_inference_batch_size=4,
    )
    assert splitter.call_args.kwargs == {"max_chunk_sec": 180.0}


@pytest.mark.parametrize("http_status", [200, 503])
def test_moss_cli_over_real_local_http(tmp_path, http_status):
    import os
    import subprocess
    import threading
    import wave
    from email import policy
    from email.parser import BytesParser
    from http.server import BaseHTTPRequestHandler, HTTPServer

    audio = tmp_path / "whole-recording.wav"
    with wave.open(str(audio), "wb") as source:
        source.setnchannels(1)
        source.setsampwidth(2)
        source.setframerate(16000)
        source.writeframes(b"\x00\x00" * 1600)
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = BytesParser(policy=policy.default).parsebytes(
                ("Content-Type: " + self.headers["Content-Type"] + "\r\n\r\n").encode()
                + body
            )
            fields = {
                part.get_param("name", header="content-disposition"): part.get_payload(
                    decode=True
                )
                for part in message.iter_parts()
            }
            received.append((self.path, self.headers.get("Authorization"), fields))
            self.send_response(http_status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "text": "fixture",
                        "segments": [
                            {
                                "start": 0.01,
                                "end": 0.09,
                                "text": "fixture",
                                "speaker": "S02",
                            }
                        ],
                    }
                ).encode()
            )

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        env = os.environ.copy()
        env.update(
            PYTHONPATH=str(ROOT),
            MOSS_VLLM_API_KEY="http-fixture-key",
            CUDA_VISIBLE_DEVICES="",
            HF_HUB_OFFLINE="1",
            TRANSFORMERS_OFFLINE="1",
        )
        try:
            process = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_PATH),
                    str(audio),
                    "--engine",
                    "moss",
                    "--vllm-base-url",
                    f"http://127.0.0.1:{server.server_port}/v1",
                    "--served-model",
                    "fixture-model",
                    "--max-completion-tokens",
                    "256",
                    "--request-timeout",
                    "5",
                ],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=45,
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(received) == 1
    path, authorization, fields = received[0]
    assert path == "/v1/audio/transcriptions"
    assert authorization == "Bearer http-fixture-key"
    assert fields["file"] == audio.read_bytes()
    assert fields["model"] == b"fixture-model"
    assert fields["response_format"] == b"diarized_json"
    assert fields["max_completion_tokens"] == b"256"
    assert "http-fixture-key" not in process.stdout + process.stderr
    output = audio.with_suffix(".moss-vllm.json")
    if http_status == 200:
        assert process.returncode == 0, process.stderr
        result = json.loads(output.read_text())
        assert result["timestamp"] == [[10, 90]]
        assert result["sentence_info"][0]["spk"] == "S02"
        assert "S02" in process.stdout
    else:
        assert process.returncode != 0
        assert not output.exists()


@pytest.mark.parametrize("suffix", ["", "_en"])
def test_notes_provide_separate_moss_recipe_and_output_contract(suffix):
    notes = SCRIPT_PATH.with_name(
        f"transcribe_vllm_offline_notes{suffix}.md"
    ).read_text()
    for required in [
        "--engine moss",
        "--vllm-base-url",
        "--served-model",
        "--max-completion-tokens",
        "MOSS_VLLM_API_KEY",
        "sentence_info",
        "raw_text",
        ".moss-vllm.json",
        "0.14.0",
        "0.27.1",
        "docs/moss_transcribe_diarize",
        "8192",
    ]:
        assert required in notes


if __name__ == "__main__":
    unittest.main()
