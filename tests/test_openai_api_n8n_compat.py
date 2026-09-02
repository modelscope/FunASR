import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPO_ROOT / "examples" / "openai_api" / "server.py"


def load_example_server():
    module_name = "funasr_example_openai_server_under_test"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, SERVER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_example_server_maps_n8n_whisper_alias_to_started_model():
    module = load_example_server()
    module.DEFAULT_MODEL = "moss-transcribe-diarize"

    assert module.resolve_openai_transcription_model("whisper-1") == "moss-transcribe-diarize"
    assert module.resolve_openai_transcription_model("sensevoice") == "sensevoice"
