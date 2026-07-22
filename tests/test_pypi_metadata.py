from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / "setup.py"


def _setup_text() -> str:
    return SETUP.read_text(encoding="utf-8")


def test_package_summary_matches_current_deployment_positioning():
    text = _setup_text()

    expected_summary = (
        'description="OpenAI-compatible speech recognition toolkit with '
        'WebSocket streaming, vLLM acceleration, and llama.cpp/GGUF edge '
        'runtime."'
    )
    assert expected_summary in text
    assert "170x realtime" not in text


def test_package_keywords_surface_deployment_discovery_terms():
    text = _setup_text()

    for keyword in [
        '"openai-compatible"',
        '"websocket"',
        '"vllm"',
        '"gguf"',
        '"llama-cpp"',
    ]:
        assert keyword in text
