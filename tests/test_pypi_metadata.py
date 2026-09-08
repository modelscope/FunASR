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


def test_core_dependencies_allow_numpy_two():
    """numpy must not be pinned to <2; the codebase is compatible with both 1.x and 2.x."""
    text = _setup_text()

    assert '"numpy<2"' not in text
    assert '"numpy"' in text


def test_no_removed_numpy_aliases_in_source():
    """np.float / np.int / etc. were removed in numpy 1.24 and must not appear in funasr code."""
    import re

    removed_aliases = re.compile(
        r"\bnp\.(float|int|bool|object|str|complex|long|unicode)\b(?!\d|_)"
    )
    source_files = list((ROOT / "funasr").rglob("*.py"))
    offenders: list[str] = []
    for path in source_files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            # Skip docstrings/comments: only flag lines that are actual code
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"') or stripped.startswith("'"):
                continue
            if removed_aliases.search(line):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: {stripped}")
    assert not offenders, f"Removed numpy aliases found:\n" + "\n".join(offenders)
