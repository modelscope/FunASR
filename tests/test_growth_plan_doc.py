from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "community_growth_20k.md"


def test_growth_plan_records_four_repo_community_template_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Add or refresh CONTRIBUTING, PR template, and issue templates",
        "modelscope/FunClip#188",
        "modelscope/FunClip#189",
        "FunAudioLLM/Fun-ASR#151",
        "FunAudioLLM/SenseVoice#327",
        "four core repositories now have `CONTRIBUTING.md`, issue templates, and PR templates",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Add or refresh CONTRIBUTING, PR template, and issue templates" not in text


def test_growth_plan_records_install_deployment_faq_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Add a short FAQ for install/deployment failures",
        "modelscope/FunASR#3366",
        "./troubleshooting.md",
        "./troubleshooting_zh.md",
        "`funasr-server`",
        "`/v1/audio/transcriptions`",
        "`WebSocket`",
        "`llama.cpp` / `GGUF`",
        "Deployment Help",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Add a short FAQ for install/deployment failures" not in text
