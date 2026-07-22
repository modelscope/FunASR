import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DOCS_WITH_CURRENT_FUNASR_INSTALL = [
    "docs/vllm_guide.md",
    "docs/vllm_guide_zh.md",
    "docs/vllm_guide_zh_v2.md",
    "examples/industrial_data_pretraining/fun_asr_nano/docs/finetune.md",
    "examples/industrial_data_pretraining/fun_asr_nano/docs/fintune_zh.md",
]


def test_current_funasr_install_commands_are_quoted():
    for relpath in DOCS_WITH_CURRENT_FUNASR_INSTALL:
        text = (ROOT / relpath).read_text()
        assert '"funasr>=1.3.23"' in text
        assert "funasr>=1.3.0" not in text
        assert not re.search(r"pip install funasr>=", text)


def test_realtime_demo_documents_partial_and_hotword_boundaries():
    text = (
        ROOT
        / "examples/industrial_data_pretraining/fun_asr_nano/docs/realtime_demo.md"
    ).read_text()

    required = [
        "data.sentences.map",
        "data.partial || \"\"",
        "partial_start_ms",
        "--partial-window-sec",
        "不是确定性文本替换",
        "HOTWORDS:Tool,客製化,季會",
        "后处理",
    ]
    for marker in required:
        assert marker in text
