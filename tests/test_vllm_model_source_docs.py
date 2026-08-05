from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
VLLM_GUIDES = [
    "docs/vllm_guide.md",
    "docs/vllm_guide_zh.md",
    "docs/vllm_guide_zh_v2.md",
]


@pytest.mark.parametrize("relpath", VLLM_GUIDES)
def test_vllm_guides_distinguish_official_and_native_model_paths(relpath):
    text = (ROOT / relpath).read_text(encoding="utf-8")
    required_markers = [
        "https://modelscope.cn/models/FunAudioLLM/Fun-ASR-Nano-2512",
        "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512",
        "model.pt",
        "Qwen3-0.6B/",
        "prepare_vllm_model_dir()",
        "Qwen3-0.6B-vllm/model.safetensors",
        'hub="ms"',
        'hub="hf"',
        "allendou/Fun-ASR-Nano-2512-vllm",
        "FunASRForConditionalGeneration",
    ]

    for marker in required_markers:
        assert marker in text, f"{relpath} is missing {marker}"
