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


@pytest.mark.parametrize(
    ("relpath", "required_markers", "stale_claim"),
    [
        (
            "docs/vllm_guide.md",
            (
                "https://github.com/modelscope/FunASR/issues/3496",
                "incomplete CTC weights",
                "timestamps or speaker diarization",
                "Use the ModelScope\ncheckpoint",
            ),
            "does contain the complete `model.pt` at its root",
        ),
        (
            "docs/vllm_guide_zh.md",
            (
                "https://github.com/modelscope/FunASR/issues/3496",
                "CTC 权重不完整",
                "时间戳或说话人分离",
                "ModelScope checkpoint",
            ),
            "实际包含完整的 `model.pt`",
        ),
        (
            "docs/vllm_guide_zh_v2.md",
            (
                "https://github.com/modelscope/FunASR/issues/3496",
                "CTC 权重不完整",
                "时间戳或说话人分离",
                "ModelScope checkpoint",
            ),
            "实际包含完整的 `model.pt`",
        ),
    ],
)
def test_vllm_guides_document_hf_incomplete_ctc_checkpoint(
    relpath, required_markers, stale_claim
):
    text = (ROOT / relpath).read_text(encoding="utf-8")

    for marker in required_markers:
        assert marker in text, f"{relpath} is missing {marker}"
    assert stale_claim not in text, f"{relpath} still claims HF has complete CTC"
