from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
GUIDES = (
    ROOT / "docs" / "moss_transcribe_diarize.md",
    ROOT / "docs" / "moss_transcribe_diarize_zh.md",
)
MATRICES = (
    ROOT / "docs" / "deployment_matrix.md",
    ROOT / "docs" / "deployment_matrix_zh.md",
    ROOT / "docs" / "deployment_matrix_ja.md",
    ROOT / "docs" / "deployment_matrix_ko.md",
)


@pytest.mark.parametrize("guide", GUIDES)
def test_moss_guides_pin_upstream_and_separate_serving_contracts(guide: Path) -> None:
    text = guide.read_text(encoding="utf-8")

    for marker in (
        "OpenMOSS/MOSS-Transcribe-Diarize",
        "cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "e8681d68e7042738ffca8ac8212bc8fcb1131ab8",
        "68b4a1d582818e67adc903bf1b8fc5a5447da2fa",
        "vllm[audio]",
        "0.23.1rc1.dev949+g68b4a1d58",
        "dbb32bcfed2e8226bedf64248a9f4a44685b293a4696d18fb4cfa701b04db912",
        "43dccc068506439cb633b382b6b98185baa837363d08cc5f7152ca89b0fdc3c8",
        "S01 -> S02 -> S01",
        "/v1/audio/transcriptions",
        "response_format=json",
        "response_format=verbose_json",
        "[S01]",
        "Apache-2.0",
        "sentence_info",
        "raw_text",
        "vad_model",
    ):
        assert marker in text, f"{guide.name} is missing {marker}"

    assert "AutoModel" in text
    assert "third-party" in text.lower() or "第三方" in text


@pytest.mark.parametrize("matrix", MATRICES)
def test_all_deployment_matrices_link_moss_guide(matrix: Path) -> None:
    text = matrix.read_text(encoding="utf-8")
    moss_row = next(
        line
        for line in text.splitlines()
        if line.startswith("| MOSS-Transcribe-Diarize |")
    )

    assert "moss_transcribe_diarize.md" in moss_row
    assert '`backend="hf"`' in moss_row
    assert '`backend="vllm"`' in moss_row
    for stale_claim in (
        "not a FunASR-owned model or `AutoModel` backend",
        "不是 FunASR 自有模型或 `AutoModel` 后端",
        "FunASR 所有 model や `AutoModel` backend ではありません",
        "FunASR 소유 model 또는 `AutoModel` backend가 아닙니다",
    ):
        assert stale_claim not in moss_row
