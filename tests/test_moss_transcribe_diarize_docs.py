import json
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
BOUNDARY_DOCS = (
    *GUIDES,
    ROOT / "docs" / "repository_roles.md",
    ROOT / "docs" / "repository_roles_zh.md",
    ROOT
    / "web-pages"
    / "product-site"
    / "legacy"
    / "en"
    / "blog"
    / "funclip-v2-2-0-moss-speaker-clipping.html",
    ROOT
    / "web-pages"
    / "product-site"
    / "legacy"
    / "blog"
    / "funclip-v2-2-0-moss-speaker-clipping.html",
)


@pytest.mark.parametrize("guide", GUIDES)
def test_moss_guides_pin_upstream_and_separate_serving_contracts(guide: Path) -> None:
    text = guide.read_text(encoding="utf-8")

    for marker in (
        "OpenMOSS/MOSS-Transcribe-Diarize",
        "cb765f2b0fe6f7a298aa2002e2281ae693d1f3c3",
        "OpenMOSS-Team/MOSS-Transcribe-Diarize",
        "e8681d68e7042738ffca8ac8212bc8fcb1131ab8",
        "6e448d0ea9bf3d88d898b65449ca6dc2aec170ac",
        "3f819f9cdae3d4eeec22f73306c9067a1ec2542e",
        "bf0d52faa2a51e7a01c6856a7a8a2d1307fd0ff711415d34168a67ffac0fa47b",
        "vllm[audio]",
        "vLLM 0.27.1",
        "68b4a1d582818e67adc903bf1b8fc5a5447da2fa",
        "dbb32bcfed2e8226bedf64248a9f4a44685b293a4696d18fb4cfa701b04db912",
        "43dccc068506439cb633b382b6b98185baa837363d08cc5f7152ca89b0fdc3c8",
        "S01 -> S02 -> S01",
        "/v1/audio/transcriptions",
        "response_format=json",
        "response_format=diarized_json",
        "response_format=verbose_json",
        'vllm_response_format="diarized_json"',
        'backend="sglang"',
        'sglang_base_url="http://127.0.0.1:8898/v1"',
        "max_new_tokens=65536",
        "[S01]",
        "Apache-2.0",
        "sentence_info",
        "raw_text",
        "vad_model",
        "max_completion_tokens=8192",
        "6561ee553c8f762aac4ebd65439d3414820761b547fa3a2edcea43b86a2abc02",
        "779899a3ce937dd7352b4db1ea53e3f6aa2cfef7109de0249082223c936f9372",
        "localai-org/moss-transcribe.cpp",
        "Open WebUI",
        "OpenAI API Base URL",
        "/v1/audio/transcriptions",
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
    assert '`backend="sglang"`' in moss_row
    for stale_claim in (
        "not a FunASR-owned model or `AutoModel` backend",
        "不是 FunASR 自有模型或 `AutoModel` 后端",
        "FunASR 所有 model や `AutoModel` backend ではありません",
        "FunASR 소유 model 또는 `AutoModel` backend가 아닙니다",
    ):
        assert stale_claim not in moss_row


def test_moss_guides_cover_funasr_service_and_offline_boundaries() -> None:
    for guide in GUIDES:
        text = guide.read_text(encoding="utf-8")
        assert "funasr-server --model moss-transcribe-diarize" in text
        assert "response_format=verbose_json" in text
        assert "examples/openai_api/docker-compose.moss.yml" in text
        assert "examples/openai_api/kubernetes/funasr-moss-api.yaml" in text
        assert "WebSocket" in text
        assert "FunClip" in text


def test_moss_service_recipes_are_pinned_and_gpu_scoped() -> None:
    dockerfile = ROOT / "examples" / "openai_api" / "Dockerfile.moss"
    compose = ROOT / "examples" / "openai_api" / "docker-compose.moss.yml"
    kubernetes = (
        ROOT / "examples" / "openai_api" / "kubernetes" / "funasr-moss-api.yaml"
    )

    assert dockerfile.is_file()
    assert compose.is_file()
    assert kubernetes.is_file()
    docker_text = dockerfile.read_text(encoding="utf-8")
    compose_text = compose.read_text(encoding="utf-8")
    kubernetes_text = kubernetes.read_text(encoding="utf-8")
    for text in (docker_text, compose_text, kubernetes_text):
        assert "moss-transcribe-diarize" in text
    assert "transformers>=5.6,<6" in docker_text
    assert "capabilities: [gpu]" in compose_text
    assert "nvidia.com/gpu" in kubernetes_text


def test_github_pages_indexes_moss_deployment_guide() -> None:
    index = (ROOT / "docs" / "index.rst").read_text(encoding="utf-8")
    assert "./moss_transcribe_diarize.md" in index
    assert "./moss_transcribe_diarize_zh.md" in index


def test_readmes_link_the_runnable_moss_service() -> None:
    readmes = [
        ROOT / "README.md",
        ROOT / "README_zh.md",
        ROOT / "README_ja.md",
        ROOT / "README_ko.md",
        ROOT / "examples" / "openai_api" / "README.md",
        ROOT / "examples" / "openai_api" / "README_zh.md",
        ROOT / "examples" / "openai_api" / "README_ja.md",
        ROOT / "examples" / "openai_api" / "README_ko.md",
    ]
    for readme in readmes:
        text = readme.read_text(encoding="utf-8")
        assert "moss-transcribe-diarize" in text
        assert "moss_transcribe_diarize" in text


def test_readme_model_zoos_expose_moss_with_its_model_card_and_guide() -> None:
    readmes = {
        "README.md": "./docs/moss_transcribe_diarize.md",
        "README_zh.md": "./docs/moss_transcribe_diarize_zh.md",
        "README_ja.md": "./docs/moss_transcribe_diarize.md",
        "README_ko.md": "./docs/moss_transcribe_diarize.md",
    }

    for name, guide in readmes.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "| **MOSS-Transcribe-Diarize** |" in text, name
        assert (
            "https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize" in text
        ), name
        assert guide in text, name


def test_model_zoo_indexes_expose_moss_with_its_model_card_and_guide() -> None:
    indexes = {
        "model_zoo/readme.md": "../docs/moss_transcribe_diarize.md",
        "model_zoo/readme_zh.md": "../docs/moss_transcribe_diarize_zh.md",
    }

    for name, guide in indexes.items():
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "MOSS-Transcribe-Diarize" in text, name
        assert (
            "https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize" in text
        ), name
        assert guide in text, name


def test_model_selection_guides_surface_offline_moss_diarization() -> None:
    guides = {
        "model_selection.md": ("./moss_transcribe_diarize.md", "offline"),
        "model_selection_zh.md": ("./moss_transcribe_diarize_zh.md", "离线"),
        "model_selection_ja.md": ("./moss_transcribe_diarize.md", "オフライン"),
        "model_selection_ko.md": ("./moss_transcribe_diarize.md", "오프라인"),
    }

    for name, (guide, offline_marker) in guides.items():
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert "MOSS-Transcribe-Diarize" in text, name
        assert guide in text, name
        assert offline_marker in text, name


def test_openai_consumer_docs_expose_moss_alias_and_boundaries() -> None:
    paths = [
        "CLIENTS.md",
        "GRADIO.md",
        "GRADIO_zh.md",
        "OPENAPI.md",
        "OPENAPI_zh.md",
        "WORKFLOWS.md",
        "WORKFLOWS_zh.md",
        "JAVASCRIPT.md",
        "JAVASCRIPT_zh.md",
        "kubernetes/README.md",
        "kubernetes/README_zh.md",
    ]
    root = ROOT / "examples" / "openai_api"
    for relative in paths:
        text = (root / relative).read_text(encoding="utf-8")
        assert "moss-transcribe-diarize" in text
        assert "moss_transcribe_diarize" in text

    spec = json.loads((root / "openapi.json").read_text(encoding="utf-8"))
    model = spec["components"]["schemas"]["TranscriptionRequest"]["properties"]["model"]
    assert "moss-transcribe-diarize" in model["enum"]


def test_use_case_showcases_surface_offline_moss_diarization() -> None:
    showcases = {
        "docs/use_case_showcase.md": (
            "Offline long-form diarized transcripts",
            "offline long audio",
            "anonymous speaker labels",
        ),
        "docs/use_case_showcase_zh.md": (
            "离线长音频一体化转写与说话人标签",
            "离线长音频",
            "匿名说话人标签",
        ),
    }

    for name, markers in showcases.items():
        text = (ROOT / name).read_text()
        assert "MOSS-Transcribe-Diarize" in text, name
        assert "moss_transcribe_diarize" in text, name
        for marker in markers:
            assert marker in text, name


def test_moss_docs_describe_anonymous_labels_not_known_person_identity() -> None:
    combined = "\n".join(path.read_text(encoding="utf-8") for path in BOUNDARY_DOCS)

    for misleading_claim in (
        "speaker identity",
        "speaker identities",
        "说话人身份",
        "身份识别",
    ):
        assert misleading_claim not in combined

    assert "anonymous speaker labels" in combined
    assert "匿名说话人标签" in combined
    assert "does not identify a known person" in combined
    assert "不能识别已知人物" in combined
