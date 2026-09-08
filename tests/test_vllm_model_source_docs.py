from pathlib import Path
import ast
import importlib.util

import pytest
from bs4 import BeautifulSoup


ROOT = Path(__file__).resolve().parents[1]
VLLM_GUIDES = [
    "docs/vllm_guide.md",
    "docs/vllm_guide_zh.md",
]


@pytest.mark.parametrize("suffix", ["", "_zh", "_ja", "_ko"])
def test_model_selection_distinguishes_checkpoint_and_service_paths(suffix):
    text = (ROOT / f"docs/model_selection{suffix}.md").read_text()
    language_suffix = "_zh" if suffix == "_zh" else ""
    for target in (f"vllm_guide{language_suffix}.md",
                   f"vllm_official_native_validation{language_suffix}.md",
                   "vllm_native_funasr_validation.md"):
        assert f"](./{target})" in text
        assert (ROOT / "docs" / target).is_file()
    for marker in ("AutoModelVLLM", "FunAudioLLM/Fun-ASR-Nano-2512-vllm",
                   "/v1/audio/transcriptions", "/v1/realtime", "2026-08-13"):
        assert marker in text
    if suffix == "_zh":
        assert "](./vllm_guide.md)" not in text


@pytest.mark.parametrize("suffix", ["", "_zh", "_ja", "_ko"])
def test_model_selection_moss_alias_matches_real_service_configuration(suffix):
    tree = ast.parse((ROOT / "examples/openai_api/server.py").read_text())
    config = next(node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "MODEL_CONFIGS" for t in node.targets))
    moss = next(value for key, value in zip(config.keys, config.values)
                if isinstance(key, ast.Constant) and key.value == "moss-transcribe-diarize")
    model = next(ast.literal_eval(value) for key, value in zip(moss.keys, moss.values)
                 if ast.literal_eval(key) == "model")
    assert model == "OpenMOSS-Team/MOSS-Transcribe-Diarize"
    text = (ROOT / f"docs/model_selection{suffix}.md").read_text()
    assert "**`moss-transcribe-diarize`**" in text
    assert f"`{model}`" in text
    assert "verbose_json" in text


@pytest.mark.parametrize("language", ["en", "zh"])
def test_model_selection_renders_distinct_local_native_and_split_links(language):
    spec = importlib.util.spec_from_file_location(
        "model_choice_documentation", ROOT / "web-pages/product-site/documentation.py")
    documentation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(documentation)
    catalogue = documentation.load_catalogue()
    entry = next(page for page in catalogue["pages"] if page["slug"] == "model-selection")
    rendered = documentation.render_source(entry, language, catalogue)
    soup = BeautifulSoup(rendered["content_html"], "html.parser")
    prefix = "/en" if language == "en" else ""
    targets = {a.get("href") for a in soup.select("a[href]")}
    for slug in ("vllm", "official-native-vllm", "native-vllm", "moss-transcribe-diarize"):
        assert f"{prefix}/docs/{slug}.html" in targets
    ids = [node["id"] for node in soup.select("[id]")]
    assert ids.count("vllm-checkpoint-paths") == 1
    assert len(ids) == len(set(ids)), "Existing and new section anchors must stay unique"


def test_documentation_hub_keeps_official_and_historical_native_entries_separate():
    text = (ROOT / "docs/README.md").read_text()
    for target in ("vllm_official_native_validation.md", "vllm_official_native_validation_zh.md",
                   "vllm_native_funasr_validation.md"):
        assert f"]({target})" in text
    assert "2026-08-13" in text


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
    ("relpath", "required_markers"),
    [
        ("docs/vllm_guide_zh.md", [
            "FunAudioLLM/Fun-ASR-Nano-2512-vllm",
            "vllm_official_native_validation_zh.md",
            "serve_realtime_ws.py",
            "VAD、partial",
            "不会注册 `/v1/realtime`",
            "不是稳定的 API 契约",
        ]),
        ("docs/vllm_guide.md", [
            "FunAudioLLM/Fun-ASR-Nano-2512-vllm",
            "vllm_official_native_validation.md",
            "serve_realtime_ws.py",
            "VAD, partial",
            "they do not register",
            "not a stable API contract",
        ]),
    ],
)
def test_primary_guide_keeps_native_vllm_and_funasr_realtime_paths_separate(
    relpath, required_markers
):
    text = (ROOT / relpath).read_text(encoding="utf-8")

    for marker in required_markers:
        assert marker in text, f"{relpath} is missing {marker}"


@pytest.mark.parametrize(
    ("relpath", "required_markers"),
    [
        (
            "docs/vllm_guide.md",
            (
                "FunAudioLLM/Fun-ASR-Nano-2512-vllm",
                "/v1/audio/transcriptions",
                "/v1/realtime",
                "realtime streaming",
            ),
        ),
        (
            "docs/vllm_guide_zh.md",
            (
                "FunAudioLLM/Fun-ASR-Nano-2512-vllm",
                "/v1/audio/transcriptions",
                "/v1/realtime",
                "实时流式识别",
            ),
        ),
    ],
)
def test_primary_vllm_guides_bound_native_transcription_to_request_response(
    relpath, required_markers
):
    text = (ROOT / relpath).read_text(encoding="utf-8")

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
