import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


DOCS_WITH_CURRENT_FUNASR_INSTALL = [
    "docs/vllm_guide.md",
    "docs/vllm_guide_zh.md",
    "docs/vllm_guide_zh_v2.md",
    "examples/industrial_data_pretraining/fun_asr_nano/docs/finetune.md",
    "examples/industrial_data_pretraining/fun_asr_nano/docs/finetune_zh.md",
]

PUBLIC_DOCS_SHOULD_USE_CURRENT_HOSTS = [
    "benchmarks/benchmark_pipeline_cer.md",
    "docs/installation/installation.md",
    "docs/installation/installation_zh.md",
    "model_zoo/modelscope_models.md",
    "model_zoo/readme.md",
    "runtime/docs/SDK_advanced_guide_offline.md",
    "runtime/docs/SDK_advanced_guide_offline_en.md",
    "runtime/docs/SDK_advanced_guide_offline_en_zh.md",
    "runtime/docs/SDK_advanced_guide_offline_gpu.md",
    "runtime/docs/SDK_advanced_guide_offline_gpu_zh.md",
    "runtime/docs/SDK_advanced_guide_offline_zh.md",
    "runtime/docs/SDK_advanced_guide_online.md",
    "runtime/docs/SDK_advanced_guide_online_zh.md",
    "runtime/python/grpc/Readme.md",
    "runtime/python/libtorch/README.md",
    "runtime/python/onnxruntime/README.md",
    "runtime/python/websocket/README.md",
    "docs/m2met2/Baseline.md",
    "examples/industrial_data_pretraining/monotonic_aligner/README_zh.md",
    "examples/industrial_data_pretraining/sense_voice/README.md",
    "examples/industrial_data_pretraining/sense_voice/README_zh.md",
    "examples/industrial_data_pretraining/sense_voice/README_ja.md",
    "runtime/html5/readme.md",
    "runtime/html5/readme_zh.md",
]


def test_current_funasr_install_commands_are_quoted():
    for relpath in DOCS_WITH_CURRENT_FUNASR_INSTALL:
        text = (ROOT / relpath).read_text()
        assert '"funasr>=1.3.26"' in text
        assert "funasr>=1.3.0" not in text
        assert not re.search(r"pip install funasr>=", text)


def test_fun_asr_nano_finetune_zh_uses_canonical_filename():
    docs_dir = ROOT / "examples/industrial_data_pretraining/fun_asr_nano/docs"
    assert (docs_dir / "finetune_zh.md").exists()
    assert "fintune_zh.md" not in (docs_dir / "finetune.md").read_text()


def test_public_docs_use_current_repository_and_docs_hosts():
    for relpath in PUBLIC_DOCS_SHOULD_USE_CURRENT_HOSTS:
        text = (ROOT / relpath).read_text()
        assert "github.com/alibaba/FunASR" not in text
        assert "alibaba-damo-academy.github.io/FunASR" not in text


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
