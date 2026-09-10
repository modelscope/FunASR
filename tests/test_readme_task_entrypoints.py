"""README entrypoint contracts; parse examples without importing or running models."""

import ast
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

from bs4 import BeautifulSoup
from markdown import Markdown
import pytest


ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = {
    "README.md": ("Why FunASR?", "Benchmark", "What's new", "Model Zoo", "offline", "anonymous", "recording"),
    "README_zh.md": ("为什么选 FunASR？", "性能评测", "最新动态", "模型列表", "离线", "匿名", "录音"),
    "README_ja.md": ("なぜFunASRを選ぶのか？", "ベンチマーク", "最新情報", "モデル一覧", "オフライン", "匿名", "録音"),
    "README_ko.md": ("왜 FunASR인가?", "벤치마크", "최신 소식", "모델 목록", "오프라인", "익명", "녹음"),
}


def read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def fences(text):
    language = None
    lines = []
    for line in text.splitlines():
        if line.startswith("```"):
            if language is None:
                language, lines = line[3:].strip(), []
            else:
                yield language, "\n".join(lines) + "\n"
                language = None
        elif language is not None:
            lines.append(line)
    assert language is None, "Unclosed code fence"


def render(text):
    def slugify(value, separator):
        return re.sub(r"[^\w\- ]", "", value.lower()).replace(" ", separator)

    return BeautifulSoup(Markdown(
        extensions=["extra", "toc"],
        extension_configs={"toc": {"slugify": slugify}},
    ).convert(text), "html.parser")


def section(name, title):
    soup = render(read(name))
    heading = soup.find(re.compile(r"^h[1-6]$"), string=title)
    assert heading is not None, title
    nodes = []
    for node in heading.next_siblings:
        if node.name and re.fullmatch(r"h[1-6]", node.name) and node.name <= heading.name:
            break
        nodes.append(str(node))
    return BeautifulSoup("".join(nodes), "html.parser")


@pytest.mark.parametrize("name", LANGUAGES)
def test_task_table_names_checkpoint_runtime_and_boundaries(name):
    body = section(name, LANGUAGES[name][0])
    table = body.find("table")
    assert table is not None
    assert len(table.select("thead th")) == 4
    assert not any(word in table.get_text() for word in ("Whisper", "Cloud APIs", "$0.006", "¥0.04"))
    for checkpoint in ("SenseVoiceSmall", "Fun-ASR-Nano", "Fun-ASR-MLT-Nano", "Paraformer", "MOSS-Transcribe-Diarize", "llama.cpp"):
        assert checkpoint in table.get_text(), checkpoint
    assert "OpenMOSS" in table.get_text()
    assert "CPU" in table.get_text() and "WebSocket" in table.get_text()
    assert all(len(row.find_all("td")) == 4 for row in table.select("tbody tr"))


@pytest.mark.parametrize("name", LANGUAGES)
def test_benchmark_links_evidence_instead_of_generalizing_vendor_performance(name):
    body = section(name, LANGUAGES[name][1])
    assert not body.find("table"), "Keep historical numeric tables in their evidence reports"
    links = [a.get("href", "") for a in body.find_all("a")]
    assert any("docs/benchmark/rtf_reproducibility.md" in href for href in links)
    assert any("benchmark.html" in href for href in links)
    assert "Whisper" not in body.get_text(), "Do not promote a cross-hardware comparison into a blanket claim"
    assert not re.search(r"[$¥]\s*0\.\d+", read(name))


@pytest.mark.parametrize("name", LANGUAGES)
def test_first_sensevoice_example_is_cpu_and_prints_real_return_fields(name):
    examples = [code for language, code in fences(read(name))
                if language == "python" and "SenseVoiceSmall" in code]
    assert examples
    tree = ast.parse(examples[0])
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    constructor = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "AutoModel")
    options = {kw.arg: ast.literal_eval(kw.value) for kw in constructor.keywords}
    assert options["device"] == "cpu"
    assert options["vad_model"] == "fsmn-vad" and options["spk_model"] == "cam++"
    generate = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "generate")
    sample = next(ast.literal_eval(kw.value) for kw in generate.keywords if kw.arg == "input")
    assert sample.startswith("https://") and "asr_example_zh.wav" in sample
    assert "asr_example_zh.wav" in read("examples/industrial_data_pretraining/paraformer/demo.py")
    assert any(isinstance(node.func, ast.Name) and node.func.id == "print" for node in calls)
    fields = {node.slice.value for node in ast.walk(tree)
              if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)}
    assert {"sentence_info", "start", "spk", "sentence"} <= fields
    assert "spk_embedding" in read(name), "Explain embeddings separately from pipeline diarization"


@pytest.mark.parametrize("name", LANGUAGES)
def test_model_zoo_retains_third_party_moss_and_recording_scope(name):
    body = section(name, LANGUAGES[name][3])
    moss = next(row for row in body.select("tr") if "MOSS-Transcribe-Diarize" in row.get_text())
    assert "OpenMOSS" in moss.get_text()
    for boundary in LANGUAGES[name][4:]:
        assert boundary in body.get_text().lower(), boundary
    assert any("docs/moss_transcribe_diarize" in a.get("href", "") for a in moss.find_all("a"))
    for row in body.select("tr"):
        if row.find("td") and row.find("td").get_text(strip=True) == "cam++":
            assert "embedding" in row.find_all("td")[1].get_text().lower()


@pytest.mark.parametrize("name", LANGUAGES)
def test_news_stays_three_items_and_keeps_release_history(name):
    body = section(name, LANGUAGES[name][2])
    assert len(body.select("li")) == 3
    assert any(a.get("href") == "https://github.com/modelscope/FunASR/releases" for a in body.find_all("a"))


@pytest.mark.parametrize("name", LANGUAGES)
def test_python_and_shell_examples_have_valid_syntax_without_execution(name):
    for language, code in fences(read(name)):
        if language == "python":
            compile(code, name, "exec")
        elif language in {"bash", "sh", "shell"}:
            result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
            assert result.returncode == 0, result.stderr
            for embedded in re.findall(r"<<'PY'\n(.*?)\nPY", code, re.S):
                ast.parse(embedded)


@pytest.mark.parametrize("name", LANGUAGES)
def test_relative_links_and_internal_navigation_resolve(name):
    soup = render(read(name))
    for link in soup.select("a[href]"):
        target = urlsplit(link["href"])
        if target.scheme or target.netloc:
            continue
        resolved = (ROOT / unquote(target.path)).resolve() if target.path else ROOT / name
        assert resolved.is_relative_to(ROOT)
        assert resolved.exists(), (name, link["href"])
        if target.fragment and resolved.suffix == ".md":
            destination = render(resolved.read_text(encoding="utf-8"))
            anchors = {node.get("id") for node in destination.select("[id]")}
            anchors.update(node.get("name") for node in destination.select("a[name]"))
            assert unquote(target.fragment) in anchors, (name, link["href"])


@pytest.mark.parametrize("name", LANGUAGES)
def test_model_zoo_distinguishes_native_transformers_and_toolkit_downloads(name):
    body = section(name, LANGUAGES[name][3])
    rows = {row.find("td").get_text(strip=True): row for row in body.select("tr")
            if row.find("td")}
    links = {link.get_text(strip=True): link.get("href")
             for link in rows["Fun-ASR-Nano"].select("a[href]")}
    assert links.get("HF / Transformers") == "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf"
    assert links.get("HF / FunASR") == "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512"
    assert links.get("GGUF") == "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF"
    mlt_links = {link.get("href") for link in rows["Fun-ASR-MLT-Nano"].select("a[href]")}
    assert "https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512" in mlt_links
    assert "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512-hf" not in mlt_links


def test_speaker_field_explanation_has_implementation_evidence():
    camp = ast.parse(read("funasr/models/campplus/model.py"))
    assert any(isinstance(node, ast.Constant) and node.value == "spk_embedding" for node in ast.walk(camp))
    auto = ast.parse(read("funasr/auto/auto_model.py"))
    strings = {node.value for node in ast.walk(auto) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert {"sentence_info", "start", "sentence"} <= strings
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "distribute_spk" for node in ast.walk(auto))
    utility = ast.parse(read("funasr/models/campplus/utils.py"))
    distribute = next(node for node in utility.body if isinstance(node, ast.FunctionDef) and node.name == "distribute_spk")
    assert any(isinstance(node, ast.Constant) and node.value == "spk" for node in ast.walk(distribute))
