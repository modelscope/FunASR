"""Native Transformers adoption documentation, without model downloads."""
import ast
import json
import re
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parents[1]
sys.path.insert(0, str(SITE))
from build import build
from export_docs import export_documentation

SLUG = "native-transformers"
BLOG = "fun-asr-nano-transformers.html"
HEAD = "fc501343edfccdc840eb8594a6cafa8185c2de53"
REVISION = "d93b302ee7fd505e1b3576120fc142fc6f7820e1"
MODEL = "FunAudioLLM/Fun-ASR-Nano-2512-hf"


def guide(suffix=""):
    path = ROOT / f"docs/transformers_native{suffix}.md"
    assert path.is_file(), "Missing native Transformers guide"
    return path.read_text()


def example(text, name):
    pattern = rf"<!-- native-example: {name} -->\s*\x60\x60\x60python\n(.*?)\x60\x60\x60"
    matches = re.findall(pattern, text, re.S)
    assert len(matches) == 1, f"Missing unique {name} example"
    return matches[0]


def test_catalogue_and_pages_export_own_both_native_guides():
    data = json.loads((SITE / "data/documentation.json").read_text())
    entries = [p for p in data["pages"] if p["slug"] == SLUG]
    assert len(entries) == 1
    assert entries[0]["source_en"] == "docs/transformers_native.md"
    assert entries[0]["source_zh"] == "docs/transformers_native_zh.md"
    from export_docs import ALIASES
    assert ALIASES[SLUG] == "native-transformers.html"


@pytest.mark.parametrize("suffix", ["", "_zh"])
def test_native_guide_pins_format_runtime_and_audio_dependency(suffix):
    text = guide(suffix)
    for expected in (MODEL, REVISION, HEAD, "torchaudio==2.10.0+cpu",
                     "torch==2.10.0+cpu", "transformers==5.17.0", "2026-09-09",
                     "vllm", "GGUF", "46180"):
        assert expected in text
    assert "huggingface/transformers/archive/" not in text
    assert "examples/transformers" in text
    assert text.index("native-example: transcribe") < text.index("native-example: processor")
    assert re.search(r"not.*(?:capacity|accuracy)|不.*(?:容量|准确率)", text, re.I)


@pytest.mark.parametrize("name", ["processor", "transcribe"])
def test_bilingual_recipes_are_identical_and_use_fixed_native_checkpoint(name):
    en = example(guide(), name)
    zh = example(guide("_zh"), name)
    assert en == zh
    tree = ast.parse(en)
    constants = {node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)}
    assert MODEL in constants and REVISION in constants
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    loaders = [node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "from_pretrained"]
    assert loaders
    for node in loaders:
        names = {kw.arg for kw in node.keywords}
        assert {"revision", "trust_remote_code"} <= names
        assert next(kw.value for kw in node.keywords if kw.arg == "trust_remote_code").value is False
    assert "apply_transcription_request" in en
    if name == "processor":
        assert not any(isinstance(node.func, ast.Attribute) and node.func.attr == "generate" for node in calls)
    else:
        assert "inference_mode" in en and "max_new_tokens" in en
        assert "generated[:, inputs.input_ids.shape[1]:]" in en
        assert "batch_decode" in en
        assert "16000" in en and "example/en.mp3" in en
        assert "272c57b82523ada6fd87095e955f8e29100979ab" in en


@pytest.mark.parametrize("suffix", ["", "_zh", "_ja", "_ko"])
def test_all_language_entry_points_reach_native_examples(suffix):
    for name in [f"README{suffix}.md", f"docs/model_selection{suffix}.md",
                 f"docs/deployment_matrix{suffix}.md", f"examples/colab/README{suffix}.md"]:
        text = (ROOT / name).read_text()
        assert "5.17.0" in text, name
        assert "5.16.1" not in text, name
    text = (ROOT / f"README{suffix}.md").read_text()
    assert "examples/transformers" in text
    assert "fun_asr_nano_transformers.ipynb" in text


def test_hf_catalogue_and_sphinx_reach_native_model():
    text = (ROOT / "model_zoo/huggingface_models.md").read_text()
    assert MODEL in text and "transformers_native.md" in text
    text = (ROOT / "docs/index.rst").read_text()
    assert "   transformers_native\n" in text
    assert "   transformers_native_zh\n" in text


@pytest.mark.parametrize("name,heading", [
    ("README.md", "What's new"), ("README_zh.md", "最新动态"),
    ("README_ja.md", "最新情報"), ("README_ko.md", "최신 소식"),
])
def test_readme_news_stays_three_items_with_native_guide(name, heading):
    text = (ROOT / name).read_text()
    section = text.split("## " + heading, 1)[1].split("\n## ", 1)[0]
    bullets = re.findall(r"^- .*", section, re.M)
    assert len(bullets) == 3
    assert any("Transformers" in row and "transformers_native" in row for row in bullets)
    assert any("1.4.15" in row for row in bullets)
    assert any("MOSS" in row for row in bullets)


@pytest.mark.parametrize("suffix", ["", "_zh"])
def test_model_zoo_and_deployment_matrix_link_native_format(suffix):
    zoo = (ROOT / f"model_zoo/readme{suffix}.md").read_text()
    matrix = (ROOT / f"docs/deployment_matrix{suffix}.md").read_text()
    assert MODEL in zoo and f"transformers_native{suffix}.md" in zoo
    assert f"transformers_native{suffix}.md" in matrix


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    output = tmp_path_factory.mktemp("native-transformers-site")
    build(output)
    pages = tmp_path_factory.mktemp("native-transformers-pages")
    export_documentation(output, pages)
    return output, pages


@pytest.mark.parametrize("prefix,suffix,pages_prefix", [("", "_zh", "zh/"), ("en/", "", "")])
def test_rendered_blog_models_guide_and_pages_are_connected(rendered, prefix, suffix, pages_prefix):
    output, pages = rendered
    route = f"/{prefix}docs/{SLUG}.html"
    article_path = output / prefix / "blog" / BLOG
    assert article_path.is_file()
    article = BeautifulSoup(article_path.read_text(), "html.parser")
    assert article.select_one(f'article a[href="{route}"]')
    assert article.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{BLOG}"
    body = article.select_one("article").get_text(" ", strip=True)
    for term in ["Transformers", "-hf", "vLLM", "GGUF", "2026-09-09"]:
        assert term in body
    index = BeautifulSoup((output / prefix / "blog/index.html").read_text(), "html.parser")
    assert len(index.select(f'a.post-card[href="/{prefix}blog/{BLOG}"]')) == 1
    models = BeautifulSoup((output / prefix / "models.html").read_text(), "html.parser")
    assert models.select_one(f'a[href="{route}"]')
    native = BeautifulSoup((output / prefix / "docs" / f"{SLUG}.html").read_text(), "html.parser")
    assert native.select_one("[data-source-link]")["href"].endswith(f"/docs/transformers_native{suffix}.md")
    exported = BeautifulSoup((pages / pages_prefix / "native-transformers.html").read_text(), "html.parser")
    assert MODEL in exported.get_text()
    assert exported.select_one('link[rel="canonical"]')["href"] == "https://www.funasr.com" + route
