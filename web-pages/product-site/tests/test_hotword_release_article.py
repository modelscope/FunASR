"""Keep the daily integration article bilingual, reproducible and bounded."""
import contextlib
import io
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "hotwords-package-boundaries"


def test_catalogue_contains_daily_article_once():
    data = json.loads((SITE / "data/blog.json").read_text())
    entries = [row for row in data["articles"] if row["slug"] == SLUG]
    assert len(entries) == 1
    assert entries[0]["reviewed"] and entries[0]["category"] == "explanations"
    assert data["selected"] == [
        "meeting-transcript-acceptance", "fun-asr-nano-transformers",
        "self-hosted-openai-whisper-api-alternative", "funasr-transcribe-long-audio",
    ]


@pytest.fixture(scope="module", params=["source", "built"])
def tree(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("hotword-article")
    build(output)
    return output


@pytest.fixture(params=["", "en/"])
def article(request, tree):
    prefix = request.param
    path = tree / prefix / "blog" / (SLUG + ".html")
    return prefix, BeautifulSoup(path.read_text(), "html.parser")


def test_article_metadata_and_package_boundaries(article):
    prefix, soup = article
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["datePublished"] == metadata["dateModified"] == "2026-11-13"
    assert metadata["headline"] == soup.select_one("article h1").get_text()
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}.html"
    links = {a["href"] for a in soup.select("article a[href]")}
    assert "https://pypi.org/project/funasr-torch/0.1.7/" in links
    assert "https://github.com/modelscope/FunASR/releases/tag/runtime-torch-v0.1.7" in links
    assert "https://github.com/modelscope/FunASR/pull/3892" in links
    text = soup.select_one("article").get_text(" ", strip=True)
    assert "funasr-torch==0.1.7" in text and "funasr-onnx==0.4.11" in text
    assert soup.select_one('[data-editorial="boundary"]')
    assert len(soup.select('[data-editorial="next-step"] a')) == 1


def test_json_example_executes_without_a_model_or_network(article):
    _, soup = article
    code = soup.select_one('code[data-language="python"]').get_text()
    scope = {}
    with contextlib.redirect_stdout(io.StringIO()) as output:
        exec(compile(code, "article-example", "exec"), scope)
    message = json.loads(output.getvalue())
    assert json.loads(message["hotwords"]) == {"\u963f\u91cc\u5df4\u5df4": 20, "\u901a\u4e49\u5b9e\u9a8c\u5ba4": 30}


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_new_article_is_discoverable_without_removing_existing_entrypoints(prefix, tmp_path):
    build(tmp_path)
    for category in ("archive", "explanations"):
        soup = BeautifulSoup((tmp_path / prefix / "blog" / category / "index.html").read_text(), "html.parser")
        cards = soup.select(f'a[href="/{prefix}blog/{SLUG}.html"]')
        assert len(cards) == 1
        assert cards[0].select_one('time[datetime="2026-11-13"]')
