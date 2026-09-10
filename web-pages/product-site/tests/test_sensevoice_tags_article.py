"""The tags story preserves raw output and avoids unsupported benchmark claims."""

import ast
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "sensevoice-emotion-language-detection"
SOURCE = "4482962437ce8ebd1f0ac5b6793d2f82d2e2955d"


@pytest.fixture(scope="module", params=["source", "built"])
def root(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("tags-story")
    build(output)
    return output


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_raw_prediction_and_display_are_separate(root, prefix):
    soup = BeautifulSoup((root / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    article = soup.select_one("article")
    assert article.select_one('[data-editorial="opening"]')
    assert article.select_one('[data-editorial="boundary"]')
    raw = article.select_one('[data-example="raw-output"]')
    display = article.select_one('[data-example="display-output"]')
    assert raw is not None and display is not None
    assert "<|zh|>" in raw.get_text() and "<|" not in display.get_text()
    assert article.select_one('[data-example="synthetic-display"]')
    assert len(article.select('[data-editorial="next-step"] a')) == 1
    assert article.select_one(f'a[href*="/{SOURCE}/benchmarks/ser/README.md"]')
    assert article.select_one('a[href*="/issues/212"]')
    assert article.select_one('a[data-installation-guide]')
    assert "PyTorch" in article.select_one('[data-editorial="environment"]').get_text()
    assert "ban_emo_unk" in article.get_text()
    assert not any(text in article.get_text() for text in ["15×", "15x", "56 out of 60", "56 段被正确"])
    assert len(article.select("h1")) == 1
    assert len(article.select("h2")) <= 5
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}.html"
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["datePublished"] == "2026-06-19"
    assert metadata["dateModified"] == "2026-09-10"
    assert metadata["headline"] == article.h1.get_text()
    image = article.select_one("figure img")
    assert image is not None and image.get("alt")
    assert (root / image["src"].lstrip("/")).is_file()


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_example_keeps_unknown_predictions_and_uses_public_sdk(prefix):
    soup = BeautifulSoup((SITE / "legacy" / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    code = soup.select_one('pre[data-example="recognize"]')
    assert code is not None
    tree = ast.parse(code.get_text())
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    model = next(n for n in calls if isinstance(n.func, ast.Name) and n.func.id == "AutoModel")
    assert {k.arg: ast.literal_eval(k.value) for k in model.keywords} == {
        "model": "iic/SenseVoiceSmall", "device": "cpu", "disable_update": True,
    }
    generate = next(n for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "generate")
    options = {k.arg: ast.literal_eval(k.value) for k in generate.keywords}
    assert options == {"input": "audio.wav", "language": "auto", "use_itn": True, "ban_emo_unk": False}
    assert any(isinstance(n.func, ast.Name) and n.func.id == "rich_transcription_postprocess" for n in calls)
    assert not any(isinstance(n.func, ast.Attribute) and n.func.attr in {"findall", "split"} for n in calls)


def test_story_is_reviewed_without_growing_the_homepage():
    data = json.loads((SITE / "data/blog.json").read_text())
    row = next(entry for entry in data["articles"] if entry["slug"] == SLUG)
    assert row["reviewed"] and row["category"] == "explanations"
    assert row["zh"]["summary"] and row["en"]["summary"]
    assert data["lead"] == "funclip-v2-2-0-moss-speaker-clipping"
    assert data["selected"] == [
        "meeting-transcript-acceptance", "fun-asr-nano-transformers",
        "self-hosted-openai-whisper-api-alternative", "funasr-transcribe-long-audio",
    ]
