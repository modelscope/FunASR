"""Subtitle readers get an explicit model and a format conversion, not invented timing."""

import json
from pathlib import Path
import shlex
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "generate-subtitles-srt-vtt-from-audio-video"


@pytest.fixture(scope="module", params=["source", "built"])
def root(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("subtitle-story")
    build(output)
    return output


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_subtitle_story_has_a_bounded_reader_path(root, prefix):
    soup = BeautifulSoup((root / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    article = soup.select_one("article")
    assert article.select_one('[data-editorial="opening"]')
    assert article.select_one('[data-editorial="timing-boundary"]')
    assert article.select_one('[data-editorial="native-boundary"] a[href$="native-transformers.html"]')
    assert article.select_one('a[data-installation-guide]')
    assert "PyTorch" in article.select_one('[data-editorial="environment"]').get_text()
    assert "torchaudio" in article.select_one('[data-editorial="environment"]').get_text()
    assert len(article.select("h1")) == 1
    assert len(article.select("h2")) <= 4
    assert len(article.select('[data-editorial="next-step"] a')) == 1
    assert not any(claim in article.get_text() for claim in [
        "下面的命令和代码都已实测", "远快于 Whisper", "中文更准", "real, usable timestamps",
    ])
    assert not article.select("pre code.language-python")
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["datePublished"] == "2026-06-18"
    assert metadata["dateModified"] == "2026-09-10"
    assert metadata["headline"] == article.h1.get_text()
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}.html"
    image = article.select_one("figure img")
    assert image and image.get("alt") and (root / image["src"].lstrip("/")).is_file()


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_examples_select_paraformer_and_convert_vtt_with_ffmpeg(prefix):
    soup = BeautifulSoup((SITE / "legacy" / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    command = soup.select_one('pre[data-example="transcribe"]')
    assert command is not None
    assert shlex.split(command.get_text()) == [
        "funasr", "audio.wav", "--model", "paraformer", "--device", "cpu",
        "--output-format", "srt", "--output-dir", "./subs",
    ]
    convert = soup.select_one('pre[data-example="convert"]')
    assert convert is not None
    assert shlex.split(convert.get_text()) == [
        "ffmpeg", "-n", "-i", "./subs/audio.srt", "./subs/audio.vtt",
    ]
    assert soup.select_one('pre[data-example="srt-output"]')
    assert soup.select_one('pre[data-example="vtt-output"]')
    assert "--spk" not in command.get_text()


def test_subtitle_story_does_not_expand_homepage_selection():
    data = json.loads((SITE / "data/blog.json").read_text())
    row = next(entry for entry in data["articles"] if entry["slug"] == SLUG)
    assert row["reviewed"] and row["category"] == "applications"
    assert data["lead"] == "funclip-v2-2-0-moss-speaker-clipping"
    assert data["selected"] == [
        "meeting-transcript-acceptance", "fun-asr-nano-transformers",
        "self-hosted-openai-whisper-api-alternative", "funasr-transcribe-long-audio",
    ]
