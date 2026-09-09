"""Selected articles answer one reader question without changing tested recipes."""

import hashlib
import json
from pathlib import Path
import re
import sys

from bs4 import BeautifulSoup
import pytest


SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

ARTICLES = {
    "funclip-v2-2-0-moss-speaker-clipping": {
        "titles": ("把多人录音变成可剪辑的字幕", "Turn a conversation into editable subtitles"),
        "published": "2026-08-31",
        "codes": ["329cac5e850c65af46316133408a35e54922a49f1bff99a17606d8278187741c", "bba936173d3caebcb982dced324785a3d45748786a1f8c0bd07db6b9515eaa9b", "9f793e6ff863cfa476a93f8324efa30d5f0a993bbec9140839a1cf0c33ca6674", "6fc5f4fd13841d22e0f30e76c0eba5a69240530895baaf628a8dc8ea7d53bd4a"],
    },
    "fun-asr-nano-transformers": {
        "titles": ("接入 Transformers，先选对权重", "Choose the right checkpoint for Transformers"),
        "published": "2026-09-09", "codes": [],
    },
    "funasr-transcribe-long-audio": {
        "titles": ("长录音为什么会漏掉结尾？", "Why can a long transcript miss the ending?"),
        "published": "2026-06-17",
        "codes": ["3a5fbb7f534c11edd7693e1e0a70cf8b66b35cf737f7669d2d6f5d19953f0745"],
    },
    "self-hosted-openai-whisper-api-alternative": {
        "titles": ("把语音转写接进自己的 API", "Add transcription to your own API"),
        "published": "2026-06-18",
        "codes": ["ffaeacf00f8f84adbc0e2d5c3291412987427592ff9f5518e82cd6e3955062e4", "f87d523ad71de6c2d5f39bb7a549031c98be8feed3663e27f3419cd97179e747", "006d3d5b258d15e2c1be5822b7268b94923d584a730f0fcdd647b9beb4372d05", "4d4613a7fa2355b3134f7d4e932bb69dfac48a1fbac2485da79815e9064cef02"],
    },
    "meeting-transcript-acceptance": {
        "titles": ("会议转写，怎样才算做好了？", "When is a meeting transcript ready to use?"),
        "published": "2026-09-09",
        "codes": ["2bfbe86981951ad5f7518cb370f77ffbf8042ae264b94e6300113866dc43671d", "c5c350081e1c145ce5701c69388d57eba97b6ec92056c2bd897dc4ccc88a33a2"],
    },
}


@pytest.fixture(scope="module", params=["source", "built"])
def tree(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("editorial-articles")
    build(output)
    return output


@pytest.fixture(params=[(slug, prefix) for slug in ARTICLES for prefix in ("", "en/")],
                ids=[slug + ("-en" if prefix else "-zh") for slug in ARTICLES for prefix in ("", "en/")])
def article(request, tree):
    slug, prefix = request.param
    soup = BeautifulSoup((tree / prefix / "blog" / (slug + ".html")).read_text(), "html.parser")
    return slug, prefix, soup


def visible(node):
    assert node is not None, "Missing editorial content, not just metadata"
    for parent in (node, *node.parents):
        assert not parent.has_attr("hidden") and parent.get("aria-hidden") != "true"
        assert not re.search(r"display\s*:\s*none|visibility\s*:\s*hidden", parent.get("style", ""))
        assert parent.name != "details" or parent.has_attr("open"), "Core conclusion must not require opening an appendix"
    text = " ".join(node.get_text(" ", strip=True).split())
    assert text
    return text


def test_title_metadata_and_existing_routes_stay_in_sync(article):
    slug, prefix, soup = article
    expected = ARTICLES[slug]["titles"][bool(prefix)]
    assert visible(soup.select_one("article h1")) == expected
    assert soup.select_one('meta[property="og:title"]')["content"] == expected
    assert soup.title.get_text().startswith(expected)
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["headline"] == expected
    assert metadata["datePublished"] == ARTICLES[slug]["published"]
    assert metadata["dateModified"] == "2026-09-09"
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{slug}.html"
    peer = "" if prefix else "en/"
    assert soup.select_one(f'link[rel="alternate"][href="https://www.funasr.com/{peer}blog/{slug}.html"]')
    desc = soup.select_one('meta[name="description"]')["content"]
    assert 20 <= len(desc) <= (220 if prefix else 110)


def test_opening_serves_a_reader_and_example_before_technical_inventory(article):
    _, prefix, soup = article
    body = soup.select_one("article")
    opening = body.select_one('[data-editorial="opening"]')
    example = body.select_one('[data-editorial="example"]')
    text = visible(opening)
    assert len(text) <= (440 if prefix else 180)
    assert not opening.select("code, table, pre"), "Do not turn the opening back into a parameter list"
    assert re.search(r"you|your|developer|team|reader|你|团队|开发者|适合", text, re.I)
    assert len(visible(example)) >= (65 if prefix else 30)
    order = {id(n): i for i, n in enumerate(body.descendants)}
    first_h2 = body.select_one("h2")
    assert order[id(opening)] < order[id(example)] < order[id(first_h2)]
    assert opening.find_parent(attrs={"data-editorial": "appendix"}) is None


def test_constraints_remain_near_conclusion_and_one_next_step(article):
    _, _, soup = article
    boundary = soup.select_one('article [data-editorial="boundary"]')
    assert re.search(r"not|cannot|only|doesn't|不是|不等于|不能|只|不保证", visible(boundary), re.I)
    assert boundary.find_parent(attrs={"data-editorial": "appendix"}) is None
    next_step = soup.select_one('article [data-editorial="next-step"]')
    assert len(next_step.select("a[href]")) == 1
    assert len(visible(next_step)) >= 25
    assert not next_step.find_next("h2"), "End with one action, not another knowledge directory"
    assert len(soup.select("article .post-list li")) <= 3


def test_existing_code_and_link_anchors_are_not_rewritten(article):
    slug, _, soup = article
    # Captured from the exact pre-edit source, identical between languages.
    blocks = [hashlib.sha256(p.get_text().encode()).hexdigest() for p in soup.select("article pre")]
    assert blocks == ARTICLES[slug]["codes"]
    if slug == "self-hosted-openai-whisper-api-alternative":
        for anchor in ("security-boundary", "api-contract"):
            assert len(soup.select(f"[id={anchor}]")) == 1


def test_media_and_historical_commands_have_honest_context(article):
    slug, prefix, soup = article
    for image in soup.select("article img"):
        figure = image.find_parent("figure")
        assert figure is not None and figure.select_one("figcaption"), "Keep the real asset's provenance next to it"
        assert len(visible(figure.select_one("figcaption"))) > 25
    if slug == "funclip-v2-2-0-moss-speaker-clipping":
        block = next(p for p in soup.select("article pre") if "pip install -U vllm" in p.get_text())
        appendix = block.find_parent(attrs={"data-editorial": "appendix"})
        assert appendix is not None
        text = appendix.get_text(" ", strip=True)
        assert re.search(r"historical|历史", text, re.I)
        assert re.search(r"not.*(?:install|compatib)|不是.*安装|不.*兼容", text, re.I)
        assert soup.select_one(f'article a[href="/{prefix}deploy/moss-transcribe-diarize.html"]')
