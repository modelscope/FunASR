"""Punctuation is an editable text candidate, not timing or a semantic guarantee."""
import ast
import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "punctuation-restoration-python"
INPUTS = [
    "我们都是木头人不许说话不许动",
    "the meeting is at 3 pm please bring your laptop and the report",
    "人不是石头人有主观价值",
]
EXPECTED = [
    {"original": INPUTS[0], "candidate": "我们都是木头人，不许说话，不许动。"},
    {"original": INPUTS[1], "candidate": " The meeting is at 3 pm, please bring your laptop and the report."},
    {"original": INPUTS[2], "candidate": "人不是石头人有主观价值。"},
]


@pytest.fixture(scope="module", params=["source", "built"])
def root(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("punctuation-story")
    build(output)
    return output


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_reader_sees_candidate_boundaries_and_one_next_step(root, prefix):
    soup = BeautifulSoup((root / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    article = soup.select_one("article")
    assert article.select_one('[data-editorial="opening"]')
    assert len(article.select("h1")) == 1
    assert len(article.select("h2")) <= 4
    assert "ct-punc" in article.select_one('[data-editorial="environment"]').get_text()
    assert article.select_one('a[data-installation-guide]')
    boundary = article.select_one('[data-editorial="timing-boundary"]').get_text().lower()
    assert ("timestamps" if prefix else "时间戳") in boundary
    assert article.select_one('[data-editorial="native-boundary"] a')["href"] == f"/{prefix}docs/native-transformers.html"
    assert article.select_one('[data-editorial="case-boundary"] a')["href"] == "https://github.com/modelscope/FunASR/issues/3644"
    assert len(article.select('[data-editorial="next-step"] a')) == 1
    assert article.select_one('[data-editorial="manual-edit"]')
    assert not any(text in article.get_text() for text in [
        "任何无标点文本", "同一个模型双语通吃", "全家桶开源(MIT)",
        "Everything below is real measured output", "anything unpunctuated",
        "commercial-friendly",
    ])
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["datePublished"] == "2026-06-23"
    assert metadata["dateModified"] == "2026-09-10"
    assert metadata["headline"] == article.h1.get_text()
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}.html"


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_example_keeps_raw_text_and_observed_counterexample(root, prefix):
    soup = BeautifulSoup((root / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    code = soup.select_one('pre[data-example="punctuate"]')
    assert code is not None
    tree = ast.parse(code.get_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    model = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "AutoModel")
    assert {kw.arg: ast.literal_eval(kw.value) for kw in model.keywords} == {
        "model": "ct-punc", "device": "cpu", "disable_update": True,
    }
    text_assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == "texts" for t in node.targets))
    assert ast.literal_eval(text_assignment.value) == INPUTS
    assert '"original": original' in code.get_text()
    assert ".strip(" not in code.get_text()
    assert json.loads(soup.select_one('pre[data-example="observed"]').get_text()) == EXPECTED
    assert "人不是石头，人有主观价值。" in soup.select_one('[data-editorial="manual-edit"]').get_text()


def test_bilingual_examples_match_and_homepage_is_not_expanded():
    examples = []
    for prefix in ["", "en/"]:
        soup = BeautifulSoup((SITE / "legacy" / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
        code = soup.select_one('pre[data-example="punctuate"]')
        assert code is not None
        examples.append(code.get_text())
    assert examples[0] == examples[1]
    data = json.loads((SITE / "data/blog.json").read_text())
    row = next(entry for entry in data["articles"] if entry["slug"] == SLUG)
    assert row["reviewed"] and row["category"] == "explanations"
    assert data["lead"] == "funclip-v2-2-0-moss-speaker-clipping"
    assert data["selected"] == [
        "meeting-transcript-acceptance", "fun-asr-nano-transformers",
        "self-hosted-openai-whisper-api-alternative", "funasr-transcribe-long-audio",
    ]
