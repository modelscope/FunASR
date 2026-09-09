"""The continual-learning story separates examples from measured evidence."""

import json
from pathlib import Path
import sys

from bs4 import BeautifulSoup
import pytest

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build

SLUG = "sensevoice-finetuning-acceptance"


@pytest.fixture(scope="module", params=["source", "built"])
def root(request, tmp_path_factory):
    if request.param == "source":
        return SITE / "legacy"
    output = tmp_path_factory.mktemp("continual-story")
    build(output)
    return output


@pytest.mark.parametrize("prefix", ["", "en/"])
def test_article_has_a_reader_path_and_honest_example(root, prefix):
    soup = BeautifulSoup((root / prefix / "blog" / f"{SLUG}.html").read_text(), "html.parser")
    article = soup.select_one("article")
    assert len(article.select("h1")) == 1
    assert article.select_one('[data-editorial="opening"]')
    example = article.select_one('[data-editorial="example"]').get_text()
    assert "synthetic" in example.lower() or "合成" in example
    boundary = article.select_one('[data-editorial="boundary"]').get_text()
    assert "not" in boundary.lower() or "不等于" in boundary
    assert len(article.select('[data-editorial="next-step"] a')) == 1
    assert article.select_one('a[href*="issues/3388#issuecomment-5578763142"]')
    assert article.select_one('a[href*="/blob/v1.4.15/examples/industrial_data_pretraining/sense_voice/CONTINUAL_FINETUNING"]')
    assert article.select_one('a[href*="/pull/3677"]')
    assert len(article.select("pre")) == 0, "Link the maintained recipe; do not duplicate it"
    assert len(article.select("tbody tr")) == 5
    image = article.select_one("figure img")
    assert image.get("alt") and (root / image["src"].lstrip("/")).is_file()
    assert article.select_one("figure figcaption")
    assert soup.select_one('link[rel="canonical"]')["href"] == f"https://www.funasr.com/{prefix}blog/{SLUG}.html"
    metadata = json.loads(soup.select_one('script[type="application/ld+json"]').get_text())
    assert metadata["datePublished"] == "2026-09-10"
    assert metadata["headline"] == article.h1.get_text()


def test_synthetic_counts_support_the_decision_not_a_benchmark():
    data = json.loads((SITE / "data/continual-eval-example.json").read_text())
    assert data["synthetic"] is True and data["old_domain_limit_pp"] == 1
    rows = data["slices"]
    assert len(rows) == 5 and all(row["reference_characters"] == 1000 for row in rows)
    assert [sum(row[name] for row in rows) / 50 for name in ("baseline_errors", "a_errors", "b_errors")] == [19, 12.4, 12.9]
    old = [row for row in rows if row["old_domain"]]
    assert max((r["a_errors"] - r["baseline_errors"]) / 10 for r in old) == 6
    assert max((r["b_errors"] - r["baseline_errors"]) / 10 for r in old) == 0.5


def test_story_is_discoverable_without_expanding_homepage():
    data = json.loads((SITE / "data/blog.json").read_text())
    row = next(entry for entry in data["articles"] if entry["slug"] == SLUG)
    assert row["reviewed"] and row["category"] == "explanations"
    assert len(data["selected"]) == 4 and SLUG not in data["selected"]
