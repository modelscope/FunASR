"""Reader-facing boundaries for the bilingual model-selection article.

Staged only. Install under web-pages/product-site/tests and run RED against
the old page before integration. These are editorial contracts, not ASR tests.
"""

import datetime
import json
from pathlib import Path

from bs4 import BeautifulSoup
import pytest


SITE = Path(__file__).resolve().parents[1]
SLUG = "which-funasr-model"
SOURCE_REF = "114292dbeff0f8f377bb165e22121cfcfb60cf65"
SECTIONS = {
    "deliverable", "ecosystem", "questions", "candidates", "selection-brief",
    "agent-contract", "verification", "sources",
}
TITLES = {
    "": "FunASR 模型怎么选？先说清你要交付什么",
    "en/": "Choosing a FunASR model starts with the result you need",
}


@pytest.fixture(params=("", "en/"))
def article(request):
    prefix = request.param
    path = SITE / "legacy" / prefix / "blog" / (SLUG + ".html")
    raw = path.read_text(encoding="utf-8")
    return prefix, raw, BeautifulSoup(raw, "html.parser")


def test_old_claims_are_not_retained_in_body_or_metadata(article):
    _, raw, _ = article
    for obsolete in (
        "CER 8.06%", "CER 7.81%", "CER 10.18%", "340×", "340x",
        "CPU 也能实时", "FunASR 全家桶开源(MIT)", "CPU 默认",
        "fastest on CPU", "real-time on CPU", "all MIT",
    ):
        assert obsolete not in raw


def test_preserves_route_publication_date_and_records_revision(article):
    prefix, raw, soup = article
    url = "https://www.funasr.com/" + prefix + "blog/" + SLUG + ".html"
    assert soup.select_one('link[rel="canonical"]')["href"] == url
    meta = json.loads(soup.select_one('script[type="application/ld+json"]').string)
    assert meta["datePublished"] == "2026-06-23"
    assert datetime.date.fromisoformat(meta["dateModified"]) >= datetime.date(2026, 11, 18)
    assert "__REVISION_DATE__" not in raw
    assert soup.select_one("h1").get_text(strip=True) == TITLES[prefix]
    assert meta["headline"] == TITLES[prefix]
    assert soup.select_one('meta[name="description"]')["content"] == meta["description"]


def test_explains_task_before_candidates_and_supplies_selection_brief(article):
    _, _, soup = article
    body = soup.select_one("article")
    headings = [node.get("id") for node in body.select("h2")]
    assert SECTIONS <= set(headings)
    assert headings.index("deliverable") < headings.index("candidates")
    assert body.select_one('[data-editorial="outcome"]')
    assert len(body.select('[data-example="selection-brief"] > li')) == 6
    assert body.select_one('[data-editorial="scope"]')
    assert not body.select("pre"), "This is a selection explainer, not an untested install recipe"


def test_sources_and_next_step_are_explicit(article):
    prefix, _, soup = article
    links = [node["href"] for node in soup.select("article a[href]")]
    root = "https://github.com/modelscope/FunASR/blob/" + SOURCE_REF + "/"
    assert root + "README.md#why-funasr" in links
    for path in ("docs/model_selection.md", "docs/repository_roles.md", "docs/deployment_matrix.md", "docs/python_api.md"):
        assert root + path in links
    assert "/" + prefix + "blog/speech-to-text-python-transcribe-audio.html" in links
    assert soup.select_one('[data-editorial="next-step"]')


def test_catalogue_marks_only_this_revised_entry_with_new_metadata():
    data = json.loads((SITE / "data/blog.json").read_text())
    matches = [row for row in data["articles"] if row["slug"] == SLUG]
    assert len(matches) == 1
    entry = matches[0]
    assert entry["reviewed"] is True
    assert entry["category"] == "selection"
    for lang, prefix in (("zh", ""), ("en", "en/")):
        assert entry[lang]["title"] == TITLES[prefix]
        assert entry[lang]["summary"]


def test_languages_have_matching_structure_and_citation_targets():
    pages = []
    for prefix in ("", "en/"):
        path = SITE / "legacy" / prefix / "blog" / (SLUG + ".html")
        pages.append(BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser"))
    assert [h.get("id") for h in pages[0].select("article h2")] == [h.get("id") for h in pages[1].select("article h2")]
    assert {a["href"] for a in pages[0].select('article a[href^="https://github.com/"]')} == {a["href"] for a in pages[1].select('article a[href^="https://github.com/"]')}
