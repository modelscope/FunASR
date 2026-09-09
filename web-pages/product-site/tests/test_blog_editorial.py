"""The blog entry is a curated publication, not the complete article archive."""

import copy
import importlib
import json
import sys
from pathlib import Path

import pytest
from bs4 import BeautifulSoup

SITE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SITE))
from build import build


@pytest.fixture(scope="module")
def output(tmp_path_factory):
    root = tmp_path_factory.mktemp("blog-editorial")
    build(root)
    return root


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_home_is_an_edited_selection(output, prefix):
    soup = BeautifulSoup((output / prefix / "blog/index.html").read_text(), "html.parser")
    main = soup.select_one("main[data-blog-view='home']") or soup.select_one("main [data-blog-view='home']")
    assert main is not None, "Render a curated blog homepage"
    assert len(main.select("a[data-blog-story]")) == 5
    assert len(main.select("[data-blog-lead]")) == 1
    assert len(main.select("[data-blog-selected] a[data-blog-story]")) == 4
    assert len({a["href"] for a in main.select("a[data-blog-story]")}) == 5
    assert main.select_one("h1").get_text(strip=True).startswith("FunASR")
    assert not main.select(".previous-release, .launch-feature")
    image = main.select_one("[data-blog-lead] img")
    assert image and image.get("alt")
    assert (output / image["src"].lstrip("/")).is_file()
    assert all(a.get("data-blog-category") != "releases" for a in main.select("a[data-blog-story]"))


def test_catalogue_covers_every_legacy_article_and_no_paused_draft():
    blog = importlib.import_module("blog")
    data = blog.load_blog(SITE)
    slugs = {entry["slug"] for entry in data["articles"]}
    for prefix in ("", "en/"):
        actual = {p.stem for p in (SITE / "legacy" / prefix / "blog").glob("*.html") if p.name != "index.html"}
        assert slugs == actual
    assert "recoverable-batch-transcription" not in slugs
    selected = [data["lead"], *data["selected"]]
    assert len(selected) == len(set(selected)) == 5
    assert all(next(e for e in data["articles"] if e["slug"] == s)["reviewed"] for s in selected)


@pytest.mark.parametrize("mutation", ("duplicate", "unsafe", "translation", "selection", "category", "missing", "image", "review"))
def test_invalid_catalogue_is_rejected(mutation):
    blog = importlib.import_module("blog")
    data = json.loads((SITE / "data/blog.json").read_text())
    if mutation == "duplicate":
        data["articles"].append(copy.deepcopy(data["articles"][0]))
    elif mutation == "unsafe":
        data["articles"][0]["slug"] = "../../secrets"
    elif mutation == "translation":
        del data["articles"][0]["en"]
    elif mutation == "selection":
        data["selected"][0] = data["lead"]
    elif mutation == "category":
        data["articles"][0]["category"] = "anything"
    elif mutation == "image":
        data["lead_image"] = "/img/not-a-real-image.png"
    elif mutation == "review":
        next(e for e in data["articles"] if e["slug"] == data["lead"])["reviewed"] = False
    else:
        data["articles"].pop()
    with pytest.raises(ValueError):
        blog.validate_blog(data, SITE)


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_archive_preserves_all_routes_and_category_membership(output, prefix):
    data = importlib.import_module("blog").load_blog(SITE)
    archive = BeautifulSoup((output / prefix / "blog/archive/index.html").read_text(), "html.parser")
    expected = {f"/{prefix}blog/{e['slug']}.html" for e in data["articles"]}
    assert {a["href"] for a in archive.select("a[data-blog-story]")} == expected
    for href in expected:
        assert (output / href.lstrip("/")).is_file()
    for category in ("applications", "selection", "explanations", "releases"):
        soup = BeautifulSoup((output / prefix / "blog" / category / "index.html").read_text(), "html.parser")
        expected_category = {f"/{prefix}blog/{e['slug']}.html" for e in data["articles"]
                             if e["category"] == category and (e["reviewed"] or category == "releases")}
        assert {a["href"] for a in soup.select("a[data-blog-story]")} == expected_category
        assert expected_category


@pytest.mark.parametrize("prefix", ("", "en/"))
def test_blog_routes_have_metadata_and_crawlable_navigation(output, prefix):
    for view in ("", "applications/", "selection/", "explanations/", "archive/", "releases/"):
        route = f"/{prefix}blog/{view}"
        soup = BeautifulSoup((output / route.lstrip("/") / "index.html").read_text(), "html.parser")
        assert soup.find("link", rel="canonical")["href"] == "https://www.funasr.com" + route
        other = "" if prefix else "en/"
        assert soup.select_one(f'link[rel="alternate"][href="https://www.funasr.com/{other}blog/{view}"]')
        assert len(soup.select("h1")) == 1
        assert soup.select_one('script[type="application/ld+json"]')
        for a in soup.select("[data-blog-navigation] a, [data-blog-more] a"):
            assert (output / a["href"].lstrip("/") / "index.html").is_file()
