"""Curated blog navigation; article bodies remain in the legacy corpus."""

import json
import re
from pathlib import Path

from bs4 import BeautifulSoup

SITE_ROOT = Path(__file__).resolve().parent
CATEGORIES = {
    "applications": {"zh": "应用实践", "en": "Applications"},
    "selection": {"zh": "选型指南", "en": "Choosing a solution"},
    "explanations": {"zh": "技术解读", "en": "Technical explanations"},
    "releases": {"zh": "版本记录", "en": "Release history"},
}


def validate_blog(data, root=SITE_ROOT):
    """Fail the build if editorial navigation loses an existing article."""
    if data.get("schema_version") != 1 or not isinstance(data.get("articles"), list):
        raise ValueError("invalid blog catalogue")
    seen = set()
    for entry in data["articles"]:
        slug = entry.get("slug", "")
        if not isinstance(slug, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", slug) or slug in seen:
            raise ValueError("invalid or duplicate blog slug")
        seen.add(slug)
        if entry.get("category") not in CATEGORIES or type(entry.get("reviewed")) is not bool:
            raise ValueError("invalid blog category or review state")
        for language, title_limit, summary_limit in (("zh", 44, 110), ("en", 100, 210)):
            text = entry.get(language)
            if not isinstance(text, dict) or not isinstance(text.get("title"), str) or not 1 <= len(text["title"].strip()) <= title_limit:
                raise ValueError("missing or overlong blog title")
            summary = text.get("summary", "")
            if not isinstance(summary, str) or len(summary) > summary_limit or (entry["reviewed"] and not summary.strip()):
                raise ValueError("missing or overlong blog summary")
    for prefix in ("", "en"):
        actual = {p.stem for p in (root / "legacy" / prefix / "blog").glob("*.html") if p.name != "index.html"}
        if seen != actual:
            raise ValueError("blog catalogue must cover the complete bilingual article corpus")
    selected = [data.get("lead"), *data.get("selected", [])]
    if len(selected) != 5 or len(set(selected)) != 5 or not set(selected) <= seen:
        raise ValueError("blog home requires one lead and four distinct selected stories")
    by_slug = {entry["slug"]: entry for entry in data["articles"]}
    if any(not by_slug[slug]["reviewed"] or by_slug[slug]["category"] == "releases" for slug in selected):
        raise ValueError("only reviewed reader stories belong on the homepage")
    image = data.get("lead_image", "")
    if not isinstance(image, str) or not image.startswith("/img/") or ".." in Path(image).parts or not (root / "legacy" / image.lstrip("/")).is_file():
        raise ValueError("blog lead requires an existing local image")
    return data


def load_blog(root=SITE_ROOT):
    return validate_blog(json.loads((root / "data/blog.json").read_text(encoding="utf-8")), root)


def _published(root, slug, language):
    prefix = "en" if language == "en" else ""
    soup = BeautifulSoup((root / "legacy" / prefix / "blog" / f"{slug}.html").read_text(encoding="utf-8"), "html.parser")
    for node in soup.select('script[type="application/ld+json"]'):
        value = json.loads(node.get_text())
        if isinstance(value, dict) and isinstance(value.get("datePublished"), str):
            date = value["datePublished"][:10]
            if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
                return date
    return ""


def blog_views(data, language, root=SITE_ROOT):
    prefix = "/en" if language == "en" else ""
    peer = "" if language == "en" else "/en"
    rows = {
        entry["slug"]: {
            **entry[language], "slug": entry["slug"], "category": entry["category"],
            "category_label": CATEGORIES[entry["category"]][language], "reviewed": entry["reviewed"],
            "href": f"{prefix}/blog/{entry['slug']}.html", "date": _published(root, entry["slug"], language),
        }
        for entry in data["articles"]
    }
    ordered = sorted(rows.values(), key=lambda row: (row["date"], row["slug"]), reverse=True)
    headings = {
        "home": {"zh": "FunASR 技术博客", "en": "FunASR Blog"},
        "archive": {"zh": "全部文章", "en": "All articles"},
        **CATEGORIES,
    }
    descriptions = {
        "home": {"zh": "从一个真实问题出发，把语音技术用起来。", "en": "Start with a real problem. Put speech technology to work."},
        "applications": {"zh": "从录音、字幕到自己的服务，走完一条应用路径。", "en": "From recordings and subtitles to your own service, one workflow at a time."},
        "selection": {"zh": "根据应用需求选方案，先弄清楚哪些能力真正匹配。", "en": "Choose a path around the capabilities your application actually needs."},
        "explanations": {"zh": "解释一个关键问题，帮助你做出更好的工程判断。", "en": "Understand one important question and make a better engineering decision."},
        "archive": {"zh": "保留全部历史文章。旧文中的版本与测量对应当时环境，当前配置请以维护中的文档为准。", "en": "The complete archive. Versions and measurements in older articles reflect their original environments; use maintained documentation for current setup."},
        "releases": {"zh": "按发布时间回看版本变化；当前安装包与完整记录见 GitHub Releases。", "en": "Changes at the time of each release. Find current packages and complete release notes on GitHub."},
    }
    for view in ("home", "applications", "selection", "explanations", "archive", "releases"):
        suffix = "" if view == "home" else view + "/"
        selected = [rows[slug] for slug in [data["lead"], *data["selected"]]]
        stories = selected if view == "home" else [row for row in ordered if view == "archive" or
                    (row["category"] == view and (row["reviewed"] or view == "releases"))]
        yield {
            "view": view, "heading": headings[view][language], "description": descriptions[view][language],
            "route": f"{prefix}/blog/{suffix}", "peer_route": f"{peer}/blog/{suffix}",
            "stories": stories, "lead": selected[0], "selected": selected[1:], "lead_image": data["lead_image"],
            "prefix": prefix,
            "topics": [{"id": name, "label": ("精选" if language == "zh" else "Selected") if name == "home" else CATEGORIES[name][language],
                        "href": f"{prefix}/blog/" + ("" if name == "home" else name + "/")}
                       for name in ("home", "applications", "selection", "explanations")],
        }
