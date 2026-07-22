import re
import urllib.parse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(
    r"(?<!!)(?:\[[^\]]*\]\(([^)]+)\)|<a\s+[^>]*href=[\"']([^\"']+)[\"'])",
    re.IGNORECASE,
)
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "site", "third_party"}


def _iter_markdown_files():
    for path in ROOT.rglob("*.md"):
        if path.is_symlink():
            continue
        if not any(part in SKIP_DIRS for part in path.parts):
            yield path


def _is_external_or_anchor(url):
    return url.startswith(
        (
            "#",
            "http://",
            "https://",
            "mailto:",
            "tel:",
            "javascript:",
            "data:",
        )
    )


def test_relative_markdown_links_point_to_existing_paths():
    missing = []

    for markdown_path in _iter_markdown_files():
        text = markdown_path.read_text(encoding="utf-8", errors="ignore")
        for match in LINK_PATTERN.finditer(text):
            raw_url = (match.group(1) or match.group(2) or "").strip()
            url = raw_url.split()[0].strip("<>")
            if not url:
                missing.append(f"{markdown_path.relative_to(ROOT)} -> empty link")
                continue
            if _is_external_or_anchor(url):
                continue

            target = urllib.parse.unquote(url.split("#", 1)[0])
            if not target:
                continue

            candidate = (
                ROOT / target.lstrip("/")
                if target.startswith("/")
                else markdown_path.parent / target
            )
            if not candidate.exists():
                missing.append(f"{markdown_path.relative_to(ROOT)} -> {raw_url}")

    assert missing == []
