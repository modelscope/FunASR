#!/usr/bin/env python3
"""Collect lightweight growth metrics for the FunASR repository.

The script uses only Python's standard library and public APIs by default.
Set GITHUB_TOKEN to raise GitHub API rate limits in CI or release workflows.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

DEFAULT_REPO = "modelscope/FunASR"
DEFAULT_PACKAGE = "funasr"


def fetch_json(url: str, headers: Optional[Dict[str, str]] = None) -> Any:
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GET {url} failed with HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"GET {url} failed: {exc.reason}") from exc


def github_headers() -> Dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "funasr-growth-metrics",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def collect_metrics(repo: str, package: str) -> Dict[str, Any]:
    owner_repo = repo.strip("/")
    github = fetch_json(f"https://api.github.com/repos/{owner_repo}", github_headers())
    pypi = fetch_json(f"https://pypi.org/pypi/{package}/json", {"User-Agent": "funasr-growth-metrics"})
    latest_release = github.get("pushed_at")
    metrics = {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "github": {
            "repo": owner_repo,
            "stars": github.get("stargazers_count"),
            "forks": github.get("forks_count"),
            "watchers": github.get("subscribers_count"),
            "open_issues": github.get("open_issues_count"),
            "default_branch": github.get("default_branch"),
            "pushed_at": latest_release,
            "html_url": github.get("html_url"),
        },
        "pypi": {
            "package": package,
            "version": pypi.get("info", {}).get("version"),
            "summary": pypi.get("info", {}).get("summary"),
            "project_url": pypi.get("info", {}).get("project_url"),
        },
    }
    return metrics


def format_markdown(metrics: Dict[str, Any], star_goal: int) -> str:
    github = metrics["github"]
    pypi = metrics["pypi"]
    stars = github.get("stars") or 0
    remaining = max(star_goal - stars, 0)
    lines = [
        f"# FunASR Growth Snapshot ({metrics['collected_at_utc']})",
        "",
        f"- GitHub stars: **{stars:,}** / {star_goal:,} ({remaining:,} remaining)",
        f"- GitHub forks: **{github.get('forks'):,}**" if github.get("forks") is not None else "- GitHub forks: n/a",
        f"- GitHub watchers: **{github.get('watchers'):,}**" if github.get("watchers") is not None else "- GitHub watchers: n/a",
        f"- Open issues: **{github.get('open_issues'):,}**" if github.get("open_issues") is not None else "- Open issues: n/a",
        f"- PyPI package: **{pypi.get('package')} {pypi.get('version')}**",
        f"- Last GitHub push: `{github.get('pushed_at')}`",
        "",
        "## Links",
        "",
        f"- Repository: {github.get('html_url')}",
        f"- PyPI: {pypi.get('project_url')}",
        "- GitHub Pages: https://modelscope.github.io/FunASR/",
        "- Trendshift: https://trendshift.io/repositories/10479",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FunASR growth metrics.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository, e.g. modelscope/FunASR")
    parser.add_argument("--pypi-package", default=DEFAULT_PACKAGE, help="PyPI package name")
    parser.add_argument("--star-goal", type=int, default=20000, help="Target GitHub star count")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        metrics = collect_metrics(args.repo, args.pypi_package)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(format_markdown(metrics, args.star_goal), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
