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
from typing import Any, Dict, Optional, Sequence

DEFAULT_REPO = "modelscope/FunASR"
DEFAULT_ECOSYSTEM_REPOS = [
    "modelscope/FunASR",
    "FunAudioLLM/Fun-ASR",
    "FunAudioLLM/SenseVoice",
    "modelscope/FunClip",
]
DEFAULT_PACKAGE = "funasr"
DEFAULT_BASELINE_STARS = 31224
DEFAULT_TARGET_ADDITIONAL_STARS = 20000


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


def collect_github_repo_metrics(repo: str) -> Dict[str, Any]:
    owner_repo = repo.strip("/")
    github = fetch_json(f"https://api.github.com/repos/{owner_repo}", github_headers())
    return {
        "repo": owner_repo,
        "stars": github.get("stargazers_count"),
        "forks": github.get("forks_count"),
        "watchers": github.get("subscribers_count"),
        "open_issues": github.get("open_issues_count"),
        "default_branch": github.get("default_branch"),
        "pushed_at": github.get("pushed_at"),
        "html_url": github.get("html_url"),
    }


def collect_metrics(repo: str, package: str) -> Dict[str, Any]:
    github = collect_github_repo_metrics(repo)
    pypi = fetch_json(f"https://pypi.org/pypi/{package}/json", {"User-Agent": "funasr-growth-metrics"})
    metrics = {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "github": github,
        "pypi": {
            "package": package,
            "version": pypi.get("info", {}).get("version"),
            "summary": pypi.get("info", {}).get("summary"),
            "project_url": pypi.get("info", {}).get("project_url"),
        },
    }
    return metrics


def collect_ecosystem_metrics(
    repos: Sequence[str],
    package: str,
    baseline_stars: int,
    target_additional_stars: int,
) -> Dict[str, Any]:
    repositories = [collect_github_repo_metrics(repo) for repo in repos]
    pypi = fetch_json(f"https://pypi.org/pypi/{package}/json", {"User-Agent": "funasr-growth-metrics"})
    total_stars = sum(repo.get("stars") or 0 for repo in repositories)
    added_stars = total_stars - baseline_stars
    remaining_to_target = max(target_additional_stars - added_stars, 0)
    return {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "ecosystem": {
            "repositories": repositories,
            "total_stars": total_stars,
            "baseline_stars": baseline_stars,
            "target_additional_stars": target_additional_stars,
            "added_stars": added_stars,
            "remaining_to_target": remaining_to_target,
        },
        "pypi": {
            "package": package,
            "version": pypi.get("info", {}).get("version"),
            "summary": pypi.get("info", {}).get("summary"),
            "project_url": pypi.get("info", {}).get("project_url"),
        },
    }


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


def format_ecosystem_markdown(metrics: Dict[str, Any]) -> str:
    ecosystem = metrics["ecosystem"]
    pypi = metrics["pypi"]
    lines = [
        f"# FunASR Ecosystem Growth Snapshot ({metrics['collected_at_utc']})",
        "",
        f"- Combined GitHub stars: **{ecosystem['total_stars']:,}**",
        f"- Baseline stars: **{ecosystem['baseline_stars']:,}**",
        f"- Added since baseline: **{ecosystem['added_stars']:,}** / {ecosystem['target_additional_stars']:,}",
        f"- Remaining to target: **{ecosystem['remaining_to_target']:,}**",
        f"- PyPI package: **{pypi.get('package')} {pypi.get('version')}**",
        "",
        "## Repositories",
        "",
        "| Repository | Stars | Forks | Open issues | Last push |",
        "|---|---:|---:|---:|---|",
    ]
    for repo in ecosystem["repositories"]:
        lines.append(
            f"| [{repo['repo']}]({repo.get('html_url')}) | "
            f"{(repo.get('stars') or 0):,} | "
            f"{(repo.get('forks') or 0):,} | "
            f"{(repo.get('open_issues') or 0):,} | "
            f"`{repo.get('pushed_at')}` |"
        )
    lines.extend(
        [
            "",
            "## Links",
            "",
            f"- PyPI: {pypi.get('project_url')}",
            "- GitHub Pages: https://modelscope.github.io/FunASR/",
            "- Trendshift: https://trendshift.io/repositories/10479",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect FunASR growth metrics.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repository, e.g. modelscope/FunASR")
    parser.add_argument(
        "--repos",
        nargs="+",
        default=DEFAULT_ECOSYSTEM_REPOS,
        help="GitHub repositories for --ecosystem mode",
    )
    parser.add_argument("--ecosystem", action="store_true", help="Collect the four-repository FunASR ecosystem")
    parser.add_argument("--pypi-package", default=DEFAULT_PACKAGE, help="PyPI package name")
    parser.add_argument("--star-goal", type=int, default=20000, help="Target GitHub star count")
    parser.add_argument("--baseline-stars", type=int, default=DEFAULT_BASELINE_STARS, help="Ecosystem baseline stars")
    parser.add_argument(
        "--target-additional-stars",
        type=int,
        default=DEFAULT_TARGET_ADDITIONAL_STARS,
        help="Ecosystem added-star target",
    )
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.ecosystem:
            metrics = collect_ecosystem_metrics(
                args.repos,
                args.pypi_package,
                args.baseline_stars,
                args.target_additional_stars,
            )
        else:
            metrics = collect_metrics(args.repo, args.pypi_package)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.ecosystem:
        print(format_ecosystem_markdown(metrics), end="")
    else:
        print(format_markdown(metrics, args.star_goal), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
