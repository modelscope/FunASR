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
DEFAULT_TARGET_DATE = "2026-09-30"
DEFAULT_INTEGRATION_PRS = [
    "huggingface/transformers#46180",
    "ray-project/ray#64053",
    "huggingface/optimum-intel#1801",
]
FAILED_CHECK_CONCLUSIONS = {"action_required", "cancelled", "failure", "startup_failure", "timed_out"}


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
    open_pull_requests = len(
        fetch_json(f"https://api.github.com/repos/{owner_repo}/pulls?state=open&per_page=100", github_headers())
    )
    open_items = github.get("open_issues_count")
    open_issues = max((open_items or 0) - open_pull_requests, 0) if open_items is not None else None
    return {
        "repo": owner_repo,
        "stars": github.get("stargazers_count"),
        "forks": github.get("forks_count"),
        "watchers": github.get("subscribers_count"),
        "open_items": open_items,
        "open_issues": open_issues,
        "open_pull_requests": open_pull_requests,
        "default_branch": github.get("default_branch"),
        "pushed_at": github.get("pushed_at"),
        "html_url": github.get("html_url"),
    }


def parse_pull_request_spec(spec: str) -> tuple[str, int]:
    if "#" not in spec:
        raise ValueError(f"pull request spec must look like owner/repo#number: {spec}")
    repo, number = spec.split("#", 1)
    repo = repo.strip("/")
    try:
        pr_number = int(number)
    except ValueError as exc:
        raise ValueError(f"pull request number must be an integer: {spec}") from exc
    if repo.count("/") != 1 or pr_number <= 0:
        raise ValueError(f"pull request spec must look like owner/repo#number: {spec}")
    return repo, pr_number


def summarize_commit_checks(repo: str, head_sha: str) -> Dict[str, Any]:
    status = fetch_json(f"https://api.github.com/repos/{repo}/commits/{head_sha}/status", github_headers())
    check_runs_payload = fetch_json(
        f"https://api.github.com/repos/{repo}/commits/{head_sha}/check-runs?per_page=100",
        github_headers(),
    )
    check_runs = check_runs_payload.get("check_runs", [])
    failed_check_runs = []
    pending_check_runs = []
    for check_run in check_runs:
        name = check_run.get("name")
        url = check_run.get("html_url") or check_run.get("details_url")
        if check_run.get("status") != "completed":
            pending_check_runs.append({"name": name, "status": check_run.get("status"), "url": url})
        elif check_run.get("conclusion") in FAILED_CHECK_CONCLUSIONS:
            failed_check_runs.append({"name": name, "conclusion": check_run.get("conclusion"), "url": url})

    status_state = status.get("state")
    if failed_check_runs or status_state in {"error", "failure"}:
        state = "failure"
    elif pending_check_runs or status_state == "pending":
        state = "pending"
    elif check_runs or status.get("statuses"):
        state = "success"
    else:
        state = "unknown"

    return {
        "state": state,
        "commit_status_state": status_state,
        "total_check_runs": check_runs_payload.get("total_count", len(check_runs)),
        "failed_check_runs": failed_check_runs,
        "pending_check_runs": pending_check_runs,
    }


def collect_pull_request_metrics(spec: str) -> Dict[str, Any]:
    repo, pr_number = parse_pull_request_spec(spec)
    pull_request = fetch_json(f"https://api.github.com/repos/{repo}/pulls/{pr_number}", github_headers())
    head = pull_request.get("head") or {}
    base = pull_request.get("base") or {}
    author = pull_request.get("user") or {}
    head_sha = head.get("sha")
    checks = summarize_commit_checks(repo, head_sha) if head_sha else {"state": "unknown"}
    return {
        "pr": f"{repo}#{pr_number}",
        "repo": repo,
        "number": pr_number,
        "title": pull_request.get("title"),
        "state": pull_request.get("state"),
        "draft": pull_request.get("draft"),
        "mergeable": pull_request.get("mergeable"),
        "mergeable_state": pull_request.get("mergeable_state"),
        "updated_at": pull_request.get("updated_at"),
        "html_url": pull_request.get("html_url"),
        "head_ref": head.get("ref"),
        "head_sha": head_sha,
        "base_ref": base.get("ref"),
        "author": author.get("login"),
        "checks": checks,
    }


def collect_integration_metrics(prs: Sequence[str]) -> Dict[str, Any]:
    return {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "integrations": [collect_pull_request_metrics(pr) for pr in prs],
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
    target_date: str = DEFAULT_TARGET_DATE,
    today: Optional[dt.date] = None,
) -> Dict[str, Any]:
    repositories = [collect_github_repo_metrics(repo) for repo in repos]
    pypi = fetch_json(f"https://pypi.org/pypi/{package}/json", {"User-Agent": "funasr-growth-metrics"})
    total_stars = sum(repo.get("stars") or 0 for repo in repositories)
    added_stars = total_stars - baseline_stars
    remaining_to_target = max(target_additional_stars - added_stars, 0)
    target_day = dt.date.fromisoformat(target_date)
    current_day = today or dt.datetime.now(dt.timezone.utc).date()
    days_remaining = max((target_day - current_day).days, 0)
    required_daily_average = (
        (remaining_to_target + days_remaining - 1) // days_remaining if days_remaining else remaining_to_target
    )
    return {
        "collected_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "ecosystem": {
            "repositories": repositories,
            "total_stars": total_stars,
            "baseline_stars": baseline_stars,
            "target_additional_stars": target_additional_stars,
            "added_stars": added_stars,
            "remaining_to_target": remaining_to_target,
            "target_date": target_date,
            "days_remaining": days_remaining,
            "required_daily_average": required_daily_average,
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
        f"- Open pull requests: **{github.get('open_pull_requests'):,}**"
        if github.get("open_pull_requests") is not None
        else "- Open pull requests: n/a",
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
        f"- Target date: **{ecosystem['target_date']}** ({ecosystem['days_remaining']:,} days remaining)",
        f"- Required daily average: **{ecosystem['required_daily_average']:,}** stars/day",
        f"- PyPI package: **{pypi.get('package')} {pypi.get('version')}**",
        "",
        "## Repositories",
        "",
        "| Repository | Stars | Forks | Open issues | Open PRs | Last push |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for repo in ecosystem["repositories"]:
        lines.append(
            f"| [{repo['repo']}]({repo.get('html_url')}) | "
            f"{(repo.get('stars') or 0):,} | "
            f"{(repo.get('forks') or 0):,} | "
            f"{(repo.get('open_issues') or 0):,} | "
            f"{(repo.get('open_pull_requests') or 0):,} | "
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


def format_integration_markdown(metrics: Dict[str, Any]) -> str:
    lines = [
        f"# FunASR External Integration Snapshot ({metrics['collected_at_utc']})",
        "",
        "| Pull request | State | Mergeable | Checks | Failed | Pending | Updated |",
        "|---|---|---|---|---:|---:|---|",
    ]
    for integration in metrics["integrations"]:
        checks = integration.get("checks", {})
        failed_count = len(checks.get("failed_check_runs") or [])
        pending_count = len(checks.get("pending_check_runs") or [])
        lines.append(
            f"| [{integration['pr']}]({integration.get('html_url')}) | "
            f"{integration.get('state')} | "
            f"{integration.get('mergeable_state') or integration.get('mergeable')} | "
            f"{checks.get('state')} | "
            f"{failed_count:,} | "
            f"{pending_count:,} | "
            f"`{integration.get('updated_at')}` |"
        )
    lines.extend(["", "## Failed or pending checks", ""])
    for integration in metrics["integrations"]:
        checks = integration.get("checks", {})
        failed = checks.get("failed_check_runs") or []
        pending = checks.get("pending_check_runs") or []
        if not failed and not pending:
            continue
        lines.append(f"### {integration['pr']}")
        for check in failed:
            url = check.get("url")
            suffix = f" ({url})" if url else ""
            lines.append(f"- Failed: {check.get('name')} [{check.get('conclusion')}]{suffix}")
        for check in pending:
            url = check.get("url")
            suffix = f" ({url})" if url else ""
            lines.append(f"- Pending: {check.get('name')} [{check.get('status')}]{suffix}")
        lines.append("")
    if lines[-1] == "## Failed or pending checks":
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


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
    parser.add_argument("--integrations", action="store_true", help="Collect tracked external integration PRs")
    parser.add_argument(
        "--integration-prs",
        nargs="+",
        default=DEFAULT_INTEGRATION_PRS,
        help="GitHub pull requests for --integrations mode, e.g. owner/repo#123",
    )
    parser.add_argument("--pypi-package", default=DEFAULT_PACKAGE, help="PyPI package name")
    parser.add_argument("--star-goal", type=int, default=20000, help="Target GitHub star count")
    parser.add_argument("--baseline-stars", type=int, default=DEFAULT_BASELINE_STARS, help="Ecosystem baseline stars")
    parser.add_argument(
        "--target-additional-stars",
        type=int,
        default=DEFAULT_TARGET_ADDITIONAL_STARS,
        help="Ecosystem added-star target",
    )
    parser.add_argument("--target-date", default=DEFAULT_TARGET_DATE, help="Ecosystem target date in YYYY-MM-DD format")
    parser.add_argument("--format", choices=["markdown", "json"], default="markdown", help="Output format")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.integrations:
            metrics = collect_integration_metrics(args.integration_prs)
        elif args.ecosystem:
            metrics = collect_ecosystem_metrics(
                args.repos,
                args.pypi_package,
                args.baseline_stars,
                args.target_additional_stars,
                args.target_date,
            )
        else:
            metrics = collect_metrics(args.repo, args.pypi_package)
    except (RuntimeError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1

    if args.format == "json":
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.integrations:
        print(format_integration_markdown(metrics), end="")
    elif args.ecosystem:
        print(format_ecosystem_markdown(metrics), end="")
    else:
        print(format_markdown(metrics, args.star_goal), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
