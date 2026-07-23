#!/usr/bin/env python3
"""Aggregate privacy-safe funasr.com traffic metrics from Nginx access logs."""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import gzip
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator, NamedTuple, Optional, Sequence
from urllib.parse import unquote, urlsplit


LOG_PATTERN = re.compile(
    r'^(?P<client_ip>\S+) \S+ \S+ \[(?P<timestamp>[^]]+)] '
    r'"(?P<method>[A-Z]+) (?P<target>\S+) [^"]+" '
    r'(?P<status>\d{3}) \S+ "(?P<referrer>[^"]*)" "(?P<user_agent>[^"]*)"$'
)
BOT_PATTERN = re.compile(
    r"bot|spider|crawler|slurp|headless|lighthouse|pagespeed|curl/|wget/|"
    r"python-requests|aiohttp|scrapy|go-http-client|uptime|monitoring|zgrab|masscan",
    re.IGNORECASE,
)
STATIC_EXTENSIONS = {
    ".avif",
    ".css",
    ".eot",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".map",
    ".mp3",
    ".mp4",
    ".ogg",
    ".pdf",
    ".png",
    ".svg",
    ".txt",
    ".wav",
    ".webm",
    ".webp",
    ".woff",
    ".woff2",
    ".xml",
    ".zip",
}
IGNORED_PREFIXES = ("/.well-known/", "/assets/", "/fonts/", "/images/", "/static/")
DOC_PREFIXES = ("/docs/", "/en/docs/", "/guide/", "/en/guide/")
DOC_FILENAMES = {
    "agent.html",
    "benchmark.html",
    "cli-tutorial.html",
    "download.html",
    "ecosystem.html",
    "llama-cpp.html",
    "models.html",
    "quickstart.html",
    "vs-whisper.html",
}
SECTIONS = ("homepage", "docs", "blog", "donors", "other")


class AccessRecord(NamedTuple):
    client_ip: str
    timestamp: dt.datetime
    method: str
    path: str
    status: int
    referrer_host: str
    user_agent: str


def normalize_path(target: str) -> str:
    path = unquote(urlsplit(target).path or "/")
    if not path.startswith("/"):
        path = f"/{path}"
    return path


def referrer_host(referrer: str) -> str:
    if not referrer or referrer == "-":
        return ""
    return (urlsplit(referrer).hostname or "").lower()


def parse_access_log_line(line: str) -> Optional[AccessRecord]:
    match = LOG_PATTERN.match(line.rstrip("\n"))
    if match is None:
        return None
    try:
        timestamp = dt.datetime.strptime(match.group("timestamp"), "%d/%b/%Y:%H:%M:%S %z")
        status = int(match.group("status"))
    except ValueError:
        return None
    return AccessRecord(
        client_ip=match.group("client_ip"),
        timestamp=timestamp,
        method=match.group("method"),
        path=normalize_path(match.group("target")),
        status=status,
        referrer_host=referrer_host(match.group("referrer")),
        user_agent=match.group("user_agent"),
    )


def is_page_path(path: str) -> bool:
    lowered = path.lower()
    if lowered in {"/favicon.ico", "/robots.txt", "/sitemap.xml"}:
        return False
    if lowered.startswith(IGNORED_PREFIXES):
        return False
    return Path(lowered).suffix not in STATIC_EXTENSIONS


def classify_page(path: str) -> str:
    lowered = path.lower()
    if lowered in {"/", "/index.html", "/en", "/en/", "/en/index.html"}:
        return "homepage"
    if lowered.startswith(("/blog/", "/en/blog/")):
        return "blog"
    if lowered in {"/donors.html", "/en/donors.html"}:
        return "donors"
    if lowered.startswith(DOC_PREFIXES) or Path(lowered).name in DOC_FILENAMES:
        return "docs"
    return "other"


def anonymized_visitor(record: AccessRecord, salt: bytes) -> bytes:
    value = f"{record.client_ip}\0{record.user_agent}".encode("utf-8", errors="replace")
    return hashlib.blake2b(value, key=salt, digest_size=16).digest()


def aggregate_records(
    records: Iterable[Optional[AccessRecord]],
    *,
    now: dt.datetime,
    days: int,
    top_limit: int,
    visitor_salt: Optional[bytes] = None,
) -> dict:
    if now.tzinfo is None:
        raise ValueError("now must include a timezone")
    if days <= 0:
        raise ValueError("days must be positive")
    if top_limit <= 0:
        raise ValueError("top_limit must be positive")

    cutoff = now - dt.timedelta(days=days)
    salt = visitor_salt or os.urandom(16)
    page_views = 0
    visitors = set()
    section_views = Counter()
    section_visitors = defaultdict(set)
    path_views = Counter()
    path_visitors = defaultdict(set)
    referrer_views = Counter()
    filtered = Counter()

    for record in records:
        if record is None:
            filtered["malformed"] += 1
            continue
        record_time = record.timestamp.astimezone(now.tzinfo)
        if record_time < cutoff or record_time > now:
            filtered["outside_window"] += 1
            continue
        if record.method not in {"GET", "HEAD"}:
            filtered["non_page_methods"] += 1
            continue
        if record.status != 200:
            filtered["non_success"] += 1
            continue
        if BOT_PATTERN.search(record.user_agent):
            filtered["bots"] += 1
            continue
        if not is_page_path(record.path):
            filtered["static_assets"] += 1
            continue

        visitor = anonymized_visitor(record, salt)
        section = classify_page(record.path)
        page_views += 1
        visitors.add(visitor)
        section_views[section] += 1
        section_visitors[section].add(visitor)
        path_views[record.path] += 1
        path_visitors[record.path].add(visitor)
        if record.referrer_host and not (
            record.referrer_host == "funasr.com" or record.referrer_host.endswith(".funasr.com")
        ):
            referrer_views[record.referrer_host] += 1

    sections = {
        section: {
            "page_views": section_views[section],
            "unique_visitors": len(section_visitors[section]),
        }
        for section in SECTIONS
    }
    top_paths = [
        {
            "path": path,
            "page_views": views,
            "unique_visitors": len(path_visitors[path]),
        }
        for path, views in sorted(path_views.items(), key=lambda item: (-item[1], item[0]))[:top_limit]
    ]
    top_referrers = [
        {"host": host, "page_views": views}
        for host, views in sorted(referrer_views.items(), key=lambda item: (-item[1], item[0]))[:top_limit]
    ]
    filter_names = (
        "malformed",
        "outside_window",
        "non_page_methods",
        "non_success",
        "bots",
        "static_assets",
    )
    return {
        "collected_at_utc": now.astimezone(dt.timezone.utc).isoformat(),
        "window_start_utc": cutoff.astimezone(dt.timezone.utc).isoformat(),
        "days": days,
        "page_views": page_views,
        "unique_visitors": len(visitors),
        "sections": sections,
        "top_paths": top_paths,
        "top_referrers": top_referrers,
        "filtered": {name: filtered[name] for name in filter_names},
        "privacy": "Aggregate counts only; raw client IPs and user agents are never emitted.",
    }


def iter_log_lines(paths: Iterable[Path]) -> Iterator[str]:
    for path in paths:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle


def resolve_log_paths(patterns: Sequence[str]) -> list[Path]:
    paths = []
    seen = set()
    for pattern in patterns:
        for match in sorted(glob.glob(pattern)):
            path = Path(match)
            if path.is_file() and path not in seen:
                paths.append(path)
                seen.add(path)
    return paths


def collect_traffic(
    paths: Sequence[Path],
    *,
    now: dt.datetime,
    days: int,
    top_limit: int,
) -> dict:
    records = (parse_access_log_line(line) for line in iter_log_lines(paths))
    result = aggregate_records(records, now=now, days=days, top_limit=top_limit)
    result["source_file_count"] = len(paths)
    return result


def parse_now(value: Optional[str]) -> dt.datetime:
    if value is None:
        return dt.datetime.now(dt.timezone.utc)
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise argparse.ArgumentTypeError("--now must include a timezone")
    return parsed


def render_text(metrics: dict) -> str:
    lines = [
        f"funasr.com traffic ({metrics['days']} days)",
        f"Page views: {metrics['page_views']:,}",
        f"Approximate unique visitors: {metrics['unique_visitors']:,}",
    ]
    for section in SECTIONS:
        values = metrics["sections"][section]
        lines.append(
            f"{section}: {values['page_views']:,} views / "
            f"{values['unique_visitors']:,} approximate visitors"
        )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--log",
        action="append",
        required=True,
        help="Access-log path or glob. Repeat for multiple files.",
    )
    parser.add_argument("--days", type=int, default=30, help="Trailing window length (default: 30).")
    parser.add_argument("--top", type=int, default=10, help="Number of top paths/referrers (default: 10).")
    parser.add_argument("--now", help="Timezone-aware ISO timestamp for reproducible backfills.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    args = parser.parse_args(argv)

    try:
        now = parse_now(args.now)
        paths = resolve_log_paths(args.log)
        if not paths:
            parser.error("no access-log files matched --log")
        metrics = collect_traffic(paths, now=now, days=args.days, top_limit=args.top)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    if args.format == "json":
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    else:
        print(render_text(metrics))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
