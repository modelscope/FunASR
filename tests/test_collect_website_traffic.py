import gzip
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_website_traffic.py"
    spec = importlib.util.spec_from_file_location("collect_website_traffic", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def access_line(
    ip,
    timestamp,
    path,
    *,
    status=200,
    referrer="-",
    user_agent="Mozilla/5.0 Chrome/142.0",
):
    return (
        f'{ip} - - [{timestamp}] "GET {path} HTTP/2.0" {status} 123 '
        f'"{referrer}" "{user_agent}"\n'
    )


def test_parse_access_log_line_normalizes_path_and_referrer():
    module = load_module()
    record = module.parse_access_log_line(
        access_line(
            "203.0.113.5",
            "23/Jul/2026:11:00:00 +0800",
            "/docs/guide.html?source=release",
            referrer="https://www.google.com/search?q=funasr",
        )
    )

    assert record.client_ip == "203.0.113.5"
    assert record.timestamp.isoformat() == "2026-07-23T11:00:00+08:00"
    assert record.method == "GET"
    assert record.path == "/docs/guide.html"
    assert record.status == 200
    assert record.referrer_host == "www.google.com"


def test_aggregate_filters_bots_assets_failures_and_old_requests():
    module = load_module()
    lines = [
        access_line("203.0.113.10", "23/Jul/2026:09:00:00 +0800", "/"),
        access_line("203.0.113.10", "23/Jul/2026:09:01:00 +0800", "/docs/quickstart.html"),
        access_line(
            "203.0.113.20",
            "23/Jul/2026:09:02:00 +0800",
            "/en/blog/local-asr.html",
            referrer="https://news.ycombinator.com/item?id=1",
        ),
        access_line("203.0.113.20", "23/Jul/2026:09:03:00 +0800", "/donors.html"),
        access_line(
            "198.51.100.1",
            "23/Jul/2026:09:04:00 +0800",
            "/docs/quickstart.html",
            user_agent="Googlebot/2.1",
        ),
        access_line("198.51.100.2", "23/Jul/2026:09:05:00 +0800", "/assets/app.js"),
        access_line("198.51.100.3", "23/Jul/2026:09:06:00 +0800", "/docs/missing.html", status=404),
        access_line("198.51.100.4", "20/Jun/2026:09:07:00 +0800", "/docs/old.html"),
    ]
    records = [module.parse_access_log_line(line) for line in lines]

    metrics = module.aggregate_records(
        records,
        now=datetime(2026, 7, 23, 12, 0, tzinfo=timezone.utc),
        days=30,
        top_limit=10,
    )

    assert metrics["page_views"] == 4
    assert metrics["unique_visitors"] == 2
    assert metrics["sections"] == {
        "homepage": {"page_views": 1, "unique_visitors": 1},
        "docs": {"page_views": 1, "unique_visitors": 1},
        "blog": {"page_views": 1, "unique_visitors": 1},
        "donors": {"page_views": 1, "unique_visitors": 1},
        "other": {"page_views": 0, "unique_visitors": 0},
    }
    assert metrics["filtered"]["bots"] == 1
    assert metrics["filtered"]["static_assets"] == 1
    assert metrics["filtered"]["non_success"] == 1
    assert metrics["filtered"]["outside_window"] == 1
    assert metrics["top_referrers"] == [{"host": "news.ycombinator.com", "page_views": 1}]
    serialized = json.dumps(metrics)
    assert "203.0.113" not in serialized
    assert "Mozilla" not in serialized


def test_iter_log_lines_supports_plain_and_gzip_files(tmp_path):
    module = load_module()
    plain = tmp_path / "access.log"
    compressed = tmp_path / "access.log.1.gz"
    plain.write_text("plain\n")
    with gzip.open(compressed, "wt") as handle:
        handle.write("compressed\n")

    assert list(module.iter_log_lines([plain, compressed])) == ["plain\n", "compressed\n"]


def test_classify_owned_product_guides_as_docs():
    module = load_module()

    for path in (
        "/quickstart.html",
        "/en/quickstart.html",
        "/models.html",
        "/en/models.html",
        "/vs-whisper.html",
        "/en/vs-whisper.html",
        "/ecosystem.html",
        "/en/llama-cpp.html",
    ):
        assert module.classify_page(path) == "docs"


def test_cli_outputs_aggregate_json_without_identifiers(tmp_path):
    script = Path(__file__).resolve().parents[1] / "scripts" / "collect_website_traffic.py"
    log = tmp_path / "access.log"
    log.write_text(
        access_line(
            "203.0.113.55",
            "23/Jul/2026:09:00:00 +0800",
            "/docs/deployment.html",
        )
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--log",
            str(log),
            "--now",
            "2026-07-23T12:00:00+00:00",
            "--days",
            "30",
            "--format",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)

    assert payload["page_views"] == 1
    assert payload["sections"]["docs"]["page_views"] == 1
    assert "203.0.113.55" not in result.stdout
