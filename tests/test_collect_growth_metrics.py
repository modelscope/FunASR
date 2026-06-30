import importlib.util
import io
import json
import sys
from datetime import date
from contextlib import redirect_stdout
from pathlib import Path


def load_growth_metrics_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "collect_growth_metrics.py"
    spec = importlib.util.spec_from_file_location("collect_growth_metrics", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_collect_ecosystem_metrics_sums_repositories_and_target_gap(monkeypatch):
    module = load_growth_metrics_module()

    github_payloads = {
        "modelscope/FunASR": {
            "stargazers_count": 18714,
            "forks_count": 3100,
            "subscribers_count": 210,
            "open_issues_count": 3,
            "default_branch": "main",
            "pushed_at": "2026-06-30T01:48:35Z",
            "html_url": "https://github.com/modelscope/FunASR",
        },
        "FunAudioLLM/Fun-ASR": {
            "stargazers_count": 1318,
            "forks_count": 120,
            "subscribers_count": 20,
            "open_issues_count": 0,
            "default_branch": "main",
            "pushed_at": "2026-06-29T13:24:05Z",
            "html_url": "https://github.com/FunAudioLLM/Fun-ASR",
        },
        "FunAudioLLM/SenseVoice": {
            "stargazers_count": 8718,
            "forks_count": 800,
            "subscribers_count": 90,
            "open_issues_count": 0,
            "default_branch": "main",
            "pushed_at": "2026-06-29T13:24:06Z",
            "html_url": "https://github.com/FunAudioLLM/SenseVoice",
        },
        "modelscope/FunClip": {
            "stargazers_count": 5872,
            "forks_count": 500,
            "subscribers_count": 70,
            "open_issues_count": 0,
            "default_branch": "main",
            "pushed_at": "2026-06-29T20:47:23Z",
            "html_url": "https://github.com/modelscope/FunClip",
        },
    }

    def fake_fetch_json(url, headers=None):
        if url.startswith("https://api.github.com/repos/"):
            repo = url.removeprefix("https://api.github.com/repos/")
            return github_payloads[repo]
        if url == "https://pypi.org/pypi/funasr/json":
            return {"info": {"version": "1.3.14", "summary": "FunASR", "project_url": "https://pypi.org/project/funasr/"}}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_ecosystem_metrics(
        ["modelscope/FunASR", "FunAudioLLM/Fun-ASR", "FunAudioLLM/SenseVoice", "modelscope/FunClip"],
        package="funasr",
        baseline_stars=31224,
        target_additional_stars=20000,
    )

    assert metrics["ecosystem"]["total_stars"] == 34622
    assert metrics["ecosystem"]["baseline_stars"] == 31224
    assert metrics["ecosystem"]["added_stars"] == 3398
    assert metrics["ecosystem"]["remaining_to_target"] == 16602
    assert [repo["repo"] for repo in metrics["ecosystem"]["repositories"]] == [
        "modelscope/FunASR",
        "FunAudioLLM/Fun-ASR",
        "FunAudioLLM/SenseVoice",
        "modelscope/FunClip",
    ]


def test_collect_ecosystem_metrics_calculates_daily_target(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url.startswith("https://api.github.com/repos/"):
            repo = url.removeprefix("https://api.github.com/repos/")
            stars = {
                "modelscope/FunASR": 18715,
                "FunAudioLLM/Fun-ASR": 1318,
                "FunAudioLLM/SenseVoice": 8718,
                "modelscope/FunClip": 5872,
            }[repo]
            return {
                "stargazers_count": stars,
                "forks_count": 0,
                "subscribers_count": 0,
                "open_issues_count": 0,
                "default_branch": "main",
                "pushed_at": "2026-06-30T00:00:00Z",
                "html_url": f"https://github.com/{repo}",
            }
        if url == "https://pypi.org/pypi/funasr/json":
            return {"info": {"version": "1.3.14", "summary": "FunASR", "project_url": "https://pypi.org/project/funasr/"}}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_ecosystem_metrics(
        ["modelscope/FunASR", "FunAudioLLM/Fun-ASR", "FunAudioLLM/SenseVoice", "modelscope/FunClip"],
        package="funasr",
        baseline_stars=31224,
        target_additional_stars=20000,
        target_date="2026-09-30",
        today=date(2026, 6, 30),
    )

    assert metrics["ecosystem"]["target_date"] == "2026-09-30"
    assert metrics["ecosystem"]["days_remaining"] == 92
    assert metrics["ecosystem"]["required_daily_average"] == 181


def test_main_outputs_ecosystem_json(monkeypatch):
    module = load_growth_metrics_module()

    repo_stars = {
        "modelscope/FunASR": 18714,
        "FunAudioLLM/Fun-ASR": 1318,
        "FunAudioLLM/SenseVoice": 8718,
        "modelscope/FunClip": 5872,
    }

    def fake_fetch_json(url, headers=None):
        if url.startswith("https://api.github.com/repos/"):
            repo = url.removeprefix("https://api.github.com/repos/")
            return {
                "stargazers_count": repo_stars[repo],
                "forks_count": 0,
                "subscribers_count": 0,
                "open_issues_count": 0,
                "default_branch": "main",
                "pushed_at": "2026-06-30T00:00:00Z",
                "html_url": f"https://github.com/{repo}",
            }
        if url == "https://pypi.org/pypi/funasr/json":
            return {"info": {"version": "1.3.14", "summary": "FunASR", "project_url": "https://pypi.org/project/funasr/"}}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_growth_metrics.py",
            "--ecosystem",
            "--format",
            "json",
            "--baseline-stars",
            "31224",
            "--target-additional-stars",
            "20000",
        ],
    )

    with redirect_stdout(io.StringIO()) as stdout:
        assert module.main() == 0

    payload = json.loads(stdout.getvalue())
    assert payload["ecosystem"]["total_stars"] == 34622
    assert payload["ecosystem"]["remaining_to_target"] == 16602
