import importlib.util
import io
import json
import sys
from datetime import date, datetime, timezone
from contextlib import redirect_stdout
from pathlib import Path


def load_growth_metrics_module():
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "collect_growth_metrics.py"
    spec = importlib.util.spec_from_file_location("collect_growth_metrics", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_integration_prs_include_sglang_omni_fun_asr():
    module = load_growth_metrics_module()

    assert "sgl-project/sglang-omni#898" in module.DEFAULT_INTEGRATION_PRS


def test_default_integration_prs_include_high_visibility_external_queue():
    module = load_growth_metrics_module()

    expected_prs = {
        "infiniflow/ragflow#16473",
        "pipecat-ai/pipecat#4844",
        "TEN-framework/ten-framework#2191",
        "activepieces/activepieces#13985",
        "Uberi/speech_recognition#903",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))


def test_collect_github_repo_metrics_splits_open_issues_and_pull_requests(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/modelscope/FunASR":
            return {
                "stargazers_count": 18716,
                "forks_count": 3100,
                "subscribers_count": 210,
                "open_issues_count": 5,
                "default_branch": "main",
                "pushed_at": "2026-06-30T02:26:33Z",
                "html_url": "https://github.com/modelscope/FunASR",
            }
        if url == "https://api.github.com/repos/modelscope/FunASR/pulls?state=open&per_page=100":
            return [{"number": 3056}, {"number": 3057}]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_github_repo_metrics("modelscope/FunASR")

    assert metrics["open_items"] == 5
    assert metrics["open_pull_requests"] == 2
    assert metrics["open_issues"] == 3


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
        if url.endswith("/pulls?state=open&per_page=100"):
            return []
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
        if url.endswith("/pulls?state=open&per_page=100"):
            return []
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


def test_collect_integration_metrics_summarizes_pull_request_checks(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/huggingface/transformers/pulls/46180":
            return {
                "number": 46180,
                "title": "Add Fun-ASR-Nano model",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "updated_at": "2026-06-29T23:45:57Z",
                "head": {"sha": "8a744ec", "ref": "add-fun-asr-nano"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/huggingface/transformers/commits/8a744ec/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/huggingface/transformers/commits/8a744ec/check-runs?per_page=100":
            return {
                "total_count": 3,
                "check_runs": [
                    {"name": "pr-ci / Check code quality", "status": "completed", "conclusion": "success"},
                    {
                        "name": "pr-ci / tests_processors / tests_processors [shard 3/8]",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/huggingface/transformers/actions/runs/1/job/2",
                    },
                    {"name": "pr-ci / slow tests", "status": "queued", "conclusion": None},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(
        ["huggingface/transformers#46180"],
        now=datetime(2026, 7, 2, 0, 0, 0, tzinfo=timezone.utc),
    )

    integration = metrics["integrations"][0]
    assert integration["pr"] == "huggingface/transformers#46180"
    assert integration["title"] == "Add Fun-ASR-Nano model"
    assert integration["next_action"] == "fix checks"
    assert integration["updated_age_days"] == 2
    assert integration["mergeable"] is True
    assert integration["checks"]["state"] == "failure"
    assert integration["checks"]["total_check_runs"] == 3
    assert integration["checks"]["failed_check_runs"] == [
        {
            "name": "pr-ci / tests_processors / tests_processors [shard 3/8]",
            "conclusion": "failure",
            "url": "https://github.com/huggingface/transformers/actions/runs/1/job/2",
        }
    ]
    assert integration["checks"]["pending_check_runs"] == [
        {"name": "pr-ci / slow tests", "status": "queued", "url": None}
    ]


def test_format_ecosystem_markdown_includes_open_pull_requests():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-06-30T02:30:00+00:00",
        "ecosystem": {
            "repositories": [
                {
                    "repo": "modelscope/FunASR",
                    "stars": 18716,
                    "forks": 3100,
                    "open_issues": 3,
                    "open_pull_requests": 2,
                    "pushed_at": "2026-06-30T02:26:33Z",
                    "html_url": "https://github.com/modelscope/FunASR",
                }
            ],
            "total_stars": 18716,
            "baseline_stars": 31224,
            "added_stars": -12508,
            "target_additional_stars": 20000,
            "remaining_to_target": 32508,
            "target_date": "2026-09-30",
            "days_remaining": 92,
            "required_daily_average": 354,
        },
        "pypi": {"package": "funasr", "version": "1.3.14", "project_url": "https://pypi.org/project/funasr/"},
    }

    output = module.format_ecosystem_markdown(metrics)

    assert "| Repository | Stars | Forks | Open issues | Open PRs | Last push |" in output
    assert "| [modelscope/FunASR](https://github.com/modelscope/FunASR) | 18,716 | 3,100 | 3 | 2 |" in output


def test_collect_integration_metrics_recommends_review_for_clean_success_pr(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/infiniflow/ragflow/pulls/16473":
            return {
                "number": 16473,
                "title": "feat(stt): add FunASR / SenseVoice provider",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/infiniflow/ragflow/pull/16473",
                "updated_at": "2026-06-30T03:16:39Z",
                "head": {"sha": "818a295", "ref": "funasr/ragflow-15526-conflict-fix"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/infiniflow/ragflow/commits/818a295/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/infiniflow/ragflow/commits/818a295/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "CodeRabbit", "status": "completed", "conclusion": "success"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["infiniflow/ragflow#16473"])

    assert metrics["integrations"][0]["next_action"] == "request review"


def test_collect_integration_metrics_surfaces_pending_cla_status(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/activepieces/activepieces/pulls/13985":
            return {
                "number": 13985,
                "title": "feat: add FunASR speech recognition piece",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/activepieces/activepieces/pull/13985",
                "updated_at": "2026-06-30T01:17:07Z",
                "head": {"sha": "f9d22ee", "ref": "funasr/activepieces-13450-conflict-fix"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/activepieces/activepieces/commits/f9d22ee/status":
            return {
                "state": "pending",
                "statuses": [
                    {
                        "context": "license/cla",
                        "state": "pending",
                        "target_url": "https://cla-assistant.io/activepieces/activepieces?pullRequest=13985",
                    }
                ],
            }
        if url == "https://api.github.com/repos/activepieces/activepieces/commits/f9d22ee/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "GitGuardian Security Checks", "status": "completed", "conclusion": "success"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["activepieces/activepieces#13985"])

    integration = metrics["integrations"][0]
    assert integration["next_action"] == "resolve CLA"
    assert integration["checks"]["pending_check_runs"] == [
        {
            "name": "license/cla",
            "status": "pending",
            "url": "https://cla-assistant.io/activepieces/activepieces?pullRequest=13985",
        }
    ]


def test_format_integration_markdown_includes_update_age_and_next_action():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "huggingface/transformers#46180",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "state": "open",
                "mergeable_state": "blocked",
                "updated_at": "2026-06-29T23:45:57Z",
                "updated_age_days": 2,
                "next_action": "fix checks",
                "checks": {"state": "failure", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert "| Pull request | State | Mergeable | Checks | Failed | Pending | Age | Action | Updated |" in output
    assert (
        "| [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180) | "
        "open | blocked | failure | 0 | 0 | 2d | fix checks | `2026-06-29T23:45:57Z` |"
    ) in output


def test_collect_issue_metrics_filters_pull_requests_and_assigns_waiting_on(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/modelscope/FunASR/issues?state=open&per_page=100":
            return [
                {
                    "number": 3038,
                    "title": "realtime websocket falls behind with two clients",
                    "html_url": "https://github.com/modelscope/FunASR/issues/3038",
                    "updated_at": "2026-06-29T19:22:38Z",
                    "comments": 4,
                    "user": {"login": "liujixingit"},
                    "labels": [{"name": "bug"}, {"name": "needs feedback"}],
                },
                {
                    "number": 3034,
                    "title": "Fun-ASR-Nano on Ascend NPU",
                    "html_url": "https://github.com/modelscope/FunASR/issues/3034",
                    "updated_at": "2026-06-29T19:49:07Z",
                    "comments": 3,
                    "user": {"login": "pubulichen"},
                    "labels": [{"name": "help wanted"}, {"name": "ready for PR"}],
                },
                {
                    "number": 3057,
                    "title": "open pull request should be ignored",
                    "pull_request": {"url": "https://api.github.com/repos/modelscope/FunASR/pulls/3057"},
                    "labels": [],
                },
            ]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_issue_metrics(["modelscope/FunASR"])

    repository = metrics["repositories"][0]
    assert repository["repo"] == "modelscope/FunASR"
    assert repository["open_issue_count"] == 2
    assert [issue["number"] for issue in repository["open_issues"]] == [3038, 3034]
    assert repository["open_issues"][0]["labels"] == ["bug", "needs feedback"]
    assert repository["open_issues"][0]["waiting_on"] == "reporter"
    assert repository["open_issues"][1]["waiting_on"] == "contributor"


def test_main_outputs_issues_json(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/modelscope/FunASR/issues?state=open&per_page=100":
            return [
                {
                    "number": 2959,
                    "title": "postprocess hotwords",
                    "html_url": "https://github.com/modelscope/FunASR/issues/2959",
                    "updated_at": "2026-06-29T19:46:28Z",
                    "comments": 2,
                    "user": {"login": "HaujetZhao"},
                    "labels": [{"name": "good first issue"}, {"name": "ready for PR"}],
                }
            ]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_growth_metrics.py",
            "--issues",
            "--repos",
            "modelscope/FunASR",
            "--format",
            "json",
        ],
    )

    with redirect_stdout(io.StringIO()) as stdout:
        assert module.main() == 0

    payload = json.loads(stdout.getvalue())
    assert payload["repositories"][0]["open_issues"][0]["number"] == 2959
    assert payload["repositories"][0]["open_issues"][0]["waiting_on"] == "contributor"


def test_main_outputs_ecosystem_json(monkeypatch):
    module = load_growth_metrics_module()

    repo_stars = {
        "modelscope/FunASR": 18714,
        "FunAudioLLM/Fun-ASR": 1318,
        "FunAudioLLM/SenseVoice": 8718,
        "modelscope/FunClip": 5872,
    }

    def fake_fetch_json(url, headers=None):
        if url.endswith("/pulls?state=open&per_page=100"):
            return []
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


def test_main_outputs_integrations_json(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/ray-project/ray/pulls/64053":
            return {
                "number": 64053,
                "title": "docs(serve): add FunASR ASR integration example",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/ray-project/ray/pull/64053",
                "updated_at": "2026-06-30T01:53:52Z",
                "head": {"sha": "780eb4c", "ref": "issue-64052"},
                "base": {"ref": "master"},
                "user": {"login": "nh-atuan"},
            }
        if url == "https://api.github.com/repos/ray-project/ray/commits/780eb4c/status":
            return {"state": "success", "statuses": [{"context": "buildkite/microcheck", "state": "success"}]}
        if url == "https://api.github.com/repos/ray-project/ray/commits/780eb4c/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "disable-automerge", "status": "completed", "conclusion": "success"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "collect_growth_metrics.py",
            "--integrations",
            "--integration-prs",
            "ray-project/ray#64053",
            "--format",
            "json",
        ],
    )

    with redirect_stdout(io.StringIO()) as stdout:
        assert module.main() == 0

    payload = json.loads(stdout.getvalue())
    integration = payload["integrations"][0]
    assert integration["pr"] == "ray-project/ray#64053"
    assert integration["checks"]["state"] == "success"
    assert integration["next_action"] == "request review"
