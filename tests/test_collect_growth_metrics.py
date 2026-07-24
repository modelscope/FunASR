import importlib.util
import io
import json
import subprocess
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

    assert "sgl-project/sglang-omni#1078" not in module.DEFAULT_INTEGRATION_PRS
    assert "sgl-project/sglang-omni#898" not in module.DEFAULT_INTEGRATION_PRS


def test_github_headers_falls_back_to_gh_auth_token(monkeypatch):
    module = load_growth_metrics_module()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", "/tmp/funasr-growth-metrics-no-token-home")
    monkeypatch.setattr(
        module,
        "OPS_GITHUB_TOKEN_PATH",
        Path("/tmp/funasr-growth-metrics-missing-ops-token"),
    )

    def fake_run(args, **kwargs):
        assert args == ["gh", "auth", "token"]
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="cli-token\n")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    headers = module.github_headers()

    assert headers["Authorization"] == "Bearer cli-token"


def test_github_headers_reads_funasr_ops_token_file(monkeypatch, tmp_path):
    module = load_growth_metrics_module()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    token_path = tmp_path / ".config" / "funasr-ops" / "github_token"
    token_path.parent.mkdir(parents=True)
    token_path.write_text("file-token\n")

    def fake_run(args, **kwargs):
        raise FileNotFoundError("gh")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    headers = module.github_headers()

    assert headers["Authorization"] == "Bearer file-token"


def test_github_headers_reads_configured_ops_token_file(monkeypatch, tmp_path):
    module = load_growth_metrics_module()
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "empty-home"))
    ops_token_path = tmp_path / "ops" / "github_token"
    ops_token_path.parent.mkdir(parents=True)
    ops_token_path.write_text("ops-file-token\n")
    monkeypatch.setattr(module, "OPS_GITHUB_TOKEN_PATH", ops_token_path)

    def fake_run(args, **kwargs):
        raise AssertionError("gh auth token should not be called when ops token exists")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    headers = module.github_headers()

    assert headers["Authorization"] == "Bearer ops-file-token"


def test_default_integration_prs_include_high_visibility_external_queue():
    module = load_growth_metrics_module()

    expected_prs = {
        "huggingface/optimum-intel#1874",
        "TEN-framework/ten-framework#2191",
        "Uberi/speech_recognition#903",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))
    assert "activepieces/activepieces#13985" not in module.DEFAULT_INTEGRATION_PRS
    assert "infiniflow/ragflow#16473" not in module.DEFAULT_INTEGRATION_PRS
    assert "pipecat-ai/pipecat#4844" not in module.DEFAULT_INTEGRATION_PRS


def test_default_integration_prs_include_new_growth_lanes():
    module = load_growth_metrics_module()

    expected_prs = {
        "huggingface/speech-to-speech#319",
        "run-llama/llama_index#21958",
        "run-llama/llama_index#21996",
        "mem0ai/mem0#5571",
        "mudler/LocalAI#10090",
        "agno-agi/agno#8501",
        "GetStream/Vision-Agents#606",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))
    assert "Significant-Gravitas/AutoGPT#13500" not in module.DEFAULT_INTEGRATION_PRS
    assert "ai4s-research/awesome-ai-for-science#69" not in module.DEFAULT_INTEGRATION_PRS


def test_default_integration_prs_include_video_discovery_lanes():
    module = load_growth_metrics_module()

    expected_prs = {
        "mahseema/awesome-ai-tools#1689",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))
    assert "tmoroney/auto-subs#629" not in module.DEFAULT_INTEGRATION_PRS


def test_default_integration_prs_include_voice_agent_and_ml_discovery_lanes():
    module = load_growth_metrics_module()

    expected_prs = {
        "livekit/agents#6176",
        "lukasmasuch/best-of-ml-python#455",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))


def test_default_integration_prs_include_high_star_awesome_discovery_lanes():
    module = load_growth_metrics_module()

    expected_prs = {
        "fighting41love/funNLP#478",
        "josephmisiti/awesome-machine-learning#1339",
        "jobbole/awesome-python-cn#141",
        "ChristosChristofidis/awesome-deep-learning#317",
        "Hannibal046/Awesome-LLM#623",
        "AiHubCN/Awesome-Chinese-LLM#103",
        "pluja/awesome-privacy#836",
        "BradyFU/Awesome-Multimodal-Large-Language-Models#280",
        "mahmoud/awesome-python-applications#227",
        "bharathgs/Awesome-pytorch-list#164",
        "owainlewis/awesome-artificial-intelligence#243",
        "steven2358/awesome-generative-ai#821",
        "crownpku/Awesome-Chinese-NLP#32",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))
    assert "WangRongsheng/awesome-LLM-resources#162" not in module.DEFAULT_INTEGRATION_PRS
    assert "vinta/awesome-python#3246" not in module.DEFAULT_INTEGRATION_PRS
    assert "RVC-Boss/GPT-SoVITS#2801" not in module.DEFAULT_INTEGRATION_PRS


def test_high_star_awesome_discovery_lanes_wait_for_maintainer_review():
    module = load_growth_metrics_module()

    expected_prs = {
        "vinta/awesome-python#3246",
        "fighting41love/funNLP#478",
        "josephmisiti/awesome-machine-learning#1339",
        "RVC-Boss/GPT-SoVITS#2801",
        "jobbole/awesome-python-cn#141",
        "ChristosChristofidis/awesome-deep-learning#317",
        "Hannibal046/Awesome-LLM#623",
        "AiHubCN/Awesome-Chinese-LLM#103",
        "pluja/awesome-privacy#836",
        "BradyFU/Awesome-Multimodal-Large-Language-Models#280",
        "mahmoud/awesome-python-applications#227",
        "bharathgs/Awesome-pytorch-list#164",
        "owainlewis/awesome-artificial-intelligence#243",
        "steven2358/awesome-generative-ai#821",
        "crownpku/Awesome-Chinese-NLP#32",
    }

    assert expected_prs.issubset(set(module.KNOWN_ASSISTED_REVIEW_REQUESTS))
    assert "WangRongsheng/awesome-LLM-resources#162" not in module.KNOWN_ASSISTED_REVIEW_REQUESTS


def test_default_integration_prs_include_missing_validated_discovery_lanes():
    module = load_growth_metrics_module()

    expected_prs = {
        "mahseema/awesome-ai-tools#1403",
        "INTERMT/Awesome-PyTorch-Chinese#5",
        "krzjoa/awesome-python-data-science#99",
        "zzw922cn/awesome-speech-recognition-speech-synthesis-papers#27",
        "Osmantic/ODS#1639",
        "faroit/awesome-python-scientific-audio#85",
        "joewongjc/type4me#207",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))
    assert "ga642381/speech-trident#31" not in module.DEFAULT_INTEGRATION_PRS


def test_missing_validated_discovery_lanes_wait_for_maintainer_review():
    module = load_growth_metrics_module()

    expected_prs = {
        "mahseema/awesome-ai-tools#1403",
        "INTERMT/Awesome-PyTorch-Chinese#5",
        "krzjoa/awesome-python-data-science#99",
        "zzw922cn/awesome-speech-recognition-speech-synthesis-papers#27",
        "Osmantic/ODS#1639",
        "faroit/awesome-python-scientific-audio#85",
        "joewongjc/type4me#207",
        "ga642381/speech-trident#31",
    }

    assert expected_prs.issubset(set(module.KNOWN_ASSISTED_REVIEW_REQUESTS))


def test_type4me_review_wait_reflects_completed_qwen3_smoke():
    module = load_growth_metrics_module()

    reason = module.KNOWN_ASSISTED_REVIEW_REQUESTS["joewongjc/type4me#207"]["reason"]

    assert "Qwen3-only ASR smoke evidence posted" in reason
    assert "keep draft" not in reason
    assert "until Qwen3-only ASR smoke test" not in reason


def test_default_integration_prs_include_mcp_discovery_lanes():
    module = load_growth_metrics_module()

    assert "punkpeye/awesome-mcp-servers#7153" in module.DEFAULT_INTEGRATION_PRS


def test_default_integration_prs_include_speech_server_lanes():
    module = load_growth_metrics_module()

    expected_prs = {
        "BerriAI/litellm-docs#610",
        "speaches-ai/speaches#658",
        "getpaseo/paseo#1634",
    }

    assert expected_prs.issubset(set(module.DEFAULT_INTEGRATION_PRS))


def test_default_integration_prs_exclude_archived_integration_prs():
    module = load_growth_metrics_module()

    assert "xinnan-tech/xiaozhi-esp32-server#3255" not in module.DEFAULT_INTEGRATION_PRS


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
        "QwenAudio/Fun-ASR": {
            "stargazers_count": 1318,
            "forks_count": 120,
            "subscribers_count": 20,
            "open_issues_count": 0,
            "default_branch": "main",
            "pushed_at": "2026-06-29T13:24:05Z",
            "html_url": "https://github.com/QwenAudio/Fun-ASR",
        },
        "QwenAudio/SenseVoice": {
            "stargazers_count": 8718,
            "forks_count": 800,
            "subscribers_count": 90,
            "open_issues_count": 0,
            "default_branch": "main",
            "pushed_at": "2026-06-29T13:24:06Z",
            "html_url": "https://github.com/QwenAudio/SenseVoice",
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
        ["modelscope/FunASR", "QwenAudio/Fun-ASR", "QwenAudio/SenseVoice", "modelscope/FunClip"],
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
        "QwenAudio/Fun-ASR",
        "QwenAudio/SenseVoice",
        "modelscope/FunClip",
    ]


def test_collect_pypi_download_metrics_sums_recent_windows(monkeypatch, tmp_path):
    module = load_growth_metrics_module()
    monkeypatch.setattr(module, "DEFAULT_PYPI_DOWNLOAD_CACHE", tmp_path / "pypi-downloads-funasr.json")

    def fake_fetch_json(url, headers=None):
        assert url == "https://pypistats.org/api/packages/funasr/overall?mirrors=false"
        return {
            "data": [
                {
                    "category": "without_mirrors",
                    "date": f"2026-07-{day:02d}",
                    "downloads": day,
                }
                for day in range(1, 32)
            ]
        }

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_pypi_download_metrics("funasr")

    assert metrics == {
        "source": "pypistats.org",
        "source_url": "https://pypistats.org/packages/funasr",
        "latest_date": "2026-07-31",
        "downloads_last_7_days": sum(range(25, 32)),
        "downloads_last_30_days": sum(range(2, 32)),
        "status": "available",
    }
    cached = json.loads((tmp_path / "pypi-downloads-funasr.json").read_text())
    assert cached["package"] == "funasr"
    assert cached["metrics"] == metrics


def test_collect_pypi_download_metrics_uses_cache_after_live_failure(monkeypatch, tmp_path):
    module = load_growth_metrics_module()
    cache_path = tmp_path / "pypi-downloads-funasr.json"
    monkeypatch.setattr(module, "DEFAULT_PYPI_DOWNLOAD_CACHE", cache_path)
    cache_path.write_text(json.dumps({
        "package": "funasr",
        "metrics": {
            "source": "pypistats.org",
            "source_url": "https://pypistats.org/packages/funasr",
            "latest_date": "2026-07-21",
            "downloads_last_7_days": 116407,
            "downloads_last_30_days": 442556,
            "status": "available",
        },
    }))

    def fake_fetch_json(url, headers=None):
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_pypi_download_metrics("funasr")

    assert metrics["status"] == "stale_available"
    assert metrics["downloads_last_7_days"] == 116407
    assert metrics["downloads_last_30_days"] == 442556
    assert metrics["cache_path"] == str(cache_path)
    assert "HTTP 429" in metrics["reason"]


def test_collect_pypi_download_metrics_reports_unavailable_without_cache(monkeypatch, tmp_path):
    module = load_growth_metrics_module()
    monkeypatch.setattr(module, "DEFAULT_PYPI_DOWNLOAD_CACHE", tmp_path / "missing-cache.json")

    def fake_fetch_json(url, headers=None):
        raise RuntimeError("HTTP 429")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_pypi_download_metrics("funasr")

    assert metrics["status"] == "unavailable"
    assert metrics["source"] == "pypistats.org"
    assert "HTTP 429" in metrics["reason"]


def test_collect_ecosystem_metrics_calculates_daily_target(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url.endswith("/pulls?state=open&per_page=100"):
            return []
        if url.startswith("https://api.github.com/repos/"):
            repo = url.removeprefix("https://api.github.com/repos/")
            stars = {
                "modelscope/FunASR": 18715,
                "QwenAudio/Fun-ASR": 1318,
                "QwenAudio/SenseVoice": 8718,
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
        ["modelscope/FunASR", "QwenAudio/Fun-ASR", "QwenAudio/SenseVoice", "modelscope/FunClip"],
        package="funasr",
        baseline_stars=31224,
        target_additional_stars=20000,
        target_date="2026-09-30",
        today=date(2026, 6, 30),
    )

    assert metrics["ecosystem"]["target_date"] == "2026-09-30"
    assert metrics["ecosystem"]["days_remaining"] == 92
    assert metrics["ecosystem"]["required_daily_average"] == 181


def test_format_ecosystem_markdown_marks_cached_pypi_download_windows():
    module = load_growth_metrics_module()

    metrics = {
        "collected_at_utc": "2026-07-23T00:00:00+00:00",
        "ecosystem": {
            "repositories": [],
            "total_stars": 35787,
            "baseline_stars": 31224,
            "target_additional_stars": 20000,
            "added_stars": 4563,
            "remaining_to_target": 15437,
            "target_date": "2026-09-30",
            "days_remaining": 70,
            "required_daily_average": 221,
        },
        "pypi": {
            "package": "funasr",
            "version": "1.3.26",
            "downloads": {
                "status": "stale_available",
                "latest_date": "2026-07-21",
                "downloads_last_7_days": 116407,
                "downloads_last_30_days": 442556,
                "source_url": "https://pypistats.org/packages/funasr",
                "reason": "HTTP 429",
            },
        },
    }

    output = module.format_ecosystem_markdown(metrics)

    assert "- PyPI downloads: **116,407** last 7 days; **442,556** last 30 days" in output
    assert "cached after live API failure: HTTP 429" in output


def test_format_ecosystem_markdown_includes_pypi_download_windows():
    module = load_growth_metrics_module()

    metrics = {
        "collected_at_utc": "2026-07-23T00:00:00+00:00",
        "ecosystem": {
            "repositories": [],
            "total_stars": 35787,
            "baseline_stars": 31224,
            "target_additional_stars": 20000,
            "added_stars": 4563,
            "remaining_to_target": 15437,
            "target_date": "2026-09-30",
            "days_remaining": 70,
            "required_daily_average": 221,
        },
        "pypi": {
            "package": "funasr",
            "version": "1.3.26",
            "project_url": "https://pypi.org/project/funasr/",
            "downloads": {
                "status": "available",
                "latest_date": "2026-07-22",
                "downloads_last_7_days": 70123,
                "downloads_last_30_days": 301234,
                "source_url": "https://pypistats.org/packages/funasr",
            },
        },
    }

    output = module.format_ecosystem_markdown(metrics)

    assert "- PyPI downloads: **70,123** last 7 days; **301,234** last 30 days" in output
    assert "through 2026-07-22" in output


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


def test_collect_integration_metrics_classifies_known_external_ci_failure(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/huggingface/transformers/pulls/46180":
            return {
                "number": 46180,
                "title": "Add Fun-ASR-Nano model",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "updated_at": "2026-06-30T04:24:15Z",
                "head": {"sha": "8ed336b", "ref": "add-fun-asr-nano"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/huggingface/transformers/commits/8ed336b/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/huggingface/transformers/commits/8ed336b/check-runs?per_page=100":
            return {
                "total_count": 99,
                "check_runs": [
                    {
                        "name": "pr-ci / tests_processors / tests_processors [shard 3/8]",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/huggingface/transformers/actions/runs/28419525291/job/84209871338",
                    },
                    {
                        "name": "pr-ci / PR CI status",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/huggingface/transformers/actions/runs/28419525291/job/84211351756",
                    },
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["huggingface/transformers#46180"])

    integration = metrics["integrations"][0]
    assert integration["checks"]["state"] == "failure"
    assert integration["known_external_failure_reason"] == (
        "LightOnOCR shared hub-cache read-only failure; PR CI status is the aggregate failure"
    )
    assert integration["known_external_failure_action"] == "wait for maintainer rerun"
    assert integration["next_action"] == "wait for maintainer rerun"


def test_collect_integration_metrics_classifies_ray_docs_infra_failure(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/ray-project/ray/pulls/64053":
            return {
                "number": 64053,
                "title": "docs(serve): add FunASR ASR integration example",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/ray-project/ray/pull/64053",
                "updated_at": "2026-07-22T13:46:40Z",
                "head": {"sha": "9b2a3b3", "ref": "issue-64052"},
                "base": {"ref": "master"},
                "user": {"login": "nh-atuan"},
            }
        if url == "https://api.github.com/repos/ray-project/ray/commits/9b2a3b3/status":
            return {
                "state": "failure",
                "statuses": [
                    {"context": "buildkite/microcheck", "state": "failure"},
                    {"context": "docs/readthedocs.com:anyscale-ray", "state": "failure"},
                ],
            }
        if url == "https://api.github.com/repos/ray-project/ray/commits/9b2a3b3/check-runs?per_page=100":
            return {"total_count": 0, "check_runs": []}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["ray-project/ray#64053"])

    integration = metrics["integrations"][0]
    assert integration["checks"]["state"] == "failure"
    assert integration["known_external_failure_reason"] == (
        "Current Ray failures are Buildkite/ReadTheDocs gates already triaged; "
        "the remaining PR-local Black fix is waiting on the contributor branch owner"
    )
    assert integration["known_external_failure_action"] == "wait for contributor branch update"
    assert integration["next_action"] == "wait for contributor branch update"


def test_collect_integration_metrics_handles_known_external_ci_aggregate_race(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/huggingface/transformers/pulls/46180":
            return {
                "number": 46180,
                "title": "Add Fun-ASR-Nano model",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "updated_at": "2026-06-30T04:15:07Z",
                "head": {"sha": "8ed336b", "ref": "add-fun-asr-nano"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/huggingface/transformers/commits/8ed336b/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/huggingface/transformers/commits/8ed336b/check-runs?per_page=100":
            return {
                "total_count": 98,
                "check_runs": [
                    {
                        "name": "pr-ci / tests_processors / tests_processors [shard 3/8]",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/huggingface/transformers/actions/runs/28419525291/job/84209871338",
                    },
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["huggingface/transformers#46180"])

    integration = metrics["integrations"][0]
    assert integration["known_external_failure_reason"] == (
        "LightOnOCR shared hub-cache read-only failure; PR CI status is the aggregate failure"
    )
    assert integration["known_external_failure_action"] == "wait for maintainer rerun"
    assert integration["next_action"] == "wait for maintainer rerun"


def test_collect_integration_metrics_treats_transformers_review_gate_as_passive_wait(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/huggingface/transformers/pulls/46180":
            return {
                "number": 46180,
                "title": "Add Fun-ASR-Nano model",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "updated_at": "2026-07-22T04:24:15Z",
                "head": {"sha": "5efce76", "ref": "add-fun-asr-nano"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/huggingface/transformers/commits/5efce76/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/huggingface/transformers/commits/5efce76/check-runs?per_page=100":
            return {
                "total_count": 2,
                "check_runs": [
                    {"name": "pr-ci / Check code quality", "status": "completed", "conclusion": "success"},
                    {"name": "pr-ci / PR CI status", "status": "completed", "conclusion": "success"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["huggingface/transformers#46180"])

    integration = metrics["integrations"][0]
    assert integration["checks"]["state"] == "success"
    assert integration["next_action"] == "wait for maintainer review"
    assert "active maintainer-held review thread" in integration["known_assisted_review_reason"]


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
        if url == "https://api.github.com/repos/example/clean-pr/pulls/42":
            return {
                "number": 42,
                "title": "Add FunASR",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/example/clean-pr/pull/42",
                "updated_at": "2026-06-30T06:44:34Z",
                "head": {"sha": "9a19b5e", "ref": "add-funasr"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/example/clean-pr/commits/9a19b5e/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/example/clean-pr/commits/9a19b5e/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "validate", "status": "completed", "conclusion": "success"},
                ],
            }
        if url == "https://api.github.com/repos/example/clean-pr/pulls/42/reviews?per_page=100":
            return []
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["example/clean-pr#42"])

    assert metrics["integrations"][0]["next_action"] == "request review"


def test_collect_integration_metrics_waits_for_merge_after_maintainer_approval(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/example/approved-pr/pulls/42":
            return {
                "number": 42,
                "title": "Add FunASR",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/example/approved-pr/pull/42",
                "updated_at": "2026-07-23T04:16:43Z",
                "head": {"sha": "1dafe5e", "ref": "add-funasr"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/example/approved-pr/commits/1dafe5e/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/example/approved-pr/commits/1dafe5e/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "validate", "status": "completed", "conclusion": "success"},
                ],
            }
        if url == "https://api.github.com/repos/example/approved-pr/pulls/42/reviews?per_page=100":
            return [
                {
                    "id": 123,
                    "state": "APPROVED",
                    "submitted_at": "2026-07-23T04:00:00Z",
                    "user": {"login": "maintainer"},
                }
            ]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["example/approved-pr#42"])

    integration = metrics["integrations"][0]
    assert integration["review_state"] == "approved"
    assert integration["approvals"] == ["maintainer"]
    assert integration["next_action"] == "wait for maintainer merge"


def test_collect_integration_metrics_surfaces_requested_review_changes(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/example/reviewed-pr/pulls/42":
            return {
                "number": 42,
                "title": "Add FunASR",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/example/reviewed-pr/pull/42",
                "updated_at": "2026-07-23T04:16:43Z",
                "head": {"sha": "824c3c5", "ref": "add-funasr"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/example/reviewed-pr/commits/824c3c5/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/example/reviewed-pr/commits/824c3c5/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "validate", "status": "completed", "conclusion": "success"},
                ],
            }
        if url == "https://api.github.com/repos/example/reviewed-pr/pulls/42/reviews?per_page=100":
            return [
                {
                    "id": 124,
                    "state": "CHANGES_REQUESTED",
                    "submitted_at": "2026-07-23T04:00:00Z",
                    "user": {"login": "maintainer"},
                }
            ]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["example/reviewed-pr#42"])

    integration = metrics["integrations"][0]
    assert integration["review_state"] == "changes_requested"
    assert integration["changes_requested_by"] == ["maintainer"]
    assert integration["next_action"] == "address review feedback"


def test_review_summary_reads_later_pages_before_deciding(monkeypatch):
    module = load_growth_metrics_module()
    first_page = [
        {
            "id": 1,
            "state": "CHANGES_REQUESTED",
            "user": {"login": "maintainer"},
        }
    ] + [
        {
            "id": review_id,
            "state": "COMMENTED",
            "user": {"login": f"reviewer-{review_id}"},
        }
        for review_id in range(2, 101)
    ]

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/example/long-pr/pulls/42/reviews?per_page=100":
            return first_page
        if url == "https://api.github.com/repos/example/long-pr/pulls/42/reviews?per_page=100&page=2":
            return [
                {
                    "id": 101,
                    "state": "APPROVED",
                    "user": {"login": "maintainer"},
                }
            ]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    summary = module.summarize_pull_request_reviews("example/long-pr", 42)

    assert summary["state"] == "approved"
    assert summary["approvals"] == ["maintainer"]
    assert summary["changes_requested_by"] == []


def test_review_summary_keeps_latest_decisive_state_per_reviewer(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        assert url == "https://api.github.com/repos/example/review-history/pulls/42/reviews?per_page=100"
        return [
            {"state": "CHANGES_REQUESTED", "user": {"login": "alice"}},
            {"state": "COMMENTED", "user": {"login": "alice"}},
            {"state": "APPROVED", "user": {"login": "alice"}},
            {"state": "APPROVED", "user": {"login": "bob"}},
            {"state": "COMMENTED", "user": {"login": "bob"}},
            {"state": "APPROVED", "user": {"login": "carol"}},
            {"state": "DISMISSED", "user": {"login": "carol"}},
        ]

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    summary = module.summarize_pull_request_reviews("example/review-history", 42)

    assert summary["state"] == "approved"
    assert summary["approvals"] == ["alice", "bob"]
    assert summary["changes_requested_by"] == []


def test_blocked_approved_pr_waits_for_maintainer_merge(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/example/blocked-pr/pulls/42":
            return {
                "number": 42,
                "title": "Add FunASR",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/example/blocked-pr/pull/42",
                "updated_at": "2026-07-23T04:16:43Z",
                "head": {"sha": "abc1234", "ref": "add-funasr"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/example/blocked-pr/commits/abc1234/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/example/blocked-pr/commits/abc1234/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "test", "status": "completed", "conclusion": "success"}
                ],
            }
        if url == "https://api.github.com/repos/example/blocked-pr/pulls/42/reviews?per_page=100":
            return [{"state": "APPROVED", "user": {"login": "maintainer"}}]
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["example/blocked-pr#42"])

    assert metrics["integrations"][0]["next_action"] == "wait for maintainer merge"


def test_wait_for_maintainer_merge_is_not_active_operator_work():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-23T06:00:00+00:00",
        "integrations": [
            {
                "pr": "huggingface/optimum-intel#1874",
                "html_url": "https://github.com/huggingface/optimum-intel/pull/1874",
                "state": "open",
                "mergeable": True,
                "mergeable_state": "clean",
                "repo_stars": 608,
                "repo_forks": 147,
                "updated_at": "2026-07-23T04:16:43Z",
                "updated_age_days": 0,
                "checks": {
                    "state": "success",
                    "failed_check_runs": [],
                    "pending_check_runs": [],
                },
                "next_action": "wait for maintainer merge",
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert "## Active operator queue" not in output


def test_known_assisted_review_requests_include_validated_discovery_prs():
    module = load_growth_metrics_module()

    expected_prs = {
        "mahseema/awesome-ai-tools#1689",
        "ai4s-research/awesome-ai-for-science#69",
    }

    assert expected_prs.issubset(set(module.KNOWN_ASSISTED_REVIEW_REQUESTS))


def test_known_assisted_review_requests_include_validated_review_gate_prs():
    module = load_growth_metrics_module()

    expected_prs = {
        "livekit/agents#6176",
        "run-llama/llama_index#21958",
        "run-llama/llama_index#21996",
        "BerriAI/litellm-docs#610",
        "mudler/LocalAI#10090",
        "ray-project/ray#64053",
        "agno-agi/agno#8501",
        "pipecat-ai/pipecat#4844",
        "GetStream/Vision-Agents#606",
        "Uberi/speech_recognition#903",
        "huggingface/speech-to-speech#319",
        "speaches-ai/speaches#658",
    }

    assert expected_prs.issubset(set(module.KNOWN_ASSISTED_REVIEW_REQUESTS))


def test_collect_integration_metrics_marks_known_assisted_review_request(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/lukasmasuch/best-of-ml-python/pulls/455":
            return {
                "number": 455,
                "title": "Add FunASR",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/lukasmasuch/best-of-ml-python/pull/455",
                "updated_at": "2026-06-30T09:16:59Z",
                "head": {"sha": "48f3070", "ref": "add-funasr"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/lukasmasuch/best-of-ml-python/commits/48f3070/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/lukasmasuch/best-of-ml-python/commits/48f3070/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {"name": "tests", "status": "completed", "conclusion": "success"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["lukasmasuch/best-of-ml-python#455"])

    integration = metrics["integrations"][0]
    assert integration["known_assisted_review_reason"] == "review evidence already posted; avoid duplicate pings"
    assert integration["next_action"] == "wait for maintainer review"


def test_collect_integration_metrics_applies_known_review_gate(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/punkpeye/awesome-mcp-servers/pulls/7153":
            return {
                "number": 7153,
                "title": "Add FunASR speech recognition MCP server",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "clean",
                "html_url": "https://github.com/punkpeye/awesome-mcp-servers/pull/7153",
                "updated_at": "2026-06-16T04:21:55Z",
                "head": {"sha": "d28680e", "ref": "add-funasr-mcp-server"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/punkpeye/awesome-mcp-servers/commits/d28680e/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/punkpeye/awesome-mcp-servers/commits/d28680e/check-runs?per_page=100":
            return {
                "total_count": 2,
                "check_runs": [
                    {"name": "check-submission", "status": "completed", "conclusion": "success"},
                    {"name": "welcome", "status": "completed", "conclusion": "skipped"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["punkpeye/awesome-mcp-servers#7153"])

    integration = metrics["integrations"][0]
    assert integration["checks"]["state"] == "success"
    assert integration["known_review_gate_reason"] == "Glama listing and score badge are live"
    assert integration["next_action"] == "wait for maintainer review"


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
    assert integration["next_action"] == "wait for author CLA"
    assert integration["checks"]["pending_check_runs"] == [
        {
            "name": "license/cla",
            "status": "pending",
            "url": "https://cla-assistant.io/activepieces/activepieces?pullRequest=13985",
        }
    ]


def test_collect_integration_metrics_classifies_autogpt_process_gate(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/Significant-Gravitas/AutoGPT/pulls/13500":
            return {
                "number": 13500,
                "title": "Support configurable transcription endpoints",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/Significant-Gravitas/AutoGPT/pull/13500",
                "updated_at": "2026-07-07T21:13:23Z",
                "head": {"sha": "ed9f137", "ref": "codex/openai-compatible-stt"},
                "base": {"ref": "dev"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/Significant-Gravitas/AutoGPT/commits/ed9f137/status":
            return {
                "state": "failure",
                "statuses": [
                    {
                        "context": "Vercel",
                        "state": "failure",
                        "target_url": "https://vercel.com/git/authorize?team=Significant%20Gravitas",
                    },
                    {
                        "context": "license/cla",
                        "state": "pending",
                        "target_url": "https://cla-assistant.io/Significant-Gravitas/AutoGPT?pullRequest=13500",
                    },
                ],
            }
        if url == "https://api.github.com/repos/Significant-Gravitas/AutoGPT/commits/ed9f137/check-runs?per_page=100":
            return {
                "total_count": 4,
                "check_runs": [
                    {"name": "check API types", "status": "completed", "conclusion": "success"},
                    {"name": "lint", "status": "completed", "conclusion": "success"},
                    {"name": "integration_test", "status": "completed", "conclusion": "success"},
                    {"name": "end-to-end tests", "status": "completed", "conclusion": "success"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["Significant-Gravitas/AutoGPT#13500"])

    integration = metrics["integrations"][0]
    assert integration["known_review_gate_reason"] == (
        "CLA must be completed by an authorized author; Vercel preview also requires AutoGPT team authorization"
    )
    assert integration["next_action"] == "wait for author CLA"
    assert {check["name"] for check in integration["checks"]["failed_check_runs"]} == {"Vercel"}
    assert {check["name"] for check in integration["checks"]["pending_check_runs"]} == {"license/cla"}


def test_collect_integration_metrics_classifies_review_bot_failure(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/TEN-framework/ten-framework/pulls/2191":
            return {
                "number": 2191,
                "title": "feat: add funasr_asr_python local FunASR ASR extension",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/TEN-framework/ten-framework/pull/2191",
                "updated_at": "2026-06-29T23:02:41Z",
                "head": {"sha": "43cbedc", "ref": "add-funasr-asr-extension"},
                "base": {"ref": "main"},
                "user": {"login": "LauraGPT"},
            }
        if url == "https://api.github.com/repos/TEN-framework/ten-framework/commits/43cbedc/status":
            return {"state": "pending", "statuses": []}
        if url == "https://api.github.com/repos/TEN-framework/ten-framework/commits/43cbedc/check-runs?per_page=100":
            return {
                "total_count": 1,
                "check_runs": [
                    {
                        "name": "claude-review",
                        "status": "completed",
                        "conclusion": "failure",
                        "html_url": "https://github.com/TEN-framework/ten-framework/actions/runs/27985905295/job/82827178776",
                    },
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["TEN-framework/ten-framework#2191"])

    integration = metrics["integrations"][0]
    assert integration["known_review_gate_reason"] == (
        "Claude review action requires maintainer permissions for fork PRs"
    )
    assert integration["next_action"] == "wait for maintainer review"
    assert integration["checks"]["failed_check_runs"][0]["name"] == "claude-review"


def test_collect_integration_metrics_classifies_vercel_authorization_gate(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/mem0ai/mem0/pulls/5571":
            return {
                "number": 5571,
                "title": "feat: add optional FunASR audio transcription helper",
                "state": "open",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "blocked",
                "html_url": "https://github.com/mem0ai/mem0/pull/5571",
                "updated_at": "2026-06-30T05:20:27Z",
                "head": {"sha": "d0b1cfa", "ref": "funasr-audio-helper"},
                "base": {"ref": "main"},
                "user": {"login": "anneheartrecord"},
            }
        if url == "https://api.github.com/repos/mem0ai/mem0/commits/d0b1cfa/status":
            return {
                "state": "failure",
                "statuses": [
                    {
                        "context": "Vercel",
                        "state": "failure",
                        "target_url": "https://vercel.com/git/authorize?team=Mem0&slug=mem0&pullRequest=5571",
                    }
                ],
            }
        if url == "https://api.github.com/repos/mem0ai/mem0/commits/d0b1cfa/check-runs?per_page=100":
            return {"total_count": 0, "check_runs": []}
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["mem0ai/mem0#5571"])

    integration = metrics["integrations"][0]
    assert integration["next_action"] == "wait for preview authorization"
    assert integration["checks"]["failed_check_runs"] == [
        {
            "name": "Vercel",
            "conclusion": "failure",
            "url": "https://vercel.com/git/authorize?team=Mem0&slug=mem0&pullRequest=5571",
        }
    ]


def test_recommend_integration_action_treats_blocked_unknown_checks_as_review_gate():
    module = load_growth_metrics_module()

    action = module.recommend_integration_action(
        {"state": "open", "draft": False, "mergeable_state": "blocked"},
        {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
    )

    assert action == "review gate"


def test_recommend_integration_action_waits_for_maintainer_after_assisted_blocked_review_gate():
    module = load_growth_metrics_module()

    action = module.recommend_integration_action(
        {"state": "open", "draft": False, "mergeable_state": "blocked"},
        {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
        known_assisted_review_reason="review evidence already posted; avoid duplicate pings",
    )

    assert action == "wait for maintainer review"


def test_recommend_integration_action_treats_clean_unknown_checks_as_request_review():
    module = load_growth_metrics_module()

    action = module.recommend_integration_action(
        {"state": "open", "draft": False, "mergeable_state": "clean"},
        {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
    )

    assert action == "request review"


def test_recommend_integration_action_treats_dirty_merge_state_as_conflict_resolution():
    module = load_growth_metrics_module()

    action = module.recommend_integration_action(
        {"state": "open", "draft": False, "mergeable_state": "dirty"},
        {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
    )

    assert action == "resolve conflicts"


def test_recommend_integration_action_prioritizes_dirty_merge_state_over_failed_checks():
    module = load_growth_metrics_module()

    action = module.recommend_integration_action(
        {"state": "open", "draft": False, "mergeable_state": "dirty"},
        {
            "state": "failure",
            "failed_check_runs": [{"name": "ci", "conclusion": "failure"}],
            "pending_check_runs": [],
        },
    )

    assert action == "resolve conflicts"


def test_recommend_integration_action_prioritizes_dirty_merge_state_over_pending_checks():
    module = load_growth_metrics_module()

    action = module.recommend_integration_action(
        {"state": "open", "draft": False, "mergeable_state": "dirty"},
        {
            "state": "pending",
            "failed_check_runs": [],
            "pending_check_runs": [{"name": "ci", "status": "queued"}],
        },
    )

    assert action == "resolve conflicts"


def test_collect_integration_metrics_treats_merged_sglang_omni_as_completed(monkeypatch):
    module = load_growth_metrics_module()

    def fake_fetch_json(url, headers=None):
        if url == "https://api.github.com/repos/sgl-project/sglang-omni/pulls/1078":
            return {
                "number": 1078,
                "title": "Add Fun-ASR serving support",
                "state": "closed",
                "draft": False,
                "mergeable": True,
                "mergeable_state": "unknown",
                "merged_at": "2026-07-18T15:11:15Z",
                "html_url": "https://github.com/sgl-project/sglang-omni/pull/1078",
                "updated_at": "2026-07-18T15:11:15Z",
                "head": {"sha": "b6ea7ab1", "ref": "add-fun-asr"},
                "base": {"ref": "main"},
                "user": {"login": "PoTaTo-Mika"},
            }
        if url == "https://api.github.com/repos/sgl-project/sglang-omni/commits/b6ea7ab1/status":
            return {"state": "success", "statuses": []}
        if url == "https://api.github.com/repos/sgl-project/sglang-omni/commits/b6ea7ab1/check-runs?per_page=100":
            return {
                "total_count": 2,
                "check_runs": [
                    {"name": "unit", "status": "completed", "conclusion": "success"},
                    {"name": "cleanup - PR CI home on close", "status": "completed", "conclusion": "cancelled"},
                ],
            }
        raise AssertionError(f"unexpected URL: {url}")

    monkeypatch.setattr(module, "fetch_json", fake_fetch_json)

    metrics = module.collect_integration_metrics(["sgl-project/sglang-omni#1078"])

    integration = metrics["integrations"][0]
    assert integration["state"] == "closed"
    assert integration["checks"]["state"] == "failure"
    assert integration["next_action"] == "merged / done"


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
                "repo_stars": 155_000,
                "repo_forks": 31_000,
                "updated_at": "2026-06-29T23:45:57Z",
                "updated_age_days": 2,
                "next_action": "fix checks",
                "checks": {"state": "failure", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert "| Pull request | Stars | Forks | State | Mergeable | Checks | Failed | Pending | Age | Action | Updated |" in output
    assert (
        "| [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180) | "
        "155,000 | 31,000 | open | blocked | failure | 0 | 0 | 2d | fix checks | `2026-06-29T23:45:57Z` |"
    ) in output


def test_format_integration_markdown_includes_known_external_failure_reason():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "huggingface/transformers#46180",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "state": "open",
                "mergeable_state": "blocked",
                "repo_stars": 162_000,
                "repo_forks": 33_000,
                "updated_at": "2026-06-30T04:24:15Z",
                "updated_age_days": 0,
                "next_action": "request rerun",
                "known_external_failure_reason": (
                    "LightOnOCR shared hub-cache read-only failure; PR CI status is the aggregate failure"
                ),
                "checks": {
                    "state": "failure",
                    "failed_check_runs": [
                        {
                            "name": "pr-ci / tests_processors / tests_processors [shard 3/8]",
                            "conclusion": "failure",
                            "url": "https://github.com/huggingface/transformers/actions/runs/28419525291/job/84209871338",
                        }
                    ],
                    "pending_check_runs": [],
                },
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert (
        "- Known external failure: LightOnOCR shared hub-cache read-only failure; "
        "PR CI status is the aggregate failure"
    ) in output


def test_format_integration_markdown_marks_missing_exposure_metrics_as_unavailable():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "deleted-org/deleted-repo#1",
                "html_url": "https://github.com/deleted-org/deleted-repo/pull/1",
                "state": "open",
                "mergeable_state": "unknown",
                "updated_at": "2026-06-29T23:45:57Z",
                "updated_age_days": None,
                "next_action": "inspect",
                "checks": {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert (
        "| [deleted-org/deleted-repo#1](https://github.com/deleted-org/deleted-repo/pull/1) | "
        "n/a | n/a | open | unknown | unknown | 0 | 0 | n/a | inspect | `2026-06-29T23:45:57Z` |"
    ) in output


def test_format_integration_markdown_marks_unknown_review_gate_as_review_gated():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "punkpeye/awesome-mcp-servers#7153",
                "html_url": "https://github.com/punkpeye/awesome-mcp-servers/pull/7153",
                "state": "open",
                "mergeable": None,
                "mergeable_state": "unknown",
                "repo_stars": 91_000,
                "repo_forks": 13_000,
                "updated_at": "2026-07-22T17:46:15Z",
                "updated_age_days": 0,
                "next_action": "wait for maintainer review",
                "known_review_gate_reason": "Glama listing and score badge are live",
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert (
        "| [punkpeye/awesome-mcp-servers#7153](https://github.com/punkpeye/awesome-mcp-servers/pull/7153) | "
        "91,000 | 13,000 | open | review-gated | success | 0 | 0 | 0d | wait for maintainer review | `2026-07-22T17:46:15Z` |"
    ) in output


def test_format_integration_markdown_marks_mergeable_blocked_review_wait_as_review_gated():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "huggingface/transformers#46180",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "state": "open",
                "mergeable": True,
                "mergeable_state": "blocked",
                "repo_stars": 162_000,
                "repo_forks": 33_000,
                "updated_at": "2026-07-22T08:50:59Z",
                "updated_age_days": 0,
                "next_action": "wait for maintainer review",
                "known_assisted_review_reason": (
                    "checks are green and the remaining blocker is an active maintainer-held review thread"
                ),
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert (
        "| [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180) | "
        "162,000 | 33,000 | open | review-gated | success | 0 | 0 | 0d | wait for maintainer review | `2026-07-22T08:50:59Z` |"
    ) in output


def test_format_integration_markdown_lists_high_exposure_priorities_by_stars():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "ray-project/ray#64053",
                "html_url": "https://github.com/ray-project/ray/pull/64053",
                "state": "open",
                "mergeable_state": "blocked",
                "repo_stars": 43_000,
                "repo_forks": 7_700,
                "updated_at": "2026-06-30T07:02:58Z",
                "updated_age_days": 0,
                "next_action": "review gate",
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "huggingface/transformers#46180",
                "html_url": "https://github.com/huggingface/transformers/pull/46180",
                "state": "open",
                "mergeable_state": "blocked",
                "repo_stars": 162_000,
                "repo_forks": 33_000,
                "updated_at": "2026-06-30T04:24:15Z",
                "updated_age_days": 0,
                "next_action": "request rerun",
                "checks": {"state": "failure", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "deleted-org/deleted-repo#1",
                "html_url": "https://github.com/deleted-org/deleted-repo/pull/1",
                "state": "open",
                "mergeable_state": "unknown",
                "updated_at": "2026-06-29T23:45:57Z",
                "updated_age_days": None,
                "next_action": "inspect",
                "checks": {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
            },
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert "## High-exposure priorities" in output
    transformers = "- [huggingface/transformers#46180](https://github.com/huggingface/transformers/pull/46180): 162,000 stars, request rerun"
    ray = "- [ray-project/ray#64053](https://github.com/ray-project/ray/pull/64053): 43,000 stars, review gate"
    assert transformers in output
    assert ray in output
    assert output.index(transformers) < output.index(ray)
    assert "deleted-org/deleted-repo#1" not in output.split("## High-exposure priorities", 1)[1].split("## Failed or pending checks", 1)[0]


def test_format_integration_markdown_lists_active_operator_queue():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "getpaseo/paseo#1634",
                "html_url": "https://github.com/getpaseo/paseo/pull/1634",
                "state": "open",
                "mergeable_state": "dirty",
                "repo_stars": 9_500,
                "repo_forks": 900,
                "updated_at": "2026-06-30T09:47:12Z",
                "updated_age_days": 0,
                "next_action": "resolve conflicts",
                "checks": {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "huggingface/optimum-intel#1801",
                "html_url": "https://github.com/huggingface/optimum-intel/pull/1801",
                "state": "open",
                "mergeable_state": "unstable",
                "repo_stars": 600,
                "repo_forks": 240,
                "updated_at": "2026-06-30T04:20:02Z",
                "updated_age_days": 0,
                "next_action": "review gate",
                "checks": {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "livekit/agents#6176",
                "html_url": "https://github.com/livekit/agents/pull/6176",
                "state": "open",
                "mergeable_state": "blocked",
                "repo_stars": 11_000,
                "repo_forks": 3_200,
                "updated_at": "2026-06-30T10:55:17Z",
                "updated_age_days": 0,
                "next_action": "wait for maintainer review",
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "activepieces/activepieces#13985",
                "html_url": "https://github.com/activepieces/activepieces/pull/13985",
                "state": "open",
                "mergeable_state": "blocked",
                "repo_stars": 23_000,
                "repo_forks": 3_800,
                "updated_at": "2026-06-30T08:46:39Z",
                "updated_age_days": 0,
                "next_action": "resolve CLA",
                "checks": {"state": "pending", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "huggingface/optimum-intel#1801",
                "html_url": "https://github.com/huggingface/optimum-intel/pull/1801",
                "state": "open",
                "mergeable_state": "unstable",
                "repo_stars": 18_000,
                "repo_forks": 2_800,
                "updated_at": "2026-06-30T04:20:02Z",
                "updated_age_days": 0,
                "next_action": "wait for maintainer preliminary PR",
                "checks": {"state": "failure", "failed_check_runs": [], "pending_check_runs": []},
            },
        ],
    }

    output = module.format_integration_markdown(metrics)

    active_queue = output.split("## Active operator queue", 1)[1].split("## Failed or pending checks", 1)[0]
    assert "- [getpaseo/paseo#1634](https://github.com/getpaseo/paseo/pull/1634): resolve conflicts" in active_queue
    assert (
        "- [huggingface/optimum-intel#1801](https://github.com/huggingface/optimum-intel/pull/1801): review gate"
        in active_queue
    )
    assert "livekit/agents#6176" not in active_queue
    assert "activepieces/activepieces#13985" not in active_queue
    assert "wait for maintainer preliminary PR" not in active_queue


def test_format_integration_markdown_excludes_satisfied_review_gates_from_handoff():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "punkpeye/awesome-mcp-servers#7153",
                "html_url": "https://github.com/punkpeye/awesome-mcp-servers/pull/7153",
                "state": "open",
                "mergeable_state": "clean",
                "repo_stars": 90_000,
                "repo_forks": 12_000,
                "updated_at": "2026-06-16T04:21:55Z",
                "updated_age_days": 16,
                "next_action": "wait for maintainer review",
                "known_review_gate_reason": "Glama listing and score badge are live",
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            },
            {
                "pr": "huggingface/optimum-intel#1801",
                "html_url": "https://github.com/huggingface/optimum-intel/pull/1801",
                "state": "open",
                "mergeable_state": "unstable",
                "repo_stars": 600,
                "repo_forks": 240,
                "updated_at": "2026-06-30T04:20:02Z",
                "updated_age_days": 0,
                "next_action": "review gate",
                "checks": {"state": "unknown", "failed_check_runs": [], "pending_check_runs": []},
            },
        ],
    }

    output = module.format_integration_markdown(metrics)

    active_queue = output.split("## Active operator queue", 1)[1].split("## Manual review gates", 1)[0]
    assert "punkpeye/awesome-mcp-servers#7153" not in active_queue
    assert "## Manual handoff gates" not in output


def test_format_integration_markdown_lists_known_review_gates():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "punkpeye/awesome-mcp-servers#7153",
                "html_url": "https://github.com/punkpeye/awesome-mcp-servers/pull/7153",
                "state": "open",
                "mergeable_state": "clean",
                "repo_stars": 90_000,
                "repo_forks": 12_000,
                "updated_at": "2026-06-16T04:21:55Z",
                "updated_age_days": 14,
                "next_action": "wait for maintainer review",
                "known_review_gate_reason": "Glama listing and score badge are live",
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert "## Manual review gates" in output
    assert (
        "- [punkpeye/awesome-mcp-servers#7153](https://github.com/punkpeye/awesome-mcp-servers/pull/7153): "
        "Glama listing and score badge are live"
    ) in output


def test_format_integration_markdown_lists_assisted_review_waits():
    module = load_growth_metrics_module()
    metrics = {
        "collected_at_utc": "2026-07-02T00:00:00+00:00",
        "integrations": [
            {
                "pr": "lukasmasuch/best-of-ml-python#455",
                "html_url": "https://github.com/lukasmasuch/best-of-ml-python/pull/455",
                "state": "open",
                "mergeable_state": "clean",
                "repo_stars": 23_652,
                "repo_forks": 3_800,
                "updated_at": "2026-06-30T09:16:59Z",
                "updated_age_days": 0,
                "next_action": "wait for maintainer review",
                "known_assisted_review_reason": "review evidence already posted; avoid duplicate pings",
                "checks": {"state": "success", "failed_check_runs": [], "pending_check_runs": []},
            }
        ],
    }

    output = module.format_integration_markdown(metrics)

    assert "## Assisted review waits" in output
    assert (
        "- [lukasmasuch/best-of-ml-python#455](https://github.com/lukasmasuch/best-of-ml-python/pull/455): "
        "review evidence already posted; avoid duplicate pings"
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
        "QwenAudio/Fun-ASR": 1318,
        "QwenAudio/SenseVoice": 8718,
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


def test_main_treats_explicit_multi_repos_as_ecosystem_json(monkeypatch):
    module = load_growth_metrics_module()

    repo_stars = {
        "modelscope/FunASR": 18714,
        "QwenAudio/Fun-ASR": 1318,
        "QwenAudio/SenseVoice": 8718,
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
            "--format",
            "json",
            "--repos",
            "modelscope/FunASR",
            "QwenAudio/Fun-ASR",
            "QwenAudio/SenseVoice",
            "modelscope/FunClip",
        ],
    )

    with redirect_stdout(io.StringIO()) as stdout:
        assert module.main() == 0

    payload = json.loads(stdout.getvalue())
    assert "github" not in payload
    assert payload["ecosystem"]["total_stars"] == 34622
    assert [repo["repo"] for repo in payload["ecosystem"]["repositories"]] == [
        "modelscope/FunASR",
        "QwenAudio/Fun-ASR",
        "QwenAudio/SenseVoice",
        "modelscope/FunClip",
    ]


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
                "base": {
                    "ref": "master",
                    "repo": {
                        "stargazers_count": 36_000,
                        "forks_count": 6_800,
                    },
                },
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
    assert integration["repo_stars"] == 36_000
    assert integration["repo_forks"] == 6_800
    assert integration["checks"]["state"] == "success"
    assert integration["next_action"] == "wait for maintainer review"
    assert integration["known_assisted_review_reason"] == "review evidence already posted; avoid duplicate pings"
