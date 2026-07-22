import importlib.util
import sys
import urllib.error
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_funasr_website_static.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("check_funasr_website_static", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_website_contract_accepts_current_public_copy():
    checker = _load_module()

    pages = {
        "https://www.funasr.com/": """
            工业级 语音识别服务
            /v1/audio/transcriptions
            vLLM 加速
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/": """
            Industrial Speech Recognition
            OpenAI-compatible
            /v1/audio/transcriptions
            vLLM Acceleration
        """,
        "https://www.funasr.com/ecosystem.html": """
            <div class="stat-num">35K+</div>
            <a href="/donors.html">功德榜</a>
            <a href="https://github.com/BerriAI/litellm">LiteLLM</a>
            <div>custom_openai</div>
            <div>54.3K stars</div>
        """,
        "https://www.funasr.com/en/ecosystem.html": """
            <div class="stat-num">35K+</div>
            <a href="/en/donors.html">Thanks</a>
            <a href="https://github.com/BerriAI/litellm">LiteLLM</a>
            <div>custom_openai</div>
            <div>54.3K stars</div>
        """,
        "https://www.funasr.com/donors.html": """
            捐赠金额用于购买服务器与 <a href="https://www.funasr.com/">www.funasr.com</a> 域名，帮助官网和社区资料持续在线。
        """,
        "https://www.funasr.com/en/donors.html": """
            Donations helped purchase servers and the <a href="https://www.funasr.com/">www.funasr.com</a> domain.
        """,
        "https://www.funasr.com/blog/funasr-cli-transcribe-command-line.html": """
            pip install -U funasr   # 推荐 funasr ≥ 1.3.26
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-cli-transcribe-command-line.html": """
            pip install -U funasr   # recommended funasr &gt;= 1.3.26
            <a href="/en/donors.html">Thanks</a>
        """,
        "https://www.funasr.com/llama-cpp.html": """
            runtime-llamacpp-v0.1.8
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            funasr-llamacpp-windows-x64-cuda.zip
            Fun-ASR-Nano
            GGUF
        """,
        "https://www.funasr.com/blog/funasr-llama-cpp-whisper-cpp-alternative.html": """
            runtime-llamacpp-v0.1.8
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            Fun-ASR-Nano-GGUF
            Vulkan / CUDA
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-llama-cpp-whisper-cpp-alternative.html": """
            runtime-llamacpp-v0.1.8
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            Fun-ASR-Nano-GGUF
            Vulkan / CUDA
            <a href="/en/donors.html">Thanks</a>
        """,
    }

    assert checker.validate_pages(pages) == []


def test_website_contract_includes_homepage_entrypoints():
    checker = _load_module()

    assert "https://www.funasr.com/" in checker.PAGE_CONTRACTS
    assert "https://www.funasr.com/en/" in checker.PAGE_CONTRACTS


def test_website_contract_reports_stale_runtime_and_star_copy():
    checker = _load_module()

    pages = {
        url: "" for url in checker.PAGE_CONTRACTS
    }
    pages["https://www.funasr.com/ecosystem.html"] = "16K+"
    pages[
        "https://www.funasr.com/blog/funasr-llama-cpp-whisper-cpp-alternative.html"
    ] = "runtime-llamacpp-v0.1.1"
    pages["https://www.funasr.com/llama-cpp.html"] = "runtime-llamacpp-v0.1.1"

    failures = checker.validate_pages(pages)

    assert any("ecosystem.html" in failure and "forbidden `16K+`" in failure for failure in failures)
    assert any(
        "funasr-llama-cpp-whisper-cpp-alternative.html" in failure
        and "forbidden `runtime-llamacpp-v0.1.1`" in failure
        for failure in failures
    )
    assert any(
        "llama-cpp.html" in failure and "forbidden `runtime-llamacpp-v0.1.1`" in failure
        for failure in failures
    )


def test_website_contract_requires_visible_donor_usage_copy():
    checker = _load_module()

    pages = {
        url: "ok" for url in checker.PAGE_CONTRACTS
    }
    pages["https://www.funasr.com/donors.html"] = """
        <head>
            <meta name="description" content="捐赠金额用于购买服务器与 www.funasr.com 域名。">
        </head>
        <body>FunASR 社区功德榜</body>
    """
    pages["https://www.funasr.com/en/donors.html"] = """
        <head>
            <meta name="description" content="Donations helped purchase servers and the www.funasr.com domain.">
        </head>
        <body>FunASR Community Thanks</body>
    """

    failures = checker.validate_pages(pages)

    assert any(
        "donors.html" in failure
        and "visible text missing `捐赠金额用于购买服务器与`" in failure
        for failure in failures
    )
    assert any(
        "en/donors.html" in failure
        and "visible text missing `Donations helped purchase servers`" in failure
        for failure in failures
    )


def test_fetch_pages_retries_transient_url_errors(monkeypatch):
    checker = _load_module()
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self):
            return b"ok"

    def fake_urlopen(url, timeout):
        calls.append((url, timeout))
        if len(calls) == 1:
            raise urllib.error.URLError("temporary SSL handshake timeout")
        return FakeResponse()

    monkeypatch.setattr(checker.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(
        checker,
        "PAGE_CONTRACTS",
        {
            "https://www.funasr.com/example.html": checker.PageContract(
                required=("ok",)
            )
        },
    )

    pages = checker.fetch_pages(timeout=3, retries=2)

    assert pages == {"https://www.funasr.com/example.html": "ok"}
    assert calls == [
        ("https://www.funasr.com/example.html", 3),
        ("https://www.funasr.com/example.html", 3),
    ]
