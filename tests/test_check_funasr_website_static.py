import importlib.util
import sys
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
            捐赠金额用于购买服务器与 <a href="https://www.funasr.com/">www.funasr.com</a> 域名
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
        "https://www.funasr.com/blog/funasr-llama-cpp-whisper-cpp-alternative.html": """
            runtime-llamacpp-v0.1.8
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            Vulkan / CUDA
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-llama-cpp-whisper-cpp-alternative.html": """
            runtime-llamacpp-v0.1.8
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            Vulkan / CUDA
            <a href="/en/donors.html">Thanks</a>
        """,
    }

    assert checker.validate_pages(pages) == []


def test_website_contract_reports_stale_runtime_and_star_copy():
    checker = _load_module()

    pages = {
        url: "" for url in checker.PAGE_CONTRACTS
    }
    pages["https://www.funasr.com/ecosystem.html"] = "16K+"
    pages[
        "https://www.funasr.com/blog/funasr-llama-cpp-whisper-cpp-alternative.html"
    ] = "runtime-llamacpp-v0.1.1"

    failures = checker.validate_pages(pages)

    assert any("ecosystem.html" in failure and "forbidden `16K+`" in failure for failure in failures)
    assert any(
        "funasr-llama-cpp-whisper-cpp-alternative.html" in failure
        and "forbidden `runtime-llamacpp-v0.1.1`" in failure
        for failure in failures
    )
