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
            <a href="https://marketplace.dify.ai/plugin/langgenius/funasr">
                FunASR 官方插件 0.1.1
            </a>
            <div>支持最大 25 MB 音频上传</div>
        """,
        "https://www.funasr.com/en/ecosystem.html": """
            <div class="stat-num">35K+</div>
            <a href="/en/donors.html">Thanks</a>
            <a href="https://github.com/BerriAI/litellm">LiteLLM</a>
            <div>custom_openai</div>
            <div>54.3K stars</div>
            <a href="https://marketplace.dify.ai/plugin/langgenius/funasr">
                FunASR plugin 0.1.1
            </a>
            <div>supports 25 MB uploads</div>
        """,
        "https://www.funasr.com/donors.html": """
            捐赠资金用于社区基础设施建设，包括购买和维护服务器，以及购买、续费和维护
            <a href="https://www.funasr.com/">www.funasr.com</a> 域名。
        """,
        "https://www.funasr.com/en/donors.html": """
            Donations fund community infrastructure, including server purchase and maintenance,
            as well as the purchase, renewal, and maintenance of the
            <a href="https://www.funasr.com/">www.funasr.com</a> domain.
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
            runtime-llamacpp-v0.1.9
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            funasr-llamacpp-windows-x64-vulkan.zip
            funasr-llamacpp-windows-x64-cuda.zip
            Fun-ASR-Nano
            GGUF
        """,
        "https://www.funasr.com/en/llama-cpp.html": """
            runtime-llamacpp-v0.1.9
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            funasr-llamacpp-windows-x64-vulkan.zip
            funasr-llamacpp-windows-x64-cuda.zip
            Fun-ASR-Nano
            GGUF
        """,
        "https://www.funasr.com/blog/funasr-llama-cpp-whisper-cpp-alternative.html": """
            runtime-llamacpp-v0.1.9
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            funasr-llamacpp-windows-x64-vulkan.zip
            Fun-ASR-Nano-GGUF
            Linux/Windows Vulkan 与 Windows CUDA
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-llama-cpp-whisper-cpp-alternative.html": """
            runtime-llamacpp-v0.1.9
            funasr-llamacpp-linux-x64-vulkan.tar.gz
            funasr-llamacpp-windows-x64-vulkan.zip
            Fun-ASR-Nano-GGUF
            Linux/Windows Vulkan and Windows CUDA
            <a href="/en/donors.html">Thanks</a>
        """,
        "https://www.funasr.com/blog/funasr-v1-3-26-openai-vllm-llama-cpp.html": """
            funasr==1.3.26
            /v1/audio/transcriptions
            RTFx 340
            runtime-llamacpp-v0.1.9
            funasr-llamacpp-windows-x64-vulkan.zip
            https://github.com/modelscope/FunASR
            https://github.com/FunAudioLLM/Fun-ASR
            https://github.com/FunAudioLLM/SenseVoice
            https://github.com/modelscope/FunClip
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-v1-3-26-openai-vllm-llama-cpp.html": """
            funasr==1.3.26
            /v1/audio/transcriptions
            RTFx 340
            runtime-llamacpp-v0.1.9
            funasr-llamacpp-windows-x64-vulkan.zip
            https://github.com/modelscope/FunASR
            https://github.com/FunAudioLLM/Fun-ASR
            https://github.com/FunAudioLLM/SenseVoice
            https://github.com/modelscope/FunClip
            <a href="/en/donors.html">Thanks</a>
        """,
        "https://www.funasr.com/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html": """
            funasr==1.3.27
            /v1/audio/transcriptions
            verbose_json.language
            language":"en
            AutoModel
            runtime-llamacpp-v0.1.9
            https://github.com/modelscope/FunASR/releases/tag/v1.3.27
            https://github.com/modelscope/FunASR
            https://github.com/FunAudioLLM/Fun-ASR
            https://github.com/FunAudioLLM/SenseVoice
            https://github.com/modelscope/FunClip
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html": """
            funasr==1.3.27
            /v1/audio/transcriptions
            verbose_json.language
            language":"en
            AutoModel
            runtime-llamacpp-v0.1.9
            https://github.com/modelscope/FunASR/releases/tag/v1.3.27
            https://github.com/modelscope/FunASR
            https://github.com/FunAudioLLM/Fun-ASR
            https://github.com/FunAudioLLM/SenseVoice
            https://github.com/modelscope/FunClip
            <a href="/en/donors.html">Thanks</a>
        """,
        "https://www.funasr.com/blog/funasr-v1-3-28-realtime-websocket-subtitles.html": """
            funasr==1.3.28
            funasr-realtime-server
            VAD STOP SenseVoice
            精确合并源码通过 118 项聚焦回归测试
            实时 WebSocket 文件还独立通过 60 项测试
            覆盖 STOP 最终解码
            SenseVoice 用户通过同一次包升级获得字幕对齐修复
            runtime-llamacpp-v0.1.9
            https://github.com/modelscope/FunASR/releases/tag/v1.3.28
            <a href="https://github.com/modelscope/FunASR">FunASR</a>
            <a href="https://github.com/FunAudioLLM/Fun-ASR">Fun-ASR</a>
            <a href="https://github.com/FunAudioLLM/SenseVoice">SenseVoice</a>
            <a href="https://github.com/modelscope/FunClip">FunClip</a>
            <a href="/donors.html">功德榜</a>
        """,
        "https://www.funasr.com/en/blog/funasr-v1-3-28-realtime-websocket-subtitles.html": """
            funasr==1.3.28
            funasr-realtime-server
            VAD STOP SenseVoice
            The exact merged source passed 118 focused regression tests.
            The realtime WebSocket file also passed all 60 tests independently
            including STOP final decode
            SenseVoice users receive the subtitle alignment fix
            runtime-llamacpp-v0.1.9
            https://github.com/modelscope/FunASR/releases/tag/v1.3.28
            <a href="https://github.com/modelscope/FunASR">FunASR</a>
            <a href="https://github.com/FunAudioLLM/Fun-ASR">Fun-ASR</a>
            <a href="https://github.com/FunAudioLLM/SenseVoice">SenseVoice</a>
            <a href="https://github.com/modelscope/FunClip">FunClip</a>
            <a href="/en/donors.html">Thanks</a>
        """,
    }

    assert checker.validate_pages(pages) == []


def test_website_contract_includes_homepage_entrypoints():
    checker = _load_module()

    assert "https://www.funasr.com/" in checker.PAGE_CONTRACTS
    assert "https://www.funasr.com/en/" in checker.PAGE_CONTRACTS
    assert "https://www.funasr.com/llama-cpp.html" in checker.PAGE_CONTRACTS
    assert "https://www.funasr.com/en/llama-cpp.html" in checker.PAGE_CONTRACTS


def test_ecosystem_contract_requires_live_dify_marketplace():
    checker = _load_module()
    url = "https://www.funasr.com/en/ecosystem.html"
    pages = {
        url: """
            <body>
            <div class="stat-num">35K+</div>
            <a href="/en/donors.html">Thanks</a>
            <a href="https://github.com/BerriAI/litellm">LiteLLM</a>
            <div>custom_openai</div>
            <div>54.3K stars</div>
            <a href="https://github.com/langgenius/dify">Dify</a>
            </body>
        """
    }

    failures = checker.validate_pages(pages)

    assert any(
        url in failure
        and "missing link `https://marketplace.dify.ai/plugin/langgenius/funasr`"
        in failure
        for failure in failures
    )
    assert any(
        url in failure and "visible text missing `FunASR plugin 0.1.1`" in failure
        for failure in failures
    )
    assert any(
        url in failure and "visible text missing `25 MB uploads`" in failure
        for failure in failures
    )


def _valid_english_ecosystem_html():
    return """
        <body>
        <div class="stat-num">35K+</div>
        <a href="/en/donors.html">Thanks</a>
        <a href="https://github.com/BerriAI/litellm">LiteLLM</a>
        <div>custom_openai</div>
        <div>54.3K stars</div>
        <a href="https://marketplace.dify.ai/plugin/langgenius/funasr">
            FunASR plugin 0.1.1
        </a>
        <div>supports 25 MB uploads</div>
        </body>
    """


def test_ecosystem_contract_rejects_longer_dify_version():
    checker = _load_module()
    url = "https://www.funasr.com/en/ecosystem.html"
    pages = {url: _valid_english_ecosystem_html()}
    pages[url] = pages[url].replace(
        "FunASR plugin 0.1.1", "FunASR plugin 0.1.10"
    )

    failures = checker.validate_pages(pages)

    assert any(url in failure and "FunASR plugin 0.1.1" in failure for failure in failures)


def test_ecosystem_contract_rejects_dify_version_suffixes():
    checker = _load_module()
    url = "https://www.funasr.com/en/ecosystem.html"

    for version in ("0.1.1-beta", "0.1.1+build"):
        pages = {url: _valid_english_ecosystem_html()}
        pages[url] = pages[url].replace("0.1.1", version)

        failures = checker.validate_pages(pages)

        assert any(
            url in failure and "FunASR plugin 0.1.1" in failure
            for failure in failures
        )


def test_ecosystem_contract_rejects_larger_dify_upload_limit():
    checker = _load_module()
    url = "https://www.funasr.com/en/ecosystem.html"
    pages = {url: _valid_english_ecosystem_html()}
    pages[url] = pages[url].replace("25 MB uploads", "125 MB uploads")

    failures = checker.validate_pages(pages)

    assert any(url in failure and "25 MB uploads" in failure for failure in failures)


def test_website_contract_includes_v1326_launch_articles():
    checker = _load_module()

    assert (
        "https://www.funasr.com/blog/funasr-v1-3-26-openai-vllm-llama-cpp.html"
        in checker.PAGE_CONTRACTS
    )
    assert (
        "https://www.funasr.com/en/blog/funasr-v1-3-26-openai-vllm-llama-cpp.html"
        in checker.PAGE_CONTRACTS
    )


def test_website_contract_includes_v1327_launch_articles():
    checker = _load_module()

    assert (
        "https://www.funasr.com/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html"
        in checker.PAGE_CONTRACTS
    )
    assert (
        "https://www.funasr.com/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html"
        in checker.PAGE_CONTRACTS
    )


def test_website_contract_includes_v1328_launch_articles():
    checker = _load_module()

    assert (
        "https://www.funasr.com/blog/funasr-v1-3-28-realtime-websocket-subtitles.html"
        in checker.PAGE_CONTRACTS
    )
    assert (
        "https://www.funasr.com/en/blog/funasr-v1-3-28-realtime-websocket-subtitles.html"
        in checker.PAGE_CONTRACTS
    )


def test_v1328_contract_rejects_language_swapped_body():
    checker = _load_module()
    url = "https://www.funasr.com/blog/funasr-v1-3-28-realtime-websocket-subtitles.html"
    pages = {
        url: """
            <body>
            funasr==1.3.28 funasr-realtime-server VAD STOP SenseVoice
            The exact merged source passed 118 focused regression tests.
            The realtime WebSocket file also passed all 60 tests independently.
            runtime-llamacpp-v0.1.9
            https://github.com/modelscope/FunASR/releases/tag/v1.3.28
            <a href="https://github.com/modelscope/FunASR">FunASR</a>
            <a href="https://github.com/FunAudioLLM/Fun-ASR">Fun-ASR</a>
            <a href="https://github.com/FunAudioLLM/SenseVoice">SenseVoice</a>
            <a href="https://github.com/modelscope/FunClip">FunClip</a>
            <a href="/donors.html">Thanks</a>
            </body>
        """
    }

    failures = checker.validate_pages(pages)

    assert any(
        url in failure
        and "visible text missing `精确合并源码通过 118 项聚焦回归测试`" in failure
        for failure in failures
    )


def test_v1328_contract_rejects_css_test_count_and_indirect_repo_link():
    checker = _load_module()
    url = "https://www.funasr.com/en/blog/funasr-v1-3-28-realtime-websocket-subtitles.html"
    pages = {
        url: """
            <style>.proof { font-weight: 600; }</style>
            <body>
            funasr==1.3.28 funasr-realtime-server VAD STOP SenseVoice
            The exact merged source passed 118 focused regression tests.
            runtime-llamacpp-v0.1.9
            <a href="https://github.com/modelscope/FunASR/releases/tag/v1.3.28">Release</a>
            <a href="https://github.com/FunAudioLLM/Fun-ASR">Fun-ASR</a>
            <a href="https://github.com/FunAudioLLM/SenseVoice">SenseVoice</a>
            <a href="https://github.com/modelscope/FunClip">FunClip</a>
            <a href="/en/donors.html">Thanks</a>
            </body>
        """
    }

    failures = checker.validate_pages(pages)

    assert any(
        url in failure
        and "visible text missing `The realtime WebSocket file also passed all 60 tests independently`"
        in failure
        for failure in failures
    )
    assert any(
        url in failure
        and "missing link `https://github.com/modelscope/FunASR`" in failure
        for failure in failures
    )


def test_v1328_contract_requires_visible_stop_and_sensevoice_copy():
    checker = _load_module()
    url = "https://www.funasr.com/en/blog/funasr-v1-3-28-realtime-websocket-subtitles.html"
    pages = {
        url: """
            <head><meta name="keywords" content="STOP"></head>
            <body>
            funasr==1.3.28 funasr-realtime-server VAD
            The exact merged source passed 118 focused regression tests.
            The realtime WebSocket file also passed all 60 tests independently.
            runtime-llamacpp-v0.1.9
            https://github.com/modelscope/FunASR/releases/tag/v1.3.28
            <a href="https://github.com/modelscope/FunASR">FunASR</a>
            <a href="https://github.com/FunAudioLLM/Fun-ASR">Fun-ASR</a>
            <a href="https://github.com/FunAudioLLM/SenseVoice">Repository</a>
            <a href="https://github.com/modelscope/FunClip">FunClip</a>
            <a href="/en/donors.html">Thanks</a>
            </body>
        """
    }

    failures = checker.validate_pages(pages)

    assert any(
        url in failure and "visible text missing `including STOP final decode`" in failure
        for failure in failures
    )
    assert any(
        url in failure
        and "visible text missing `SenseVoice users receive the subtitle alignment fix`"
        in failure
        for failure in failures
    )


def test_static_asset_contract_rejects_non_png_payloads():
    checker = _load_module()
    banner_url = "https://www.funasr.com/img/banner.4f436d19.png"
    logo_url = "https://www.funasr.com/logo.png"

    assert banner_url in checker.STATIC_ASSET_CONTRACTS
    assert logo_url in checker.STATIC_ASSET_CONTRACTS

    failures = checker.validate_assets(
        {
            banner_url: b"<html>not an image</html>",
            logo_url: b"\x89PNG\r\n\x1a\n" + b"x" * 200,
        }
    )

    assert failures == [f"{banner_url}: response is not a valid PNG"]


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
            <meta name="description" content="捐赠资金用于社区基础设施建设，包括购买和维护服务器，以及购买、续费和维护 www.funasr.com 域名。">
        </head>
        <body>FunASR 社区功德榜</body>
    """
    pages["https://www.funasr.com/en/donors.html"] = """
        <head>
            <meta name="description" content="Donations fund community infrastructure, including server purchase and maintenance, plus the purchase, renewal, and maintenance of the www.funasr.com domain.">
        </head>
        <body>FunASR Community Thanks</body>
    """

    failures = checker.validate_pages(pages)

    assert any(
        "donors.html" in failure
        and "visible text missing `购买和维护服务器`" in failure
        for failure in failures
    )
    assert any(
        "en/donors.html" in failure
        and "visible text missing `server purchase and maintenance`" in failure
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
