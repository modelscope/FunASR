#!/usr/bin/env python3
"""Check public funasr.com pages for growth-critical copy regressions."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import re
import sys
import time
import urllib.request
from dataclasses import dataclass


BASE_URL = "https://www.funasr.com"


@dataclass(frozen=True)
class PageContract:
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()
    visible_required: tuple[str, ...] = ()
    visible_patterns: tuple[tuple[str, str], ...] = ()
    required_links: tuple[str, ...] = ()
    required_images: tuple[str, ...] = ()


@dataclass(frozen=True)
class StaticAssetContract:
    signature: bytes
    media_type: str
    min_bytes: int = 128


PAGE_CONTRACTS: dict[str, PageContract] = {
    f"{BASE_URL}/": PageContract(
        required=(
            "工业级",
            "/v1/audio/transcriptions",
            "vLLM",
            "/donors.html",
        ),
    ),
    f"{BASE_URL}/en/": PageContract(
        required=(
            "Industrial Speech Recognition",
            "OpenAI-compatible",
            "/v1/audio/transcriptions",
            "vLLM",
        ),
    ),
    f"{BASE_URL}/ecosystem.html": PageContract(
        required=(
            "35K+",
            "/donors.html",
            "LiteLLM",
            "custom_openai",
            "54.3K stars",
        ),
        visible_patterns=(
            ("FunASR 官方插件 0.1.1", r"FunASR 官方插件 0\.1\.1(?![\w.+-])"),
            ("最大 25 MB 音频上传", r"最大 (?<!\d)25 MB 音频上传"),
        ),
        required_links=("https://marketplace.dify.ai/plugin/langgenius/funasr",),
        forbidden=("16K+",),
    ),
    f"{BASE_URL}/en/ecosystem.html": PageContract(
        required=(
            "35K+",
            "/en/donors.html",
            "LiteLLM",
            "custom_openai",
            "54.3K stars",
        ),
        visible_patterns=(
            ("FunASR plugin 0.1.1", r"FunASR plugin 0\.1\.1(?![\w.+-])"),
            ("25 MB uploads", r"(?<!\d)25 MB uploads"),
        ),
        required_links=("https://marketplace.dify.ai/plugin/langgenius/funasr",),
        forbidden=("16K+",),
    ),
    f"{BASE_URL}/donors.html": PageContract(
        required=(
            "捐赠资金用于社区基础设施建设",
            "购买和维护服务器",
            "购买、续费和维护",
            "www.funasr.com",
            "域名",
        ),
        visible_required=(
            "捐赠资金用于社区基础设施建设",
            "购买和维护服务器",
            "购买、续费和维护",
            "www.funasr.com",
            "域名",
        ),
    ),
    f"{BASE_URL}/en/donors.html": PageContract(
        required=(
            "Donations fund community infrastructure",
            "server purchase and maintenance",
            "purchase, renewal, and maintenance",
            "www.funasr.com",
            "domain",
        ),
        visible_required=(
            "Donations fund community infrastructure",
            "server purchase and maintenance",
            "purchase, renewal, and maintenance",
            "www.funasr.com",
            "domain",
        ),
    ),
    f"{BASE_URL}/blog/funasr-cli-transcribe-command-line.html": PageContract(
        required=("推荐 funasr ≥ 1.3.26", "/donors.html"),
        forbidden=("1.3.10",),
    ),
    f"{BASE_URL}/en/blog/funasr-cli-transcribe-command-line.html": PageContract(
        required=("recommended funasr &gt;= 1.3.26", "/en/donors.html"),
        forbidden=("1.3.10",),
    ),
    f"{BASE_URL}/llama-cpp.html": PageContract(
        required=(
            "runtime-llamacpp-v0.1.9",
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "funasr-llamacpp-windows-x64-cuda.zip",
            "Fun-ASR-Nano",
            "GGUF",
        ),
        forbidden=("runtime-llamacpp-v0.1.1",),
    ),
    f"{BASE_URL}/en/llama-cpp.html": PageContract(
        required=(
            "runtime-llamacpp-v0.1.9",
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "funasr-llamacpp-windows-x64-cuda.zip",
            "Fun-ASR-Nano",
            "GGUF",
        ),
        forbidden=("runtime-llamacpp-v0.1.1",),
    ),
    f"{BASE_URL}/blog/funasr-llama-cpp-whisper-cpp-alternative.html": PageContract(
        required=(
            "runtime-llamacpp-v0.1.9",
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "Fun-ASR-Nano-GGUF",
            "Linux/Windows Vulkan 与 Windows CUDA",
            "/donors.html",
        ),
        forbidden=("runtime-llamacpp-v0.1.1",),
    ),
    f"{BASE_URL}/en/blog/funasr-llama-cpp-whisper-cpp-alternative.html": PageContract(
        required=(
            "runtime-llamacpp-v0.1.9",
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "Fun-ASR-Nano-GGUF",
            "Linux/Windows Vulkan and Windows CUDA",
            "/en/donors.html",
        ),
        forbidden=("runtime-llamacpp-v0.1.1",),
    ),
    f"{BASE_URL}/blog/funasr-v1-3-26-openai-vllm-llama-cpp.html": PageContract(
        required=(
            "funasr==1.3.26",
            "/v1/audio/transcriptions",
            "RTFx 340",
            "runtime-llamacpp-v0.1.9",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "https://github.com/modelscope/FunASR",
            "https://github.com/QwenAudio/Fun-ASR",
            "https://github.com/QwenAudio/SenseVoice",
            "https://github.com/modelscope/FunClip",
            "/donors.html",
        ),
        forbidden=("funasr==1.3.10", "runtime-llamacpp-v0.1.1"),
    ),
    f"{BASE_URL}/en/blog/funasr-v1-3-26-openai-vllm-llama-cpp.html": PageContract(
        required=(
            "funasr==1.3.26",
            "/v1/audio/transcriptions",
            "RTFx 340",
            "runtime-llamacpp-v0.1.9",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "https://github.com/modelscope/FunASR",
            "https://github.com/QwenAudio/Fun-ASR",
            "https://github.com/QwenAudio/SenseVoice",
            "https://github.com/modelscope/FunClip",
            "/en/donors.html",
        ),
        forbidden=("funasr==1.3.10", "runtime-llamacpp-v0.1.1"),
    ),
    f"{BASE_URL}/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html": PageContract(
        required=(
            "funasr==1.3.27",
            "/v1/audio/transcriptions",
            "verbose_json.language",
            'language":"en',
            "AutoModel",
            "runtime-llamacpp-v0.1.9",
            "https://github.com/modelscope/FunASR/releases/tag/v1.3.27",
            "https://github.com/modelscope/FunASR",
            "https://github.com/QwenAudio/Fun-ASR",
            "https://github.com/QwenAudio/SenseVoice",
            "https://github.com/modelscope/FunClip",
            "/donors.html",
        ),
        forbidden=("funasr==1.3.26", "runtime-llamacpp-v0.1.1"),
    ),
    f"{BASE_URL}/en/blog/funasr-v1-3-27-language-metadata-vllm-fallback.html": PageContract(
        required=(
            "funasr==1.3.27",
            "/v1/audio/transcriptions",
            "verbose_json.language",
            'language":"en',
            "AutoModel",
            "runtime-llamacpp-v0.1.9",
            "https://github.com/modelscope/FunASR/releases/tag/v1.3.27",
            "https://github.com/modelscope/FunASR",
            "https://github.com/QwenAudio/Fun-ASR",
            "https://github.com/QwenAudio/SenseVoice",
            "https://github.com/modelscope/FunClip",
            "/en/donors.html",
        ),
        forbidden=("funasr==1.3.26", "runtime-llamacpp-v0.1.1"),
    ),
    f"{BASE_URL}/blog/funasr-v1-3-28-realtime-websocket-subtitles.html": PageContract(
        required=(
            "funasr==1.3.28",
            "funasr-realtime-server",
            "VAD",
            "runtime-llamacpp-v0.1.9",
            "https://github.com/modelscope/FunASR/releases/tag/v1.3.28",
            "/donors.html",
        ),
        visible_required=(
            "精确合并源码通过 118 项聚焦回归测试",
            "实时 WebSocket 文件还独立通过 60 项测试",
            "覆盖 STOP 最终解码",
            "SenseVoice 用户通过同一次包升级获得字幕对齐修复",
        ),
        required_links=(
            "https://github.com/modelscope/FunASR",
            "https://github.com/QwenAudio/Fun-ASR",
            "https://github.com/QwenAudio/SenseVoice",
            "https://github.com/modelscope/FunClip",
        ),
        forbidden=("funasr==1.3.27", "runtime-llamacpp-v0.1.1"),
    ),
    f"{BASE_URL}/en/blog/funasr-v1-3-28-realtime-websocket-subtitles.html": PageContract(
        required=(
            "funasr==1.3.28",
            "funasr-realtime-server",
            "VAD",
            "runtime-llamacpp-v0.1.9",
            "https://github.com/modelscope/FunASR/releases/tag/v1.3.28",
            "/en/donors.html",
        ),
        visible_required=(
            "The exact merged source passed 118 focused regression tests",
            "The realtime WebSocket file also passed all 60 tests independently",
            "including STOP final decode",
            "SenseVoice users receive the subtitle alignment fix",
        ),
        required_links=(
            "https://github.com/modelscope/FunASR",
            "https://github.com/QwenAudio/Fun-ASR",
            "https://github.com/QwenAudio/SenseVoice",
            "https://github.com/modelscope/FunClip",
        ),
        forbidden=("funasr==1.3.27", "runtime-llamacpp-v0.1.1"),
    ),
    f"{BASE_URL}/blog/funclip-v2-1-0-video-clipping-release.html": PageContract(
        required=(
            "FunClip v2.1.0",
            "FunClip-2.1.0.tar.gz",
            "FunClip-2.1.0.zip",
            "SHA256SUMS",
            "funasr>=1.3.29",
            "TwelveLabs Pegasus",
        ),
        visible_required=(
            "FunClip v2.1.0",
            "FunClip-2.1.0.tar.gz",
            "FunClip-2.1.0.zip",
            "SHA256SUMS",
            "funasr>=1.3.29",
            "TwelveLabs Pegasus",
            "为什么发布的是源码归档，而不是 wheel",
            "v2.1.0 的精确合并源码通过 50 项测试，另有 1 项跳过",
            "干净 Python 3.12 环境中的 6 项发布契约测试",
            "GitHub Actions 首次发布和主动重跑均成功",
            "Star FunClip 仓库",
        ),
        required_links=(
            "https://github.com/modelscope/FunClip",
            "https://github.com/modelscope/FunClip/releases/tag/v2.1.0",
            "https://github.com/modelscope/FunClip/releases/download/v2.1.0/FunClip-2.1.0.tar.gz",
            "https://github.com/modelscope/FunClip/releases/download/v2.1.0/FunClip-2.1.0.zip",
            "https://github.com/modelscope/FunClip/releases/download/v2.1.0/SHA256SUMS",
            "https://huggingface.co/spaces/FunAudioLLM/FunClip",
            "https://github.com/modelscope/FunASR/releases/tag/v1.3.29",
            "/donors.html",
        ),
        required_images=("/img/funclip-v2-1-0-interface.jpg",),
        forbidden=("funasr>=1.3.28",),
    ),
    f"{BASE_URL}/en/blog/funclip-v2-1-0-video-clipping-release.html": PageContract(
        required=(
            "FunClip v2.1.0",
            "FunClip-2.1.0.tar.gz",
            "FunClip-2.1.0.zip",
            "SHA256SUMS",
            "funasr>=1.3.29",
            "TwelveLabs Pegasus",
        ),
        visible_required=(
            "FunClip v2.1.0",
            "FunClip-2.1.0.tar.gz",
            "FunClip-2.1.0.zip",
            "SHA256SUMS",
            "funasr>=1.3.29",
            "TwelveLabs Pegasus",
            "Why these are source archives, not a wheel",
            "The exact v2.1.0 merge passed 50 tests with 1 skipped",
            "Six release-contract tests also passed in a clean Python 3.12 environment",
            "Both the initial GitHub Actions release and a deliberate rerun succeeded",
            "star the FunClip repository",
        ),
        required_links=(
            "https://github.com/modelscope/FunClip",
            "https://github.com/modelscope/FunClip/releases/tag/v2.1.0",
            "https://github.com/modelscope/FunClip/releases/download/v2.1.0/FunClip-2.1.0.tar.gz",
            "https://github.com/modelscope/FunClip/releases/download/v2.1.0/FunClip-2.1.0.zip",
            "https://github.com/modelscope/FunClip/releases/download/v2.1.0/SHA256SUMS",
            "https://huggingface.co/spaces/FunAudioLLM/FunClip",
            "https://github.com/modelscope/FunASR/releases/tag/v1.3.29",
            "/en/donors.html",
        ),
        required_images=("/img/funclip-v2-1-0-interface.jpg",),
        forbidden=("funasr>=1.3.28",),
    ),
}


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
JPEG_SIGNATURE = b"\xff\xd8\xff"

STATIC_ASSET_CONTRACTS: dict[str, StaticAssetContract] = {
    f"{BASE_URL}/img/banner.4f436d19.png": StaticAssetContract(
        PNG_SIGNATURE,
        "PNG",
    ),
    f"{BASE_URL}/logo.png": StaticAssetContract(PNG_SIGNATURE, "PNG"),
    f"{BASE_URL}/img/funclip-v2-1-0-interface.jpg": StaticAssetContract(
        JPEG_SIGNATURE,
        "JPEG",
        min_bytes=100_000,
    ),
}


_HIDDEN_TAGS = frozenset(("head", "script", "style", "template", "noscript"))
_VOID_TAGS = frozenset(
    (
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "source",
        "track",
        "wbr",
    )
)


@dataclass
class _ElementState:
    tag: str
    hidden: bool
    link: str | None = None
    has_visible_content: bool = False


class _VisibleContentCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.text: list[str] = []
        self.links: set[str] = set()
        self.images: set[str] = set()
        self._stack: list[_ElementState] = []

    @property
    def _parent_hidden(self) -> bool:
        return bool(self._stack and self._stack[-1].hidden)

    def _mark_link_content(self) -> None:
        for state in reversed(self._stack):
            if state.link is not None:
                state.has_visible_content = True
                return

    @staticmethod
    def _has_hidden_attribute(attrs: dict[str, str | None]) -> bool:
        style = re.sub(r"\s+", "", attrs.get("style") or "").lower()
        return (
            "hidden" in attrs
            or (attrs.get("aria-hidden") or "").lower() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
            or "opacity:0" in style
        )

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value for name, value in attrs}
        hidden = (
            self._parent_hidden
            or tag in _HIDDEN_TAGS
            or self._has_hidden_attribute(attributes)
        )
        if not hidden:
            if tag == "img" and attributes.get("src"):
                self.images.add(attributes["src"])
                self._mark_link_content()
        if tag not in _VOID_TAGS:
            link = attributes.get("href") if tag == "a" and not hidden else None
            self._stack.append(_ElementState(tag=tag, hidden=hidden, link=link))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        attributes = {name.lower(): value for name, value in attrs}
        hidden = (
            self._parent_hidden
            or tag in _HIDDEN_TAGS
            or self._has_hidden_attribute(attributes)
        )
        if not hidden and tag == "img" and attributes.get("src"):
            self.images.add(attributes["src"])
            self._mark_link_content()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        for index in range(len(self._stack) - 1, -1, -1):
            if self._stack[index].tag == tag:
                state = self._stack[index]
                if state.link is not None and state.has_visible_content:
                    self.links.add(state.link)
                del self._stack[index:]
                return

    def handle_data(self, data: str) -> None:
        if not self._parent_hidden:
            self.text.append(data)
            if data.strip():
                self._mark_link_content()


def _collect_visible_content(html: str) -> _VisibleContentCollector:
    parser = _VisibleContentCollector()
    parser.feed(html)
    parser.close()
    return parser


def extract_visible_text(html: str) -> str:
    parser = _collect_visible_content(html)
    return re.sub(r"\s+", " ", " ".join(parser.text)).strip()


def extract_links(html: str) -> set[str]:
    return _collect_visible_content(html).links


def extract_images(html: str) -> set[str]:
    return _collect_visible_content(html).images


def validate_pages(pages: dict[str, str]) -> list[str]:
    failures: list[str] = []
    for url, contract in PAGE_CONTRACTS.items():
        text = pages.get(url)
        if text is None:
            failures.append(f"{url}: page was not fetched")
            continue
        for needle in contract.required:
            if needle not in text:
                failures.append(f"{url}: missing `{needle}`")
        visible_text = ""
        if contract.visible_required or contract.visible_patterns:
            visible_text = extract_visible_text(text)
        if contract.visible_required:
            for needle in contract.visible_required:
                if needle not in visible_text:
                    failures.append(f"{url}: visible text missing `{needle}`")
        if contract.visible_patterns:
            for label, pattern in contract.visible_patterns:
                if re.search(pattern, visible_text) is None:
                    failures.append(f"{url}: visible text missing `{label}`")
        if contract.required_links:
            links = extract_links(text)
            for target in contract.required_links:
                if target not in links:
                    failures.append(f"{url}: missing link `{target}`")
        if contract.required_images:
            images = extract_images(text)
            for target in contract.required_images:
                if target not in images:
                    failures.append(f"{url}: missing image `{target}`")
        for needle in contract.forbidden:
            if needle in text:
                failures.append(f"{url}: forbidden `{needle}`")
    return failures


def validate_assets(assets: dict[str, bytes]) -> list[str]:
    failures: list[str] = []
    for url, contract in STATIC_ASSET_CONTRACTS.items():
        body = assets.get(url)
        if body is None:
            failures.append(f"{url}: asset was not fetched")
        elif not body.startswith(contract.signature) or len(body) < contract.min_bytes:
            failures.append(f"{url}: response is not a valid {contract.media_type}")
    return failures


def _fetch_bytes(url: str, timeout: float, retries: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                return response.read()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(0.25 * (attempt + 1), 1.0))
    raise last_error


def _fetch_url(url: str, timeout: float, retries: int) -> str:
    return _fetch_bytes(url, timeout=timeout, retries=retries).decode(
        "utf-8", errors="replace"
    )


def fetch_pages(timeout: float, retries: int = 3) -> dict[str, str]:
    pages: dict[str, str] = {}
    for url in PAGE_CONTRACTS:
        pages[url] = _fetch_url(url, timeout=timeout, retries=retries)
    return pages


def fetch_assets(timeout: float, retries: int = 3) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    for url in STATIC_ASSET_CONTRACTS:
        assets[url] = _fetch_bytes(url, timeout=timeout, retries=retries)
    return assets


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    failures = validate_pages(fetch_pages(timeout=args.timeout, retries=args.retries))
    failures.extend(
        validate_assets(fetch_assets(timeout=args.timeout, retries=args.retries))
    )
    if failures:
        print("funasr.com static page contract failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(
        "funasr.com static contract passed for "
        f"{len(PAGE_CONTRACTS)} pages and {len(STATIC_ASSET_CONTRACTS)} assets"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
