#!/usr/bin/env python3
"""Check public funasr.com pages for growth-critical copy regressions."""

from __future__ import annotations

import argparse
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
            "https://github.com/FunAudioLLM/Fun-ASR",
            "https://github.com/FunAudioLLM/SenseVoice",
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
            "https://github.com/FunAudioLLM/Fun-ASR",
            "https://github.com/FunAudioLLM/SenseVoice",
            "https://github.com/modelscope/FunClip",
            "/en/donors.html",
        ),
        forbidden=("funasr==1.3.10", "runtime-llamacpp-v0.1.1"),
    ),
}


def extract_visible_text(html: str) -> str:
    body = html.split("<body", 1)[-1]
    body = re.sub(
        r"<script[\s\S]*?</script>|<style[\s\S]*?</style>",
        " ",
        body,
        flags=re.IGNORECASE,
    )
    body = re.sub(r"<[^>]+>", " ", body)
    return re.sub(r"\s+", " ", body).strip()


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
        if contract.visible_required:
            visible_text = extract_visible_text(text)
            for needle in contract.visible_required:
                if needle not in visible_text:
                    failures.append(f"{url}: visible text missing `{needle}`")
        for needle in contract.forbidden:
            if needle in text:
                failures.append(f"{url}: forbidden `{needle}`")
    return failures


def _fetch_url(url: str, timeout: float, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                body = response.read()
            return body.decode("utf-8", errors="replace")
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(min(0.25 * (attempt + 1), 1.0))
    raise last_error


def fetch_pages(timeout: float, retries: int = 3) -> dict[str, str]:
    pages: dict[str, str] = {}
    for url in PAGE_CONTRACTS:
        pages[url] = _fetch_url(url, timeout=timeout, retries=retries)
    return pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    args = parser.parse_args(argv)

    failures = validate_pages(fetch_pages(timeout=args.timeout, retries=args.retries))
    if failures:
        print("funasr.com static page contract failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"funasr.com static page contract passed for {len(PAGE_CONTRACTS)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
