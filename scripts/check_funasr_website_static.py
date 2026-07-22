#!/usr/bin/env python3
"""Check public funasr.com pages for growth-critical copy regressions."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from dataclasses import dataclass


BASE_URL = "https://www.funasr.com"


@dataclass(frozen=True)
class PageContract:
    required: tuple[str, ...]
    forbidden: tuple[str, ...] = ()


PAGE_CONTRACTS: dict[str, PageContract] = {
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
        required=("服务器", "www.funasr.com", "域名"),
    ),
    f"{BASE_URL}/en/donors.html": PageContract(
        required=("servers", "www.funasr.com", "domain"),
    ),
    f"{BASE_URL}/blog/funasr-cli-transcribe-command-line.html": PageContract(
        required=("推荐 funasr ≥ 1.3.26", "/donors.html"),
        forbidden=("1.3.10",),
    ),
    f"{BASE_URL}/en/blog/funasr-cli-transcribe-command-line.html": PageContract(
        required=("recommended funasr &gt;= 1.3.26", "/en/donors.html"),
        forbidden=("1.3.10",),
    ),
    f"{BASE_URL}/blog/funasr-llama-cpp-whisper-cpp-alternative.html": PageContract(
        required=(
            "runtime-llamacpp-v0.1.8",
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "Vulkan / CUDA",
            "/donors.html",
        ),
        forbidden=("runtime-llamacpp-v0.1.1",),
    ),
    f"{BASE_URL}/en/blog/funasr-llama-cpp-whisper-cpp-alternative.html": PageContract(
        required=(
            "runtime-llamacpp-v0.1.8",
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "Vulkan / CUDA",
            "/en/donors.html",
        ),
        forbidden=("runtime-llamacpp-v0.1.1",),
    ),
}


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
        for needle in contract.forbidden:
            if needle in text:
                failures.append(f"{url}: forbidden `{needle}`")
    return failures


def fetch_pages(timeout: float) -> dict[str, str]:
    pages: dict[str, str] = {}
    for url in PAGE_CONTRACTS:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read()
        pages[url] = body.decode("utf-8", errors="replace")
    return pages


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)

    failures = validate_pages(fetch_pages(timeout=args.timeout))
    if failures:
        print("funasr.com static page contract failed:", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1

    print(f"funasr.com static page contract passed for {len(PAGE_CONTRACTS)} pages")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
