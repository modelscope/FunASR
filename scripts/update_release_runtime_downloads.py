#!/usr/bin/env python3
"""Add prebuilt runtime download links to a Python package GitHub release."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


RUNTIME_SECTION_HEADING = "## Runtime downloads"


def run_gh_json(args: list[str]) -> Any:
    completed = subprocess.run(
        ["gh", *args],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return json.loads(completed.stdout)


def run_gh(args: list[str]) -> None:
    subprocess.run(["gh", *args], check=True)


def platform_label(asset_name: str) -> str:
    if "linux-arm64" in asset_name:
        return "Linux arm64"
    if "linux-x64-vulkan" in asset_name:
        return "Linux x64 Vulkan"
    if "linux-x64-avx2" in asset_name:
        return "Linux x64 AVX2"
    if "linux-x64" in asset_name:
        return "Linux x64 portable"
    if "macos-arm64" in asset_name:
        return "macOS arm64"
    if "windows-x64-vulkan" in asset_name:
        return "Windows x64 Vulkan"
    if "windows-x64-cuda" in asset_name:
        return "Windows x64 CUDA"
    if "windows-x64-avx2" in asset_name:
        return "Windows x64 AVX2"
    if "windows-x64" in asset_name:
        return "Windows x64 portable"
    return asset_name


def latest_runtime_tag(repository: str) -> str:
    releases = run_gh_json(
        [
            "release",
            "list",
            "--repo",
            repository,
            "--limit",
            "50",
            "--json",
            "tagName,publishedAt,isDraft,isPrerelease",
        ]
    )
    for release in releases:
        if release["tagName"].startswith("runtime-llamacpp-v") and not release.get(
            "isDraft"
        ):
            return release["tagName"]
    raise RuntimeError("No published runtime-llamacpp release found")


def load_release(repository: str, tag: str) -> dict[str, Any]:
    return run_gh_json(
        [
            "release",
            "view",
            tag,
            "--repo",
            repository,
            "--json",
            "tagName,name,body,url,assets",
        ]
    )


def build_runtime_downloads_section(
    *, python_tag: str, runtime_release: dict[str, Any]
) -> str:
    assets = sorted(runtime_release.get("assets") or [], key=lambda asset: asset["name"])
    runtime_assets = [
        asset for asset in assets if asset["name"].startswith("funasr-llamacpp-")
    ]
    if not runtime_assets:
        raise RuntimeError(f"{runtime_release['tagName']} has no llama.cpp assets")

    python_version = python_tag.removeprefix("v")
    lines = [
        RUNTIME_SECTION_HEADING,
        "",
        (
            "This Python release pairs with the current prebuilt llama.cpp / GGUF "
            f"runtime release: [{runtime_release['tagName']}]({runtime_release['url']})."
        ),
        "",
        (
            "Use these assets when you want the self-contained `llama-funasr-*` "
            "binaries instead of the Python package. The runtime release is shared "
            f"by the current Python releases; `{python_tag}` itself is the PyPI package release."
        ),
        "",
        "| Platform | Asset | SHA-256 |",
        "|---|---|---|",
    ]
    for asset in runtime_assets:
        digest = (asset.get("digest") or "").removeprefix("sha256:")
        lines.append(
            f"| {platform_label(asset['name'])} | "
            f"[{asset['name']}]({asset['url']}) | "
            f"`{digest}` |"
        )

    lines.extend(
        [
            "",
            (
                "Quick start: download one asset, unpack it, then run the bundled "
                "`download-funasr-model.sh <sensevoice|paraformer|nano>` helper and "
                "one of `llama-funasr-cli`, `llama-funasr-sensevoice`, or "
                "`llama-funasr-paraformer`."
            ),
            "",
            "For Python users, install from PyPI:",
            "",
            "```bash",
            f'python -m pip install -U "funasr=={python_version}"',
            "```",
        ]
    )
    return "\n".join(lines)


def merge_release_body(
    *, current_body: str, python_tag: str, runtime_release: dict[str, Any]
) -> str:
    body_without_runtime = current_body.split(RUNTIME_SECTION_HEADING, maxsplit=1)[
        0
    ].rstrip()
    runtime_section = build_runtime_downloads_section(
        python_tag=python_tag, runtime_release=runtime_release
    )
    return f"{body_without_runtime}\n\n{runtime_section}\n"


def update_release_body(repository: str, python_tag: str, runtime_tag: str | None) -> None:
    runtime_tag = runtime_tag or latest_runtime_tag(repository)
    python_release = load_release(repository, python_tag)
    runtime_release = load_release(repository, runtime_tag)
    merged = merge_release_body(
        current_body=python_release.get("body") or "",
        python_tag=python_tag,
        runtime_release=runtime_release,
    )

    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as handle:
        handle.write(merged)
        notes_path = Path(handle.name)
    try:
        run_gh(
            [
                "release",
                "edit",
                python_tag,
                "--repo",
                repository,
                "--notes-file",
                str(notes_path),
            ]
        )
    finally:
        notes_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("python_tag", help="Python package release tag, for example v1.3.26")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "modelscope/FunASR"),
        help="GitHub repository in owner/name form",
    )
    parser.add_argument(
        "--runtime-tag",
        default=None,
        help="Runtime release tag. Defaults to the latest runtime-llamacpp-v* release.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_release_body(args.repo, args.python_tag, args.runtime_tag)


if __name__ == "__main__":
    main()
