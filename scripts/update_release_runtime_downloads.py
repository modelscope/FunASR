#!/usr/bin/env python3
"""Attach prebuilt runtimes and their download links to a Python GitHub release."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


RUNTIME_SECTION_HEADING = "## Runtime downloads"
RUNTIME_SECTION_START = "<!-- funasr-runtime-downloads:start -->"
RUNTIME_SECTION_END = "<!-- funasr-runtime-downloads:end -->"


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


def parse_published_at(value: str, *, label: str) -> datetime:
    if not value:
        raise RuntimeError(f"{label} has no publication timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise RuntimeError(
            f"{label} has invalid publication timestamp: {value}"
        ) from error


def latest_runtime_tag(repository: str, *, published_before: str) -> str:
    releases = run_gh_json(
        [
            "release",
            "list",
            "--repo",
            repository,
            "--limit",
            "1000",
            "--json",
            "tagName,publishedAt,isDraft,isPrerelease",
        ]
    )
    cutoff = parse_published_at(published_before, label="Python release")
    eligible = []
    for release in releases:
        if (
            release["tagName"].startswith("runtime-llamacpp-v")
            and not release.get("isDraft")
            and not release.get("isPrerelease")
        ):
            published_at = parse_published_at(
                release.get("publishedAt") or "", label=release["tagName"]
            )
            if published_at <= cutoff:
                eligible.append((published_at, release["tagName"]))
    if eligible:
        return max(eligible)[1]
    raise RuntimeError(
        f"No stable runtime-llamacpp release published by {published_before} found"
    )


def load_release(repository: str, tag: str) -> dict[str, Any]:
    return run_gh_json(
        [
            "release",
            "view",
            tag,
            "--repo",
            repository,
            "--json",
            "tagName,name,body,url,assets,publishedAt,isDraft,isPrerelease",
        ]
    )


def runtime_assets(release: dict[str, Any]) -> list[dict[str, Any]]:
    assets = release.get("assets") or []
    selected = [
        asset for asset in assets if asset["name"].startswith("funasr-llamacpp-")
    ]
    if not selected:
        raise RuntimeError(f"{release['tagName']} has no llama.cpp assets")
    return selected


def validate_runtime_release(
    *, runtime_release: dict[str, Any], python_release: dict[str, Any]
) -> None:
    runtime_tag = runtime_release["tagName"]
    if runtime_release.get("isDraft"):
        raise RuntimeError(f"{runtime_tag} is a draft")
    if runtime_release.get("isPrerelease"):
        raise RuntimeError(f"{runtime_tag} is a prerelease")
    runtime_published_at = parse_published_at(
        runtime_release.get("publishedAt") or "", label=runtime_tag
    )
    python_published_at = parse_published_at(
        python_release.get("publishedAt") or "", label=python_release["tagName"]
    )
    if runtime_published_at > python_published_at:
        raise RuntimeError(
            f"{runtime_tag} was published after {python_release['tagName']}"
        )


def asset_sha256(asset: dict[str, Any]) -> str:
    digest = asset.get("digest")
    if not isinstance(digest, str) or not digest.startswith("sha256:"):
        raise RuntimeError(f"{asset['name']} has no SHA-256 digest")
    sha256 = digest.removeprefix("sha256:")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", sha256):
        raise RuntimeError(f"{asset['name']} has an invalid SHA-256 digest")
    return sha256.lower()


def runtime_assets_to_upload(
    *, python_release: dict[str, Any], runtime_release: dict[str, Any]
) -> list[dict[str, Any]]:
    existing = {asset["name"]: asset for asset in (python_release.get("assets") or [])}
    missing = []
    for source_asset in runtime_assets(runtime_release):
        source_digest = asset_sha256(source_asset)
        target_asset = existing.get(source_asset["name"])
        if target_asset is None:
            missing.append(source_asset)
            continue
        target_digest = asset_sha256(target_asset)
        if target_digest != source_digest:
            raise RuntimeError(
                f"{python_release['tagName']} already has {source_asset['name']} "
                "with a different digest"
            )
    return missing


def verify_asset_digest(asset_path: Path, asset: dict[str, Any]) -> None:
    expected = asset_sha256(asset)
    with asset_path.open("rb") as handle:
        actual = hashlib.file_digest(handle, "sha256").hexdigest()
    if actual != expected:
        raise RuntimeError(
            f"SHA-256 mismatch for {asset['name']}: expected {expected}, got {actual}"
        )


def sync_runtime_assets(
    *,
    repository: str,
    python_tag: str,
    runtime_release: dict[str, Any],
    python_release: dict[str, Any],
) -> None:
    missing = runtime_assets_to_upload(
        python_release=python_release,
        runtime_release=runtime_release,
    )
    if not missing:
        return

    with tempfile.TemporaryDirectory() as temp_dir:
        download_dir = Path(temp_dir)
        paths = []
        for asset in missing:
            run_gh(
                [
                    "release",
                    "download",
                    runtime_release["tagName"],
                    "--repo",
                    repository,
                    "--pattern",
                    asset["name"],
                    "--dir",
                    str(download_dir),
                ]
            )
            asset_path = download_dir / asset["name"]
            verify_asset_digest(asset_path, asset)
            paths.append(str(asset_path))

        run_gh(
            [
                "release",
                "upload",
                python_tag,
                *paths,
                "--repo",
                repository,
            ]
        )


def build_runtime_downloads_section(
    *, python_tag: str, runtime_release: dict[str, Any]
) -> str:
    selected_assets = sorted(
        runtime_assets(runtime_release), key=lambda asset: asset["name"]
    )

    python_version = python_tag.removeprefix("v")
    repository_url = runtime_release["url"].split("/releases/tag/", maxsplit=1)[0]
    lines = [
        RUNTIME_SECTION_HEADING,
        "",
        (
            "This Python release pairs with the current prebuilt llama.cpp / GGUF "
            f"runtime release: [{runtime_release['tagName']}]({runtime_release['url']})."
        ),
        "",
        (
            "The same verified runtime assets are attached directly to this Python "
            "release so users can find the package and self-contained "
            "`llama-funasr-*` binaries in one place."
        ),
        "",
        "| Platform | Asset | SHA-256 |",
        "|---|---|---|",
    ]
    for asset in selected_assets:
        digest = asset_sha256(asset)
        asset_url = f"{repository_url}/releases/download/{python_tag}/{asset['name']}"
        lines.append(
            f"| {platform_label(asset['name'])} | "
            f"[{asset['name']}]({asset_url}) | "
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
    runtime_section = build_runtime_downloads_section(
        python_tag=python_tag, runtime_release=runtime_release
    )
    managed_section = (
        f"{RUNTIME_SECTION_START}\n{runtime_section}\n{RUNTIME_SECTION_END}"
    )

    marker_start = current_body.find(RUNTIME_SECTION_START)
    marker_end = current_body.find(RUNTIME_SECTION_END)
    if (marker_start == -1) != (marker_end == -1):
        raise RuntimeError("Release body has incomplete runtime download markers")
    if marker_start != -1:
        if marker_end < marker_start:
            raise RuntimeError("Release body has invalid runtime download markers")
        prefix = current_body[:marker_start]
        suffix = current_body[marker_end + len(RUNTIME_SECTION_END) :]
    else:
        legacy_start = current_body.find(RUNTIME_SECTION_HEADING)
        if legacy_start == -1:
            prefix = current_body
            suffix = ""
        else:
            next_heading = current_body.find(
                "\n## ", legacy_start + len(RUNTIME_SECTION_HEADING)
            )
            legacy_end = len(current_body) if next_heading == -1 else next_heading + 1
            prefix = current_body[:legacy_start]
            suffix = current_body[legacy_end:]

    parts = [
        part for part in (prefix.rstrip(), managed_section, suffix.strip()) if part
    ]
    return "\n\n".join(parts) + "\n"


def update_release(repository: str, python_tag: str, runtime_tag: str | None) -> None:
    python_release = load_release(repository, python_tag)
    published_at = python_release.get("publishedAt")
    if not published_at:
        raise RuntimeError(f"{python_tag} has no publication timestamp")
    runtime_tag = runtime_tag or latest_runtime_tag(
        repository, published_before=published_at
    )
    runtime_release = load_release(repository, runtime_tag)
    validate_runtime_release(
        runtime_release=runtime_release,
        python_release=python_release,
    )
    sync_runtime_assets(
        repository=repository,
        python_tag=python_tag,
        runtime_release=runtime_release,
        python_release=python_release,
    )
    python_release = load_release(repository, python_tag)
    remaining = runtime_assets_to_upload(
        python_release=python_release,
        runtime_release=runtime_release,
    )
    if remaining:
        names = ", ".join(asset["name"] for asset in remaining)
        raise RuntimeError(f"{python_tag} is still missing runtime assets: {names}")
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
    parser.add_argument(
        "python_tag", help="Python package release tag, for example v1.3.26"
    )
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
    update_release(args.repo, args.python_tag, args.runtime_tag)


if __name__ == "__main__":
    main()
