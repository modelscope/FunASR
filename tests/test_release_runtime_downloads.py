import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def sha256_digest(value):
    return "sha256:" + hashlib.sha256(value.encode()).hexdigest()


def load_release_script():
    script_path = ROOT / "scripts" / "update_release_runtime_downloads.py"
    spec = importlib.util.spec_from_file_location(
        "update_release_runtime_downloads", script_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def runtime_release_payload():
    return {
        "tagName": "runtime-llamacpp-v0.1.9",
        "url": "https://github.com/modelscope/FunASR/releases/tag/runtime-llamacpp-v0.1.9",
        "publishedAt": "2026-07-22T00:00:00Z",
        "isDraft": False,
        "isPrerelease": False,
        "assets": [
            {
                "name": "funasr-llamacpp-linux-arm64.tar.gz",
                "url": "https://example.test/linux-arm64.tar.gz",
                "digest": sha256_digest("linuxarm64"),
            },
            {
                "name": "funasr-llamacpp-linux-x64-avx2.tar.gz",
                "url": "https://example.test/linux-avx2.tar.gz",
                "digest": sha256_digest("linuxavx2"),
            },
            {
                "name": "funasr-llamacpp-linux-x64-vulkan.tar.gz",
                "url": "https://example.test/linux-vulkan.tar.gz",
                "digest": sha256_digest("linuxvulkan"),
            },
            {
                "name": "funasr-llamacpp-linux-x64.tar.gz",
                "url": "https://example.test/linux-x64.tar.gz",
                "digest": sha256_digest("linuxx64"),
            },
            {
                "name": "funasr-llamacpp-macos-arm64.tar.gz",
                "url": "https://example.test/macos-arm64.tar.gz",
                "digest": sha256_digest("macosarm64"),
            },
            {
                "name": "funasr-llamacpp-windows-x64-avx2.zip",
                "url": "https://example.test/windows-avx2.zip",
                "digest": sha256_digest("windowsavx2"),
            },
            {
                "name": "funasr-llamacpp-windows-x64-cuda.zip",
                "url": "https://example.test/windows-cuda.zip",
                "digest": sha256_digest("windowscuda"),
            },
            {
                "name": "funasr-llamacpp-windows-x64-vulkan.zip",
                "url": "https://example.test/windows-vulkan.zip",
                "digest": sha256_digest("windowsvulkan"),
            },
            {
                "name": "funasr-llamacpp-windows-x64.zip",
                "url": "https://example.test/windows-x64.zip",
                "digest": sha256_digest("windowsx64"),
            },
        ],
    }


def python_release_payload(assets=None):
    return {
        "tagName": "v1.3.27",
        "publishedAt": "2026-07-24T01:00:00Z",
        "assets": assets or [],
    }


def test_release_on_tag_workflow_adds_runtime_assets_and_downloads():
    workflow = (ROOT / ".github" / "workflows" / "release-on-tag.yml").read_text(
        encoding="utf-8"
    )

    assert "scripts/update_release_runtime_downloads.py" in workflow
    assert "Runtime assets and downloads" in workflow
    assert "gh release view" in workflow
    assert "if !" in workflow


def test_latest_runtime_tag_is_stable_for_python_release_time():
    module = load_release_script()
    calls = []
    module.run_gh_json = lambda args: (
        calls.append(args)
        or [
            {
                "tagName": "runtime-llamacpp-v0.2.0",
                "publishedAt": "2026-07-25T00:00:00Z",
                "isDraft": False,
                "isPrerelease": False,
            },
            {
                "tagName": "runtime-llamacpp-v0.1.10-rc1",
                "publishedAt": "2026-07-23T00:00:00Z",
                "isDraft": False,
                "isPrerelease": True,
            },
            {
                "tagName": "runtime-llamacpp-v0.1.9",
                "publishedAt": "2026-07-22T00:00:00Z",
                "isDraft": False,
                "isPrerelease": False,
            },
        ]
    )

    tag = module.latest_runtime_tag(
        "modelscope/FunASR", published_before="2026-07-24T01:00:00Z"
    )

    assert tag == "runtime-llamacpp-v0.1.9"
    assert calls[0][calls[0].index("--limit") + 1] == "1000"


def test_runtime_assets_to_upload_returns_missing_assets():
    module = load_release_script()

    missing = module.runtime_assets_to_upload(
        python_release=python_release_payload(),
        runtime_release=runtime_release_payload(),
    )

    assert [asset["name"] for asset in missing] == [
        asset["name"] for asset in runtime_release_payload()["assets"]
    ]


def test_runtime_assets_to_upload_skips_matching_assets():
    module = load_release_script()
    runtime_release = runtime_release_payload()
    existing = [
        {
            "name": asset["name"],
            "digest": asset["digest"],
        }
        for asset in runtime_release["assets"]
    ]

    missing = module.runtime_assets_to_upload(
        python_release=python_release_payload(existing),
        runtime_release=runtime_release,
    )

    assert missing == []


def test_runtime_assets_to_upload_recovers_partially_uploaded_release():
    module = load_release_script()
    runtime_release = runtime_release_payload()
    uploaded = runtime_release["assets"][:3]

    missing = module.runtime_assets_to_upload(
        python_release=python_release_payload(uploaded),
        runtime_release=runtime_release,
    )

    assert [asset["name"] for asset in missing] == [
        asset["name"] for asset in runtime_release["assets"][3:]
    ]


def test_runtime_assets_to_upload_rejects_digest_mismatch():
    module = load_release_script()
    runtime_release = runtime_release_payload()
    source_asset = runtime_release["assets"][0]

    with pytest.raises(RuntimeError, match="different digest"):
        module.runtime_assets_to_upload(
            python_release=python_release_payload(
                [{"name": source_asset["name"], "digest": sha256_digest("different")}]
            ),
            runtime_release=runtime_release,
        )


def test_runtime_assets_to_upload_rejects_missing_source_digest():
    module = load_release_script()
    runtime_release = runtime_release_payload()
    runtime_release["assets"][0].pop("digest")

    with pytest.raises(RuntimeError, match="has no SHA-256 digest"):
        module.runtime_assets_to_upload(
            python_release=python_release_payload(),
            runtime_release=runtime_release,
        )


def test_runtime_assets_to_upload_rejects_missing_existing_digest():
    module = load_release_script()
    runtime_release = runtime_release_payload()
    source_asset = runtime_release["assets"][0]

    with pytest.raises(RuntimeError, match="has no SHA-256 digest"):
        module.runtime_assets_to_upload(
            python_release=python_release_payload([{"name": source_asset["name"]}]),
            runtime_release=runtime_release,
        )


def test_runtime_assets_to_upload_rejects_malformed_digest():
    module = load_release_script()
    runtime_release = runtime_release_payload()
    runtime_release["assets"][0]["digest"] = "sha256:not-a-real-digest"

    with pytest.raises(RuntimeError, match="invalid SHA-256 digest"):
        module.runtime_assets_to_upload(
            python_release=python_release_payload(),
            runtime_release=runtime_release,
        )


def test_verify_asset_digest_checks_downloaded_bytes(tmp_path):
    module = load_release_script()
    asset_path = tmp_path / "funasr-llamacpp-linux-x64.tar.gz"
    asset_path.write_bytes(b"verified runtime")
    expected = hashlib.sha256(b"verified runtime").hexdigest()
    asset = {"name": asset_path.name, "digest": f"sha256:{expected}"}

    module.verify_asset_digest(asset_path, asset)

    asset["digest"] = "sha256:" + ("0" * 64)
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        module.verify_asset_digest(asset_path, asset)


def test_sync_runtime_assets_downloads_verifies_and_uploads_missing_asset():
    module = load_release_script()
    payload = b"runtime archive"
    digest = hashlib.sha256(payload).hexdigest()
    asset_name = "funasr-llamacpp-linux-x64.tar.gz"
    runtime_release = {
        "tagName": "runtime-llamacpp-v0.1.9",
        "assets": [
            {
                "name": asset_name,
                "digest": f"sha256:{digest}",
            }
        ],
    }
    calls = []

    def fake_run_gh(args):
        calls.append(args)
        if args[:2] == ["release", "download"]:
            download_dir = Path(args[args.index("--dir") + 1])
            (download_dir / asset_name).write_bytes(payload)

    module.run_gh = fake_run_gh

    module.sync_runtime_assets(
        repository="modelscope/FunASR",
        python_tag="v1.3.27",
        runtime_release=runtime_release,
        python_release=python_release_payload(),
    )

    assert calls[0][:3] == [
        "release",
        "download",
        "runtime-llamacpp-v0.1.9",
    ]
    assert calls[0][calls[0].index("--pattern") + 1] == asset_name
    assert calls[1][:3] == ["release", "upload", "v1.3.27"]
    assert calls[1][-2:] == ["--repo", "modelscope/FunASR"]


def test_update_release_syncs_assets_before_editing_notes():
    module = load_release_script()
    python_release_before = python_release_payload()
    python_release_before["body"] = "stale body"
    python_release_after = python_release_payload(runtime_release_payload()["assets"])
    python_release_after["body"] = "fresh body"
    runtime_release = runtime_release_payload()
    events = []
    python_loads = 0

    def fake_load_release(repository, tag):
        nonlocal python_loads
        assert repository == "modelscope/FunASR"
        if tag != "v1.3.27":
            return runtime_release
        python_loads += 1
        return python_release_before if python_loads == 1 else python_release_after

    def fake_sync_runtime_assets(**kwargs):
        assert kwargs["python_release"] is python_release_before
        assert kwargs["runtime_release"] is runtime_release
        events.append("sync")

    def fake_run_gh(args):
        assert args[:3] == ["release", "edit", "v1.3.27"]
        notes_path = Path(args[args.index("--notes-file") + 1])
        notes = notes_path.read_text(encoding="utf-8")
        assert "fresh body" in notes
        assert "stale body" not in notes
        assert "## Runtime downloads" in notes
        events.append("edit")

    module.load_release = fake_load_release
    module.sync_runtime_assets = fake_sync_runtime_assets
    module.run_gh = fake_run_gh

    module.update_release(
        repository="modelscope/FunASR",
        python_tag="v1.3.27",
        runtime_tag="runtime-llamacpp-v0.1.9",
    )

    assert events == ["sync", "edit"]
    assert python_loads == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("isDraft", True, "is a draft"),
        ("isPrerelease", True, "is a prerelease"),
        ("publishedAt", "2026-07-25T00:00:00Z", "was published after"),
    ],
)
def test_update_release_rejects_explicit_unstable_or_future_runtime(
    field, value, message
):
    module = load_release_script()
    python_release = python_release_payload()
    runtime_release = runtime_release_payload()
    runtime_release[field] = value
    module.load_release = lambda repository, tag: (
        python_release if tag == "v1.3.27" else runtime_release
    )
    module.sync_runtime_assets = lambda **kwargs: pytest.fail(
        "invalid runtime must be rejected before upload"
    )

    with pytest.raises(RuntimeError, match=message):
        module.update_release(
            repository="modelscope/FunASR",
            python_tag="v1.3.27",
            runtime_tag="runtime-llamacpp-v0.1.9",
        )


@pytest.mark.parametrize(
    ("assets_after_upload", "message"),
    [
        ([], "still missing runtime assets"),
        (
            [
                {
                    "name": "funasr-llamacpp-linux-arm64.tar.gz",
                    "digest": sha256_digest("different"),
                }
            ],
            "different digest",
        ),
    ],
)
def test_update_release_does_not_edit_notes_when_post_upload_verification_fails(
    assets_after_upload, message
):
    module = load_release_script()
    python_release_before = python_release_payload()
    python_release_after = python_release_payload(assets_after_upload)
    runtime_release = runtime_release_payload()
    python_loads = 0
    edit_calls = []

    def fake_load_release(repository, tag):
        nonlocal python_loads
        if tag != "v1.3.27":
            return runtime_release
        python_loads += 1
        return python_release_before if python_loads == 1 else python_release_after

    module.load_release = fake_load_release
    module.sync_runtime_assets = lambda **kwargs: None
    module.run_gh = lambda args: edit_calls.append(args)

    with pytest.raises(RuntimeError, match=message):
        module.update_release(
            repository="modelscope/FunASR",
            python_tag="v1.3.27",
            runtime_tag="runtime-llamacpp-v0.1.9",
        )

    assert edit_calls == []


def test_runtime_download_section_lists_all_prebuilt_assets():
    module = load_release_script()

    section = module.build_runtime_downloads_section(
        python_tag="v1.3.26",
        runtime_release=runtime_release_payload(),
    )

    assert "## Runtime downloads" in section
    assert "runtime-llamacpp-v0.1.9" in section
    assert "attached directly to this Python release" in section
    assert 'python -m pip install -U "funasr==1.3.26"' in section
    assert section.count("| [funasr-llamacpp-") == 9
    assert "Windows x64 CUDA" in section
    assert "Windows x64 Vulkan" in section
    assert "Linux x64 Vulkan" in section
    assert f"`{sha256_digest('windowscuda').removeprefix('sha256:')}`" in section
    assert f"`{sha256_digest('windowsvulkan').removeprefix('sha256:')}`" in section
    assert (
        "https://github.com/modelscope/FunASR/releases/download/v1.3.26/"
        "funasr-llamacpp-linux-arm64.tar.gz"
    ) in section


def test_merge_release_body_replaces_stale_runtime_download_section():
    module = load_release_script()
    old_body = """## What's Changed
* old release note

## Runtime downloads

old runtime table
"""

    merged = module.merge_release_body(
        current_body=old_body,
        python_tag="v1.3.26",
        runtime_release=runtime_release_payload(),
    )

    assert "* old release note" in merged
    assert "old runtime table" not in merged
    assert merged.count("## Runtime downloads") == 1
    assert "funasr-llamacpp-windows-x64-cuda.zip" in merged
    assert "funasr-llamacpp-windows-x64-vulkan.zip" in merged


def test_merge_release_body_preserves_sections_after_managed_runtime_downloads():
    module = load_release_script()
    current = """## What's Changed
* release note

## Runtime downloads

old runtime table

## Security notes

Keep this manual section.
"""

    merged = module.merge_release_body(
        current_body=current,
        python_tag="v1.3.27",
        runtime_release=runtime_release_payload(),
    )
    merged_again = module.merge_release_body(
        current_body=merged,
        python_tag="v1.3.27",
        runtime_release=runtime_release_payload(),
    )

    assert "## Security notes\n\nKeep this manual section." in merged
    assert merged.count("<!-- funasr-runtime-downloads:start -->") == 1
    assert merged.count("<!-- funasr-runtime-downloads:end -->") == 1
    assert merged_again == merged
