import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


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
        "assets": [
            {
                "name": "funasr-llamacpp-linux-arm64.tar.gz",
                "url": "https://example.test/linux-arm64.tar.gz",
                "digest": "sha256:linuxarm64",
            },
            {
                "name": "funasr-llamacpp-linux-x64-avx2.tar.gz",
                "url": "https://example.test/linux-avx2.tar.gz",
                "digest": "sha256:linuxavx2",
            },
            {
                "name": "funasr-llamacpp-linux-x64-vulkan.tar.gz",
                "url": "https://example.test/linux-vulkan.tar.gz",
                "digest": "sha256:linuxvulkan",
            },
            {
                "name": "funasr-llamacpp-linux-x64.tar.gz",
                "url": "https://example.test/linux-x64.tar.gz",
                "digest": "sha256:linuxx64",
            },
            {
                "name": "funasr-llamacpp-macos-arm64.tar.gz",
                "url": "https://example.test/macos-arm64.tar.gz",
                "digest": "sha256:macosarm64",
            },
            {
                "name": "funasr-llamacpp-windows-x64-avx2.zip",
                "url": "https://example.test/windows-avx2.zip",
                "digest": "sha256:windowsavx2",
            },
            {
                "name": "funasr-llamacpp-windows-x64-cuda.zip",
                "url": "https://example.test/windows-cuda.zip",
                "digest": "sha256:windowscuda",
            },
            {
                "name": "funasr-llamacpp-windows-x64-vulkan.zip",
                "url": "https://example.test/windows-vulkan.zip",
                "digest": "sha256:windowsvulkan",
            },
            {
                "name": "funasr-llamacpp-windows-x64.zip",
                "url": "https://example.test/windows-x64.zip",
                "digest": "sha256:windowsx64",
            },
        ],
    }


def test_release_on_tag_workflow_adds_runtime_downloads_to_latest_release():
    workflow = (ROOT / ".github" / "workflows" / "release-on-tag.yml").read_text(
        encoding="utf-8"
    )

    assert "scripts/update_release_runtime_downloads.py" in workflow
    assert "Runtime downloads" in workflow


def test_runtime_download_section_lists_all_prebuilt_assets():
    module = load_release_script()

    section = module.build_runtime_downloads_section(
        python_tag="v1.3.26",
        runtime_release=runtime_release_payload(),
    )

    assert "## Runtime downloads" in section
    assert "runtime-llamacpp-v0.1.9" in section
    assert 'python -m pip install -U "funasr==1.3.26"' in section
    assert section.count("| [funasr-llamacpp-") == 9
    assert "Windows x64 CUDA" in section
    assert "Windows x64 Vulkan" in section
    assert "Linux x64 Vulkan" in section
    assert "`windowscuda`" in section
    assert "`windowsvulkan`" in section


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
