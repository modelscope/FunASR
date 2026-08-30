from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCKER_DIR = ROOT / "runtime" / "dockerfile"


def test_online_cpu_dockerfile_rebuilds_current_checkout_from_pinned_base():
    dockerfile = (DOCKER_DIR / "Dockerfile.online.cpu").read_text()

    required = [
        "funasr-runtime-sdk-online-cpu-0.1.13@sha256:2a54c20f",
        "COPY . /workspace/FunASR",
        "pip install --no-cache-dir -e ./",
        "cmake -S runtime/websocket -B runtime/websocket/build",
        "-DONNXRUNTIME_DIR=",
        "-DFFMPEG_DIR=",
        "online-cpu-entrypoint.sh",
        "EXPOSE 10095",
    ]
    for marker in required:
        assert marker in dockerfile


def test_online_cpu_entrypoint_execs_server_and_supports_overrides():
    entrypoint = (DOCKER_DIR / "online-cpu-entrypoint.sh").read_text()

    required = [
        "set -euo pipefail",
        "FUNASR_MODEL_DIR",
        "FUNASR_ONLINE_MODEL_DIR",
        "FUNASR_VAD_DIR",
        "FUNASR_PUNC_DIR",
        "FUNASR_PORT",
        "exec",
        "funasr-wss-server-2pass",
        '"$@"',
    ]
    for marker in required:
        assert marker in entrypoint


def test_online_cpu_guides_document_source_build_and_image_boundary():
    guides = [
        ROOT / "runtime" / "docs" / "SDK_advanced_guide_online.md",
        ROOT / "runtime" / "docs" / "SDK_advanced_guide_online_zh.md",
    ]

    for guide in guides:
        text = guide.read_text()
        assert "Dockerfile.online.cpu" in text
        assert "docker build" in text
        assert "0.1.13" in text
        assert "10095" in text


def test_online_cpu_docker_workflow_builds_and_smokes_the_server():
    workflow = (
        ROOT / ".github" / "workflows" / "test-online-cpu-docker.yml"
    ).read_text()

    required = [
        "runtime/dockerfile/Dockerfile.online.cpu",
        "timeout-minutes: 120",
        "docker/setup-buildx-action@v3",
        "docker/build-push-action@v6",
        "platforms: linux/amd64",
        "load: true",
        "cache-from: type=gha,scope=online-cpu-amd64",
        "cache-to: type=gha,mode=max,scope=online-cpu-amd64",
        "funasr-wss-server-2pass",
        "--help",
        "bash -n runtime/dockerfile/online-cpu-entrypoint.sh",
    ]
    for marker in required:
        assert marker in workflow


def test_online_cpu_fallback_manifest_lists_current_public_images():
    manifest = (
        ROOT / "runtime" / "docs" / "docker_online_cpu_zh_lists"
    ).read_text()
    images = [
        line.strip()
        for line in manifest.splitlines()
        if line.strip().startswith("funasr-runtime-sdk-online-cpu-")
    ]

    assert images[:2] == [
        "funasr-runtime-sdk-online-cpu-0.1.13",
        "funasr-runtime-sdk-online-cpu-0.1.12",
    ]
