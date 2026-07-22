from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "build-llamacpp-binaries.yml"


def test_windows_cuda_release_asset_is_in_matrix():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: windows-x64-cuda" in workflow
    assert "cuda: true" in workflow
    assert "windows-x64-cuda" in workflow
    assert "cuda_architectures: '86'" in workflow
    assert "build_target: llama-funasr-sensevoice" in workflow
    assert "timeout_minutes: 90" in workflow


def test_linux_vulkan_release_asset_is_in_matrix():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: linux-x64-vulkan" in workflow
    assert "vulkan: true" in workflow
    assert "build_target: llama-funasr-sensevoice" in workflow
    assert "timeout_minutes: 90" in workflow


def test_windows_cuda_build_uses_cuda_toolkit_and_flags():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Jimver/cuda-toolkit" in workflow
    assert "if: matrix.cuda" in workflow
    assert "-DGGML_CUDA=ON" in workflow
    assert "-DGGML_CUDA_FORCE_CUBLAS=ON" in workflow
    assert "-DGGML_CUDA_FA=OFF" in workflow
    assert "-DGGML_CUDA_NCCL=OFF" in workflow
    assert "CMAKE_CUDA_ARCHITECTURES=${{ matrix.cuda_architectures }}" in workflow
    assert "cmake \"${cmake_args[@]}\"" in workflow
    assert "--target \"${{ matrix.build_target }}\"" in workflow


def test_linux_vulkan_build_uses_vulkan_sdk_and_flags():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "libvulkan-dev" in workflow
    assert "vulkan-tools" in workflow
    assert "if: matrix.vulkan" in workflow
    assert "-DGGML_VULKAN=ON" in workflow


def test_release_notes_explain_cpu_and_cuda_windows_assets():
    readme = (ROOT / "runtime" / "llama.cpp" / "README.md").read_text(encoding="utf-8")

    assert "windows-x64-cuda" in readme
    assert "--backend cuda" in readme
    assert "Windows CUDA" in readme
    assert "CUDA architecture 86" in readme
    assert "Build from source" in readme
    assert "other GPU architectures" in readme
    assert "CMAKE_CUDA_ARCHITECTURES=120" in readme
    assert "sm_120" in readme


def test_release_notes_explain_vulkan_linux_asset():
    readme = (ROOT / "runtime" / "llama.cpp" / "README.md").read_text(encoding="utf-8")

    assert "linux-x64-vulkan" in readme
    assert "--backend vulkan" in readme
    assert "Linux Vulkan" in readme
    assert "GGML_VULKAN=ON" in readme



def test_github_release_notes_mention_vulkan_asset():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    release_notes = workflow.split('--notes "', maxsplit=1)[1]

    assert "linux-x64-vulkan" in release_notes
    assert "--backend vulkan" in release_notes
    assert "GGML_VULKAN=ON" in release_notes

def test_readme_documents_lightweight_http_server():
    readme = (ROOT / "runtime" / "llama.cpp" / "README.md").read_text(encoding="utf-8")

    assert "server/funasr_gguf_server.py" in readme
    assert "/v1/audio/transcriptions" in readme
    assert "--binary ./build/bin/llama-funasr-sensevoice" in readme
    assert "one subprocess per request" in readme
