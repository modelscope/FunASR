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


def test_windows_vulkan_release_asset_is_in_matrix():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    entry = workflow.split("name: windows-x64-vulkan", maxsplit=1)[1].split(
        "          - os:", maxsplit=1
    )[0]

    assert "vulkan: true" in entry
    assert "windows_vulkan: true" in entry
    assert "build_target: llama-funasr-sensevoice" in entry
    assert "timeout_minutes: 90" in entry
    assert "-DGGML_VULKAN=ON" in entry
    assert "-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded" in entry
    assert "-DGGML_OPENMP=OFF" in entry


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


def test_windows_vulkan_build_uses_pinned_full_sdk():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "if: matrix.vulkan && runner.os == 'Linux'" in workflow
    assert (
        "humbletim/install-vulkan-sdk@30ba978f977e81b72d091fc8888feb1fb26f9aff"
        in workflow
    )
    assert "if: matrix.windows_vulkan" in workflow
    assert "version: '1.4.309.0'" in workflow
    assert "glslc --version" in workflow
    assert "KhronosGroup/SPIRV-Headers.git" in workflow
    assert "09913f088a1197aba4aefd300a876b2ebbaa3391" in workflow
    assert "-DSPIRV_HEADERS_ENABLE_INSTALL=ON" in workflow
    assert "-DSPIRV_HEADERS_ENABLE_TESTS=OFF" in workflow
    assert "CMAKE_PREFIX_PATH=" in workflow


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


def test_release_notes_explain_vulkan_windows_asset():
    readme = (ROOT / "runtime" / "llama.cpp" / "README.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "windows-x64-vulkan" in readme
    assert "Windows Vulkan" in readme
    assert "Vulkan SDK" in readme
    assert "SPIRV-Headers" in readme
    assert "--backend vulkan" in readme
    assert "windows-x64-vulkan" in root_readme



def test_github_release_notes_mention_vulkan_asset():
    workflow = WORKFLOW.read_text(encoding="utf-8")
    release_notes = workflow.split('--notes "', maxsplit=1)[1]

    assert "linux-x64-vulkan" in release_notes
    assert "windows-x64-vulkan" in release_notes
    assert "--backend vulkan" in release_notes
    assert "GGML_VULKAN=ON" in release_notes

def test_readme_documents_lightweight_http_server():
    readme = (ROOT / "runtime" / "llama.cpp" / "README.md").read_text(encoding="utf-8")

    assert "server/funasr_gguf_server.py" in readme
    assert "/v1/audio/transcriptions" in readme
    assert "--binary ./build/bin/llama-funasr-sensevoice" in readme
    assert "one subprocess per request" in readme
