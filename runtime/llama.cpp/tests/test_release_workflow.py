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
