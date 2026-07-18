from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "build-llamacpp-binaries.yml"


def test_windows_cuda_release_asset_is_in_matrix():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: windows-x64-cuda" in workflow
    assert "cuda: true" in workflow
    assert "windows-x64-cuda" in workflow


def test_windows_cuda_build_uses_cuda_toolkit_and_flags():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "Jimver/cuda-toolkit@v0.2.35" in workflow
    assert "if: matrix.cuda" in workflow
    assert "-DGGML_CUDA=ON" in workflow
    assert "-DGGML_CUDA_FA=OFF" in workflow
    assert "-DGGML_CUDA_NCCL=OFF" in workflow


def test_release_notes_explain_cpu_and_cuda_windows_assets():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "windows-x64-cuda" in workflow
    assert "--backend cuda" in workflow
    assert "Windows CUDA" in workflow
