from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / ".github" / "workflows" / "build-llamacpp-binaries.yml"
RUNTIME_CMAKE = ROOT / "runtime" / "llama.cpp" / "CMakeLists.txt"


def test_windows_cuda_release_asset_is_in_matrix():
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "name: windows-x64-cuda" in workflow
    assert "cuda: true" in workflow
    assert "windows-x64-cuda" in workflow
    assert "cuda_architectures: '86'" in workflow
    assert "build_target: llama-funasr-sensevoice" in workflow


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


def test_cublas_release_build_skips_long_mmq_template_sources():
    cmake = RUNTIME_CMAKE.read_text(encoding="utf-8")

    assert "GGML_CUDA_FORCE_CUBLAS" in cmake
    assert "pruning ggml CUDA template sources" in cmake
    assert "if (GGML_CUDA_FA)" in cmake
    assert 'if (NOT GGML_CUDA_FORCE_CUBLAS)' in cmake
    assert 'template-instances/fattn-tile*.cu' in cmake
    assert 'template-instances/fattn-mma*.cu' in cmake
    assert 'template-instances/mmq*.cu' in cmake
    assert 'template-instances/mmf*.cu' in cmake


def test_release_notes_explain_cpu_and_cuda_windows_assets():
    readme = (ROOT / "runtime" / "llama.cpp" / "README.md").read_text(encoding="utf-8")

    assert "windows-x64-cuda" in readme
    assert "--backend cuda" in readme
    assert "Windows CUDA" in readme
    assert "CUDA architecture 86" in readme
    assert "Build from source" in readme
    assert "other GPU architectures" in readme
