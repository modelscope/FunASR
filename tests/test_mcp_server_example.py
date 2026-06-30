import json
from pathlib import Path


def test_mcp_server_example_has_glama_ready_dockerfile():
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "mcp_server"
    dockerfile = example_dir / "Dockerfile"

    assert dockerfile.is_file()

    dockerfile_text = dockerfile.read_text()
    assert "pip install" in dockerfile_text
    assert "funasr" in dockerfile_text
    assert "COPY funasr_mcp.py /app/funasr_mcp.py" in dockerfile_text
    assert 'CMD ["python", "/app/funasr_mcp.py"]' in dockerfile_text


def test_mcp_server_example_has_glama_metadata():
    example_dir = Path(__file__).resolve().parents[1] / "examples" / "mcp_server"
    metadata_path = example_dir / "glama.json"

    metadata = json.loads(metadata_path.read_text())

    assert metadata["repository"] == "https://github.com/modelscope/FunASR"
    assert metadata["runtime"] == "python"
    assert metadata["license"] == "MIT"
    assert metadata["mcpServers"]["funasr"]["command"] == "python"
    assert metadata["mcpServers"]["funasr"]["args"] == ["/app/funasr_mcp.py"]


def test_repository_root_has_glama_metadata_for_mcp_directory_scanners():
    repo_root = Path(__file__).resolve().parents[1]
    metadata = json.loads((repo_root / "glama.json").read_text())

    assert metadata["repository"] == "https://github.com/modelscope/FunASR"
    assert metadata["runtime"] == "python"
    assert metadata["license"] == "MIT"
    assert metadata["mcpServers"]["funasr"]["command"] == "python"
    assert metadata["mcpServers"]["funasr"]["args"] == [
        "examples/mcp_server/funasr_mcp.py"
    ]
