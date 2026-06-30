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
