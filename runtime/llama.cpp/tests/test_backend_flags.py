from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SENSEVOICE = ROOT / "sensevoice" / "funasr-sensevoice" / "funasr-sensevoice.cpp"


def test_sensevoice_exposes_backend_flag():
    source = SENSEVOICE.read_text(encoding="utf-8")

    assert "--backend" in source
    assert "cpu|cuda|vulkan" in source


def test_sensevoice_does_not_hardcode_cpu_graph_backend():
    source = SENSEVOICE.read_text(encoding="utf-8")
    run_seg_body = source.split("auto run_seg=", maxsplit=1)[1].split("int64_t t0=", maxsplit=1)[0]

    assert "graph_be.backend" in run_seg_body
    assert "graph_be.buffer_type" in run_seg_body
    assert "ggml_backend_cpu_init()" not in run_seg_body
    assert "ggml_backend_cpu_buffer_type()" not in run_seg_body


def test_sensevoice_vulkan_backend_has_dedicated_error_message():
    source = SENSEVOICE.read_text(encoding="utf-8")

    assert 'name=="vulkan"' in source
    assert "GGML_VULKAN=ON" in source
    assert "unsupported backend '%s' (expected cpu|cuda|vulkan)" in source
