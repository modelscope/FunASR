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


def test_sensevoice_prefers_discrete_gpu_and_falls_back_to_matching_igpu():
    source = SENSEVOICE.read_text(encoding="utf-8")
    selector = source.split(
        "static ggml_backend_dev_t find_gpu_backend_device", maxsplit=1
    )[1].split("static graph_backend make_graph_backend", maxsplit=1)[0]

    assert "GGML_BACKEND_DEVICE_TYPE_IGPU" in selector
    assert "integrated_fallback" in selector
    assert "return integrated_fallback" in selector
    discrete_return = "if(type==GGML_BACKEND_DEVICE_TYPE_GPU) return dev;"
    integrated_save = "if(!integrated_fallback) integrated_fallback=dev;"
    assert discrete_return in selector
    assert integrated_save in selector
    assert selector.index(discrete_return) < selector.index(integrated_save)
    assert selector.index(integrated_save) < selector.index(
        "return integrated_fallback"
    )


def test_sensevoice_checks_backend_before_resolving_buffer_type():
    source = SENSEVOICE.read_text(encoding="utf-8")
    initializer = source.split(
        "static graph_backend initialize_device_backend", maxsplit=1
    )[1].split("static graph_backend make_graph_backend", maxsplit=1)[0]

    init_call = "out.backend=ggml_backend_dev_init(dev,nullptr);"
    null_check = "if(!out.backend)"
    buffer_type = "ggml_backend_get_default_buffer_type(out.backend)"
    assert init_call in initializer
    assert null_check in initializer
    assert buffer_type in initializer
    assert initializer.index(init_call) < initializer.index(null_check)
    assert initializer.index(null_check) < initializer.index(buffer_type)


def test_sensevoice_flushes_device_initialization_boundaries_to_stderr():
    source = SENSEVOICE.read_text(encoding="utf-8")
    initializer = source.split(
        "static graph_backend initialize_device_backend", maxsplit=1
    )[1].split("static graph_backend make_graph_backend", maxsplit=1)[0]

    assert "initializing %s backend on %s (%s)" in initializer
    assert "initialized %s backend on %s; resolving buffer type" in initializer
    assert "fflush(stderr);" in initializer
