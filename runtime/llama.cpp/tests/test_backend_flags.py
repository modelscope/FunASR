from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SENSEVOICE = ROOT / "sensevoice" / "funasr-sensevoice" / "funasr-sensevoice.cpp"


def test_sensevoice_exposes_backend_flag():
    source = SENSEVOICE.read_text(encoding="utf-8")

    assert "--backend" in source
    assert "cpu|cuda|vulkan" in source


def test_sensevoice_reads_query_embeddings_using_their_ggml_type():
    source = SENSEVOICE.read_text(encoding="utf-8")

    assert 'ggml_tensor*embed=m.g("embed.weight")' in source
    assert "embed->type==GGML_TYPE_F32" in source
    assert "embed->type==GGML_TYPE_F16" in source
    assert "ggml_fp16_to_fp32" in source
    assert source.count("ggml_backend_tensor_get(embed") == 2
    assert "embed->data" not in source
    assert 'float*emb=(float*)m.g("embed.weight")->data' not in source


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


def test_sensevoice_traces_post_backend_pipeline_boundaries():
    source = SENSEVOICE.read_text(encoding="utf-8")

    boundaries = [
        "loading model metadata",
        "model ready",
        "loading audio",
        "audio ready",
        "running VAD",
        "VAD ready",
        "building graph",
        "graph built",
        "allocating graph",
        "graph allocated",
        "compute starting",
        "compute complete",
    ]
    for boundary in boundaries:
        assert boundary in source

    assert "static void trace_stage(" in source
    trace_stage = source.split("static void trace_stage(", maxsplit=1)[1].split("}", maxsplit=1)[0]
    assert "vfprintf(stderr" in trace_stage
    assert "fflush(stderr);" in trace_stage

    for before, after in (
        ("loading model metadata", "model ready"),
        ("loading audio", "audio ready"),
        ("running VAD", "VAD ready"),
        ("building graph", "graph built"),
        ("allocating graph", "graph allocated"),
        ("compute starting", "compute complete"),
    ):
        assert source.index(before) < source.index(after)


def test_sensevoice_uploads_model_weights_to_selected_backend_before_compute():
    source = SENSEVOICE.read_text(encoding="utf-8")

    assert "static bool load_model_weights(" in source
    loader = source.split("static bool load_model_weights(", maxsplit=1)[1].split(
        "static ggml_tensor*", maxsplit=1
    )[0]

    assert "gguf_init_from_file" in loader
    assert "ggml_dup_tensor" in loader
    assert "ggml_backend_alloc_ctx_tensors_from_buft" in loader
    assert "GGML_BACKEND_BUFFER_USAGE_WEIGHTS" in loader
    assert "gguf_get_data_offset" in loader
    assert "ggml_backend_tensor_set" in loader

    model_load = source.split("// load model", maxsplit=1)[1].split(
        "auto run_seg=", maxsplit=1
    )[0]
    assert "load_model_weights(gguf_path,graph_be.buffer_type,m)" in model_load
    assert "gguf_init_params gp={false,&m.ctx_w}" not in model_load
