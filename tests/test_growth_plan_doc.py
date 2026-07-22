from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "docs" / "community_growth_20k.md"


def test_growth_plan_records_four_repo_community_template_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Add or refresh CONTRIBUTING, PR template, and issue templates",
        "modelscope/FunClip#188",
        "modelscope/FunClip#189",
        "FunAudioLLM/Fun-ASR#151",
        "FunAudioLLM/SenseVoice#327",
        "four core repositories now have `CONTRIBUTING.md`, issue templates, and PR templates",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Add or refresh CONTRIBUTING, PR template, and issue templates" not in text


def test_growth_plan_records_install_deployment_faq_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Add a short FAQ for install/deployment failures",
        "modelscope/FunASR#3366",
        "./troubleshooting.md",
        "./troubleshooting_zh.md",
        "`funasr-server`",
        "`/v1/audio/transcriptions`",
        "`WebSocket`",
        "`llama.cpp` / `GGUF`",
        "Deployment Help",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Add a short FAQ for install/deployment failures" not in text


def test_growth_plan_records_github_metadata_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Confirm GitHub repo description and topics mention ASR",
        "`modelscope/FunASR` topics include `mcp-server`, `openai-compatible-api`, `streaming-asr`, and `vllm`",
        "`FunAudioLLM/Fun-ASR` topics include `fun-asr-nano`, `llama-cpp`, `gguf`, and `multilingual-asr`",
        "`FunAudioLLM/SenseVoice` topics include `sensevoice`, `language-identification`, `audio-event-detection`, and `speech-understanding`",
        "`modelscope/FunClip` topics include `video-transcription`, `auto-subtitles`, `ai-video-editing`, and `funclip`",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Confirm GitHub repo description and topics mention ASR" not in text


def test_growth_plan_records_api_curl_examples_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Publish API server curl examples",
        "`examples/openai_api/README.md`",
        "`examples/openai_api/README_zh.md`",
        "`examples/openai_api/smoke_test.sh`",
        "`examples/openai_api/smoke_test.py`",
        "`/v1/audio/transcriptions`",
        "`-F file=@sample.wav`",
        "`-F model=sensevoice`",
        "`response_format=verbose_json`",
        "`examples/openai_api/POSTMAN.md`",
        "`examples/openai_api/OPENAPI.md`",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Publish API server curl examples" not in text


def test_growth_plan_records_websocket_streaming_examples_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Publish WebSocket streaming examples",
        "`docs/vllm_guide.md`",
        "`docs/vllm_guide_zh.md`",
        "`examples/industrial_data_pretraining/fun_asr_nano/docs/realtime_demo.md`",
        "`serve_realtime_ws.py`",
        "`ws://localhost:10095`",
        "`client_python.py --server ws://localhost:10095 --mic`",
        "`client_python.py --server ws://localhost:10095 --file audio.wav`",
        "`partial`",
        "`partial_start_ms`",
        "`is_final`",
        "`realtime_ws_benchmark.py`",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Publish WebSocket streaming examples" not in text


def test_growth_plan_records_vllm_guide_gpu_validation():
    text = PLAN.read_text()

    required_markers = [
        "[x] Validate vLLM guide on one GPU server",
        "funasr-3239-vllm0191-py312-20260717",
        "`torch 2.10.0+cu128`",
        "`vllm 0.19.1`",
        "`funasr 1.3.26`",
        "ModelScope snapshot",
        "`Qwen3-0.6B`",
        "`model.pt`",
        "`CUDA_VISIBLE_DEVICES=0`",
        "`examples/industrial_data_pretraining/fun_asr_nano/demo_vllm.py`",
        "`gpu_memory_utilization=0.35`",
        "Model loaded in 82.6s",
        "1 files in 1.95s",
        "开饭时间早上九点至下午五点。",
        "HF `Fun-ASR-Nano-2512-hf` snapshot",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Validate vLLM guide on one GPU server" not in text
