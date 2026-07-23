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


def test_growth_plan_records_readme_quick_start_cpu_gpu_validation():
    text = PLAN.read_text()

    required_markers = [
        "[x] Verify README quick start on clean CPU and GPU environments",
        "clean CPU venv",
        "clean GPU venv",
        "README Quick Start SenseVoiceSmall CPU",
        "README Quick Start Fun-ASR-Nano GPU",
        "`torch 2.13.0+cu130`",
        "`torchaudio 2.11.0+cu130`",
        "`torch 2.11.0+cu128`",
        "`torch.version.cuda 12.8`",
        "`torchaudio 2.11.0+cu128`",
        "`funasr 1.3.26`",
        "`AutoModel(model=\"iic/SenseVoiceSmall\"",
        "`AutoModel(model=\"FunAudioLLM/Fun-ASR-Nano-2512\"",
        "`torch.cuda.is_available()` was `False`",
        "install matching PyTorch / torchaudio CUDA wheels from pytorch.org",
        "verify `torch.cuda.is_available()` before using `device=\"cuda\"`",
        "`cuda_available True`",
        "`NVIDIA H100 80GB HBM3`",
        "`rtf_avg: 0.237`",
        "欢迎大家来体验达摩院推出的语音识别模型",
        "readme-quickstart-cpu-20260723.log",
        "readme-quickstart-gpu-20260723.log",
        "readme-quickstart-gpu-cu128-20260723.log",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Verify README quick start on clean CPU and GPU environments" not in text
    assert "Do not mark this complete until a clean GPU venv reports CUDA visible" not in text


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


def test_growth_plan_records_release_note_terminal_outputs():
    text = PLAN.read_text()

    required_markers = [
        "[x] Record concise terminal outputs for release notes",
        "Combined GitHub stars: **35,787**",
        "Added since baseline: **4,563** / 20,000",
        "PyPI package: **funasr 1.3.26**",
        "funasr.com static page contract passed for 9 pages",
        "validate_docker.sh syntax ok",
        "Model loaded in 82.6s",
        "Results: 1 samples, total inference time: 1.95s",
        "Text: 开饭时间早上九点至下午五点。",
        "Release notes can now cite the growth snapshot, website patrol, Docker helper guard, and H100 vLLM smoke",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Record concise terminal outputs for release notes" not in text


def test_growth_plan_records_sharp_github_release_focus():
    text = PLAN.read_text()

    required_markers = [
        "[x] Create a GitHub release focused on one sharp value proposition",
        "v1.3.26",
        "https://github.com/modelscope/FunASR/releases/tag/v1.3.26",
        "Release focus: OpenAI-compatible ASR deployment",
        "run the OpenAI-compatible transcription API",
        "`python -m pip install -U \"funasr==1.3.26\"`",
        "`/v1/audio/transcriptions`",
        "`serve_realtime_ws.py`",
        "`docs/vllm_guide.md`",
        "`vllm 0.19.1`",
        "`runtime-llamacpp-v0.1.8`",
        "self-contained `llama-funasr-*` binary",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Create a GitHub release focused on one sharp value proposition" not in text


def test_growth_plan_records_discussion_pin_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Pin a GitHub discussion for the release",
        "`modelscope/FunASR#3376`",
        "authenticated Discussions admin UI",
        "`Discussion has successfully been pinned`",
        "`Edit pinned discussion`",
        "`Unpin discussion`",
        "top of the public Discussions page",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Pin a GitHub discussion for the release" not in text


def test_growth_plan_records_issue_triage_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Triage all new issues within 48 hours",
        "`modelscope/FunASR`",
        "`FunAudioLLM/Fun-ASR`",
        "`FunAudioLLM/SenseVoice`",
        "`modelscope/FunClip`",
        "`0` open pull requests",
        "Fun-ASR, SenseVoice, and FunClip had `0` open issues",
        "#3302",
        "`question` + `needs feedback`",
        "#3104",
        "`enhancement` + `needs maintainer decision`",
        "No unlabelled or unowned open issue remained",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Triage all new issues within 48 hours" not in text


def test_growth_plan_records_top_support_questions_completion():
    text = PLAN.read_text()
    troubleshooting = (ROOT / "docs" / "troubleshooting.md").read_text()
    troubleshooting_zh = (ROOT / "docs" / "troubleshooting_zh.md").read_text()

    required_markers = [
        "[x] Convert top 3 support questions into docs",
        "latest 80 FunASR issues",
        "install / hub / model-id selection",
        "llama.cpp / GGUF / CUDA-Vulkan runtime packages",
        "realtime / VAD / vLLM / server output behavior",
        "#3321",
        "#3298",
        "#3101",
        "Top support questions",
        "`docs/troubleshooting.md`",
        "`docs/troubleshooting_zh.md`",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Convert top 3 support questions into docs" not in text
    assert "## Top support questions from recent issues" in troubleshooting
    assert "Which install or hub path should I use?" in troubleshooting
    assert "Which runtime package should I run on CPU, CUDA, Vulkan, or GGUF?" in troubleshooting
    assert "Why is realtime, VAD, vLLM, or server output delayed" in troubleshooting
    assert "## 最近 issue 里的 Top 支持问题" in troubleshooting_zh
    assert "应该用哪个安装命令、模型 id 或 hub？" in troubleshooting_zh


def test_growth_plan_records_modelscope_card_sync_blocker():
    text = PLAN.read_text()

    required_markers = [
        "[ ] Update ModelScope model cards and demos",
        "`scripts/sync_modelscope_model_cards.py`",
        "`FunAudioLLM/Fun-ASR-Nano-2512`",
        "`README.md`, `README_zh.md`",
        "`iic/SenseVoiceSmall`",
        "`python -m pip install -U \"funasr>=1.3.26\" modelscope`",
        "`torch.cuda.is_available()`",
        "modelscope-card-sync-20260723",
        "HTTP 400 from the ModelScope LFS batch API",
        "`No credentials found`",
        "https://modelscope.cn/my/myaccesstoken",
    ]
    for marker in required_markers:
        assert marker in text

    assert (ROOT / "scripts" / "sync_modelscope_model_cards.py").exists()


def test_growth_plan_records_pypi_description_audit():
    text = PLAN.read_text()

    required_markers = [
        "[x] Update PyPI release description if needed",
        "PyPI `funasr 1.3.26`",
        "long description already comes from `README.md`",
        "`OpenAI-compatible`",
        "`/v1/audio/transcriptions`",
        "`vllm`",
        "`runtime-llamacpp-v0.1.8`",
        "`GGUF`",
        "`funasr==1.3.26`",
        "package summary was prepared for the next PyPI publish",
        "OpenAI-compatible speech recognition toolkit with WebSocket streaming, vLLM acceleration, and llama.cpp/GGUF edge runtime",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Update PyPI release description if needed" not in text


def test_growth_plan_records_release_discussion_distribution():
    text = PLAN.read_text()

    required_markers = [
        "[x] Share the release in relevant developer communities",
        "modelscope/FunASR#3376",
        "https://github.com/modelscope/FunASR/discussions/3376",
        "FunASR v1.3.26: OpenAI-compatible ASR deployment, vLLM, and llama.cpp runtime",
        "Announcements",
        "`/v1/audio/transcriptions`",
        "`runtime-llamacpp-v0.1.8`",
        "[x] Pin a GitHub discussion for the release",
        "`Discussion has successfully been pinned`",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Share the release in relevant developer communities" not in text


def test_growth_plan_records_homepage_entrypoint_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Update homepage hero and docs entry points",
        "https://www.funasr.com/",
        "https://www.funasr.com/en/",
        "Industrial Speech Recognition",
        "`OpenAI-compatible`",
        "`/v1/audio/transcriptions`",
        "`vLLM`",
        "`/donors.html`",
        "funasr.com static page contract passed for 13 pages",
        "funasr-v1-3-26-openai-vllm-llama-cpp.html",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Update homepage hero and docs entry points" not in text


def test_growth_plan_records_metrics_tracking_completion():
    text = PLAN.read_text()

    required_markers = [
        "[x] Track stars, PyPI downloads, issue volume, and docs traffic.",
        "PyPIStats",
        "`downloads_last_7_days`",
        "`downloads_last_30_days`",
        "`collect_growth_metrics.py`",
        "GitHub stars, open issues, and open pull requests",
        "`117,850` downloads in 7 days",
        "`445,755` downloads in 30 days",
        "`2026-07-22`",
        "pypi-downloads-funasr.json",
        "`stale_available`",
        "SSL/429 outages",
        "`scripts/collect_website_traffic.py`",
        "`python scripts/collect_website_traffic.py --log '/var/log/nginx/access.log*' --days 30 --format json`",
        "plain or gzip Nginx logs",
        "raw client IPs and user agents are never emitted",
        "over 15 rotated logs",
        "`88,239` page views / `47,084` approximate unique visitors",
        "docs at `15,501` / `11,843`",
        "blog at `27,901` / `22,716`",
        "`21,915` / `12,668`",
        "docs at `4,638` / `3,612`",
        "blog at `6,901` / `5,590`",
    ]
    for marker in required_markers:
        assert marker in text

    assert "[ ] Track stars, PyPI downloads, issue volume, and docs traffic." not in text
    assert "docs traffic remains unavailable" not in text
