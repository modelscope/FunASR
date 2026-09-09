import io
import json
import re
import runpy
import shlex
import shutil
import subprocess
import sys
import threading
import wave
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

ROOT_READMES = {
    "README.md": "Deploy",
    "README_zh.md": "部署",
    "README_ja.md": "デプロイ",
    "README_ko.md": "배포",
}


def _deployment_blocks(readme):
    text = (ROOT / readme).read_text()
    section = text.split(f"## {ROOT_READMES[readme]}\n", 1)[1].split("\n## ", 1)[0]
    return re.findall(r"^```bash\n(.*?)^```", section, re.M | re.S)


def _recipe_commands(block):
    commands = []
    for line in block.replace("\\\n", " ").splitlines():
        lexer = shlex.shlex(line, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
        current = []
        for token in tokens:
            if token == "&&":
                assert current
                commands.append(current)
                current = []
            else:
                current.append(token)
        if current:
            commands.append(current)
    return commands


@pytest.fixture(scope="module")
def server_parser():
    # Load the real CLI parser without importing FunASR or downloading models.
    return runpy.run_path(str(ROOT / "funasr/bin/server.py"))["build_parser"]()


@pytest.mark.parametrize("readme", ROOT_READMES)
def test_root_readme_http_recipe_is_loopback_and_model_consistent(readme, server_parser):
    blocks = _deployment_blocks(readme)
    starts = [c for b in blocks for c in _recipe_commands(b) if c[0] == "funasr-server"]
    assert len(starts) == 1, "MOSS preparation belongs to its separate environment guide"
    command = starts[0]
    assert {"--host", "--port", "--model", "--device"}.issubset(command)
    args = server_parser.parse_args(command[1:])
    assert (args.host, args.port, args.model, args.device) == ("127.0.0.1", 8000, "sensevoice", "cpu")
    curl_commands = [c for b in blocks for c in _recipe_commands(b) if c[0] == "curl"]
    assert len(curl_commands) == 2
    download, request = curl_commands
    assert {"--fail", "--location", "-o", "sample.wav"}.issubset(download)
    assert download[download.index("-o") + 1] == "sample.wav"
    assert "&&" in next(b for b in blocks if "curl" in b), "Do not send a stale sample after a failed download"
    assert "--fail-with-body" in request
    assert f"http://{args.host}:{args.port}/v1/audio/transcriptions" in request
    forms = [request[i + 1] for i, token in enumerate(request) if token == "-F"]
    assert forms == ["file=@sample.wav", f"model={args.model}", "response_format=verbose_json"]


@pytest.mark.parametrize("readme", ROOT_READMES)
def test_root_readme_cpu_install_is_isolated_without_vllm(readme):
    commands = _recipe_commands(_deployment_blocks(readme)[0])
    assert commands[:2] == [["python3.11", "-m", "venv", ".venv-funasr-http"],
                            [".", ".venv-funasr-http/bin/activate"]]
    installs = [c for c in commands if c[:4] == ["python", "-m", "pip", "install"]]
    assert installs == [["python", "-m", "pip", "install", "torch", "torchaudio"],
                        ["python", "-m", "pip", "install", "funasr", "fastapi", "uvicorn", "python-multipart"]]
    assert ["python", "-m", "pip", "check"] in commands
    text = (ROOT / readme).read_text()
    assert not re.findall(r"`funasr-server[^`]*`", text), "Link to a prepared recipe instead of an unprepared inline launch"
    suffix = "_zh" if readme == "README_zh.md" else ""
    for link in [f"./docs/moss_transcribe_diarize{suffix}.md",
                 f"./examples/openai_api/SECURITY{suffix}.md"]:
        assert f"]({link})" in text
        assert (ROOT / link).is_file()


def test_root_readme_http_commands_are_identical_in_four_languages():
    recipes = [[_recipe_commands(b) for b in _deployment_blocks(readme)[:2]] for readme in ROOT_READMES]
    assert all(recipe == recipes[0] for recipe in recipes[1:])


def test_server_help_describes_auto_selection_and_a_matching_sdk_recipe(server_parser):
    default = server_parser.parse_args([])
    assert (default.host, default.port, default.device, default.model) == ("0.0.0.0", 8000, "cuda", "auto")
    module = runpy.run_path(str(ROOT / "funasr/bin/server.py"))
    usage = module["__doc__"]
    help_text = server_parser.format_help()
    for text in [usage, help_text]:
        assert "cuda* -> fun-asr-nano; other devices -> sensevoice" in text
        assert "default: sensevoice" not in text.lower()
        commands = re.findall(r"^\s*(funasr-server[^#\n]*)", text, re.M)
        parsed = [server_parser.parse_args(shlex.split(c)[1:]) for c in commands]
        assert any(a.model == "sensevoice" and a.device == "cpu" and a.host == "127.0.0.1" for a in parsed)
        assert all("--host" in shlex.split(c) for c in commands)
        assert all(a.host == "127.0.0.1" for a in parsed)
    assert 'with open("a.wav", "rb") as audio:' in help_text
    assert 'model="sensevoice", file=audio' in help_text
    assert "python -m pip install openai" in help_text
    assert "placeholder, not authentication" in help_text


def test_readme_venv_activation_works_in_posix_sh(tmp_path):
    env = tmp_path / ".venv-funasr-http"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(env)], check=True, timeout=30)
    activation = _deployment_blocks("README.md")[0].splitlines()[1]
    result = subprocess.run(["sh", "-c", activation + " && command -v python"],
                            cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(env / "bin/python")


@pytest.mark.parametrize("download_status,transcription_status", [(200, 200), (503, 200), (200, 422)])
def test_readme_curl_recipe_uploads_exact_sample_and_propagates_errors(
    tmp_path, download_status, transcription_status
):
    assert shutil.which("curl"), "curl is required to check the published shell recipe"
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * 160)
    sample = buffer.getvalue()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            requests.append(("GET", self.path))
            body = sample if download_status == 200 else b"sample unavailable"
            self.send_response(download_status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            self.connection.settimeout(5)
            body = self.rfile.read(int(self.headers["Content-Length"]))
            message = BytesParser(policy=email_policy).parsebytes(
                f"Content-Type: {self.headers['Content-Type']}\r\n\r\n".encode() + body
            )
            fields = {part.get_param("name", header="content-disposition"): part.get_payload(decode=True)
                      for part in message.iter_parts()}
            requests.append(("POST", self.path, fields))
            response = json.dumps({"text": "fixture"} if transcription_status == 200 else {"error": "fixture failure"}).encode()
            self.send_response(transcription_status)
            self.send_header("Content-Length", str(len(response)))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(response)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        recipe = _deployment_blocks("README.md")[1]
        download, request = _recipe_commands(recipe)
        source_url = next(token for token in download if token.startswith("https://"))
        target_url = next(token for token in request if token.startswith("http://"))
        origin = f"http://127.0.0.1:{server.server_port}"
        recipe = recipe.replace(source_url, origin + "/sample.wav").replace(target_url, origin + "/v1/audio/transcriptions")
        (tmp_path / "sample.wav").write_bytes(b"stale sample must not be uploaded")
        result = subprocess.run(["bash", "-c", recipe], cwd=tmp_path, capture_output=True, text=True, timeout=15)
        assert requests[0] == ("GET", "/sample.wav")
        if download_status != 200:
            assert result.returncode != 0
            assert len(requests) == 1
        else:
            assert requests[1] == ("POST", "/v1/audio/transcriptions", {
                "file": sample, "model": b"sensevoice", "response_format": b"verbose_json",
            })
            assert len(requests) == 2
            assert (result.returncode == 0) == (transcription_status == 200)
            assert json.loads(result.stdout) == ({"text": "fixture"} if transcription_status == 200 else {"error": "fixture failure"})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


DOCS_WITH_CURRENT_FUNASR_INSTALL = [
    "examples/industrial_data_pretraining/fun_asr_nano/docs/finetune.md",
    "examples/industrial_data_pretraining/fun_asr_nano/docs/finetune_zh.md",
]

PUBLIC_DOCS_SHOULD_USE_CURRENT_HOSTS = [
    "benchmarks/benchmark_pipeline_cer.md",
    "docs/installation/installation.md",
    "docs/installation/installation_zh.md",
    "model_zoo/modelscope_models.md",
    "model_zoo/readme.md",
    "runtime/docs/SDK_advanced_guide_offline.md",
    "runtime/docs/SDK_advanced_guide_offline_en.md",
    "runtime/docs/SDK_advanced_guide_offline_en_zh.md",
    "runtime/docs/SDK_advanced_guide_offline_gpu.md",
    "runtime/docs/SDK_advanced_guide_offline_gpu_zh.md",
    "runtime/docs/SDK_advanced_guide_offline_zh.md",
    "runtime/docs/SDK_advanced_guide_online.md",
    "runtime/docs/SDK_advanced_guide_online_zh.md",
    "runtime/python/grpc/Readme.md",
    "runtime/python/libtorch/README.md",
    "runtime/python/onnxruntime/README.md",
    "runtime/python/websocket/README.md",
    "docs/m2met2/Baseline.md",
    "examples/industrial_data_pretraining/monotonic_aligner/README_zh.md",
    "examples/industrial_data_pretraining/sense_voice/README.md",
    "examples/industrial_data_pretraining/sense_voice/README_zh.md",
    "examples/industrial_data_pretraining/sense_voice/README_ja.md",
    "runtime/html5/readme.md",
    "runtime/html5/readme_zh.md",
]

PUBLIC_ENTRYPOINTS_SHOULD_USE_CURRENT_REPO_URLS = [
    "model_zoo/modelscope_models.md",
    "model_zoo/modelscope_models_zh.md",
    "model_zoo/readme.md",
    "web-pages/src/views/home/index.vue",
    "web-pages/src/views/home/lxwjzxfw.vue",
    "web-pages/src/views/home/sstx.vue",
    "runtime/deploy_tools/funasr-runtime-deploy-offline-cpu-en.sh",
    "runtime/deploy_tools/funasr-runtime-deploy-offline-cpu-zh.sh",
    "runtime/deploy_tools/funasr-runtime-deploy-online-cpu-zh.sh",
    "runtime/docs/SDK_tutorial.md",
    "runtime/docs/SDK_tutorial_en.md",
    "runtime/docs/SDK_tutorial_en_zh.md",
    "runtime/docs/SDK_tutorial_online.md",
    "runtime/docs/SDK_tutorial_online_zh.md",
    "runtime/docs/SDK_tutorial_zh.md",
    "runtime/python/http/server.py",
    "runtime/python/libtorch/setup.py",
    "runtime/python/onnxruntime/setup.py",
]


def test_current_funasr_install_commands_are_quoted():
    for relpath in DOCS_WITH_CURRENT_FUNASR_INSTALL:
        text = (ROOT / relpath).read_text()
        assert '"funasr>=1.3.26"' in text
        assert "funasr>=1.3.0" not in text
        assert not re.search(r"pip install funasr>=", text)


@pytest.mark.parametrize("relpath", ["docs/vllm_guide.md", "docs/vllm_guide_zh.md"])
def test_vllm_service_install_uses_a_pinned_checkout(relpath):
    text = (ROOT / relpath).read_text()
    blocks = re.findall(r"^```bash\n(.*?)^```", text, re.M | re.S)
    install = blocks[0]
    assert 'python -m pip install "vllm==0.19.1"' in install
    assert "git clone https://github.com/modelscope/FunASR.git FunASR-vllm" in install
    assert re.search(r"^git checkout --detach [0-9a-f]{40}$", install, re.M)
    assert "python -m pip install -e ." in install
    assert "python -m pip check" in install
    assert not re.search(r"pip install.*funasr[>=]", install)


def test_vllm_install_translations_have_identical_commands():
    recipes = []
    for relpath in ("docs/vllm_guide.md", "docs/vllm_guide_zh.md"):
        text = (ROOT / relpath).read_text()
        install = re.findall(r"^```bash\n(.*?)^```", text, re.M | re.S)[0]
        recipes.append([
            tokens for line in install.splitlines()
            if (tokens := shlex.split(line, comments=True))
        ])
    assert recipes[0] == recipes[1]


def test_fun_asr_nano_finetune_zh_uses_canonical_filename():
    docs_dir = ROOT / "examples/industrial_data_pretraining/fun_asr_nano/docs"
    assert (docs_dir / "finetune_zh.md").exists()
    assert "fintune_zh.md" not in (docs_dir / "finetune.md").read_text()


def test_public_docs_use_current_repository_and_docs_hosts():
    for relpath in PUBLIC_DOCS_SHOULD_USE_CURRENT_HOSTS:
        text = (ROOT / relpath).read_text()
        assert "github.com/alibaba/FunASR" not in text
        assert "alibaba-damo-academy.github.io/FunASR" not in text


def test_public_entrypoints_use_current_repository_urls():
    forbidden = [
        "github.com/alibaba-damo-academy/FunASR",
        "raw.githubusercontent.com/alibaba-damo-academy/FunASR",
    ]

    for relpath in PUBLIC_ENTRYPOINTS_SHOULD_USE_CURRENT_REPO_URLS:
        text = (ROOT / relpath).read_text()
        for marker in forbidden:
            assert marker not in text


def test_troubleshooting_faq_is_linked_from_readmes():
    readme_links = [
        ("README.md", "./docs/troubleshooting.md"),
        ("README_zh.md", "./docs/troubleshooting_zh.md"),
    ]

    for readme, link in readme_links:
        text = (ROOT / readme).read_text()
        assert link in text


def test_troubleshooting_faq_covers_common_install_and_deploy_failures():
    docs = [
        (ROOT / "docs/troubleshooting.md").read_text(),
        (ROOT / "docs/troubleshooting_zh.md").read_text(),
    ]
    required_markers = [
        "torch",
        "torchaudio",
        "ModelScope",
        "Hugging Face",
        "funasr-server",
        "/v1/audio/transcriptions",
        "WebSocket",
        "llama.cpp",
        "GGUF",
        "Deployment Help",
    ]

    for text in docs:
        for marker in required_markers:
            assert marker in text


def test_tutorials_keep_model_license_boundaries_model_card_specific():
    docs = {
        "en": (ROOT / "docs/tutorial/README.md").read_text(),
        "zh": (ROOT / "docs/tutorial/README_zh.md").read_text(),
    }

    assert "Each model weight has its own license" in docs["en"]
    assert "model card explicitly links" in docs["en"]
    assert "每个模型权重都有各自的许可" in docs["zh"]
    assert "模型卡明确链接" in docs["zh"]

    for text in docs.values():
        assert "自由使用、复制、修改和分享FunASR模型" not in text
        assert "free to use, copy, modify, and share FunASR models" not in text


def test_public_docs_do_not_advertise_stale_release_or_star_copy():
    checked_docs = [
        "docs/repository_roles.md",
        "docs/repository_roles_zh.md",
        "docs/blog_whisper_vs_funasr_zh.md",
        "runtime/llama.cpp/README.md",
    ]
    forbidden = [
        "GitHub tags go up to `v1.3.13`",
        "PyPI has published `1.3.14`",
        "GitHub tag 至 `v1.3.13`",
        "PyPI 已发布 `1.3.14`",
        "（16K+ stars）",
        "runtime-llamacpp-v0.1.7",
    ]

    for relpath in checked_docs:
        text = (ROOT / relpath).read_text()
        for marker in forbidden:
            assert marker not in text


def test_readme_model_tables_use_current_modelscope_entries():
    readmes = [
        (ROOT / "README.md").read_text(),
        (ROOT / "README_zh.md").read_text(),
        (ROOT / "README_ja.md").read_text(),
        (ROOT / "README_ko.md").read_text(),
    ]

    current_entries = [
        "models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
        "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
        "models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary",
        "models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary",
    ]
    stale_entries = [
        "models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
        "models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
        "models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary",
        "models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary",
    ]

    for text in readmes:
        assert current_entries[0] in text
        assert stale_entries[0] not in text

    for text in readmes[:2]:
        for marker in current_entries:
            assert marker in text
        for marker in stale_entries:
            assert marker not in text


def test_model_zoo_tables_use_current_paraformer_modelscope_entry():
    docs = [
        (ROOT / "model_zoo/modelscope_models.md").read_text(),
        (ROOT / "model_zoo/modelscope_models_zh.md").read_text(),
        (ROOT / "model_zoo/readme.md").read_text(),
    ]

    current = (
        "models/iic/"
        "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        "/summary"
    )
    stale = (
        "models/damo/"
        "speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"
        "/summary"
    )

    for text in docs:
        assert current in text
        assert stale not in text


def test_full_modelscope_tables_use_current_core_entries():
    docs = [
        (ROOT / "model_zoo/modelscope_models.md").read_text(),
        (ROOT / "model_zoo/modelscope_models_zh.md").read_text(),
    ]

    current_entries = [
        "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
        "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
        "models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary",
        "models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary",
    ]
    stale_entries = [
        "models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
        "models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
        "models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary",
        "models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary",
    ]

    for text in docs:
        for marker in current_entries:
            assert marker in text
        for marker in stale_entries:
            assert marker not in text


def test_runtime_benchmark_docs_use_current_core_modelscope_entries():
    docs = [
        (ROOT / "benchmarks/benchmark_pipeline_cer.md").read_text(),
        (ROOT / "runtime/docs/benchmark_libtorch.md").read_text(),
        (ROOT / "runtime/docs/benchmark_onnx.md").read_text(),
        (ROOT / "runtime/docs/benchmark_onnx_cpp.md").read_text(),
    ]

    current_entries = [
        "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
        "models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
        "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
        "models/iic/speech_fsmn_vad_zh-cn-16k-common-onnx/summary",
        "models/iic/punc_ct-transformer_zh-cn-common-vocab272727-onnx/summary",
    ]
    stale_entries = [
        entry.replace("models/iic/", "models/damo/") for entry in current_entries
    ]

    combined = "\n".join(docs)
    for marker in current_entries:
        assert marker in combined
    for marker in stale_entries:
        assert marker not in combined


def test_model_zoo_landing_tables_link_huggingface_repos():
    docs = [
        (
            (ROOT / "model_zoo/readme.md").read_text(),
            [
                "https://huggingface.co/funasr/paraformer-zh",
                "https://huggingface.co/funasr/paraformer-zh-streaming",
                "https://huggingface.co/funasr/ct-punc",
                "https://huggingface.co/funasr/fsmn-vad",
            ],
        ),
        (
            (ROOT / "model_zoo/readme_zh.md").read_text(),
            [
                "https://huggingface.co/funasr/paraformer-zh",
                "https://huggingface.co/funasr/paraformer-zh-streaming",
            ],
        ),
    ]

    for text, required_hf_links in docs:
        for link in required_hf_links:
            assert link in text


def test_model_zoo_landing_tables_use_current_core_modelscope_entries():
    docs = [
        (
            (ROOT / "model_zoo/readme.md").read_text(),
            [
                "models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
                "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
                "models/iic/punc_ct-transformer_cn-en-common-vocab471067-large/summary",
                "models/iic/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary",
            ],
            [
                "models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
                "models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
                "models/damo/punc_ct-transformer_cn-en-common-vocab471067-large/summary",
                "models/damo/speech_fsmn_vad_zh-cn-16k-common-pytorch/summary",
            ],
        ),
        (
            (ROOT / "model_zoo/readme_zh.md").read_text(),
            [
                "models/iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
                "models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
            ],
            [
                "models/damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch/summary",
                "models/damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online/summary",
            ],
        ),
    ]

    for text, current_entries, stale_entries in docs:
        for marker in current_entries:
            assert marker in text
        for marker in stale_entries:
            assert marker not in text


def test_readme_model_tables_surface_public_gguf_entries():
    readmes = [
        (ROOT / "README.md").read_text(),
        (ROOT / "README_zh.md").read_text(),
        (ROOT / "README_ja.md").read_text(),
        (ROOT / "README_ko.md").read_text(),
    ]

    for text in readmes:
        assert "https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-GGUF" in text
        assert "https://huggingface.co/FunAudioLLM/SenseVoiceSmall-GGUF" in text
        assert "https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-GGUF" not in text


def test_top_level_readmes_surface_current_release_and_edge_runtime():
    release_version = (ROOT / "funasr" / "version.txt").read_text().strip()
    readmes = {
        name: (ROOT / name).read_text()
        for name in ("README.md", "README_zh.md", "README_ja.md", "README_ko.md")
    }

    for name, text in readmes.items():
        assert f'python -m pip install -U "funasr=={release_version}"' in text, name
        assert (
            f"https://github.com/modelscope/FunASR/releases/tag/v{release_version}"
            in text
        ), name
        assert "runtime-llamacpp-v0.2.6" in text, name

    assert "https://www.funasr.com/en/deploy/llama-cpp.html" in readmes["README.md"]
    assert "https://www.funasr.com/deploy/llama-cpp.html" in readmes["README_zh.md"]
    for name in ("README_ja.md", "README_ko.md"):
        assert "https://www.funasr.com/en/deploy/llama-cpp.html" in readmes[name]

    for name in ("README.md", "README_zh.md"):
        text = readmes[name]
        for asset in (
            "funasr-llamacpp-linux-x64-vulkan.tar.gz",
            "funasr-llamacpp-windows-x64-vulkan.zip",
            "funasr-llamacpp-windows-x64-cuda.zip",
            "funasr-llamacpp-windows-x64-cuda-blackwell.zip",
        ):
            assert (
                f"releases/download/runtime-llamacpp-v0.2.6/{asset}" in text
            ), name
        assert "releases/download/runtime-llamacpp-v0.2.1/" not in text, name


def test_top_level_readme_news_stays_concise():
    headings = {
        "README.md": "## What's new",
        "README_zh.md": "## 最新动态",
        "README_ja.md": "## 最新情報",
        "README_ko.md": "## 최신 소식",
    }

    for name, heading in headings.items():
        text = (ROOT / name).read_text()
        news = text.split(heading, 1)[1].split("\n---", 1)[0]
        assert news.count("\n- ") <= 5, name
        assert "https://github.com/modelscope/FunASR/releases" in news, name


def test_repository_roadmap_tracks_current_delivery_and_open_work():
    release_version = (ROOT / "funasr" / "version.txt").read_text().strip()
    docs = [
        (ROOT / "docs/repository_roles.md").read_text(),
        (ROOT / "docs/repository_roles_zh.md").read_text(),
    ]

    for text in docs:
        assert f"funasr=={release_version}" in text
        assert f"releases/tag/v{release_version}" in text
        assert "v1.3.26" not in text
        assert "runtime-llamacpp-v0.2.6" in text
        assert "MOSS-Transcribe-Diarize" in text
        assert "https://github.com/modelscope/FunASR/issues/3496" in text
        assert "https://github.com/modelscope/FunASR/issues/3528" in text
        assert "https://github.com/modelscope/FunASR/issues/3479" in text
        assert "https://github.com/huggingface/transformers/pull/46180" in text

    assert "speaker identities" not in docs[0]
    assert "说话人身份识别" not in docs[1]


def test_repository_roadmap_exposes_live_contribution_entry_points():
    docs = [
        (ROOT / "docs/repository_roles.md").read_text(),
        (ROOT / "docs/repository_roles_zh.md").read_text(),
    ]
    live_queries = [
        "is%3Aissue+is%3Aopen+label%3A%22help+wanted%22",
        "is%3Aissue+is%3Aopen+label%3A%22ready+for+PR%22",
    ]

    for text in docs:
        for query in live_queries:
            assert query in text
        assert "needs feedback" in text

    assert "exact commit" in docs[0]
    assert "acceptance evidence" in docs[0]
    assert "exact commit" in docs[1]
    assert "验收证据" in docs[1]

    contributing = (ROOT / "CONTRIBUTING.md").read_text()
    assert "## Find a task" in contributing
    for query in live_queries:
        assert query in contributing
    assert "needs feedback" in contributing


def test_realtime_demo_documents_partial_and_hotword_boundaries():
    text = (
        ROOT
        / "examples/industrial_data_pretraining/fun_asr_nano/docs/realtime_demo.md"
    ).read_text()

    required = [
        "data.sentences.map",
        "data.partial || \"\"",
        "partial_start_ms",
        "--partial-window-sec",
        "不是确定性文本替换",
        "HOTWORDS:Tool,客製化,季會",
        "后处理",
    ]
    for marker in required:
        assert marker in text
