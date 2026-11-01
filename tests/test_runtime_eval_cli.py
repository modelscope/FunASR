"""CLI boundary tests; backend stand-ins do not claim native ASR/GPU coverage."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "runtime/python/utils"
NAMES = ["test_cer.py", "test_rtf.py", "test_rtf_gpu.py"]


def invoke(name, *args, env=None):
    return subprocess.run(
        [sys.executable, "-S", str(SCRIPTS / name), *map(str, args)],
        text=True,
        capture_output=True,
        timeout=30,
        env=env,
    )


@pytest.mark.parametrize("name", NAMES)
def test_import_without_runtime_dependencies_or_cli_arguments(name):
    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "import runpy; runpy.run_path(" + repr(str(SCRIPTS / name)) + ")",
        ],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert not result.stdout


@pytest.mark.parametrize("name", NAMES)
def test_help_without_runtime_dependencies(name):
    result = invoke(name, "--help")
    assert result.returncode == 0, result.stderr
    assert "--model_dir" in result.stdout and "--wav_file" in result.stdout


@pytest.mark.parametrize("name", NAMES)
def test_missing_required_arguments_are_parser_errors(name):
    result = invoke(name)
    assert result.returncode == 2
    assert "--model_dir" in result.stderr and "--wav_file" in result.stderr


def helper():
    path = SCRIPTS / "runtime_eval.py"
    assert path.exists(), "Shared runtime CLI boundary is missing"
    spec = importlib.util.spec_from_file_location("runtime_eval_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wav_list_preserves_spaces_and_tabs(tmp_path):
    scp = tmp_path / "wav.scp"
    scp.write_text("first\t/some path/a.wav\n\nsecond  /b path/b.wav\n")
    assert helper().read_wav_scp(scp) == [
        ("first", "/some path/a.wav"),
        ("second", "/b path/b.wav"),
    ]


@pytest.mark.parametrize("content", ["", "\n \n", "missing-path\n"])
def test_invalid_wav_list_is_rejected(tmp_path, content):
    scp = tmp_path / "wav.scp"
    scp.write_text(content)
    with pytest.raises(ValueError):
        helper().read_wav_scp(scp)


@pytest.fixture
def backends(tmp_path):
    # Only model execution and media decoding are external here. Native device
    # placement and inference have separate real-backend validation.
    code = """
import os
class Session:
    def get_providers(self):
        return ["CPUExecutionProvider"] if os.environ.get("CPU_FALLBACK") else ["CUDAExecutionProvider", "CPUExecutionProvider"]
class Engine:
    session = Session()
class Paraformer:
    def __init__(self, model_dir, batch_size=1, device_id="-1", **kwargs):
        assert str(device_id) == os.environ["EXPECTED_DEVICE"], device_id
        assert batch_size == int(os.environ.get("EXPECTED_BATCH", "1"))
        self.ort_infer = Engine()
    def __call__(self, paths):
        if os.environ.get("EMPTY_RESULT"):
            return []
        if isinstance(paths, str):
            paths = [paths]
        assert all(path.endswith("audio file.wav") for path in paths), paths
        return [{"preds": ("fixture text", ["fixture", "text"])} for _ in paths]
"""
    packages = tmp_path / "packages"
    packages.mkdir()
    for name in ["funasr_onnx", "funasr_torch"]:
        target = packages / name
        target.mkdir()
        (target / "__init__.py").write_text(code)
    (packages / "librosa.py").write_text("def load(path, sr=None):\n    return [0] * 16000, 8000\n")
    (packages / "torch.py").write_text(
        "class CUDA:\n    def synchronize(self, device=None): pass\n" "cuda = CUDA()\n"
    )
    scp = tmp_path / "wav.scp"
    scp.write_text("a /some path/audio file.wav\nb\t/other/audio file.wav\n")
    env = dict(os.environ, PYTHONPATH=str(packages), EXPECTED_DEVICE="-1")
    return packages, scp, env


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("backend", ["onnx", "torch"])
def test_cli_uses_only_selected_backend_and_passes_device(tmp_path, backends, name, backend):
    packages, scp, env = backends
    other = "funasr_torch" if backend == "onnx" else "funasr_onnx"
    (packages / other / "__init__.py").write_text('raise RuntimeError("wrong backend imported")\n')
    args = ["--model_dir", "local-model", "--wav_file", scp, "--backend", backend]
    if name == "test_cer.py":
        args += ["--output_dir", tmp_path / "output"]
    else:
        args += ["--warmup", "1"]
    if name == "test_rtf_gpu.py":
        env.update(EXPECTED_DEVICE="2", EXPECTED_BATCH="2")
        args += ["--device_id", "2", "--batch_size", "2"]
    result = invoke(name, *args, env=env)
    assert result.returncode == 0, result.stderr
    if name == "test_cer.py":
        assert (tmp_path / "output/text").read_text() == "a fixture text\nb fixture text\n"
        assert (tmp_path / "output/token").read_text() == "a fixture text\nb fixture text\n"
    else:
        assert "total_time_wav_ms: 4000" in result.stdout
        assert "num: 1," in result.stdout
        assert "total_rtf:" in result.stdout


def test_gpu_cli_defaults_to_device_zero(backends):
    _, scp, env = backends
    env["EXPECTED_DEVICE"] = "0"
    result = invoke(
        "test_rtf_gpu.py", "--model_dir", "local-model", "--wav_file", scp, "--warmup", "0", env=env
    )
    assert result.returncode == 0, result.stderr


def test_gpu_cli_refuses_silent_onnx_cpu_fallback(backends):
    _, scp, env = backends
    env.update(EXPECTED_DEVICE="0", CPU_FALLBACK="1")
    result = invoke(
        "test_rtf_gpu.py", "--model_dir", "local-model", "--wav_file", scp, "--warmup", "0", env=env
    )
    assert result.returncode != 0
    assert "CUDAExecutionProvider" in result.stderr
    assert "total_rtf:" not in result.stdout


@pytest.mark.parametrize(
    "option,value",
    [("--batch_size", "0"), ("--warmup", "-1"), ("--device_id", "-1"), ("--backend", "bogus")],
)
def test_gpu_cli_rejects_invalid_arguments(option, value, backends):
    _, scp, env = backends
    result = invoke(
        "test_rtf_gpu.py", "--model_dir", "local-model", "--wav_file", scp, option, value, env=env
    )
    assert result.returncode == 2, result.stderr


@pytest.mark.parametrize("name", ["test_rtf.py", "test_rtf_gpu.py"])
def test_incomplete_inference_is_not_a_successful_benchmark(name, backends):
    _, scp, env = backends
    env.update(EMPTY_RESULT="1", EXPECTED_DEVICE="0" if "gpu" in name else "-1")
    result = invoke(name, "--model_dir", "local-model", "--wav_file", scp, "--warmup", "0", env=env)
    assert result.returncode != 0
    assert "inference returned" in result.stderr
    assert "total_rtf:" not in result.stdout


def test_zero_duration_is_rejected(backends):
    packages, scp, env = backends
    (packages / "librosa.py").write_text("def load(path, sr=None): return [], 16000\n")
    result = invoke(
        "test_rtf.py", "--model_dir", "local-model", "--wav_file", scp, "--warmup", "0", env=env
    )
    assert result.returncode != 0
    assert "duration" in result.stderr
    assert "total_rtf:" not in result.stdout


@pytest.mark.parametrize(
    "value,wanted",
    [("t", True), ("f", False), ("True", True), ("False", False), ("1", True), ("0", False)],
)
def test_quantize_retains_legacy_boolean_spellings(value, wanted):
    args = helper().parse_args(
        ["--model_dir", "local-model", "--wav_file", "wav.scp", "--quantize", value]
    )
    assert args.quantize is wanted


def test_rtf_uses_measured_duration_for_warmup_and_total(backends, monkeypatch, capsys):
    _, scp, _ = backends
    module = helper()
    ticks = iter([0.0, 1.0, 2.0, 3.0])
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    monkeypatch.setitem(
        sys.modules, "librosa", SimpleNamespace(load=lambda *a, **k: ([0] * 16000, 8000))
    )
    monkeypatch.setattr(
        module,
        "load_model",
        lambda args: lambda paths: [{}] * (1 if isinstance(paths, str) else len(paths)),
    )
    args = module.parse_args(
        ["--model_dir", "local-model", "--wav_file", str(scp), "--warmup", "1"]
    )
    module.benchmark(args, batched=True)
    output = capsys.readouterr().out
    assert "avg: 1.0, rtf: 0.5" in output
    assert "total_rtf: 0.25" in output


@pytest.mark.parametrize("name", NAMES)
def test_module_mode_help(name):
    result = subprocess.run(
        [sys.executable, "-S", "-m", "runtime.python.utils." + Path(name).stem, "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--model_dir" in result.stdout
