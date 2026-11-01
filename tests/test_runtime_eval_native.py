"""Offline synthetic graphs exercise real frontends, devices and CLI inference.

These fixed weights are test fixtures, not pretrained ASR or quality benchmarks.
Set FUNASR_TEST_DEVICE to a CUDA index for the same tests on real GPU hardware.
"""

import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
DEVICE = os.environ.get("FUNASR_TEST_DEVICE", "-1")


@pytest.fixture(scope="module")
def native_fixture(tmp_path_factory):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("kaldi_native_fbank")
    directory = tmp_path_factory.mktemp("runtime-native")

    class AcousticFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.arange(20).reshape(4, 5).float() * 0.00001)
            self.register_buffer("bias", torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0]))

        def forward(self, speech, lengths):
            return speech[:, :2] @ self.weight + self.bias, torch.full_like(lengths, 2)

    model = AcousticFixture().eval()
    speech = torch.zeros(2, 10, 4)
    lengths = torch.tensor([10, 8], dtype=torch.int32)
    torch.jit.trace(model, (speech, lengths)).save(str(directory / "model.torchscript"))
    torch.onnx.export(
        model,
        (speech, lengths),
        str(directory / "model.onnx"),
        dynamo=False,
        input_names=["speech", "speech_lengths"],
        output_names=["logits", "lengths"],
        dynamic_axes={
            "speech": {0: "B", 1: "T"},
            "speech_lengths": {0: "B"},
            "logits": {0: "B"},
            "lengths": {0: "B"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(str(directory / "model.onnx"))
    (directory / "tokens.json").write_text(
        json.dumps(["<blank>", "<s>", "</s>", "fixture", "unused"])
    )
    (directory / "config.yaml").write_text(
        json.dumps(
            {
                "frontend_conf": {"n_mels": 4, "lfr_m": 1, "lfr_n": 1, "dither": 0.0},
                "model_conf": {"predictor_bias": 0},
            }
        )
    )
    (directory / "am.mvn").write_text(
        "<AddShift> 4 4\n<LearnRateCoef> 1 [ 0 0 0 0 ]\n"
        "<Rescale> 4 4\n<LearnRateCoef> 1 [ 1 1 1 1 ]\n"
    )
    paths = []
    for index, count in enumerate([16000, 8000]):
        path = directory / f"audio {index}.wav"
        signal = (
            np.zeros(count, dtype="<i2")
            if index == 0
            else (np.sin(np.arange(count) * 2 * np.pi * 440 / 16000) * 3000).astype("<i2")
        )
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(signal.tobytes())
        paths.append(path)
    scp = directory / "wav.scp"
    scp.write_text("".join(f"utt{i}\t{p}\n" for i, p in enumerate(paths)))
    return directory, paths, scp


@pytest.mark.parametrize("backend", ["torch", "onnx"])
def test_real_backend_uses_requested_device_and_decodes(native_fixture, monkeypatch, backend):
    directory, paths, _ = native_fixture
    location = "libtorch" if backend == "torch" else "onnxruntime"
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python" / location))
    module = importlib.import_module("funasr_" + backend)
    model = module.Paraformer(
        str(directory), batch_size=2, device_id=DEVICE, intra_op_num_threads=1
    )
    if backend == "torch":
        expected = "cpu" if DEVICE == "-1" else "cuda:" + DEVICE
        assert {str(item.device) for item in model.ort_infer.buffers()} == {expected}
    else:
        provider = "CPUExecutionProvider" if DEVICE == "-1" else "CUDAExecutionProvider"
        assert model.ort_infer.session.get_providers()[0] == provider
    result = model([str(p) for p in paths])
    assert len(result) == 2
    assert [row["preds"][0] for row in result] == ["fixture fixture", "fixture fixture"]
    print(json.dumps({"backend": backend, "device": DEVICE, "raw": result}))


@pytest.mark.parametrize("backend", ["torch", "onnx"])
@pytest.mark.parametrize("script", ["test_cer.py", "test_rtf.py"])
def test_real_cli_outputs_and_audio_duration(native_fixture, tmp_path, backend, script):
    directory, _, scp = native_fixture
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "runtime/python/libtorch"),
            str(ROOT / "runtime/python/onnxruntime"),
            env.get("PYTHONPATH", ""),
        ]
    )
    command = [
        sys.executable,
        str(ROOT / "runtime/python/utils" / script),
        "--model_dir",
        str(directory),
        "--wav_file",
        str(scp),
        "--backend",
        backend,
        "--device_id",
        DEVICE,
    ]
    if script == "test_cer.py":
        command += ["--output_dir", str(tmp_path / "out")]
    else:
        command += ["--warmup", "1"]
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    if script == "test_cer.py":
        assert (tmp_path / "out/text").read_text() == "utt0 fixture fixture\nutt1 fixture fixture\n"
        assert (
            tmp_path / "out/token"
        ).read_text() == "utt0 fixture fixture\nutt1 fixture fixture\n"
    else:
        assert "total_time_wav_ms: 1500" in result.stdout
        assert "total_rtf:" in result.stdout
    print(
        json.dumps(
            {
                "backend": backend,
                "script": script,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "device": DEVICE,
            }
        )
    )
