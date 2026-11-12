"""Offline synthetic graphs exercise real frontends, devices and CLI inference.

These fixed weights are test fixtures, not pretrained ASR or quality benchmarks.
Set FUNASR_TEST_DEVICE to a CUDA index for the same tests on real GPU hardware.
"""

import importlib
import json
import os
from pathlib import Path
import shutil
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


@pytest.fixture(scope="module")
def hotword_fixture(tmp_path_factory):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("kaldi_native_fbank")
    directory = tmp_path_factory.mktemp("runtime-hotword")
    tokens = ["<blank>", "<s>", "</s>", "你", "好", "魔", "搭", "<unk>"]

    class EmbedderFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("weight", torch.ones(len(tokens), 512))

        def forward(self, hotword):
            # [num_hotwords, 10] -> [10, num_hotwords, 512], like the exported model_eb
            return torch.nn.functional.embedding(hotword.long(), self.weight).transpose(0, 1)

    class BackboneFixture(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("bias", torch.tensor([0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0]))

        def forward(self, speech, speech_lengths, bias_embed):
            # bias_embed [batch, num_hotwords, 512] is added as zeros; every frame predicts token 3
            hotword_bias = bias_embed.sum(dim=(1, 2), keepdim=True) * 0.0
            logits = speech[..., :1] * 0.0 + hotword_bias + self.bias
            return logits, (speech_lengths // 4).to(torch.int32)

    embedder, backbone = EmbedderFixture().eval(), BackboneFixture().eval()
    hotword = torch.zeros(3, 10, dtype=torch.int32)
    speech = torch.zeros(2, 12, 4)
    lengths = torch.tensor([12, 8], dtype=torch.int32)
    bias_embed = torch.zeros(2, 3, 512)
    torch.jit.trace(embedder, (hotword,)).save(str(directory / "model_eb.torchscript"))
    torch.jit.trace(backbone, (speech, lengths, bias_embed)).save(
        str(directory / "model_bb.torchscript")
    )
    torch.onnx.export(
        embedder,
        (hotword,),
        str(directory / "model_eb.onnx"),
        dynamo=False,
        input_names=["hotword"],
        output_names=["hw_embed"],
        dynamic_axes={"hotword": {0: "num_hotwords"}, "hw_embed": {1: "num_hotwords"}},
        opset_version=17,
    )
    torch.onnx.export(
        backbone,
        (speech, lengths, bias_embed),
        str(directory / "model.onnx"),
        dynamo=False,
        input_names=["speech", "speech_lengths", "bias_embed"],
        output_names=["logits", "token_num"],
        dynamic_axes={
            "speech": {0: "batch_size", 1: "feats_length"},
            "speech_lengths": {0: "batch_size"},
            "bias_embed": {0: "batch_size", 1: "num_hotwords"},
            "logits": {0: "batch_size", 1: "logits_length"},
            "token_num": {0: "batch_size"},
        },
        opset_version=17,
    )
    for name in ("model_eb.onnx", "model.onnx"):
        onnx.checker.check_model(str(directory / name))
    (directory / "tokens.json").write_text(json.dumps(tokens))
    (directory / "config.yaml").write_text(
        json.dumps(
            {
                "frontend_conf": {"n_mels": 4, "lfr_m": 1, "lfr_n": 1, "dither": 0.0},
                "model_conf": {"predictor_bias": 1},
            }
        )
    )
    (directory / "am.mvn").write_text(
        "<AddShift> 4 4\n<LearnRateCoef> 1 [ 0 0 0 0 ]\n"
        "<Rescale> 4 4\n<LearnRateCoef> 1 [ 1 1 1 1 ]\n"
    )
    paths = []
    for index, count in enumerate([4000, 8000, 12000]):
        path = directory / f"hotword {index}.wav"
        signal = (np.sin(np.arange(count) * 2 * np.pi * 440 / 16000) * 3000).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(signal.tobytes())
        paths.append(str(path))
    return directory, paths


@pytest.mark.parametrize("backend", ["torch", "onnx"])
@pytest.mark.parametrize("wrapper", ["ContextualParaformer", "SeacoParaformer"])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_hotword_wrappers_decode_every_batch(
    hotword_fixture, monkeypatch, backend, wrapper, batch_size
):
    directory, paths = hotword_fixture
    location = "libtorch" if backend == "torch" else "onnxruntime"
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python" / location))
    module = importlib.import_module("funasr_" + backend + ".paraformer_bin")
    # CPU only: on CUDA the Torch hotword wrapper moves its inputs, not these CPU-traced modules.
    model = getattr(module, wrapper)(str(directory), batch_size=batch_size, intra_op_num_threads=1)
    hotwords = "魔搭 你好"
    alone = [model(path, hotwords)[0] for path in paths]
    assert [row["preds"][0] for row in alone] == ["你" * 4, "你" * 11, "你" * 17]
    # More files than batch_size: every batch must return its results.
    assert model(paths, hotwords) == alone


@pytest.fixture(scope="module")
def timestamp_fixture(hotword_fixture, tmp_path_factory):
    np = pytest.importorskip("numpy")
    source, _ = hotword_fixture
    directory = tmp_path_factory.mktemp("runtime-timestamp")
    # The graphs only have to load: the timestamp test replaces the acoustic model's session.
    for name, target in [
        ("tokens.json", "tokens.json"),
        ("model_eb.onnx", "model_eb.onnx"),
        ("model.onnx", "model.onnx"),
        ("model_bb.torchscript", "model.torchscript"),
    ]:
        shutil.copy(source / name, directory / target)
    (directory / "config.yaml").write_text(
        json.dumps(
            {
                "frontend_conf": {"n_mels": 4, "lfr_m": 7, "lfr_n": 6, "dither": 0.0},
                "model_conf": {"predictor_bias": 1},
            }
        )
    )
    (directory / "am.mvn").write_text(
        "<AddShift> 28 28\n<LearnRateCoef> 1 [ " + "0 " * 28 + "]\n"
        "<Rescale> 28 28\n<LearnRateCoef> 1 [ " + "1 " * 28 + "]\n"
    )
    paths = []
    for seconds in (1, 3):
        path = directory / f"tone {seconds}s.wav"
        tone = np.sin(np.arange(16000 * seconds) * 2 * np.pi * 440 / 16000) * 3000
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(tone.astype("<i2").tobytes())
        paths.append(str(path))
    return directory, paths


def timestamp_outputs(feats, feats_len):
    """BiCifParaformer-style outputs for 你 好, padded to the longest row like the exported graph.

    Each row's three CIF peaks lie inside its own 3 * feats_len upsampled frames; the
    frames after them are zero padding.
    """
    import numpy as np

    batch = len(feats_len)
    am_scores = np.zeros((batch, 3, 8), dtype=np.float32)
    am_scores[:, [0, 1, 2], [3, 4, 2]] = 1.0  # 你 好 </s>
    us_alphas = np.zeros((batch, 3 * feats.shape[1]), dtype=np.float32)
    us_peaks = np.zeros_like(us_alphas)
    for row, feat_len in enumerate(np.asarray(feats_len).tolist()):
        frames = 3 * feat_len
        us_alphas[row, :frames] = 3.0 / frames
        us_peaks[row, np.linspace(5, frames - 6, 3).round().astype(int)] = 1.0
    return [am_scores, np.full(batch, 3, dtype=np.int32), us_alphas, us_peaks]


@pytest.mark.parametrize(
    "backend, wrapper",
    [
        ("onnx", "Paraformer"),
        ("onnx", "ContextualParaformer"),
        ("onnx", "SeacoParaformer"),
        ("torch", "Paraformer"),
    ],
)
def test_batched_timestamps_ignore_padding(timestamp_fixture, monkeypatch, backend, wrapper):
    torch = pytest.importorskip("torch")
    directory, paths = timestamp_fixture
    location = "libtorch" if backend == "torch" else "onnxruntime"
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python" / location))
    module = importlib.import_module("funasr_" + backend + ".paraformer_bin")
    model = getattr(module, wrapper)(str(directory), batch_size=2, intra_op_num_threads=1)
    if backend == "torch":
        model.ort_infer = lambda feats, feats_len: tuple(
            torch.from_numpy(output) for output in timestamp_outputs(feats, feats_len)
        )
    elif wrapper == "Paraformer":
        model.ort_infer = lambda inputs: timestamp_outputs(*inputs)
    else:
        model.ort_infer_bb = lambda inputs: timestamp_outputs(*inputs[:2])
    hotwords = () if wrapper == "Paraformer" else ("魔搭",)
    alone = [model(path, *hotwords)[0] for path in paths]
    # The 1 s tone ends at 1000 ms, the 3 s tone at 3000 ms.
    assert [row["timestamp"] for row in alone] == [
        [[70, 470], [470, 945]],
        [[70, 670], [1450, 2925]],
    ]
    # Batched with the 3 s tone, the 1 s tone must not take on the padded frames.
    assert model(paths, *hotwords) == alone


@pytest.mark.parametrize("extension", ["flac", "ogg", "mp3"])
def test_onnx_wrappers_read_non_wav_paths(native_fixture, tmp_path, monkeypatch, extension):
    np = pytest.importorskip("numpy")
    librosa = pytest.importorskip("librosa")
    soundfile = pytest.importorskip("soundfile")
    if extension.upper() not in soundfile.available_formats():
        pytest.skip(f"libsndfile {soundfile.__libsndfile_version__} cannot write {extension}")
    directory, _, _ = native_fixture
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python/onnxruntime"))
    funasr_onnx = importlib.import_module("funasr_onnx")
    online = importlib.import_module("funasr_onnx.paraformer_online_bin")
    audio = tmp_path / f"talk.{extension}"
    tone = np.sin(np.arange(48000) * 2 * np.pi * 440 / 16000).astype(np.float32) * 0.1
    soundfile.write(str(audio), tone, 16000)
    # A different recording with the same stem must not be read in its place.
    soundfile.write(str(tmp_path / "talk.wav"), np.zeros(16000, dtype=np.float32), 16000)
    files = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    expected = librosa.load(str(audio), sr=16000)[0]
    wrappers = [
        funasr_onnx.Paraformer,
        funasr_onnx.ContextualParaformer,
        funasr_onnx.SeacoParaformer,
        online.Paraformer,
        funasr_onnx.Fsmn_vad,
        funasr_onnx.SenseVoiceSmall,
    ]
    for wrapper in wrappers:
        # load_data uses no model state, so skip __init__ and its model files.
        waveforms = wrapper.__new__(wrapper).load_data(str(audio), 16000)
        assert len(waveforms) == 1
        name = f"{wrapper.__module__}.{wrapper.__name__}"
        np.testing.assert_array_equal(waveforms[0], expected, err_msg=name)
    model = funasr_onnx.Paraformer(str(directory), device_id=DEVICE, intra_op_num_threads=1)
    assert model(str(audio)) == model(expected)
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == files


@pytest.mark.parametrize("backend", ["torch", "onnx"])
@pytest.mark.parametrize("wrapper", ["ContextualParaformer", "SeacoParaformer"])
def test_hotword_wrappers_skip_empty_and_cut_long_hotwords(
    hotword_fixture, monkeypatch, backend, wrapper
):
    directory, paths = hotword_fixture
    location = "libtorch" if backend == "torch" else "onnxruntime"
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python" / location))
    module = importlib.import_module("funasr_" + backend + ".paraformer_bin")
    model = getattr(module, wrapper)(str(directory), intra_op_num_threads=1)
    # Like the C++ runtime: skip empty hotwords, and keep the first 10 characters of a longer
    # one (the exported model_eb takes 10 per hotword). Ids: <s> 1, 你 3, 好 4, 魔 5, 搭 6.
    long_hotwords = " 魔搭  你好魔搭你好魔搭你好魔搭 "
    matrix, positions = model.proc_hotword(long_hotwords)
    assert matrix.tolist() == [
        [5, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        [3, 4, 5, 6, 3, 4, 5, 6, 3, 4],
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert positions.tolist() == [1, 9, 0]
    matrix, positions = model.proc_hotword("")
    assert matrix.tolist() == [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    assert positions.tolist() == [0]
    # Both decode every file; the 12-character hotword used to stop the call with a ValueError.
    for hotwords in (long_hotwords, ""):
        result = model(paths, hotwords)
        assert [row["preds"][0] for row in result] == ["你" * 4, "你" * 11, "你" * 17]
