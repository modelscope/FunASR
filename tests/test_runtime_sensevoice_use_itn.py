"""SenseVoiceSmall in funasr_onnx and funasr_torch sends the textnorm query that use_itn selects.

The synthetic graph returns its language and textnorm queries as CTC tokens, and the fixture
SentencePiece model names those ids, so each result shows the queries the wrapper sent.
The graph and the tokenizer are test fixtures, not a pretrained SenseVoice model.
"""

import importlib
import io
import json
from pathlib import Path
import wave

import pytest


ROOT = Path(__file__).resolve().parents[1]
# SentencePiece keeps ids 0-2 for <unk>, <s> and </s>, so the user-defined symbols take ids
# 3-15, the query ids of the wrappers' lid_dict and textnorm_dict.
QUERY_TOKENS = {4: "<|en|>", 14: "<|withitn|>", 15: "<|woitn|>"}


@pytest.fixture(scope="module")
def sensevoice_fixture(tmp_path_factory):
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    onnx = pytest.importorskip("onnx")
    spm = pytest.importorskip("sentencepiece")
    pytest.importorskip("onnxruntime")
    pytest.importorskip("kaldi_native_fbank")
    directory = tmp_path_factory.mktemp("runtime-sensevoice")

    class QueryEcho(torch.nn.Module):
        def forward(self, speech, speech_lengths, language, textnorm):
            # Three frames: the language query, blank and the textnorm query.
            queries = torch.stack([language, torch.zeros_like(language), textnorm], dim=1)
            logits = torch.nn.functional.one_hot(queries.long(), 16).float()
            lengths = torch.full_like(speech_lengths, 3, dtype=torch.int32)
            # 0 * speech keeps speech as an input of the ONNX graph.
            return logits + 0.0 * speech.mean(), lengths

    model = QueryEcho().eval()
    inputs = (
        torch.zeros(2, 10, 4),
        torch.tensor([10, 8], dtype=torch.int32),
        torch.tensor([0, 4], dtype=torch.int32),
        torch.tensor([14, 15], dtype=torch.int32),
    )
    torch.jit.trace(model, inputs).save(str(directory / "model.torchscript"))
    torch.onnx.export(
        model,
        inputs,
        str(directory / "model.onnx"),
        dynamo=False,
        input_names=["speech", "speech_lengths", "language", "textnorm"],
        output_names=["ctc_logits", "encoder_out_lens"],
        dynamic_axes={
            "speech": {0: "B", 1: "T"},
            "speech_lengths": {0: "B"},
            "language": {0: "B"},
            "textnorm": {0: "B"},
            "ctc_logits": {0: "B"},
            "encoder_out_lens": {0: "B"},
        },
        opset_version=17,
    )
    onnx.checker.check_model(str(directory / "model.onnx"))
    bpemodel = io.BytesIO()
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(["abc"]),
        model_writer=bpemodel,
        model_type="char",
        vocab_size=20,
        user_defined_symbols=[QUERY_TOKENS.get(i, f"<|query{i}|>") for i in range(3, 16)],
        minloglevel=2,
    )
    (directory / "chn_jpn_yue_eng_ko_spectok.bpe.model").write_bytes(bpemodel.getvalue())
    (directory / "config.yaml").write_text(
        json.dumps({"frontend_conf": {"n_mels": 4, "lfr_m": 1, "lfr_n": 1, "dither": 0.0}})
    )
    (directory / "am.mvn").write_text(
        "<AddShift> 4 4\n<LearnRateCoef> 1 [ 0 0 0 0 ]\n"
        "<Rescale> 4 4\n<LearnRateCoef> 1 [ 1 1 1 1 ]\n"
    )
    paths = []
    for index in range(2):
        path = directory / f"audio {index}.wav"
        signal = (np.sin(np.arange(8000) * 2 * np.pi * 440 / 16000) * 3000).astype("<i2")
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16000)
            output.writeframes(signal.tobytes())
        paths.append(str(path))
    return directory, paths


@pytest.mark.parametrize("backend", ["onnx", "torch"])
@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({}, "<|woitn|>"),
        ({"use_itn": False}, "<|woitn|>"),
        ({"use_itn": True}, "<|withitn|>"),
        ({"textnorm": "withitn"}, "<|withitn|>"),
        ({"textnorm": "woitn", "use_itn": True}, "<|woitn|>"),
        ({"language": "en", "use_itn": True}, "<|en|><|withitn|>"),
    ],
    ids=[
        "default",
        "use_itn_false",
        "use_itn_true",
        "textnorm_withitn",
        "textnorm_wins",
        "language_en",
    ],
)
def test_sensevoice_use_itn_selects_textnorm(
    sensevoice_fixture, monkeypatch, backend, batch_size, kwargs, expected
):
    directory, paths = sensevoice_fixture
    location = "libtorch" if backend == "torch" else "onnxruntime"
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python" / location))
    module = importlib.import_module("funasr_" + backend)
    model = module.SenseVoiceSmall(str(directory), batch_size=batch_size, intra_op_num_threads=1)
    # Two files, so with batch_size=1 the second file is decoded in a batch of its own.
    assert model(paths, **kwargs) == [expected, expected]


@pytest.mark.parametrize("backend", ["onnx", "torch"])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize(
    "language_kind, textnorm_kind",
    [
        ("scalar", "list"),
        ("scalar", "file"),
        ("list", "scalar"),
        ("file", "scalar"),
        ("list", "list"),
        ("file", "file"),
    ],
)
def test_sensevoice_mixed_scalar_and_per_utterance_tags(
    sensevoice_fixture, monkeypatch, tmp_path, backend, batch_size, language_kind, textnorm_kind
):
    directory, paths = sensevoice_fixture
    location = "libtorch" if backend == "torch" else "onnxruntime"
    monkeypatch.syspath_prepend(str(ROOT / "runtime/python" / location))
    module = importlib.import_module("funasr_" + backend)
    model = module.SenseVoiceSmall(str(directory), batch_size=batch_size, intra_op_num_threads=1)
    kwargs = {"use_itn": True}
    for name, kind, values in [
        ("language", language_kind, ["en", "zh", "ko"]),
        ("textnorm", textnorm_kind, ["withitn", "woitn", "woitn"]),
    ]:
        if kind == "list":
            kwargs[name] = values
        elif kind == "file":
            tag_file = tmp_path / (name + ".txt")
            tag_file.write_text("\n".join(values) + "\n")
            kwargs[name] = str(tag_file)
    if language_kind == "scalar":
        expected = ["<|withitn|>", "<|woitn|>", "<|woitn|>"]
    elif textnorm_kind == "scalar":
        expected = ["<|en|><|withitn|>", "<|query3|><|withitn|>", "<|query12|><|withitn|>"]
    else:
        expected = ["<|en|><|withitn|>", "<|query3|><|woitn|>", "<|query12|><|woitn|>"]
    # Three routing entries also exercise the final partial batch at batch_size=2.
    assert model(paths + [paths[0]], **kwargs) == expected
