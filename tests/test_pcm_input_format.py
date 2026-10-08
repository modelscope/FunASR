"""Explicit PCM routing tests; no model weights or inference are required."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from funasr.auto.auto_model import AutoModel, prepare_data_iterator
from funasr.utils import load_utils


def _ambiguous_pcm():
    data = bytearray(6400)
    for offset in (0, 417):
        data[offset : offset + 4] = b"\xff\xfb\x90\x00"
    return bytes(data)


def _expected(data):
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def _forbid_decode(*args, **kwargs):
    raise AssertionError("container decoding must not run for explicit PCM")


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00\x80\xff\xff\x00\x00\x01\x00\xff\x7f",
        b"RIFF\x00\x00\x00\x00WAVE\x00\x00",
        b"ID3\x00",
        _ambiguous_pcm(),
    ],
    ids=["empty", "limits", "wav-like", "id3-like", "mpeg-like"],
)
def test_explicit_pcm_never_probes_or_decodes(data, monkeypatch):
    monkeypatch.setattr(load_utils, "_is_audio_container", _forbid_decode)
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    actual = load_utils.load_bytes(data, input_format="pcm_s16le")
    assert actual.dtype == np.float32
    np.testing.assert_array_equal(actual, _expected(data))


def test_ambiguous_pcm_really_matches_automatic_container_detection(monkeypatch):
    data = _ambiguous_pcm()
    assert load_utils._is_audio_container(data)
    decoded = np.array([0.25, -0.25], dtype=np.float32)
    monkeypatch.setattr(
        load_utils, "load_audio_text_image_video", lambda *a, **kw: decoded
    )
    np.testing.assert_array_equal(load_utils.load_bytes(data), decoded)


@pytest.mark.parametrize("value", ["pcm", "wav", None])
def test_invalid_byte_format_is_rejected(value):
    with pytest.raises(ValueError, match="input_format"):
        load_utils.load_bytes(b"\x00\x00", input_format=value)


def test_partial_pcm_sample_is_rejected_without_probing(monkeypatch):
    monkeypatch.setattr(load_utils, "_is_audio_container", _forbid_decode)
    with pytest.raises(ValueError, match="complete 2-byte samples"):
        load_utils.load_bytes(b"\x00", input_format="pcm_s16le")


@pytest.mark.parametrize("batch", [False, True])
def test_iterator_decodes_explicit_pcm_and_preserves_keys(batch, monkeypatch):
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    data = _ambiguous_pcm()
    value = [data, data] if batch else data
    keys, samples = prepare_data_iterator(
        value, input_format="pcm_s16le", key="session"
    )
    assert keys == ["session"] * (2 if batch else 1)
    for sample in samples:
        np.testing.assert_array_equal(sample, _expected(data))


def test_multimodal_iterator_forwards_byte_format(monkeypatch):
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    data = _ambiguous_pcm()
    keys, values = prepare_data_iterator(
        ([data], ["context"]), data_type=("sound", "text"),
        input_format="pcm_s16le"
    )
    assert len(keys) == len(values) == 1
    np.testing.assert_array_equal(values[0][0], _expected(data))
    assert values[0][1] == "context"


def test_default_batch_behavior_is_unchanged():
    data = [_ambiguous_pcm()]
    _, values = prepare_data_iterator(data)
    assert values is data


def test_scp_filelist_skips_blank_lines(tmp_path):
    list_path = tmp_path / "wav.scp"
    list_path.write_text("utt1 /a.wav\n\n \t \nutt2 /b.wav\n", encoding="utf-8")
    keys, values = prepare_data_iterator(str(list_path))
    assert keys == ["utt1", "utt2"]
    assert values == ["/a.wav", "/b.wav"]


def test_jsonl_filelist_skips_blank_lines(tmp_path):
    list_path = tmp_path / "data.jsonl"
    list_path.write_text(
        '{"key": "k1", "source": "/a.wav"}\n\n \t \n{"key": "k2", "source": "/b.wav"}\n',
        encoding="utf-8",
    )
    keys, values = prepare_data_iterator(str(list_path))
    assert keys == ["k1", "k2"]
    assert values == ["/a.wav", "/b.wav"]


class _RecordingModel(torch.nn.Module):
    def __init__(self, vad=False):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.vad = vad
        self.segments = []
        self.calls = []

    def inference(self, data_in, key, **kwargs):
        self.calls.append((data_in, kwargs))
        if self.vad:
            results = [{"key": k, "value": self.segments} for k in key]
        else:
            results = [{"key": k, "text": "ok"} for k in key]
        return results, {"batch_data_time": 0.2}


def _wrapper(vad=False, default_format=None):
    wrapper = AutoModel.__new__(AutoModel)
    wrapper.model = _RecordingModel()
    wrapper.vad_model = _RecordingModel(vad=True) if vad else None
    wrapper.punc_model = None
    wrapper.spk_model = None
    wrapper.kwargs = {
        "device": "cpu", "disable_pbar": True, "model": "pcm-routing-probe",
        "frontend": SimpleNamespace(fs=16000), "ncpu": torch.get_num_threads(),
    }
    if default_format is not None:
        wrapper.kwargs["input_format"] = default_format
    wrapper.vad_kwargs = {"device": "cpu", "disable_pbar": True}
    wrapper._store_base_configs()
    return wrapper


@pytest.mark.parametrize("vad", [False, True])
@pytest.mark.parametrize("at_construction", [False, True])
def test_generate_forwards_pcm_format_without_model_download(
    vad, at_construction, monkeypatch
):
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    wrapper = _wrapper(vad, "pcm_s16le" if at_construction else None)
    options = {} if at_construction else {"input_format": "pcm_s16le"}
    data = _ambiguous_pcm()
    result = wrapper.generate(input=data, fs=16000, **options)
    recorder = wrapper.vad_model if vad else wrapper.model
    np.testing.assert_array_equal(recorder.calls[0][0][0], _expected(data))
    assert result[0]["text"] == ("" if vad else "ok")


def test_generate_pcm_flag_does_not_leak_to_next_call(monkeypatch):
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    wrapper = _wrapper()
    data = _ambiguous_pcm()
    wrapper.generate(input=data, input_format="pcm_s16le")
    with pytest.raises(RuntimeError, match="container-formatted"):
        wrapper.generate(input=data)


def test_generate_vad_segments_reach_asr_with_same_samples(monkeypatch):
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    wrapper = _wrapper(vad=True)
    wrapper.vad_model.segments = [[0, 200]]
    data = _ambiguous_pcm()
    result = wrapper.generate(input=data, input_format="pcm_s16le", fs=16000)
    assert result[0]["text"] == "ok"
    np.testing.assert_array_equal(wrapper.model.calls[0][0][0], _expected(data))


def test_streaming_options_are_preserved(monkeypatch):
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    wrapper = _wrapper()
    wrapper.generate(
        input=_ambiguous_pcm(), input_format="pcm_s16le",
        fs=8000, cache={"marker": 7}, is_final=True, chunk_size=200,
    )
    samples, options = wrapper.model.calls[0]
    assert len(samples[0]) == 3200
    assert options["fs"] == 8000
    assert options["cache"] == {"marker": 7}
    assert options["is_final"] is True
    assert options["chunk_size"] == 200


@pytest.mark.parametrize("at_construction", [False, True])
def test_export_preparation_forwards_pcm_format(at_construction, monkeypatch):
    import importlib

    auto_module = importlib.import_module("funasr.auto.auto_model")
    monkeypatch.setattr(load_utils, "load_audio_text_image_video", _forbid_decode)
    wrapper = _wrapper(default_format="pcm_s16le" if at_construction else None)
    data = _ambiguous_pcm()

    def capture_export(model, data_in, **kwargs):
        np.testing.assert_array_equal(data_in[0], _expected(data))
        return "prepared"

    monkeypatch.setattr(auto_module.export_utils, "export", capture_export)
    options = {} if at_construction else {"input_format": "pcm_s16le"}
    assert wrapper.export(input=data, **options) == "prepared"


@pytest.mark.parametrize("source_rate", [8000, 48000])
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("at_construction", [False, True])
@pytest.mark.parametrize("input_kind", ["array", "pcm"])
def test_vad_source_rates_are_not_reapplied_to_resampled_segments(
    source_rate, batch, at_construction, input_kind, monkeypatch
):
    count = source_rate // 5
    pcm = np.round(
        np.sin(2 * np.pi * 440 * np.arange(count) / source_rate) * 12000
    ).astype("<i2").tobytes()
    waveform = _expected(pcm)
    expected = load_utils.load_audio_text_image_video(
        waveform, fs=16000, audio_fs=source_rate
    )
    monkeypatch.setattr(load_utils, "_is_audio_container", _forbid_decode)
    wrapper = _wrapper(vad=True)
    wrapper.vad_model.segments = [[0, 200]]
    if at_construction:
        wrapper.kwargs["fs"] = source_rate
        wrapper._store_base_configs()
    options = {} if at_construction else {"fs": source_rate}
    if input_kind == "pcm":
        options["input_format"] = "pcm_s16le"
    value = pcm if input_kind == "pcm" else waveform
    results = wrapper.generate(input=[value, value] if batch else value, **options)
    assert len(results) == (2 if batch else 1)
    for _, vad_options in wrapper.vad_model.calls:
        assert vad_options["fs"] == source_rate
    for samples, asr_options in wrapper.model.calls:
        assert asr_options["fs"] == 16000
        np.testing.assert_allclose(samples[0], expected, atol=1e-6, rtol=0)
        loaded = load_utils.load_audio_text_image_video(
            samples[0], fs=16000, audio_fs=asr_options["fs"]
        )
        assert len(loaded) == 3200
        np.testing.assert_allclose(loaded, expected, atol=1e-6, rtol=0)


@pytest.mark.skipif(not load_utils.is_ffmpeg_installed(), reason="ffmpeg is required")
@pytest.mark.parametrize("source_rate", [8000, 48000])
@pytest.mark.parametrize("batch", [False, True])
def test_vad_ffmpeg_fallback_preserves_resampled_audio(
    source_rate, batch, tmp_path, monkeypatch
):
    import soundfile as sf

    samples = np.round(
        np.sin(2 * np.pi * 440 * np.arange(source_rate) / source_rate) * 12000
    ).astype("<i2")
    path = tmp_path / "audio.wav"
    sf.write(path, samples, source_rate, subtype="PCM_16")
    expected = load_utils._load_audio_ffmpeg(str(path), sr=16000)

    def decoder_unavailable(*args, **kwargs):
        raise RuntimeError("decoder unavailable")

    monkeypatch.setattr(load_utils.torchaudio, "load", decoder_unavailable)
    monkeypatch.setattr(sf, "read", decoder_unavailable)
    wrapper = _wrapper(vad=True)
    wrapper.vad_model.segments = [[0, 1000]]
    value = [str(path), str(path)] if batch else str(path)

    results = wrapper.generate(input=value, fs=source_rate)

    assert len(results) == (2 if batch else 1)
    for _, vad_options in wrapper.vad_model.calls:
        assert vad_options["fs"] == source_rate
    for segments, asr_options in wrapper.model.calls:
        assert asr_options["fs"] == 16000
        assert len(segments[0]) == 16000
        np.testing.assert_array_equal(segments[0], expected)


class _RecordingSpeaker(torch.nn.Module):
    """Use CAMPPlus preprocessing with a stand-in embedding network."""

    def __init__(self, **kwargs):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def inference(self, data_in, key, **kwargs):
        from funasr.models.campplus.model import CAMPPlus

        self.calls.append(kwargs)
        return CAMPPlus.inference(self, data_in, key=key, **kwargs)

    def forward(self, features):
        return torch.ones(features.shape[0], 4)


@pytest.mark.parametrize(
    "base_rate,call_rate,speaker_rate,source_rate",
    [
        (None, None, None, 16000),
        (8000, None, None, 8000),
        (None, 8000, None, 8000),
        (None, 48000, None, 48000),
        (8000, 48000, 8000, 48000),
        (48000, 8000, 48000, 8000),
        (16000, None, 48000, 16000),
    ],
)
@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("input_kind", ["array", "pcm"])
def test_vad_speaker_uses_resampled_rate(
    base_rate, call_rate, speaker_rate, source_rate, batch, input_kind, monkeypatch
):
    from funasr.models.campplus import model as campplus_model

    feature_inputs = []

    def capture_features(audio):
        feature_inputs.extend(sample.numpy().copy() for sample in audio)
        return torch.zeros(len(audio), 20, 80), [20] * len(audio), [len(a) for a in audio]

    monkeypatch.setattr(campplus_model, "extract_feature", capture_features)
    wrapper = _wrapper(vad=True)
    wrapper.vad_model.segments = [[0, 200]]
    wrapper.spk_model = _RecordingSpeaker()
    wrapper.spk_kwargs = {"device": "cpu"}
    wrapper.kwargs["return_spk_res"] = False
    if base_rate is not None:
        wrapper.kwargs["fs"] = base_rate
    if speaker_rate is not None:
        wrapper.spk_kwargs["fs"] = speaker_rate
    wrapper._store_base_configs()

    pcm = np.round(
        np.sin(2 * np.pi * 440 * np.arange(source_rate // 5) / source_rate) * 12000
    ).astype("<i2").tobytes()
    waveform = _expected(pcm)
    resampled = load_utils.load_audio_text_image_video(
        waveform, fs=16000, audio_fs=source_rate
    )
    # sv_chunk pads the already-resampled 200 ms segment to 1.5 seconds.
    expected = np.pad(resampled, (0, 24000 - len(resampled)))
    options = {} if call_rate is None else {"fs": call_rate}
    if input_kind == "pcm":
        options["input_format"] = "pcm_s16le"
    value = pcm if input_kind == "pcm" else waveform
    results = wrapper.generate(input=[value, value] if batch else value, **options)
    assert len(results) == (2 if batch else 1)
    assert len(feature_inputs) == len(results)
    for audio in feature_inputs:
        assert len(audio) == 24000
        np.testing.assert_allclose(audio, expected, atol=1e-6, rtol=0)
    for _, vad_options in wrapper.vad_model.calls:
        assert vad_options["fs"] == source_rate
    for _, asr_options in wrapper.model.calls:
        assert asr_options["fs"] == 16000
    for speaker_options in wrapper.spk_model.calls:
        assert speaker_options.get("fs", 16000) == 16000
    assert wrapper._base_kwargs_map["kwargs"].get("fs") == base_rate
    assert wrapper._base_kwargs_map["spk_kwargs"].get("fs") == speaker_rate

    # A subsequent call without an override must use the constructor source rate.
    feature_inputs.clear()
    next_rate = base_rate or 16000
    wrapper.generate(input=np.zeros(next_rate // 5, dtype=np.float32))
    assert wrapper.vad_model.calls[-1][1]["fs"] == next_rate
    assert wrapper.model.calls[-1][1]["fs"] == 16000
    assert wrapper.spk_model.calls[-1].get("fs", 16000) == 16000
    assert len(feature_inputs[0]) == 24000


def test_speaker_rate_override_preserves_caller_config(monkeypatch):
    import copy
    import importlib

    from funasr.register import tables

    auto_module = importlib.import_module("funasr.auto.auto_model")

    class ConfiguredModel(_RecordingModel):
        def __init__(self, model, **kwargs):
            super().__init__(vad=model == "rate-test-vad")
            self.segments = [[0, 200]]

    monkeypatch.setitem(tables.model_classes, "rate-test-asr", ConfiguredModel)
    monkeypatch.setitem(tables.model_classes, "rate-test-vad", ConfiguredModel)
    monkeypatch.setitem(tables.model_classes, "rate-test-speaker", _RecordingSpeaker)
    monkeypatch.setitem(
        tables.frontend_classes, "rate-test-frontend", lambda **kw: SimpleNamespace(fs=16000)
    )
    monkeypatch.setattr(
        auto_module, "ClusterBackend", lambda **kw: torch.nn.Identity(), raising=False
    )
    monkeypatch.setattr(auto_module, "download_model", _forbid_decode)
    caller_config = {"model_conf": {}, "device": "cpu", "fs": 48000}
    original = copy.deepcopy(caller_config)
    wrapper = AutoModel(
        model="rate-test-asr", model_conf={}, frontend="rate-test-frontend",
        vad_model="rate-test-vad", vad_kwargs={"model_conf": {}},
        spk_model="rate-test-speaker", spk_kwargs=caller_config,
        fs=48000, device="cpu", ncpu=torch.get_num_threads(),
        disable_update=True, disable_pbar=True, return_spk_res=False,
    )
    for rate in (8000, 48000):
        options = {"fs": rate} if rate == 8000 else {}
        wrapper.generate(input=np.zeros(rate // 5, dtype=np.float32), **options)
        assert wrapper.vad_model.calls[-1][1]["fs"] == rate
        assert wrapper.spk_model.calls[-1]["fs"] == 16000
        assert caller_config == original
        assert wrapper._base_kwargs_map["spk_kwargs"]["fs"] == 48000
