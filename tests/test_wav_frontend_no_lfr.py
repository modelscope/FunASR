import pytest
import torch

from funasr.frontends.wav_frontend import WavFrontendOnline
from funasr.models.fsmn_vad_streaming import model as vad_model
from funasr.utils import fbank as kaldi


def make_frontend(tmp_path, cmvn):
    cmvn_file = None
    if cmvn:
        cmvn_file = tmp_path / "no_lfr.mvn"
        cmvn_file.write_text(
            "<AddShift> 80 80\n<LearnRateCoef> 1 [ "
            + " ".join(["0.25"] * 80)
            + " ]\n<Rescale> 80 80\n<LearnRateCoef> 1 [ "
            + " ".join(["2"] * 80)
            + " ]\n"
        )
    return WavFrontendOnline(cmvn_file=cmvn_file, dither=0.0)


@pytest.mark.parametrize("cmvn", (False, True))
@pytest.mark.parametrize("is_final", (False, True))
@pytest.mark.parametrize("lengths", ((5, 3, 0), (0, 0)))
def test_no_lfr_reports_valid_consumed_frames(tmp_path, cmvn, is_final, lengths):
    frontend = make_frontend(tmp_path, cmvn)
    inputs = torch.arange(len(lengths) * 7 * 80, dtype=torch.float32).reshape(
        len(lengths), 7, 80
    )
    original = inputs.clone()
    cache = {}

    feats, actual_lengths, consumed = frontend.forward_lfr_cmvn(
        inputs, torch.tensor(lengths), is_final=is_final, cache=cache
    )

    assert actual_lengths.tolist() == list(lengths)
    assert consumed.tolist() == list(lengths)
    assert feats.shape == (len(lengths), max(lengths), 80)
    assert feats.dtype == torch.float32
    assert cache == {}
    for index, length in enumerate(lengths):
        expected = original[index, :length]
        if cmvn:
            expected = (expected + 0.25) * 2
        torch.testing.assert_close(feats[index, :length], expected, rtol=0, atol=0)
        assert torch.count_nonzero(feats[index, length:]) == 0


@pytest.mark.parametrize("cmvn", (False, True))
@pytest.mark.parametrize("empty_final", (False, True))
@pytest.mark.parametrize("chunks", ((1600,), (400,) * 4, (160,) * 10, (80, 320, 40, 1160)))
def test_no_lfr_stream_matches_full_filterbank(tmp_path, cmvn, empty_final, chunks):
    frontend = make_frontend(tmp_path, cmvn)
    audio = 0.2 * torch.sin(torch.arange(1600, dtype=torch.float32) * (2 * torch.pi * 440 / 16000))
    expected = kaldi.fbank(
        audio.unsqueeze(0) * (1 << 15), num_mel_bins=80, frame_length=25, frame_shift=10,
        sample_frequency=16000, window_type="hamming", dither=0.0, energy_floor=0.0,
    )
    if cmvn:
        expected = (expected + 0.25) * 2
    cache, outputs, offset, emitted = {}, [], 0, 0
    calls = chunks + ((0,) if empty_final else ())
    for index, size in enumerate(calls):
        chunk = audio[offset : offset + size]
        offset += size
        feats, lengths = frontend(
            chunk.unsqueeze(0), torch.tensor([size]), cache=cache,
            is_final=index == len(calls) - 1, return_waveform=True,
        )
        count = int(lengths[0]) if lengths.numel() else 0
        if count:
            outputs.append(feats[0, :count])
            start = emitted * 160
            stop = start + (count - 1) * 160 + 400
            torch.testing.assert_close(cache["aligned_waveforms"][0], audio[start:stop], rtol=0, atol=0)
            emitted += count
        else:
            assert cache["aligned_waveforms"].numel() == 0
        assert cache["reserve_waveforms"].numel() == 0
    assert emitted == expected.shape[0] == 8
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("size", (0, 160, 399))
@pytest.mark.parametrize("empty_final", (False, True))
def test_no_lfr_short_stream_has_no_frames(tmp_path, size, empty_final):
    frontend = make_frontend(tmp_path, False)
    cache = {}
    for index, length in enumerate((size, 0) if empty_final else (size,)):
        feats, lengths = frontend(
            torch.zeros(1, length), torch.tensor([length]), cache=cache,
            is_final=not empty_final or index == 1, return_waveform=True,
        )
        assert feats.numel() == 0
        assert not lengths.numel() or lengths.sum() == 0
        assert cache["aligned_waveforms"].numel() == 0


@pytest.mark.parametrize("lfr_m", (1, 5))
@pytest.mark.parametrize("offset_ms", (0, 1000))
@pytest.mark.parametrize("chunk_ms", (20, 100, 5000))
def test_offline_vad_finalizes_last_nonempty_chunk(lfr_m, offset_ms, chunk_ms):
    class ScriptedEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frames = 0

        def forward(self, feats, cache=None):
            time_ms = (torch.arange(feats.shape[1]) + self.frames) * 10
            speech = ((time_ms >= 500 + offset_ms) & (time_ms < 1500 + offset_ms)) | (
                (time_ms >= 2500 + offset_ms) & (time_ms < 3200 + offset_ms)
            )
            self.frames += feats.shape[1]
            probability = torch.where(speech, 0.999, 0.001)
            return torch.stack((1 - probability, probability), dim=-1).unsqueeze(0)

    model = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
    torch.nn.Module.__init__(model)
    model.encoder = ScriptedEncoder()
    model.vad_opts = vad_model.VADXOptions()
    audio = 0.2 * torch.sin(
        torch.arange((4000 + offset_ms) * 16, dtype=torch.float32)
        * (2 * torch.pi * 440 / 16000)
    )
    frontend = WavFrontendOnline(lfr_m=lfr_m, lfr_n=1, dither=0.0)
    result, _ = model.inference(
        [audio], key=["fixture"], cache={}, frontend=frontend, device="cpu",
        chunk_size=chunk_ms, is_streaming_input=False, is_final=True, dynamic_silence=True,
    )
    assert model.encoder.frames == (4000 + offset_ms) // 10 - 2
    assert result == [{"key": "fixture", "value": [[240 + offset_ms, 3980 + offset_ms]]}]
