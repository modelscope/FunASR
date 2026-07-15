from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from funasr.frontends.wav_frontend import WavFrontendOnline
from funasr.models.fsmn_vad_streaming import model as vad_model


class DummyEncoder(torch.nn.Module):
    def forward(self, feats, cache=None):
        return torch.zeros(feats.shape[0], feats.shape[1], 2)


class DummyWindowDetector:
    def Reset(self):
        return None


def build_vad_and_cache():
    vad = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
    torch.nn.Module.__init__(vad)
    vad.encoder = DummyEncoder()
    vad.vad_opts = SimpleNamespace(
        frame_length_ms=25,
        frame_in_ms=10,
        sample_rate=16000,
        nn_eval_block_size=0,
    )
    vad.DetectCommonFrames = lambda cache=None: 0
    vad.DetectLastFrames = lambda cache=None: 0

    cache = {
        "encoder": {},
        "stats": vad_model.Stats(
            sil_pdf_ids=[],
            max_end_sil_frame_cnt_thresh=0,
            speech_noise_thres=0.5,
        ),
        "windows_detector": DummyWindowDetector(),
    }
    return vad, cache


def run_aligned_frontend_windows(vad, cache):
    source = torch.arange(4000, dtype=torch.float32) / 4000.0
    calls = (
        (source[0:560], 2),
        (source[320:1520], 6),
        (source[1280:2480], 6),
    )
    for waveform, frame_count in calls:
        vad.forward(
            feats=torch.zeros(1, frame_count, 4),
            waveform=waveform.unsqueeze(0),
            cache=cache,
            is_final=False,
        )
    return source


def test_streaming_buffers_only_append_audio_aligned_with_score_frames():
    vad, cache = build_vad_and_cache()
    source = run_aligned_frontend_windows(vad, cache)
    stats = cache["stats"]

    expected_waveform = source[:2480]
    expected_frames = expected_waveform.unfold(0, 400, 160)
    expected_decibel = 10 * torch.log10(expected_frames.square().sum(dim=1) + 0.000001)

    assert stats.scores.shape[1] == 14
    assert len(stats.decibel) == stats.scores.shape[1]
    assert torch.allclose(torch.tensor(stats.decibel), expected_decibel)
    assert torch.equal(stats.data_buf_all, expected_waveform)
    assert torch.equal(stats.data_buf, expected_waveform)


def test_reset_detection_drops_completed_aligned_frames():
    vad, cache = build_vad_and_cache()
    source = run_aligned_frontend_windows(vad, cache)
    stats = cache["stats"]

    segment = vad_model.E2EVadSpeechBufWithDoa()
    segment.Reset()
    segment.end_ms = 80
    segment.contain_seg_end_point = True
    stats.output_data_buf.append(segment)
    stats.data_buf_start_frame = 8

    vad.ResetDetection(cache=cache)

    assert stats.last_drop_frames == 8
    assert stats.scores.shape[1] == 6
    assert len(stats.decibel) == 6
    assert torch.equal(stats.data_buf_all, source[1280:2480])
    assert torch.equal(stats.data_buf, stats.data_buf_all)

    vad.forward(
        feats=torch.zeros(1, 2, 4),
        waveform=source[2240:2800].unsqueeze(0),
        cache=cache,
        is_final=False,
    )

    expected_waveform = source[1280:2800]
    expected_frames = expected_waveform.unfold(0, 400, 160)
    expected_decibel = 10 * torch.log10(expected_frames.square().sum(dim=1) + 0.000001)
    assert stats.scores.shape[1] == 8
    assert len(stats.decibel) == 8
    assert torch.allclose(torch.tensor(stats.decibel), expected_decibel)
    assert torch.equal(stats.data_buf_all, expected_waveform)
    assert torch.equal(stats.data_buf, expected_waveform)


@pytest.mark.parametrize(
    ("final_size", "expected_frame_counts", "expected_ranges", "expected_sample_count"),
    (
        (160, (2, 6, 6, 3), ((0, 560), (320, 1520), (1280, 2480), (2240, 2960)), 2960),
        (0, (2, 6, 6, 2), ((0, 560), (320, 1520), (1280, 2480), (2240, 2800)), 2800),
    ),
)
def test_real_frontend_exposes_exact_waveforms_for_final_chunks(
    final_size,
    expected_frame_counts,
    expected_ranges,
    expected_sample_count,
):
    frontend = WavFrontendOnline(
        fs=16000,
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        lfr_m=5,
        lfr_n=1,
        dither=0.0,
        upsacle_samples=False,
    )
    frontend_cache = {}
    vad, vad_cache = build_vad_and_cache()
    source = torch.arange(4000, dtype=torch.float32) / 4000.0
    offset = 0

    for index, (chunk_size, frame_count, sample_range) in enumerate(
        zip((960, 960, 960, final_size), expected_frame_counts, expected_ranges)
    ):
        chunk = source[offset : offset + chunk_size]
        offset += chunk_size
        is_final = index == len(expected_frame_counts) - 1
        feats, _ = frontend(
            chunk.unsqueeze(0),
            torch.tensor([chunk.numel()]),
            cache=frontend_cache,
            is_final=is_final,
            return_waveform=True,
        )
        aligned_waveform = frontend_cache["aligned_waveforms"]

        assert feats.shape[1] == frame_count
        assert torch.equal(aligned_waveform[0], source[slice(*sample_range)])
        assert frontend_cache["waveform_buffer"].shape[1] <= 640

        vad.forward(
            feats=feats,
            waveform=aligned_waveform,
            cache=vad_cache,
            is_final=is_final,
        )

    expected_waveform = source[:expected_sample_count]
    expected_frames = expected_waveform.unfold(0, 400, 160)
    expected_decibel = 10 * torch.log10(expected_frames.square().sum(dim=1) + 0.000001)
    stats = vad_cache["stats"]

    assert stats.scores.shape[1] == sum(expected_frame_counts)
    assert len(stats.decibel) == stats.scores.shape[1]
    assert torch.allclose(torch.tensor(stats.decibel), expected_decibel)
    assert torch.equal(stats.data_buf_all, expected_waveform)


def test_compute_decibel_keeps_positional_cache_call_compatible():
    vad, cache = build_vad_and_cache()
    waveform = torch.arange(560, dtype=torch.float32) / 560.0
    cache["stats"].waveform = waveform.unsqueeze(0)

    vad.ComputeDecibel(cache)

    assert len(cache["stats"].decibel) == 2
    assert torch.equal(cache["stats"].data_buf_all, waveform)


def test_real_frontend_tracks_initial_padding_across_small_chunks():
    frontend = WavFrontendOnline(
        fs=16000,
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        lfr_m=5,
        lfr_n=1,
        dither=0.0,
        upsacle_samples=False,
    )
    cache = {}
    source = torch.arange(2000, dtype=torch.float32) / 2000.0
    emitted_frames = 0
    input_offset = 0

    for index, (chunk_size, expected_frames) in enumerate(
        zip((480, 480, 480, 160), (0, 2, 3, 3))
    ):
        chunk = source[input_offset : input_offset + chunk_size]
        input_offset += chunk_size
        feats, _ = frontend(
            chunk.unsqueeze(0),
            torch.tensor([chunk.numel()]),
            cache=cache,
            is_final=index == 3,
            return_waveform=True,
        )

        assert (feats.shape[1] if feats.ndim == 3 else 0) == expected_frames
        assert cache["waveform_buffer"].shape[1] <= 640
        if expected_frames == 0:
            assert cache["aligned_waveforms"].numel() == 0
            continue

        aligned_sample_count = (expected_frames - 1) * 160 + 400
        expected_waveform = source[
            emitted_frames * 160 : emitted_frames * 160 + aligned_sample_count
        ]
        assert torch.equal(cache["aligned_waveforms"][0], expected_waveform)
        emitted_frames += expected_frames

    assert emitted_frames == 8


def test_frontend_cursor_does_not_advance_past_received_audio():
    frontend = WavFrontendOnline(
        fs=16000,
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        lfr_m=7,
        lfr_n=6,
        dither=0.0,
        upsacle_samples=False,
    )
    cache = {}
    source = torch.arange(2400, dtype=torch.float32) / 2400.0

    first_feats, _ = frontend(
        source[:880].unsqueeze(0),
        torch.tensor([880]),
        cache=cache,
        is_final=False,
        return_waveform=True,
    )

    assert first_feats.shape[1] == 1
    assert torch.equal(cache["aligned_waveforms"][0], source[:400])
    assert cache["waveform_buffer_start_sample"] == 880

    second_feats, _ = frontend(
        source[880:1840].unsqueeze(0),
        torch.tensor([960]),
        cache=cache,
        is_final=False,
        return_waveform=True,
    )

    assert second_feats.shape[1] == 1
    assert torch.equal(cache["aligned_waveforms"][0], source[960:1360])


def test_zero_score_frontend_batch_is_a_model_noop():
    frontend = WavFrontendOnline(
        fs=16000,
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        lfr_m=5,
        lfr_n=1,
        dither=0.0,
        upsacle_samples=False,
    )
    frontend_cache = {}
    source = torch.arange(480, dtype=torch.float32) / 480.0
    feats, _ = frontend(
        source.unsqueeze(0),
        torch.tensor([source.numel()]),
        cache=frontend_cache,
        is_final=False,
        return_waveform=True,
    )
    vad, cache = build_vad_and_cache()

    assert feats.numel() == 0
    assert (
        vad.forward(
            feats=feats,
            waveform=frontend_cache["aligned_waveforms"],
            cache=cache,
            is_final=False,
        )
        == []
    )
    assert cache["stats"].data_buf_all is None


def test_consumed_silence_history_is_compacted_without_an_endpoint():
    vad, cache = build_vad_and_cache()

    def consume_all_frames(cache=None):
        cache["stats"].data_buf_start_frame = cache["stats"].frm_cnt

    vad.DetectCommonFrames = consume_all_frames
    source = torch.arange(100 * 960 + 400, dtype=torch.float32) / 100000.0

    for index in range(100):
        start = index * 960
        vad.forward(
            feats=torch.zeros(1, 6, 4),
            waveform=source[start : start + 1200].unsqueeze(0),
            cache=cache,
            is_final=False,
        )

    stats = cache["stats"]
    assert stats.last_drop_frames == stats.frm_cnt == 600
    assert stats.data_buf_all.numel() == 240
    assert stats.data_buf.numel() == 240
    assert stats.scores.shape[1] == 0
    assert stats.decibel == []


def test_waveform_alignment_is_opt_in_for_other_frontend_consumers():
    frontend = WavFrontendOnline(
        fs=16000,
        n_mels=80,
        frame_length=25,
        frame_shift=10,
        lfr_m=5,
        lfr_n=1,
        dither=0.0,
        upsacle_samples=False,
    )
    cache = {}
    waveform = torch.zeros(1, 960)

    frontend(
        waveform,
        torch.tensor([waveform.shape[1]]),
        cache=cache,
        is_final=False,
    )

    assert cache["waveform_buffer"] is None
    assert cache["aligned_waveforms"].numel() == 0


def test_misaligned_waveform_fails_before_mutating_model_buffers():
    vad, cache = build_vad_and_cache()
    cache["stats"].waveform = torch.zeros(1, 880)

    with pytest.raises(RuntimeError, match="score frames and waveform samples"):
        vad.ComputeDecibel(cache=cache, frame_count=2)

    assert cache["stats"].data_buf_all is None
    assert cache["stats"].decibel == []


def test_low_decibel_frame_advances_detector_once_with_absolute_index():
    vad, cache = build_vad_and_cache()
    stats = cache["stats"]
    stats.decibel = [-100.0]
    stats.scores = torch.zeros(1, 1, 2)
    stats.frm_cnt = 101
    stats.last_drop_frames = 100
    vad.vad_opts.decibel_thres = 0.0
    vad.vad_opts.nn_eval_block_size = 1
    detected_frames = []
    vad.DetectOneFrame = lambda frame_state, frame_index, is_final, cache=None: (
        detected_frames.append(frame_index)
    )

    vad_model.FsmnVADStreaming.DetectCommonFrames(vad, cache=cache)

    assert detected_frames == [100]


@pytest.mark.parametrize("supports_alignment", (False, True))
def test_inference_negotiates_aligned_waveform_support(supports_alignment):
    vad = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
    vad.vad_opts = SimpleNamespace(speech_to_sil_time_thres=100)
    vad.forward = lambda **batch: []
    cache = {
        "frontend": {},
        "prev_samples": torch.empty(0),
        "encoder": {},
        "stats": SimpleNamespace(),
    }
    frontend = SimpleNamespace(fs=16000, frame_shift=10, lfr_n=1)
    if supports_alignment:
        frontend.supports_aligned_waveforms = True
    extract_kwargs = []

    def fake_extract_fbank(*args, **kwargs):
        extract_kwargs.append(kwargs)
        cache["frontend"]["waveforms"] = torch.zeros(1, 560)
        return torch.zeros(1, 2, 4), torch.tensor([2])

    with (
        patch.object(
            vad_model,
            "load_audio_text_image_video",
            return_value=[torch.zeros(960)],
        ),
        patch.object(vad_model, "extract_fbank", side_effect=fake_extract_fbank),
    ):
        vad_model.FsmnVADStreaming.inference(
            vad,
            torch.zeros(960),
            frontend=frontend,
            cache=cache,
            key=["utt"],
            chunk_size=60,
            is_final=False,
            device="cpu",
            dynamic_silence=False,
        )

    assert extract_kwargs[0].get("return_waveform", False) is supports_alignment


def test_inference_rejects_missing_waveform_before_forward():
    vad = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
    vad.vad_opts = SimpleNamespace(speech_to_sil_time_thres=100)
    forward_batches = []
    vad.forward = lambda **batch: forward_batches.append(batch) or []
    cache = {
        "frontend": {},
        "prev_samples": torch.empty(0),
        "encoder": {},
        "stats": SimpleNamespace(),
    }
    frontend = SimpleNamespace(fs=16000, frame_shift=10, lfr_n=1)

    with (
        patch.object(
            vad_model,
            "load_audio_text_image_video",
            return_value=[torch.zeros(960)],
        ),
        patch.object(
            vad_model,
            "extract_fbank",
            return_value=(torch.zeros(1, 2, 4), torch.tensor([2])),
        ),
        pytest.raises(
            RuntimeError,
            match="must provide aligned_waveforms or waveforms",
        ),
    ):
        vad_model.FsmnVADStreaming.inference(
            vad,
            torch.zeros(960),
            frontend=frontend,
            cache=cache,
            key=["utt"],
            chunk_size=60,
            is_final=False,
            device="cpu",
            dynamic_silence=False,
        )

    assert forward_batches == []
