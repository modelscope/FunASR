import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from funasr.models.fsmn_vad_streaming import model as vad_model


class TestFsmnVadDynamicSilence(unittest.TestCase):
    def _run_inference(self, **kwargs):
        vad = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
        vad.vad_opts = SimpleNamespace(speech_to_sil_time_thres=100)
        vad.forward = lambda **batch: []

        cache = {
            "frontend": {},
            "prev_samples": torch.empty(0),
            "encoder": {},
            "stats": SimpleNamespace(
                vad_state_machine=vad_model.VadStateMachine.kVadInStateInSpeechSegment,
                max_end_sil_frame_cnt_thresh=200,
                speech_noise_thres=0.6,
            ),
        }
        frontend = SimpleNamespace(fs=16000, frame_shift=10, lfr_n=1)

        def fake_extract_fbank(*args, **kwargs):
            cache["frontend"]["waveforms"] = torch.zeros(1, 16000)
            return torch.zeros(1, 1, 80), torch.tensor([100])

        with (
            patch.object(
                vad_model,
                "load_audio_text_image_video",
                return_value=[torch.zeros(16000)],
            ),
            patch.object(vad_model, "extract_fbank", side_effect=fake_extract_fbank),
        ):
            vad_model.FsmnVADStreaming.inference(
                vad,
                torch.zeros(16000),
                frontend=frontend,
                cache=cache,
                key=["utt"],
                chunk_size=1000,
                is_final=False,
                device="cpu",
                **kwargs,
            )

        return cache

    def test_explicit_max_end_silence_time_keeps_fixed_threshold_by_default(self):
        cache = self._run_inference(max_end_silence_time=300)

        self.assertEqual(cache["stats"].max_end_sil_frame_cnt_thresh, 200)
        self.assertIsNone(cache["stats"].silence_schedule)
        self.assertEqual(cache["stats"].speech_noise_thres, 0.6)

    def test_explicit_dynamic_silence_still_enables_schedule(self):
        cache = self._run_inference(max_end_silence_time=300, dynamic_silence=True)

        self.assertEqual(cache["stats"].silence_schedule, vad_model.DEFAULT_SILENCE_SCHEDULE)
        self.assertEqual(cache["stats"].speech_noise_thres, 0.5)
        # The threshold is no longer advanced per chunk; DetectOneFrame applies the schedule.
        self.assertEqual(cache["stats"].max_end_sil_frame_cnt_thresh, 200)
        self.assertNotIn("_dynamic_accumulated_ms", cache)


class TestFsmnVadDynamicSilencePerFrame(unittest.TestCase):
    """Drive DetectOneFrame with scripted window-detector transitions."""

    def _end_frames(self, speech_ms, silence_ms, schedule, start_frame=0, fixed_thresh_ms=650):
        vad = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
        torch.nn.Module.__init__(vad)
        vad.vad_opts = vad_model.VADXOptions()
        ends = []

        def on_voice_end(end_frame, fake_result, is_last_frame, cache=None):
            ends.append(end_frame)

        def reset_detection(cache=None):
            cache["stats"].confirmed_start_frame = -1
            cache["stats"].vad_state_machine = vad_model.VadStateMachine.kVadInStateStartPointNotDetected

        vad.OnVoiceEnd = on_voice_end
        vad.PopDataToOutputBuf = lambda *args, **kwargs: None
        vad.PopDataBufTillFrame = lambda *args, **kwargs: None
        vad.LatencyFrmNumAtStartPoint = lambda cache=None: 0
        vad.ResetDetection = reset_detection

        stats = vad_model.Stats(
            sil_pdf_ids=[0], max_end_sil_frame_cnt_thresh=fixed_thresh_ms, speech_noise_thres=0.5
        )
        stats.silence_schedule = schedule
        speech_frames, silence_frames = speech_ms // 10, silence_ms // 10
        changes = (
            [vad_model.AudioChangeState.kChangeStateSil2Speech]
            + [vad_model.AudioChangeState.kChangeStateSpeech2Speech] * (speech_frames - 1)
            + [vad_model.AudioChangeState.kChangeStateSpeech2Sil]
            + [vad_model.AudioChangeState.kChangeStateSil2Sil] * silence_frames
        )
        script = iter(changes)
        cache = {
            "stats": stats,
            "windows_detector": SimpleNamespace(DetectOneFrame=lambda *a, **k: next(script)),
        }
        for i in range(len(changes)):
            vad.DetectOneFrame(vad_model.FrameState.kFrameStateSpeech, start_frame + i, False, cache=cache)
        return [e - start_frame for e in ends]

    def test_schedule_follows_current_segment_duration(self):
        schedule = vad_model.DEFAULT_SILENCE_SCHEDULE
        # Thresholds below are the schedule values minus speech_to_sil_time_thres (150 ms).
        # 5 s into a segment: 2000 ms -> 1850 ms, so 900 ms of silence does not end it.
        self.assertEqual(self._end_frames(5000, 900, schedule), [])
        # 15 s into a segment: 1000 ms -> 850 ms, so 900 ms ends it and 700 ms does not.
        self.assertEqual(len(self._end_frames(15000, 900, schedule)), 1)
        self.assertEqual(self._end_frames(15000, 700, schedule), [])
        # 55 s into a segment: 200 ms -> 50 ms, so 300 ms ends it.
        self.assertEqual(len(self._end_frames(55000, 300, schedule)), 1)

    def test_schedule_does_not_depend_on_absolute_position(self):
        schedule = vad_model.DEFAULT_SILENCE_SCHEDULE
        for speech_ms, silence_ms in [(5000, 900), (15000, 900), (55000, 300)]:
            at_zero = self._end_frames(speech_ms, silence_ms, schedule)
            shifted = self._end_frames(speech_ms, silence_ms, schedule, start_frame=6000)
            self.assertEqual(at_zero, shifted)

    def test_without_schedule_threshold_stays_fixed(self):
        # 650 ms fixed threshold: 700 ms of silence ends a 5 s segment, 600 ms does not.
        self.assertEqual(len(self._end_frames(5000, 700, None)), 1)
        self.assertEqual(self._end_frames(5000, 600, None), [])


if __name__ == "__main__":
    unittest.main()
