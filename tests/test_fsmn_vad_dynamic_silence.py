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
        self.assertNotIn("_dynamic_accumulated_ms", cache)

    def test_explicit_dynamic_silence_still_enables_schedule(self):
        cache = self._run_inference(max_end_silence_time=300, dynamic_silence=True)

        self.assertEqual(cache["stats"].max_end_sil_frame_cnt_thresh, 1900)
        self.assertEqual(cache["_dynamic_accumulated_ms"], 1000)


if __name__ == "__main__":
    unittest.main()
