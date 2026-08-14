import unittest
from importlib.util import find_spec
from unittest.mock import patch

import torch

from funasr.auto.auto_model import AutoModel
from funasr.models.silero_vad.model import SileroVad


@unittest.skipUnless(find_spec("silero_vad"), "silero-vad is not installed")
class TestSileroVadAdapter(unittest.TestCase):
    def _timestamps_stub(self, waveform, model, sampling_rate, **options):
        self.assertEqual(sampling_rate, 16000)
        self.assertEqual(options["threshold"], 0.6)
        return [{"start": 1600, "end": 17600}]

    def _load_stub(self, *args, **kwargs):
        self.assertEqual(kwargs, {"onnx": False})
        return torch.nn.Identity()

    @patch("silero_vad.get_speech_timestamps")
    @patch("silero_vad.load_silero_vad")
    def test_returns_funasr_millisecond_segments_and_honors_max_length(
        self, load_model, timestamps
    ):
        load_model.side_effect = self._load_stub
        timestamps.side_effect = self._timestamps_stub
        model = SileroVad()
        results, metadata = model.inference(
            data_in=[torch.zeros(32000)],
            key=["sample"],
            silero_threshold=0.6,
            max_single_segment_time=500,
        )

        self.assertEqual(
            results, [{"key": "sample", "value": [[100, 600], [600, 1100]]}]
        )
        self.assertEqual(metadata["batch_data_time"], 2.0)
        load_model.assert_called_once_with(onnx=False)

    @patch("silero_vad.get_speech_timestamps")
    @patch("silero_vad.load_silero_vad")
    def test_rejects_unsupported_sampling_rate(self, load_model, timestamps):
        load_model.side_effect = self._load_stub
        model = SileroVad()
        with self.assertRaisesRegex(ValueError, "8000 or 16000"):
            model.inference(data_in=[torch.zeros(16000)], silero_sampling_rate=44100)

    @patch("silero_vad.get_speech_timestamps")
    @patch("silero_vad.load_silero_vad")
    def test_auto_model_alias_uses_the_existing_vad_build_path(
        self, load_model, timestamps
    ):
        load_model.side_effect = self._load_stub
        model, resolved = AutoModel.build_model(model="silero-vad", device="cpu")
        self.assertIsInstance(model, SileroVad)
        self.assertEqual(resolved["model"], "SileroVad")

    @patch("silero_vad.get_speech_timestamps")
    @patch("silero_vad.load_silero_vad")
    def test_waveform_follows_the_adapter_device(self, load_model, timestamps):
        load_model.side_effect = self._load_stub

        def timestamps_stub(waveform, model, sampling_rate, **options):
            self.assertEqual(waveform.device.type, "meta")
            return []

        timestamps.side_effect = timestamps_stub
        model = SileroVad().to("meta")
        results, _ = model.inference(data_in=[torch.zeros(16000)], key=["sample"])
        self.assertEqual(results, [{"key": "sample", "value": []}])

    @patch("silero_vad.get_speech_timestamps")
    @patch("silero_vad.load_silero_vad")
    def test_onnx_waveform_stays_on_cpu(self, load_model, timestamps):
        load_model.return_value = object()

        def timestamps_stub(waveform, model, sampling_rate, **options):
            self.assertEqual(waveform.device.type, "cpu")
            return []

        timestamps.side_effect = timestamps_stub
        model = SileroVad(silero_onnx=True).to("meta")
        results, _ = model.inference(data_in=[torch.zeros(16000)], key=["sample"])
        self.assertEqual(results, [{"key": "sample", "value": []}])
        load_model.assert_called_once_with(onnx=True)

    def test_rejects_negative_max_segment_length(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            SileroVad._split_long_segments([[100, 1100]], -500)

    def test_rejects_sub_millisecond_max_segment_length(self):
        with self.assertRaisesRegex(ValueError, "positive"):
            SileroVad._split_long_segments([[100, 1100]], 0.5)


if __name__ == "__main__":
    unittest.main()
