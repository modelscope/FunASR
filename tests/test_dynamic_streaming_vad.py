import unittest
from types import SimpleNamespace

import torch

from funasr.models.fsmn_vad_streaming import model as vad_model
from funasr.models.fsmn_vad_streaming.dynamic_vad import DynamicStreamingVAD


class _ThresholdAwareModel:
    """Small AutoModel stand-in that uses the production cache initializer."""

    sample_rate = 16000

    def __init__(self):
        self.model = vad_model.FsmnVADStreaming.__new__(vad_model.FsmnVADStreaming)
        self.model.vad_opts = SimpleNamespace(
            window_size_ms=200,
            sil_to_speech_time_thres=150,
            speech_to_sil_time_thres=150,
            frame_in_ms=10,
            sil_pdf_ids=[0],
            max_end_silence_time=800,
            speech_noise_thres=0.5,
        )

    def generate(self, input, cache, **kwargs):
        if not cache:
            self.model.init_cache(cache, **kwargs)

        audio = torch.cat((cache.get("_test_audio", torch.empty(0)), input[0]))
        cache["_test_audio"] = audio

        speech_indices = torch.nonzero(audio.abs() > 0.5)
        if not len(speech_indices) or cache.get("_test_emitted"):
            return [{"value": []}]

        last_speech_sample = speech_indices[-1].item()
        trailing_silence_ms = int(
            (len(audio) - last_speech_sample - 1) * 1000 / self.sample_rate
        )
        stats = cache["stats"]
        threshold_ms = (
            stats.max_end_sil_frame_cnt_thresh
            + self.model.vad_opts.speech_to_sil_time_thres
        )
        if trailing_silence_ms < threshold_ms:
            return [{"value": []}]

        cache["_test_emitted"] = True
        speech_end_ms = int((last_speech_sample + 1) * 1000 / self.sample_rate)
        return [{"value": [[0, speech_end_ms]]}]


class TestDynamicStreamingVadFirstCall(unittest.TestCase):
    def _new_vad(self):
        return DynamicStreamingVAD(
            _ThresholdAwareModel(),
            silence_schedule=[(float("inf"), 10000)],
            speech_noise_thres=0.73,
        )

    def test_first_feed_initializes_wrapper_thresholds(self):
        vad = self._new_vad()

        vad.feed(torch.ones(960))

        self.assertEqual(vad.cache["stats"].max_end_sil_frame_cnt_thresh, 9850)
        self.assertAlmostEqual(vad.cache["stats"].speech_noise_thres, 0.73)

    def test_first_feed_is_chunking_invariant(self):
        audio = torch.cat((torch.ones(16000), torch.zeros(32000)))

        one_chunk_vad = self._new_vad()
        one_chunk_segments = one_chunk_vad.feed(audio)

        split_vad = self._new_vad()
        split_segments = split_vad.feed(audio[:960])
        split_segments.extend(split_vad.feed(audio[960:]))

        self.assertEqual(one_chunk_segments, split_segments)
        self.assertEqual(one_chunk_segments, [])


if __name__ == "__main__":
    unittest.main()
