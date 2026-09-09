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


class _SignalModel(_ThresholdAwareModel):
    def __init__(self, signals):
        super().__init__()
        self.signals = iter(signals)
        self.thresholds = []

    def generate(self, input, cache, **kwargs):
        if not cache:
            self.model.init_cache(cache, **kwargs)
        self.thresholds.append(
            cache["stats"].max_end_sil_frame_cnt_thresh
            + self.model.vad_opts.speech_to_sil_time_thres
        )
        return [{"value": next(self.signals, [])}]


class TestDynamicStreamingVadSpeechDuration(unittest.TestCase):
    def test_initial_silence_does_not_advance_speech_schedule(self):
        vad = DynamicStreamingVAD(_SignalModel([[]]))

        vad.feed(torch.zeros(60 * 16000))

        self.assertEqual(vad.current_duration_ms, 0)
        self.assertEqual(vad.current_threshold_ms, 2000)

    def test_speech_after_initial_silence_uses_short_utterance_threshold(self):
        model = _SignalModel([[], [[60000, -1]]])
        vad = DynamicStreamingVAD(model)
        vad.feed(torch.zeros(60 * 16000))

        vad.feed(torch.ones(16000))

        self.assertEqual(model.thresholds, [2000, 2000])
        self.assertEqual(vad.current_duration_ms, 1000)
        self.assertEqual(vad.current_threshold_ms, 2000)

    def test_inter_utterance_silence_does_not_age_next_utterance(self):
        model = _SignalModel([[[0, -1]], [[-1, 1000]], [], [[22000, -1]]])
        vad = DynamicStreamingVAD(model)
        vad.feed(torch.ones(16000))
        vad.feed(torch.zeros(2 * 16000))
        vad.feed(torch.zeros(19 * 16000))

        vad.feed(torch.ones(16000))

        self.assertEqual(model.thresholds[-1], 2000)
        self.assertEqual(vad.current_duration_ms, 1000)

    def test_duration_uses_detected_start_and_keeps_long_speech_schedule(self):
        vad = DynamicStreamingVAD(_SignalModel([[], [[6000, -1]], []]))
        vad.feed(torch.zeros(5 * 16000))
        vad.feed(torch.ones(2 * 16000))
        self.assertEqual(vad.current_duration_ms, 1000)

        vad.feed(torch.ones(6 * 16000))

        self.assertEqual(vad.current_duration_ms, 7000)
        self.assertEqual(vad.current_threshold_ms, 1500)

    def test_submillisecond_packets_do_not_lose_elapsed_samples(self):
        vad = DynamicStreamingVAD(_SignalModel([[[0, -1]], [], []]))
        for count in (1, 15, 160):
            vad.feed(torch.ones(count))

        self.assertEqual(vad.current_duration_ms, 11)

    def test_last_open_signal_in_packet_defines_current_duration(self):
        vad = DynamicStreamingVAD(_SignalModel([[[0, 1000], [2000, -1]]]))

        self.assertEqual(vad.feed(torch.ones(10 * 16000)), [[0, 1000]])

        self.assertEqual(vad.current_duration_ms, 8000)
        self.assertEqual(vad.current_threshold_ms, 1500)

    def test_reset_starts_a_new_sample_clock(self):
        vad = DynamicStreamingVAD(_SignalModel([[], [[0, -1]]]))
        vad.feed(torch.zeros(60 * 16000))
        vad.reset()

        vad.feed(torch.ones(16000))

        self.assertEqual(vad.current_duration_ms, 1000)
        self.assertEqual(vad.current_threshold_ms, 2000)

    def test_backdated_start_in_previous_packet_counts_entire_segment(self):
        vad = DynamicStreamingVAD(_SignalModel([[], [[4800, -1]]]))
        vad.feed(torch.zeros(5 * 16000))

        vad.feed(torch.ones(16000))

        self.assertEqual(vad.current_duration_ms, 1200)

    def test_start_end_start_in_one_packet_uses_last_start(self):
        vad = DynamicStreamingVAD(
            _SignalModel([[[1000, -1], [-1, 2000], [3000, -1]]])
        )

        self.assertEqual(vad.feed(torch.ones(10 * 16000)), [[1000, 2000]])
        self.assertEqual(vad.current_speech_start, 3000)
        self.assertEqual(vad.current_duration_ms, 7000)

    def test_finalize_without_end_preserves_start_for_service_fallback(self):
        vad = DynamicStreamingVAD(_SignalModel([[[500, -1]], []]))
        vad.feed(torch.ones(16000))

        self.assertEqual(vad.finalize(), [])

        self.assertEqual(vad.current_speech_start, 500)
        self.assertTrue(vad.is_speaking)

    def test_feed_after_finalize_starts_a_new_stream(self):
        model = _SignalModel([[], [], [[0, -1]]])
        vad = DynamicStreamingVAD(model)
        vad.feed(torch.zeros(60 * 16000))
        vad.finalize()

        vad.feed(torch.ones(16000))

        self.assertEqual(vad.current_duration_ms, 1000)
        self.assertEqual(model.thresholds[-1], 2000)

    def test_feed_after_direct_final_clears_previous_stream(self):
        model = _SignalModel([[[0, 1000], [2000, -1]], [[0, -1]]])
        vad = DynamicStreamingVAD(model)
        self.assertEqual(vad.feed(torch.ones(60 * 16000), is_final=True), [[0, 1000]])
        self.assertEqual(vad.current_speech_start, 2000)
        old_cache = vad.cache

        vad.feed(torch.ones(16000))

        self.assertEqual(vad.current_duration_ms, 1000)
        self.assertEqual(vad.confirmed_segments, [])
        self.assertIsNot(vad.cache, old_cache)
        self.assertEqual(model.thresholds[-1], 2000)

    def test_repeated_finalize_preserves_final_state_without_feeding_again(self):
        model = _SignalModel([[[0, 100], [500, -1]], []])
        vad = DynamicStreamingVAD(model)
        vad.feed(torch.ones(16000))
        vad.finalize()
        final_cache = vad.cache

        self.assertEqual(vad.finalize(), [])

        self.assertEqual(vad.current_speech_start, 500)
        self.assertEqual(vad.confirmed_segments, [[0, 100]])
        self.assertIs(vad.cache, final_cache)
        self.assertEqual(len(model.thresholds), 2)

    def test_finalize_after_process_preserves_results_and_fallback_state(self):
        model = _SignalModel([[[0, 30]], [[80, -1]]])
        vad = DynamicStreamingVAD(model)
        self.assertEqual(vad.process(torch.ones(1920)), [[0, 30]])

        self.assertEqual(vad.finalize(), [])

        self.assertEqual(vad.confirmed_segments, [[0, 30]])
        self.assertEqual(vad.current_speech_start, 80)
        self.assertEqual(len(model.thresholds), 2)


if __name__ == "__main__":
    unittest.main()
