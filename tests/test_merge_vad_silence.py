"""Regression tests for merge_vad emitting pure-silence segments.

merge_vad sorts segment starts and ends into one list. When consecutive speech
segments are each close to max_length, it cut at the end of one segment and
again at the start of the next, so the pause between them was returned as its
own segment and sent to the ASR model (paraformer-zh decodes such segments into
garbage like "the the the ...").
"""

import random
import unittest

from funasr.utils.vad_utils import merge_vad


def _overlaps_speech(seg, vad):
    return any(min(seg[1], e) > max(seg[0], s) for s, e in vad)


class TestMergeVadSilence(unittest.TestCase):
    def assert_valid(self, vad, out):
        # No pure-silence segments.
        for seg in out:
            self.assertTrue(_overlaps_speech(seg, vad), f"pure-silence segment {seg} in {out}")
        # Output is sorted and non-overlapping.
        for a, b in zip(out, out[1:]):
            self.assertLessEqual(a[1], b[0], out)
        # Every speech segment is fully covered by exactly one output segment.
        for s, e in vad:
            covering = [seg for seg in out if seg[0] <= s and e <= seg[1]]
            self.assertEqual(len(covering), 1, f"speech {[s, e]} not covered once by {out}")

    def test_issue_example_no_silence_segments(self):
        vad = [[142870, 156260], [159720, 172040], [175490, 188040]]
        out = merge_vad(vad, 15000)
        self.assertEqual(out, vad)
        self.assert_valid(vad, out)

    def test_no_leading_silence_segment(self):
        # Speech starting after max_length used to produce [0, first_start].
        vad = [[20000, 30000], [31000, 40000]]
        out = merge_vad(vad, 15000)
        self.assert_valid(vad, out)
        self.assertNotIn([0, 20000], out)

    def test_empty_input(self):
        self.assertEqual(merge_vad([], 15000), [])

    def test_single_segment(self):
        self.assertEqual(merge_vad([[1000, 5000]], 15000), [[1000, 5000]])

    def test_single_segment_longer_than_max_length(self):
        self.assertEqual(merge_vad([[1000, 40000]], 15000), [[1000, 40000]])

    def test_long_segment_among_short_ones(self):
        vad = [[0, 2000], [3000, 40000], [41000, 43000]]
        out = merge_vad(vad, 15000)
        self.assert_valid(vad, out)

    def test_short_segments_are_still_merged(self):
        vad = [[0, 4000], [5000, 9000], [10000, 14000]]
        self.assertEqual(merge_vad(vad, 15000), [[0, 14000]])

    def test_random_inputs(self):
        rng = random.Random(0)
        for _ in range(500):
            t, vad = rng.randint(0, 20000), []
            for _ in range(rng.randint(2, 30)):
                d = rng.choice([rng.randint(200, 3000), rng.randint(8000, 16000), rng.randint(15000, 30000)])
                vad.append([t, t + d])
                t += d + rng.randint(10, 5000)
            for max_length in (5000, 15000, 20000):
                self.assert_valid(vad, merge_vad(vad, max_length))


if __name__ == "__main__":
    unittest.main()
