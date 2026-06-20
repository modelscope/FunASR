"""Unit tests for the GLM-ASR vLLM fp16 degraded-output guard.

Regression guard: ``fp16`` can produce degraded or garbage transcription for
GLM-ASR (numerical overflow in the audio embedding path), mirroring the
documented Fun-ASR-Nano behaviour. Requesting it must warn once so users are
not silently handed poor output, while leaving the requested value untouched.

The helper is dependency-free, so these tests run without a GPU, torch, or
vLLM.
"""

import logging
import unittest

from funasr.models.glm_asr import vllm_utils
from funasr.models.glm_asr.vllm_utils import warn_if_degraded_dtype


class TestWarnIfDegradedDtype(unittest.TestCase):
    def setUp(self):
        # Reset the once-per-process warning flag so each test is independent.
        vllm_utils._warned_fp16 = False

    def test_returns_value_unchanged(self):
        for dtype in ("bf16", "fp16", "fp32", "something-else"):
            self.assertEqual(warn_if_degraded_dtype(dtype), dtype)

    def test_fp16_warns(self):
        with self.assertLogs(vllm_utils.logger, level=logging.WARNING) as cm:
            warn_if_degraded_dtype("fp16")
        self.assertEqual(len(cm.records), 1)
        self.assertIn("fp16", cm.output[0])

    def test_fp16_warns_only_once(self):
        with self.assertLogs(vllm_utils.logger, level=logging.WARNING) as cm:
            warn_if_degraded_dtype("fp16")
            # Subsequent calls must not emit additional warnings.
            warn_if_degraded_dtype("fp16")
        self.assertEqual(len(cm.records), 1)

    def test_safe_values_do_not_warn(self):
        # Capture records directly (assertNoLogs is only available on 3.10+).
        records = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Collect(level=logging.WARNING)
        vllm_utils.logger.addHandler(handler)
        try:
            warn_if_degraded_dtype("bf16")
            warn_if_degraded_dtype("fp32")
        finally:
            vllm_utils.logger.removeHandler(handler)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
