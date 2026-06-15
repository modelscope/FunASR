"""Unit tests for Fun-ASR-Nano vLLM repetition-penalty handling.

Regression guard for issue #2948: a repetition penalty other than 1.0 is
incompatible with vLLM prompt-embeds mode and aborts the engine with a CUDA
"scatter gather index out of bounds" assertion. The serving paths must never
forward such a value to ``SamplingParams`` while ``enable_prompt_embeds=True``.

The helper is dependency-free, so these tests run without a GPU or vLLM.
"""

import logging
import unittest

from funasr.models.fun_asr_nano import vllm_utils
from funasr.models.fun_asr_nano.vllm_utils import (
    NEUTRAL_REPETITION_PENALTY,
    resolve_repetition_penalty,
)


class TestResolveRepetitionPenalty(unittest.TestCase):
    def setUp(self):
        # Reset the once-per-process warning flag so each test is independent.
        vllm_utils._warned_prompt_embeds = False

    def test_neutral_value_passes_through(self):
        self.assertEqual(resolve_repetition_penalty(1.0), 1.0)

    def test_none_maps_to_neutral(self):
        self.assertEqual(
            resolve_repetition_penalty(None), NEUTRAL_REPETITION_PENALTY
        )

    def test_nonneutral_is_clamped_in_prompt_embeds_mode(self):
        # The exact value that triggers the #2948 crash.
        self.assertEqual(
            resolve_repetition_penalty(1.3, prompt_embeds=True),
            NEUTRAL_REPETITION_PENALTY,
        )

    def test_nonneutral_preserved_for_token_prompts(self):
        # Regular token-prompt decoding can safely apply the penalty.
        self.assertEqual(
            resolve_repetition_penalty(1.3, prompt_embeds=False), 1.3
        )

    def test_warns_once_in_prompt_embeds_mode(self):
        with self.assertLogs(vllm_utils.logger, level=logging.WARNING) as cm:
            resolve_repetition_penalty(1.3, prompt_embeds=True)
            # Subsequent clamps must not emit additional warnings.
            resolve_repetition_penalty(1.5, prompt_embeds=True)
        self.assertEqual(len(cm.records), 1)
        self.assertIn("2948", cm.output[0])

    def test_no_warning_when_value_is_safe(self):
        # Capture records directly (assertNoLogs is only available on 3.10+).
        records = []

        class _Collect(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = _Collect(level=logging.WARNING)
        vllm_utils.logger.addHandler(handler)
        try:
            resolve_repetition_penalty(1.0, prompt_embeds=True)
            resolve_repetition_penalty(1.3, prompt_embeds=False)
        finally:
            vllm_utils.logger.removeHandler(handler)
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
