"""Unit tests for GLM-ASR vLLM sampling-parameter handling.

These tests exercise ``GLMASRVLLMEngine.generate`` without a GPU or a real
vLLM installation: the vLLM entry points are stubbed in ``sys.modules`` and the
audio/encoder/engine collaborators are mocked, so only the sampling-parameter
wiring is under test.
"""

import re
import sys
import types
import unittest
from unittest import mock


def _install_vllm_stub():
    """Install a minimal ``vllm`` stub whose SamplingParams records kwargs."""

    captured = {}

    class _RecordingSamplingParams:
        def __init__(self, **kwargs):
            captured.clear()
            captured.update(kwargs)

    class _EmbedsPrompt:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.SamplingParams = _RecordingSamplingParams
    vllm_mod.LLM = object
    inputs_mod = types.ModuleType("vllm.inputs")
    inputs_mod.EmbedsPrompt = _EmbedsPrompt
    data_mod = types.ModuleType("vllm.inputs.data")
    data_mod.EmbedsPrompt = _EmbedsPrompt

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.inputs"] = inputs_mod
    sys.modules["vllm.inputs.data"] = data_mod
    return captured


class GLMASRSamplingParamsTest(unittest.TestCase):
    def setUp(self):
        self.captured = _install_vllm_stub()
        from funasr.models.glm_asr.inference_vllm import GLMASRVLLMEngine

        # Build an engine without running __init__ (no model load / GPU needed).
        engine = GLMASRVLLMEngine.__new__(GLMASRVLLMEngine)
        engine.device = "cpu"
        engine._encode_audio = mock.Mock(return_value="audio_embeds")
        engine._build_prompt_embeds = mock.Mock(
            return_value=mock.Mock(float=lambda: "embeds")
        )

        token_out = types.SimpleNamespace(token_ids=[1, 2, 3])
        vllm_output = types.SimpleNamespace(outputs=[token_out])
        engine.vllm_engine = mock.Mock()
        engine.vllm_engine.generate = mock.Mock(return_value=[vllm_output])
        engine.tokenizer = mock.Mock()
        engine.tokenizer.decode = mock.Mock(return_value="hello world")
        self.engine = engine

    def test_defaults_preserve_greedy_behavior(self):
        results = self.engine.generate("a.wav")
        self.assertEqual(results, [{"key": "a", "text": "hello world"}])
        self.assertEqual(self.captured["temperature"], 0.0)
        self.assertEqual(self.captured["top_p"], 1.0)
        self.assertEqual(self.captured["top_k"], -1)
        self.assertEqual(self.captured["repetition_penalty"], 1.0)
        self.assertEqual(self.captured["max_tokens"], 500)

    def test_caller_sampling_params_are_forwarded(self):
        self.engine.generate(
            "a.wav", max_new_tokens=128, temperature=0.7, top_p=0.9, top_k=20
        )
        self.assertEqual(self.captured["max_tokens"], 128)
        self.assertEqual(self.captured["temperature"], 0.7)
        self.assertEqual(self.captured["top_p"], 0.9)
        self.assertEqual(self.captured["top_k"], 20)

    def test_non_positive_top_k_is_normalized_to_disabled(self):
        self.engine.generate("a.wav", top_k=0)
        self.assertEqual(self.captured["top_k"], -1)

    def test_repetition_penalty_is_forced_neutral_in_prompt_embeds_mode(self):
        # A non-neutral repetition_penalty would crash vLLM prompt-embeds mode
        # (issue #2948), so it must be coerced back to 1.0 rather than forwarded.
        self.engine.generate("a.wav", repetition_penalty=1.3)
        self.assertEqual(self.captured["repetition_penalty"], 1.0)

    def test_neutral_repetition_penalty_passes_through(self):
        self.engine.generate("a.wav", repetition_penalty=1.0)
        self.assertEqual(self.captured["repetition_penalty"], 1.0)


class SafeRepetitionPenaltyTest(unittest.TestCase):
    def setUp(self):
        _install_vllm_stub()
        import funasr.models.glm_asr.inference_vllm as mod

        self.mod = mod
        # Reset the process-wide warn-once flag between tests.
        mod._warned_rep_penalty = False

    def test_neutral_and_none_map_to_one(self):
        self.assertEqual(self.mod._safe_repetition_penalty(1.0), 1.0)
        self.assertEqual(self.mod._safe_repetition_penalty(None), 1.0)

    def test_non_neutral_is_coerced_and_warns_once(self):
        with self.assertLogs(self.mod.logger, level="WARNING") as cm:
            self.assertEqual(self.mod._safe_repetition_penalty(1.5), 1.0)
        self.assertTrue(any("2948" in line for line in cm.output))
        self.assertTrue(self.mod._warned_rep_penalty)


if __name__ == "__main__":
    unittest.main()
