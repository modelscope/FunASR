"""Unit tests for GLM-ASR vLLM result-key deduplication.

These tests exercise ``GLMASRVLLMEngine.generate`` without a GPU or a real
vLLM installation: the vLLM entry points are stubbed in ``sys.modules`` and the
audio/encoder/engine collaborators are mocked, so only the result-key wiring is
under test. The pure ``_dedup_keys`` helper is also tested directly.
"""

import sys
import types
import unittest
from unittest import mock


def _install_vllm_stub():
    """Install a minimal ``vllm`` stub so ``inference_vllm`` imports without GPU."""

    class _SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class _EmbedsPrompt:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    vllm_mod = types.ModuleType("vllm")
    vllm_mod.SamplingParams = _SamplingParams
    vllm_mod.LLM = object
    inputs_mod = types.ModuleType("vllm.inputs")
    inputs_mod.EmbedsPrompt = _EmbedsPrompt
    data_mod = types.ModuleType("vllm.inputs.data")
    data_mod.EmbedsPrompt = _EmbedsPrompt

    sys.modules["vllm"] = vllm_mod
    sys.modules["vllm.inputs"] = inputs_mod
    sys.modules["vllm.inputs.data"] = data_mod


class DedupKeysHelperTest(unittest.TestCase):
    def setUp(self):
        _install_vllm_stub()
        from funasr.models.glm_asr import inference_vllm

        self.inference_vllm = inference_vllm
        # Reset the warn-once flag so warning-related assertions are deterministic.
        inference_vllm._warned_dup_keys = False

    def test_collision_free_input_unchanged(self):
        keys = ["a", "b", "c"]
        self.assertEqual(self.inference_vllm._dedup_keys(keys), ["a", "b", "c"])

    def test_repeated_keys_get_deterministic_suffix(self):
        keys = ["seg", "seg", "other", "seg"]
        self.assertEqual(
            self.inference_vllm._dedup_keys(keys),
            ["seg", "seg_1", "other", "seg_2"],
        )

    def test_suffix_does_not_clash_with_existing_key(self):
        # A naive "seg" -> "seg_1" scheme would re-collide with the literal
        # "seg_1" input; the result must stay globally unique.
        keys = ["seg", "seg_1", "seg"]
        result = self.inference_vllm._dedup_keys(keys)
        self.assertEqual(result, ["seg", "seg_1", "seg_2"])
        self.assertEqual(len(set(result)), len(result))

    def test_empty_input(self):
        self.assertEqual(self.inference_vllm._dedup_keys([]), [])

    def test_sentinel_keys_preserved(self):
        keys = ["sample_0", "sample_1", "sample_0"]
        self.assertEqual(
            self.inference_vllm._dedup_keys(keys),
            ["sample_0", "sample_1", "sample_0_1"],
        )

    def test_does_not_mutate_input(self):
        keys = ["seg", "seg"]
        self.inference_vllm._dedup_keys(keys)
        self.assertEqual(keys, ["seg", "seg"])


class DedupResultKeysTest(unittest.TestCase):
    def setUp(self):
        _install_vllm_stub()
        from funasr.models.glm_asr.inference_vllm import GLMASRVLLMEngine

        # Build an engine without running __init__ (no model load / GPU needed).
        engine = GLMASRVLLMEngine.__new__(GLMASRVLLMEngine)
        engine.device = "cpu"
        engine._encode_audio = mock.Mock(return_value="audio_embeds")
        engine._build_prompt_embeds = mock.Mock(
            return_value=mock.Mock(float=lambda: "embeds")
        )
        engine.tokenizer = mock.Mock()
        engine.tokenizer.decode = mock.Mock(return_value="hello world")
        self.engine = engine

    def _set_outputs(self, n):
        token_out = types.SimpleNamespace(token_ids=[1, 2, 3])
        outs = [types.SimpleNamespace(outputs=[token_out]) for _ in range(n)]
        self.engine.vllm_engine = mock.Mock()
        self.engine.vllm_engine.generate = mock.Mock(return_value=outs)

    def test_duplicate_basenames_get_unique_keys(self):
        self._set_outputs(2)
        results = self.engine.generate(["spk1/segment.wav", "spk2/segment.wav"])
        self.assertEqual([r["key"] for r in results], ["segment", "segment_1"])
        # No transcript is dropped when results are folded into a {key: text} dict.
        self.assertEqual(len({r["key"] for r in results}), len(results))

    def test_distinct_basenames_unchanged(self):
        self._set_outputs(2)
        results = self.engine.generate(["a.wav", "b.wav"])
        self.assertEqual([r["key"] for r in results], ["a", "b"])

    def test_single_input_key(self):
        self._set_outputs(1)
        results = self.engine.generate("only.wav")
        self.assertEqual(results, [{"key": "only", "text": "hello world"}])


if __name__ == "__main__":
    unittest.main()
