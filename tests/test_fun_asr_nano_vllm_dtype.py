import logging
import sys
import types

import torch

from funasr.models.fun_asr_nano import inference_vllm
from funasr.models.fun_asr_nano import inference_vllm_streaming


class _FakeLLM:
    calls = []

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def get_tokenizer(self):
        return object()


def _install_fake_vllm(monkeypatch):
    vllm = types.ModuleType("vllm")
    vllm.LLM = _FakeLLM
    vllm.SamplingParams = object
    inputs = types.ModuleType("vllm.inputs")
    inputs.EmbedsPrompt = object
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.inputs", inputs)


def test_fp16_keeps_audio_compute_but_promotes_vllm_to_bf16(monkeypatch, caplog):
    _FakeLLM.calls.clear()
    _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(inference_vllm, "prepare_vllm_model_dir", lambda path: path)
    monkeypatch.setattr(
        inference_vllm.FunASRNanoVLLM,
        "_load_audio_components",
        lambda self, model_dir, **kwargs: None,
    )
    monkeypatch.setattr(
        inference_vllm.FunASRNanoVLLM,
        "_load_embedding_layer",
        lambda self, model_dir: None,
    )

    def load_streaming_audio(self, model_dir):
        self.frontend = types.SimpleNamespace(fs=16000)

    monkeypatch.setattr(
        inference_vllm_streaming.FunASRNanoStreamingVLLM,
        "_load_audio_components",
        load_streaming_audio,
    )
    monkeypatch.setattr(
        inference_vllm_streaming.FunASRNanoStreamingVLLM,
        "_load_embedding_layer",
        lambda self, model_dir: None,
    )

    with caplog.at_level(logging.WARNING):
        offline = inference_vllm.FunASRNanoVLLM("/tmp/model", dtype="fp16")
        streaming = inference_vllm_streaming.FunASRNanoStreamingVLLM(
            "/tmp/model", dtype="fp16"
        )

    assert offline.torch_dtype is torch.float16
    assert streaming.torch_dtype is torch.float16
    assert [call["dtype"] for call in _FakeLLM.calls] == ["bfloat16", "bfloat16"]
    assert "audio components remain in float16" in caplog.text
    assert "dtype='fp32'" in caplog.text


def test_vllm_dtype_mapping_preserves_supported_values(caplog):
    with caplog.at_level(logging.WARNING):
        assert inference_vllm._resolve_vllm_dtype("bf16") == "bfloat16"
        assert inference_vllm._resolve_vllm_dtype("fp32") == "float32"
        assert inference_vllm._resolve_vllm_dtype("custom") == "custom"

    assert not caplog.text
