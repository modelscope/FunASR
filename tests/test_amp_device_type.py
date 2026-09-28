"""Regression tests for the device type used by the AMP shim.

The shim used to pin ``device_type="cuda"``, which ``torch.amp`` turns into a
silent no-op on any other accelerator (it only warns "CUDA is not available ...
Disabling autocast"), so ``use_fp16``/``use_bf16`` training ran in float32 on
Ascend NPU / XPU / MPS builds, and ``GradScaler(enabled=True)`` was silently
disabled as well.
"""

import importlib.util
import types
from pathlib import Path

import pytest
import torch

AMP_PATH = Path(__file__).resolve().parents[1] / "funasr" / "utils" / "amp.py"


def _load_amp():
    spec = importlib.util.spec_from_file_location("funasr_amp_device_type", AMP_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class _Device:
    """Stand-in for a ``torch.device`` of a backend that may be unavailable."""

    def __init__(self, device_type):
        self.type = device_type


class _Accelerator:
    def __init__(self, device_type, error=None):
        self._device_type = device_type
        self._error = error

    def current_accelerator(self):
        if self._error is not None:
            raise self._error
        if self._device_type is None:
            return None
        return _Device(self._device_type)


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self._result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result


def _patch_accelerator(monkeypatch, device_type, error=None):
    monkeypatch.setattr(
        torch, "accelerator", _Accelerator(device_type, error=error), raising=False
    )


@pytest.mark.parametrize(
    "device_type,expected",
    [("cuda", "cuda"), ("npu", "npu"), ("xpu", "xpu"), ("mps", "mps")],
)
def test_resolve_amp_device_type_keeps_the_active_accelerator(monkeypatch, device_type, expected):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, device_type)

    assert amp.resolve_amp_device_type() == expected


@pytest.mark.parametrize("device_type", [None, "mtia", "meta"])
def test_resolve_amp_device_type_falls_back_to_cuda(monkeypatch, device_type):
    """No accelerator (CPU-only builds) keeps the historical no-op default."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, device_type)

    assert amp.resolve_amp_device_type() == "cuda"


def test_resolve_amp_device_type_survives_a_missing_or_failing_accelerator(monkeypatch):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu", error=RuntimeError("no accelerator compiled"))
    assert amp.resolve_amp_device_type() == "cuda"

    monkeypatch.delattr(torch, "accelerator", raising=False)
    assert amp.resolve_amp_device_type() == "cuda"


def test_autocast_resolves_the_device_type_for_the_legacy_signature(monkeypatch):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    recorder = _Recorder(result="ctx")
    monkeypatch.setattr(amp, "_amp_autocast", recorder)

    assert amp.autocast(enabled=True, dtype=torch.float16) == "ctx"
    assert recorder.calls == [
        (("npu",), {"dtype": torch.float16, "enabled": True, "cache_enabled": True})
    ]


def test_autocast_keeps_an_explicit_device_type(monkeypatch):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    recorder = _Recorder(result="ctx")
    monkeypatch.setattr(amp, "_amp_autocast", recorder)

    amp.autocast(False)
    amp.autocast(True, torch.bfloat16, False, "cuda")
    amp.autocast(enabled=False, device_type="cuda")

    assert [args[0] for args, _ in recorder.calls] == ["npu", "cuda", "cuda"]


def test_grad_scaler_prefers_the_accelerator_amp_module(monkeypatch):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _Recorder()
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(amp=types.SimpleNamespace(GradScaler=accelerator_scaler)),
        raising=False,
    )
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler(enabled=True)

    assert generic_scaler.calls == []
    assert accelerator_scaler.calls == [((), {"enabled": True})]


def test_grad_scaler_passes_the_resolved_device_to_the_generic_implementation(monkeypatch):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "mps")
    # a backend without its own amp module must not shadow the generic scaler
    monkeypatch.setattr(torch, "mps", types.SimpleNamespace(), raising=False)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler(enabled=True)
    amp.GradScaler("cuda", enabled=False)

    assert generic_scaler.calls == [
        ((), {"enabled": True, "device": "mps"}),
        (("cuda",), {"enabled": False}),
    ]


def test_grad_scaler_keeps_cuda_and_cpu_on_the_generic_implementation(monkeypatch):
    amp = _load_amp()
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    _patch_accelerator(monkeypatch, "cuda")
    amp.GradScaler(enabled=True)
    assert generic_scaler.calls == [((), {"enabled": True, "device": "cuda"})]

    _patch_accelerator(monkeypatch, None)
    amp.GradScaler(enabled=True)
    assert generic_scaler.calls[-1] == ((), {"enabled": True, "device": "cuda"})
