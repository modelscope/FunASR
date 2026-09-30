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


class _NpuGradScalerStub:
    """``torch.npu.amp.GradScaler`` stand-in with torch_npu's real signature.

    The parameter order matters: torch_npu inserts ``dynamic`` before
    ``enabled``
    (https://github.com/Ascend/pytorch/blob/6d59edda7452b26267eecbb6c328d876a6521ada/torch_npu/npu/amp/grad_scaler.py#L93-L99),
    so a ``*args, **kwargs`` recorder cannot tell a correctly translated call
    from one whose positional tail landed one parameter too early.
    """

    def __init__(
        self,
        init_scale=2.0 ** 16,
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        dynamic=True,
        enabled=True,
    ):
        self.init_scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.dynamic = dynamic
        self._enabled = enabled

    def is_enabled(self):
        return self._enabled


def _patch_npu_scaler(monkeypatch, scaler=None):
    """Register an NPU AMP scaler (an argument recorder by default)."""
    if scaler is None:
        scaler = _Recorder()
    monkeypatch.setattr(
        torch,
        "npu",
        types.SimpleNamespace(amp=types.SimpleNamespace(GradScaler=scaler)),
        raising=False,
    )
    return scaler


def test_grad_scaler_honours_an_explicit_keyword_device(monkeypatch):
    """``GradScaler(device="cpu", ...)`` must not reach the NPU implementation.

    The accelerator scaler has no ``device`` parameter, so forwarding it would
    raise ``TypeError``; the generic implementation must be selected instead.
    """
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler(device="cpu", enabled=False)

    assert accelerator_scaler.calls == []
    assert generic_scaler.calls == [((), {"device": "cpu", "enabled": False})]


def test_grad_scaler_honours_an_explicit_positional_device(monkeypatch):
    """``GradScaler("cpu", ...)`` keeps ``"cpu"`` as the device, not as ``init_scale``."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler("cpu", enabled=False)

    assert accelerator_scaler.calls == []
    assert generic_scaler.calls == [(("cpu",), {"enabled": False})]


def test_grad_scaler_honours_an_explicit_torch_device(monkeypatch):
    """A ``torch.device`` is resolved by its type, like ``torch.amp.GradScaler``."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    device = torch.device("cpu")
    amp.GradScaler(device, enabled=False)

    assert accelerator_scaler.calls == []
    assert generic_scaler.calls == [((device,), {"enabled": False})]


def test_grad_scaler_drops_the_device_for_the_accelerator_implementation(monkeypatch):
    """An explicit accelerator device selects its scaler without forwarding ``device``."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler(device="npu", enabled=True)
    amp.GradScaler("npu", 1024.0)

    assert generic_scaler.calls == []
    assert accelerator_scaler.calls == [
        ((), {"enabled": True}),
        ((), {"init_scale": 1024.0}),
    ]


@pytest.mark.parametrize("enabled", [False, True])
def test_grad_scaler_translates_the_generic_positional_tail_for_the_npu_scaler(
    monkeypatch, enabled
):
    """``GradScaler("npu", 1024.0, 2.0, 0.5, 2000, enabled)`` keeps ``enabled``.

    The last parameter of the generic signature is ``enabled``, but torch_npu
    inserts ``dynamic`` before it, so forwarding the positional tail unchanged
    hands ``False`` to ``dynamic`` and leaves the scaler enabled — a caller
    asking to disable scaling gets an enabled static scaler instead.
    """
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    _patch_npu_scaler(monkeypatch, scaler=_NpuGradScalerStub)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    scaler = amp.GradScaler("npu", 1024.0, 2.0, 0.5, 2000, enabled)

    assert generic_scaler.calls == []
    assert isinstance(scaler, _NpuGradScalerStub)
    assert scaler.is_enabled() is enabled
    assert scaler.init_scale == 1024.0
    assert scaler.growth_factor == 2.0
    assert scaler.backoff_factor == 0.5
    assert scaler.growth_interval == 2000
    assert scaler.dynamic is True


def test_grad_scaler_translates_arguments_by_name_before_dispatch(monkeypatch):
    """The accelerator scaler is called with parameter names, not a shifted tail."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler("npu", 1024.0, 2.0, 0.5, 2000, False)

    assert generic_scaler.calls == []
    assert accelerator_scaler.calls == [
        (
            (),
            {
                "init_scale": 1024.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "enabled": False,
            },
        )
    ]


def test_grad_scaler_keeps_a_keyword_enabled_flag_for_the_npu_scaler(monkeypatch):
    """The keyword form keeps meaning the same thing on every scaler."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    _patch_npu_scaler(monkeypatch, scaler=_NpuGradScalerStub)

    assert amp.GradScaler(enabled=False).is_enabled() is False
    assert amp.GradScaler(device="npu", enabled=False).is_enabled() is False
    assert amp.GradScaler("npu", enabled=True).is_enabled() is True


def test_grad_scaler_keeps_an_explicit_cuda_device_on_the_generic_implementation(monkeypatch):
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler(device="cuda")
    amp.GradScaler("cuda:0", enabled=False)

    assert accelerator_scaler.calls == []
    assert generic_scaler.calls == [
        ((), {"device": "cuda"}),
        (("cuda:0",), {"enabled": False}),
    ]


def test_grad_scaler_without_a_device_still_prefers_the_accelerator(monkeypatch):
    """No explicit device keeps the previous behaviour: the active accelerator wins."""
    amp = _load_amp()
    _patch_accelerator(monkeypatch, "npu")
    accelerator_scaler = _patch_npu_scaler(monkeypatch)
    generic_scaler = _Recorder()
    monkeypatch.setattr(amp, "_amp_grad_scaler", generic_scaler)

    amp.GradScaler(enabled=True)

    assert generic_scaler.calls == []
    assert accelerator_scaler.calls == [((), {"enabled": True})]
