import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

from funasr.utils.amp import autocast


@pytest.mark.parametrize("args", [(False,), (False, torch.float16, False)])
def test_autocast_preserves_legacy_positional_arguments(args):
    with autocast(*args):
        pass


def test_amp_import_falls_back_when_torch_amp_is_missing(monkeypatch):
    legacy_autocast = object()
    legacy_grad_scaler = object()
    torch_stub = types.ModuleType("torch")
    torch_stub.__path__ = []
    cuda_stub = types.ModuleType("torch.cuda")
    cuda_stub.__path__ = []
    cuda_amp_stub = types.ModuleType("torch.cuda.amp")
    cuda_amp_stub.autocast = legacy_autocast
    cuda_amp_stub.GradScaler = legacy_grad_scaler
    torch_stub.cuda = cuda_stub
    cuda_stub.amp = cuda_amp_stub

    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.cuda", cuda_stub)
    monkeypatch.setitem(sys.modules, "torch.cuda.amp", cuda_amp_stub)

    module_path = Path(__file__).parents[1] / "funasr" / "utils" / "amp.py"
    spec = importlib.util.spec_from_file_location("funasr_amp_legacy_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.autocast is legacy_autocast
    assert module.GradScaler is legacy_grad_scaler
