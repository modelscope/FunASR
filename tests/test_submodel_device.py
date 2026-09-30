"""CPU-only regressions using real AutoModel construction and inference orchestration.

Accelerator availability is mocked only to exercise configuration labels. Stand-in
models retain CPU weights; these tests do not claim CUDA/MPS hardware execution.
"""

import copy
import logging
import unittest
from unittest.mock import patch

import numpy as np
import pytest
import torch

from funasr.auto.auto_model import AutoModel
from funasr.register import tables


class RecordingModel(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.role = model
        self.calls = []

    def to(self, device):
        self.placement = str(device)
        return self  # Configuration probe only: weights always stay on CPU.

    def inference(self, data_in, key=None, **kwargs):
        if kwargs["device"] != self.placement:
            raise AssertionError(f"{self.role}: feature device differs from model placement")
        if kwargs["device"] == "cpu":
            features = torch.zeros(1, device=kwargs["device"])
            if features.device != self.weight.device:
                raise AssertionError("CPU feature/weight mismatch")
        self.calls.append(copy.deepcopy(kwargs))
        results = []
        for index, _ in enumerate(data_in):
            if self.role == "device-test-vad":
                result = {"key": key[index], "value": [[0, 1000]]}
            elif self.role == "device-test-spk":
                result = {"spk_embedding": torch.ones(1, 4)}
            elif self.role == "device-test-punc":
                result = {"text": "hello.", "punc_array": [2]}
            else:
                result = {"text": "hello", "timestamp": [[0, 1000]]}
            results.append(result)
        return results, {"batch_data_time": 1}


class TestSubmodelDevice(unittest.TestCase):
    def setUp(self):
        self.old_threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, self.old_threads)
        registry = {f"device-test-{role}": RecordingModel for role in ("asr", "vad", "punc", "spk")}
        self.start_patch(patch.dict(tables.model_classes, registry))
        self.start_patch(patch("torch.cuda.is_available", return_value=True))
        self.start_patch(patch("funasr.auto.auto_model.ClusterBackend", create=True))
        # Reject accidental model-hub access. model_conf={} uses the local registry.
        self.start_patch(
            patch("funasr.auto.auto_model.download_model", side_effect=AssertionError("network"))
        )
        self.audio = np.zeros(16000, dtype=np.float32)

    def start_patch(self, patcher):
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def make_model(self, with_vad=True, device="cuda:0", configs=None):
        configs = (
            configs
            if configs is not None
            else {
                role: {"model_conf": {}, "device": "cpu", f"{role}_only": "kept"}
                for role in ("vad", "punc", "spk")
            }
        )
        kwargs = dict(
            model="device-test-asr",
            model_conf={},
            device=device,
            ncpu=1,
            disable_update=True,
            disable_pbar=True,
            return_spk_res=False,
        )
        for role in ("vad", "punc", "spk") if with_vad else ("punc",):
            kwargs[f"{role}_model"] = f"device-test-{role}"
            kwargs[f"{role}_kwargs"] = configs[role]
        return AutoModel(**kwargs)

    def test_explicit_devices_and_caller_configs_survive_two_calls(self):
        configs = {
            role: {"model_conf": {"nested": [1]}, "device": "cpu", f"{role}_only": "kept"}
            for role in ("vad", "punc", "spk")
        }
        original = copy.deepcopy(configs)
        model = self.make_model(configs=configs)
        self.assertEqual(configs, original)
        first = model.generate(self.audio, device="cuda:7", hotword="runtime", batch_size_s=2)
        second = model.generate(self.audio)
        self.assertEqual(first[0]["text"], "hello.")
        self.assertEqual(second[0]["text"], "hello.")
        for role in ("vad", "punc", "spk"):
            child = getattr(model, f"{role}_model")
            self.assertEqual([call["device"] for call in child.calls], ["cpu", "cpu"])
            self.assertEqual(child.calls[0][f"{role}_only"], "kept")
            self.assertEqual(child.calls[0]["hotword"], "runtime")
            self.assertEqual(child.calls[0]["batch_size_s"], 2)
            self.assertNotIn("hotword", child.calls[1])
            self.assertNotIn("batch_size_s", child.calls[1])
            self.assertEqual(getattr(model, f"{role}_kwargs")["device"], "cpu")
        self.assertEqual([call["device"] for call in model.model.calls], ["cuda:0", "cuda:0"])
        self.assertEqual(configs, original)

    def test_punctuation_without_vad_preserves_placement(self):
        model = self.make_model(with_vad=False)
        model.generate(self.audio, device="cuda:7", hotword="runtime")
        model.generate(self.audio)
        self.assertEqual([call["device"] for call in model.punc_model.calls], ["cpu", "cpu"])
        self.assertEqual(model.punc_model.calls[0]["hotword"], "runtime")
        self.assertNotIn("hotword", model.punc_model.calls[1])

    def test_omitted_devices_inherit_resolved_asr_device(self):
        for available, expected in ((True, "cuda:0"), (False, "cpu")):
            with self.subTest(available=available), patch(
                "torch.cuda.is_available", return_value=available
            ):
                configs = {role: {"model_conf": {}} for role in ("vad", "punc", "spk")}
                model = self.make_model(configs=configs)
                model.generate(self.audio, device="cuda:7")
                model.generate(self.audio)
                for role in ("vad", "punc", "spk"):
                    child = getattr(model, f"{role}_model")
                    self.assertEqual(child.placement, expected)
                    self.assertEqual([call["device"] for call in child.calls], [expected, expected])
                self.assertEqual(configs, {role: {"model_conf": {}} for role in configs})

    def test_unavailable_explicit_submodel_devices_fall_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            configs = {
                role: {"model_conf": {}, "device": "cuda:0"} for role in ("vad", "punc", "spk")
            }
            model = self.make_model(device="cpu", configs=configs)
            model.generate(self.audio, device="cuda:7")
            model.generate(self.audio)
        for role in configs:
            self.assertEqual(getattr(model, f"{role}_model").placement, "cpu")
            self.assertEqual(
                [call["device"] for call in getattr(model, f"{role}_model").calls], ["cpu", "cpu"]
            )
            self.assertEqual(configs[role]["device"], "cuda:0")


@pytest.fixture
def fallback_model(monkeypatch):
    from funasr.auto.auto_model import _current_blas_threads, _limit_blas_threads

    original_threads = torch.get_num_threads()
    original_blas = _current_blas_threads()
    for role in ("asr", "vad", "punc", "spk"):
        monkeypatch.setitem(tables.model_classes, f"device-test-{role}", RecordingModel)

    def reject_download(**kwargs):
        raise AssertionError("device tests must not access model hubs")

    monkeypatch.setattr("funasr.auto.auto_model.download_model", reject_download)
    with patch("funasr.auto.auto_model.ClusterBackend", create=True):
        def build(**kwargs):
            return AutoModel(
                model="device-test-asr",
                model_conf={},
                ncpu=1,
                disable_update=True,
                disable_pbar=True,
                **kwargs,
            )

        yield build
    if original_blas is not None:
        _limit_blas_threads(original_blas)
    torch.set_num_threads(original_threads)


_ACCELERATOR_PROBES = [
    ("cuda:0", "torch.cuda.is_available"),
    ("xpu:0", "torch.xpu.is_available"),
    ("mps", "torch.backends.mps.is_available"),
    ("npu:0", "funasr.auto.auto_model.is_npu_available"),
]


@pytest.mark.parametrize("device,probe", _ACCELERATOR_PROBES)
def test_unavailable_accelerator_warns_and_preserves_cpu_fallback(
    fallback_model, caplog, device, probe
):
    with patch(probe, return_value=False), caplog.at_level(logging.WARNING):
        model = fallback_model(device=device, batch_size=8)
    assert model.kwargs["device"] == model.model.placement == "cpu"
    assert model.kwargs["batch_size"] == 1
    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING
    message = caplog.records[0].getMessage()
    assert device in message
    assert "device-test-asr" in message
    assert "falling back to CPU" in message
    assert "PyTorch" in message


@pytest.mark.parametrize("device,probe", _ACCELERATOR_PROBES)
def test_available_accelerator_does_not_warn(fallback_model, caplog, device, probe):
    with patch(probe, return_value=True), caplog.at_level(logging.WARNING):
        model = fallback_model(device=device, batch_size=8)
    assert model.kwargs["device"] == model.model.placement == device
    assert model.kwargs["batch_size"] == 8
    assert not caplog.records


@pytest.mark.parametrize("device,probe", _ACCELERATOR_PROBES)
@pytest.mark.parametrize("available", [False, True])
def test_ngpu_zero_is_intentional_cpu_use(
    fallback_model, caplog, device, probe, available
):
    with patch(probe, return_value=available), caplog.at_level(logging.WARNING):
        model = fallback_model(device=device, ngpu=0, batch_size=8)
    assert model.kwargs["device"] == model.model.placement == "cpu"
    assert model.kwargs["batch_size"] == 1
    assert not caplog.records


def test_explicit_cpu_does_not_warn(fallback_model, caplog):
    with caplog.at_level(logging.WARNING):
        model = fallback_model(device="cpu", batch_size=8)
    assert model.kwargs["device"] == model.model.placement == "cpu"
    assert model.kwargs["batch_size"] == 8
    assert not caplog.records


def test_default_cuda_fallback_warns_once_with_inherited_submodels(fallback_model, caplog):
    configs = {
        f"{role}_{kind}": f"device-test-{role}" if kind == "model" else {"model_conf": {}}
        for role in ("vad", "punc", "spk")
        for kind in ("model", "kwargs")
    }
    original = copy.deepcopy(configs)
    with patch("torch.cuda.is_available", return_value=False), caplog.at_level(logging.WARNING):
        model = fallback_model(**configs)
    assert model.kwargs["device"] == "cpu"
    for role in ("vad", "punc", "spk"):
        assert getattr(model, f"{role}_kwargs")["device"] == "cpu"
        assert getattr(model, f"{role}_model").placement == "cpu"
    assert configs == original
    assert len(caplog.records) == 1
    assert "cuda" in caplog.records[0].getMessage()


def test_explicit_submodel_fallback_identifies_the_submodel(fallback_model, caplog):
    config = {"model_conf": {}, "device": "cuda:0"}
    original = copy.deepcopy(config)
    with patch("torch.cuda.is_available", return_value=False), caplog.at_level(logging.WARNING):
        model = fallback_model(
            device="cpu", punc_model="device-test-punc", punc_kwargs=config
        )
    assert model.punc_kwargs["device"] == model.punc_model.placement == "cpu"
    assert config == original
    assert len(caplog.records) == 1
    assert "device-test-punc" in caplog.records[0].getMessage()


def test_fallback_respects_application_logging_level(fallback_model, caplog):
    with patch("torch.cuda.is_available", return_value=False), caplog.at_level(logging.ERROR):
        model = fallback_model(device="cuda:0")
    assert model.kwargs["device"] == "cpu"
    assert not caplog.records


if __name__ == "__main__":
    unittest.main()
