import unittest
from unittest import mock

import torch

from funasr.train_utils import device_funcs


class _FakeAccelerator:
    """Stand-in for ``torch.cuda`` / ``torch.npu`` / ``torch.xpu``."""

    def __init__(self, available):
        self._available = available
        self.device_calls = []
        self.empty_cache_calls = 0

    def is_available(self):
        return self._available

    def set_device(self, index):
        self.device_calls.append(index)

    def empty_cache(self):
        self.empty_cache_calls += 1


def _fake_modules(**availability):
    return {
        device_type: _FakeAccelerator(availability.get(device_type, False))
        for device_type in device_funcs._ACCELERATOR_DEVICE_TYPES
    }


class TestAcceleratorResolution(unittest.TestCase):
    """The training entry points must bind devices without assuming CUDA."""

    def _patch(self, **availability):
        modules = _fake_modules(**availability)
        patcher = mock.patch.object(
            device_funcs,
            "_accelerator_module",
            side_effect=lambda device_type: modules.get(device_type),
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        return modules

    def test_device_type_prefers_cuda_then_npu_then_xpu(self):
        self._patch(cuda=True, npu=True, xpu=True)
        self.assertEqual(device_funcs.get_accelerator_device_type(), "cuda")
        self._patch(npu=True, xpu=True)
        self.assertEqual(device_funcs.get_accelerator_device_type(), "npu")
        self._patch(xpu=True)
        self.assertEqual(device_funcs.get_accelerator_device_type(), "xpu")
        self._patch()
        self.assertEqual(device_funcs.get_accelerator_device_type(), "cpu")

    def test_accelerator_device_follows_active_backend(self):
        self._patch()
        self.assertEqual(device_funcs.accelerator_device(0), torch.device("cpu"))
        self._patch(cuda=True)
        self.assertEqual(device_funcs.accelerator_device(3), torch.device("cuda", 3))

    def test_set_accelerator_device_binds_local_rank(self):
        modules = self._patch(npu=True)
        resolved = device_funcs.set_accelerator_device(3)
        self.assertEqual(resolved, "npu")
        self.assertEqual(modules["npu"].device_calls, [3])

    def test_set_accelerator_device_is_noop_without_accelerator(self):
        modules = self._patch()
        self.assertEqual(device_funcs.set_accelerator_device(1), "cpu")
        self.assertTrue(all(not module.device_calls for module in modules.values()))

    def test_default_dist_backend_matches_accelerator(self):
        self._patch(npu=True)
        self.assertEqual(device_funcs.get_default_dist_backend(), "hccl")
        self._patch(cuda=True)
        self.assertEqual(device_funcs.get_default_dist_backend(), "nccl")
        self._patch()
        self.assertEqual(device_funcs.get_default_dist_backend(), "nccl")

    def test_npu_backend_needs_no_capability_probe(self):
        """HCCL is the canonical backend name for Ascend NPU, so a build that
        exposes no ``is_hccl_available()`` probe still resolves to it."""
        self._patch(npu=True)
        with mock.patch.object(device_funcs, "_dist_backend_available", return_value=False):
            self.assertEqual(device_funcs.get_default_dist_backend(), "hccl")

    def test_default_dist_backend_for_xpu_is_xccl(self):
        """An XPU run gets XCCL instead of the historical NCCL default, which an
        XPU-only build (no NCCL) cannot initialise at all."""
        self._patch(xpu=True)
        self.assertEqual(device_funcs.get_default_dist_backend(), "xccl")
        with mock.patch.object(
            device_funcs,
            "_dist_backend_available",
            side_effect=lambda backend: backend == "xccl",
        ):
            self.assertEqual(device_funcs.get_default_dist_backend(), "xccl")

    def test_xpu_backend_prefers_xccl_over_the_legacy_ccl_name(self):
        """When the build exposes both probes, the current XCCL name wins."""
        self._patch(xpu=True)
        with mock.patch.object(device_funcs, "_dist_backend_available", return_value=True):
            self.assertEqual(device_funcs.get_default_dist_backend(), "xccl")

    def test_xpu_backend_falls_back_to_the_legacy_ccl_name(self):
        """A build whose probe only knows the former oneCCL name still resolves."""
        self._patch(xpu=True)
        with mock.patch.object(
            device_funcs,
            "_dist_backend_available",
            side_effect=lambda backend: backend == "ccl",
        ):
            self.assertEqual(device_funcs.get_default_dist_backend(), "ccl")

    def test_clear_accelerator_cache_dispatches_to_active_backend(self):
        modules = self._patch(npu=True)
        device_funcs.clear_accelerator_cache(torch.device("cpu"))
        self.assertEqual(modules["npu"].empty_cache_calls, 0)
        device_funcs.clear_accelerator_cache(None)
        self.assertEqual(modules["npu"].empty_cache_calls, 1)


if __name__ == "__main__":
    unittest.main()
