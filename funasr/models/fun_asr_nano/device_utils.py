# Copyright FunASR (https://github.com/modelscope/FunASR). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

"""Device helpers for Fun-ASR-Nano runtime paths."""

_SUPPORTED_AUTOCAST_DEVICE_TYPES = {"cuda", "xpu", "mps", "npu"}


def _device_type_from_value(device):
    """Resolve a device type without requiring optional backend registration."""
    if device is None:
        return "cpu"

    device_type = getattr(device, "type", None)
    if device_type:
        return str(device_type).lower()

    if isinstance(device, str):
        return device.split(":", 1)[0].lower()

    return str(device).split(":", 1)[0].lower()


def resolve_autocast_device_type(device):
    """Return the torch.autocast device_type for a Fun-ASR-Nano device.

    PyTorch builds without torch_npu may reject ``torch.device("npu:0")`` before
    torch_npu registers the backend. Parse strings directly so NPU requests do
    not fall back to CPU autocast, which only supports bf16 and caused #3034.
    """
    device_type = _device_type_from_value(device)
    if device_type in _SUPPORTED_AUTOCAST_DEVICE_TYPES:
        return device_type
    return "cpu"
