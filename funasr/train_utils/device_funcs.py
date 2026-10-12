import dataclasses
import warnings

import numpy as np
import torch


def to_device(data, device=None, dtype=None, non_blocking=False, copy=False):
    """Change the device of object recursively"""
    if isinstance(data, dict):
        return {k: to_device(v, device, dtype, non_blocking, copy) for k, v in data.items()}
    elif dataclasses.is_dataclass(data) and not isinstance(data, type):
        return type(data)(
            *[to_device(v, device, dtype, non_blocking, copy) for v in dataclasses.astuple(data)]
        )
    # maybe namedtuple. I don't know the correct way to judge namedtuple.
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[to_device(o, device, dtype, non_blocking, copy) for o in data])
    elif isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device, dtype, non_blocking, copy) for v in data)
    elif isinstance(data, np.ndarray):
        return to_device(torch.from_numpy(data), device, dtype, non_blocking, copy)
    elif isinstance(data, torch.Tensor):
        return data.to(device, dtype, non_blocking, copy)
    else:
        return data


def force_gatherable(data, device):
    """Change object to gatherable in torch.nn.DataParallel recursively

    The difference from to_device() is changing to torch.Tensor if float or int
    value is found.

    The restriction to the returned value in DataParallel:
        The object must be
        - torch.cuda.Tensor
        - 1 or more dimension. 0-dimension-tensor sends warning.
        or a list, tuple, dict.

    """
    if isinstance(data, dict):
        return {k: force_gatherable(v, device) for k, v in data.items()}
    # DataParallel can't handle NamedTuple well
    elif isinstance(data, tuple) and type(data) is not tuple:
        return type(data)(*[force_gatherable(o, device) for o in data])
    elif isinstance(data, (list, tuple, set)):
        return type(data)(force_gatherable(v, device) for v in data)
    elif isinstance(data, np.ndarray):
        return force_gatherable(torch.from_numpy(data), device)
    elif isinstance(data, torch.Tensor):
        if data.dim() == 0:
            # To 1-dim array
            data = data[None]
        return data.to(device)
    elif isinstance(data, float):
        return torch.tensor([data], dtype=torch.float, device=device)
    elif isinstance(data, int):
        return torch.tensor([data], dtype=torch.long, device=device)
    elif data is None:
        return None
    else:
        warnings.warn(f"{type(data)} may not be gatherable by DataParallel")
        return data


_ACCELERATOR_DEVICE_TYPES = ("cuda", "npu", "xpu")

#: Default ``torch.distributed`` backend for each accelerator, most preferred
#: first. XCCL (oneCCL) is what current torch exposes for XPU; builds that
#: predate the 2.6 rename know the same backend as ``"ccl"``.
_ACCELERATOR_DIST_BACKENDS = {"npu": ("hccl",), "xpu": ("xccl", "ccl")}


def _accelerator_module(device_type):
    """Return the torch submodule owning ``device_type``, or ``None``.

    ``torch_npu`` registers ``torch.npu`` lazily, so it is imported on demand;
    builds without it simply have no ``torch.npu`` attribute. ``torch.cuda`` and
    ``torch.xpu`` exist on every build, but their ``is_available()`` reports
    ``False`` when that backend is missing.
    """
    if device_type == "npu":
        try:
            import torch_npu  # noqa: F401
        except ImportError:
            return None
    return getattr(torch, device_type, None)


def get_accelerator_device_type():
    """Return the device type of the active accelerator, or ``"cpu"``.

    Probes the same backends FunASR accepts for ``AutoModel`` device fallback
    (CUDA, then Ascend NPU, then XPU). CPU-only torch builds report ``"cpu"`` so
    callers can keep a device-agnostic, no-op codepath.
    """
    for device_type in _ACCELERATOR_DEVICE_TYPES:
        module = _accelerator_module(device_type)
        if module is None:
            continue
        try:
            if module.is_available():
                return device_type
        except Exception:
            continue
    return "cpu"


def accelerator_device(local_rank=0):
    """Return the ``torch.device`` for ``local_rank`` on the active accelerator."""
    device_type = get_accelerator_device_type()
    if device_type == "cpu":
        return torch.device("cpu")
    return torch.device(device_type, int(local_rank))


def set_accelerator_device(local_rank=0):
    """Bind this process to ``local_rank`` on the active accelerator.

    Replaces the CUDA-only ``torch.cuda.set_device(local_rank)``, which raises
    (``module 'torch._C' has no attribute '_cuda_setDevice'``) on a build
    without CUDA even when an Ascend NPU is present. Returns the resolved device
    type; for ``"cpu"`` nothing is bound, matching the previous CPU behaviour.
    """
    device_type = get_accelerator_device_type()
    if device_type not in _ACCELERATOR_DEVICE_TYPES:
        return device_type
    module = _accelerator_module(device_type)
    set_device = getattr(module, "set_device", None)
    if set_device is not None:
        set_device(int(local_rank))
    return device_type


def _dist_backend_available(backend):
    """Ask torch whether ``backend`` is usable: ``is_<backend>_available()``.

    A build that does not expose the probe cannot confirm anything, so it
    reports ``False`` and the caller keeps the accelerator's canonical backend
    name instead of inventing availability.
    """
    distributed = getattr(torch, "distributed", None)
    probe = getattr(distributed, "is_{}_available".format(backend), None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:
        return False


def get_default_dist_backend():
    """Return the ``torch.distributed`` backend for the active accelerator.

    CUDA keeps the historical ``"nccl"``, Ascend NPU uses HCCL and XPU uses
    XCCL (``"ccl"`` on builds older than the oneCCL rename). Candidates are
    probed with ``torch.distributed.is_<backend>_available()`` where available
    so the selected backend matches the build, but a non-CUDA accelerator never
    falls back to NCCL. This is only the default — an explicit ``backend=...``
    still wins.
    """
    candidates = _ACCELERATOR_DIST_BACKENDS.get(get_accelerator_device_type())
    if not candidates:
        return "nccl"
    for backend in candidates:
        if _dist_backend_available(backend):
            return backend
    return candidates[0]


def clear_accelerator_cache(device=None):
    """Release cached allocator memory for ``device`` (no-op on CPU).

    ``torch.cuda.empty_cache()`` is a silent no-op on a build without CUDA, so
    the end-of-epoch release used by the training entry points did nothing on
    Ascend NPU / XPU. Dispatch to the accelerator that owns ``device`` instead.
    """
    device_type = getattr(device, "type", None) or get_accelerator_device_type()
    if device_type not in _ACCELERATOR_DEVICE_TYPES:
        return
    module = _accelerator_module(device_type)
    empty_cache = getattr(module, "empty_cache", None)
    if empty_cache is not None:
        empty_cache()
