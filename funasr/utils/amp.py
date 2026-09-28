"""Compatibility shim for torch.amp (autocast / GradScaler).

``torch.cuda.amp.autocast`` has been deprecated since torch 2.4 and
``torch.cuda.amp.GradScaler`` since torch 2.3 — both still work but emit
a ``FutureWarning`` on every use, and the replacement device-agnostic
APIs live in ``torch.amp`` (``autocast('cuda', ...)`` since 2.0,
``GradScaler('cuda', ...)`` since 2.3).

This module re-exports the non-deprecated ``torch.amp`` names when the
installed torch provides them, and falls back to ``torch.cuda.amp`` on
older torch versions. Import it instead of ``torch.cuda.amp`` directly:

    from funasr.utils.amp import autocast, GradScaler

Note: ``torch.amp.autocast`` requires a ``device_type`` argument (e.g.
``'cuda'``) that the deprecated ``torch.cuda.amp.autocast`` did not, so
the shim resolves ``device_type`` for callers that use the old signature
(``with autocast(enabled=..., dtype=...)``) — see
``resolve_amp_device_type``. Callers that know the device can still pass
``device_type=...`` explicitly.

Device type resolution
----------------------

The deprecated ``torch.cuda.amp`` API could only target CUDA, so the first
version of this shim pinned ``device_type="cuda"``. Off CUDA that is not an
error: ``torch.amp`` only warns

    UserWarning: CUDA is not available or torch_xla is imported. Disabling autocast.

and the context becomes a no-op, so ``use_fp16``/``use_bf16`` training silently
ran in float32 on Ascend NPU / XPU / MPS builds. ``GradScaler(enabled=True)``
was silently disabled the same way, so loss scaling was missing too.

Both now default to the accelerator that is actually active
(``torch.accelerator``) — the same device types that
``funasr/models/fun_asr_nano/device_utils.py`` accepts for the Nano inference
path — and keep the historical ``"cuda"`` (a no-op on CPU-only builds) when no
accelerator is available, so CPU behaviour is unchanged.

An explicit device always wins over the detected accelerator:
``GradScaler(device="cpu", enabled=False)`` and ``GradScaler("cpu", ...)``
keep whichever implementation understands that device, exactly as the
``torch.amp.GradScaler`` export they replace did. On an accelerator that ships
its own scaler the ``device`` argument is consumed by this shim (that
constructor takes no ``device``) while every other argument is forwarded
unchanged.
"""

import torch

torch_amp = getattr(torch, "amp", None)

# Device types that FunASR is willing to enter an AMP context on; anything else
# keeps the historical "cuda" no-op instead of risking an unsupported context.
_ACCELERATOR_DEVICE_TYPES = ("cuda", "npu", "xpu", "mps")

# Backends that ship their own AMP scaler. Everything else — including ``cpu``,
# whose ``torch.cpu.amp`` is only a deprecated re-export of the generic scaler —
# goes through ``torch.amp.GradScaler``.
_NATIVE_SCALER_DEVICE_TYPES = ("npu", "xpu")


def resolve_amp_device_type():
    """Return the ``torch.amp`` device type of the active accelerator.

    ``torch.accelerator`` reports the accelerator torch was built/registered
    for (``cuda``/``npu``/``xpu``/``mps``); it is ``None`` on CPU-only builds,
    where ``"cuda"`` is kept as the historical no-op default.
    """
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is not None:
        try:
            current = accelerator.current_accelerator()
        except (AttributeError, RuntimeError):
            current = None
        device_type = getattr(current, "type", None) if current is not None else None
        if device_type and str(device_type).lower() in _ACCELERATOR_DEVICE_TYPES:
            return str(device_type).lower()
    return "cuda"


def _resolve_device_type(device):
    """Return the device type of an explicitly passed device, or ``None``.

    Accepts whatever ``torch.amp.GradScaler`` accepts (``str``, ``torch.device``),
    so ``"npu"``, ``"cuda:0"`` and ``torch.device("cpu")`` all resolve.
    """
    if device is None:
        return None
    device_type = getattr(device, "type", None)
    if device_type is None:
        device_type = str(device).split(":")[0]
    device_type = str(device_type).strip().lower()
    return device_type or None


if torch_amp is not None and hasattr(torch_amp, "autocast") and hasattr(
    torch_amp, "GradScaler"
):
    from torch.amp import GradScaler as _amp_grad_scaler
    from torch.amp import autocast as _amp_autocast

    def autocast(enabled=True, dtype=None, cache_enabled=True, device_type=None):
        """torch.amp.autocast with the legacy CUDA autocast signature."""
        if device_type is None:
            device_type = resolve_amp_device_type()
        return _amp_autocast(
            device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )

    def _grad_scaler_class(device_type):
        """GradScaler implementation to use for ``device_type``.

        Accelerators that ship their own AMP scaler (``torch.npu.amp.GradScaler``
        from torch_npu, ``torch.xpu.amp.GradScaler``) are preferred: the generic
        ``torch.amp.GradScaler("npu")`` builds its scale on the device but then
        fails in ``update()`` (``aclnnAmpUpdateScale`` error 561002 on CANN 9.1.0
        with torch_npu 2.15), while ``torch.npu.amp.GradScaler`` runs a full
        fp16 step. CUDA and CPU stay on ``torch.amp.GradScaler``, i.e. the
        non-deprecated API this shim exists for — note that ``torch.cpu.amp``
        does exist on recent torch but is a deprecated re-export of the generic
        scaler with the device pinned, so it must not be picked up here.
        """
        if device_type in _NATIVE_SCALER_DEVICE_TYPES:
            amp_module = getattr(getattr(torch, device_type, None), "amp", None)
            grad_scaler = getattr(amp_module, "GradScaler", None)
            if grad_scaler is not None:
                return grad_scaler
        return _amp_grad_scaler

    def _caller_device(args, kwargs):
        """Return ``(device, where)`` for the device the caller passed, if any.

        ``torch.amp.GradScaler``'s first positional parameter is ``device``, so
        ``GradScaler("cpu", ...)`` and ``GradScaler(device="cpu", ...)`` name a
        device while anything else leaves the scaler defaults in place.
        """
        if "device" in kwargs:
            return kwargs["device"], "keyword"
        if args:
            return args[0], "positional"
        return None, None

    def GradScaler(*args, **kwargs):
        """Return the AMP ``GradScaler`` for the requested / active device.

        ``torch.amp.GradScaler``-compatible: an explicit ``device`` (positional
        or keyword) selects the implementation for that device and is forwarded
        untouched to the generic scaler. Without an explicit device the active
        accelerator decides, and the device is filled in for the generic
        implementation only — accelerator-specific scalers default to their own
        device and their constructor takes no ``device`` argument.
        """
        device, where = _caller_device(args, kwargs)
        device_type = (
            resolve_amp_device_type() if device is None else _resolve_device_type(device)
        )
        scaler_class = _grad_scaler_class(device_type)
        if scaler_class is _amp_grad_scaler:
            if device is None:
                kwargs["device"] = device_type
            return scaler_class(*args, **kwargs)
        # The accelerator implementation owns its device and has no ``device``
        # parameter, so drop ours before forwarding the remaining arguments.
        if where == "keyword":
            kwargs.pop("device", None)
        elif where == "positional":
            args = args[1:]
        return scaler_class(*args, **kwargs)

else:
    from torch.cuda.amp import autocast, GradScaler

__all__ = ["autocast", "GradScaler", "resolve_amp_device_type"]
