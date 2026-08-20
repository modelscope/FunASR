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
the shim supplies ``device_type="cuda"`` by default for callers that use
the old signature (``with autocast(enabled=..., dtype=...)``).
"""

import torch

if hasattr(torch.amp, "autocast") and hasattr(torch.amp, "GradScaler"):
    from torch.amp import autocast as _amp_autocast
    from torch.amp import GradScaler

    def autocast(*args, **kwargs):
        """torch.amp.autocast with device_type defaulting to "cuda"."""
        if not args and "device_type" not in kwargs:
            kwargs["device_type"] = "cuda"
        return _amp_autocast(*args, **kwargs)

else:
    from torch.cuda.amp import autocast, GradScaler

__all__ = ["autocast", "GradScaler"]
