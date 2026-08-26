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

torch_amp = getattr(torch, "amp", None)
if torch_amp is not None and hasattr(torch_amp, "autocast") and hasattr(
    torch_amp, "GradScaler"
):
    from torch.amp import autocast as _amp_autocast
    from torch.amp import GradScaler

    def autocast(enabled=True, dtype=None, cache_enabled=True):
        """torch.amp.autocast with the legacy CUDA autocast signature."""
        return _amp_autocast(
            "cuda", dtype=dtype, enabled=enabled, cache_enabled=cache_enabled
        )

else:
    from torch.cuda.amp import autocast, GradScaler

__all__ = ["autocast", "GradScaler"]
