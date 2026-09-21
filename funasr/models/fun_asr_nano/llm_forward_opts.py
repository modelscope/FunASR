"""Opt-in switches for the Fun-ASR-Nano LLM forward, read from ``llm_conf``; both default to off.

``llm_conf.sdpa_backends``: a list of ``flash`` / ``efficient`` / ``math``. The decoder forward runs
under ``torch.nn.attention.sdpa_kernel`` with these backends: the process-wide SDPA flags are set on
entry and restored on return, so another thread calling scaled_dot_product_attention meanwhile sees
the same selection (leave it unset when several models share one process concurrently).
``llm_conf.torch_compile``: run the decoder stack through ``torch.compile(dynamic=True)`` for
inputs on a CUDA device, decided per call; a CPU decoder and a 1-sequence batch stay eager.
"""

import torch

SDPA_BACKENDS = {"flash": "FLASH_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}


def resolve_sdpa_backends(names):
    """``llm_conf.sdpa_backends`` (a list of names, or ``None``) -> ``SDPBackend`` members."""
    if names is None:
        return None
    from torch.nn.attention import SDPBackend  # a pybind enum: getattr, not subscript

    backends = []
    for name in names:
        member = SDPA_BACKENDS.get(name.lower())
        if member is None:
            raise ValueError(
                f"llm_conf.sdpa_backends: unknown backend {name!r}, "
                f"choose from {sorted(SDPA_BACKENDS)}"
            )
        backends.append(getattr(SDPBackend, member))
    return backends


def configure_llm_forward(llm, llm_conf):
    """Install the switches ``llm_conf`` asks for on ``llm.model.forward`` (nothing by default)."""
    backends = resolve_sdpa_backends(llm_conf.get("sdpa_backends", None))
    compile_requested = bool(llm_conf.get("torch_compile", False))
    if backends is None and not compile_requested:
        return

    decoder = llm.model
    forward = eager = decoder.forward

    if compile_requested:
        compiled = torch.compile(eager, dynamic=True)

        def forward(*args, **kw):
            # Per call, not at construction: the trainer builds the model on the CPU and moves it
            # to the GPU afterwards. A 1-sequence batch would get its own specialised graph.
            x = kw.get("inputs_embeds", kw.get("input_ids"))
            if x is None and args:
                x = args[0]
            if x is None or x.device.type != "cuda" or x.shape[0] == 1:
                return eager(*args, **kw)
            return compiled(*args, **kw)

    if backends is not None:
        from torch.nn.attention import sdpa_kernel

        inner = forward

        def forward(*args, **kw):
            with sdpa_kernel(backends):  # wraps the compiled call, so Dynamo traces under it
                return inner(*args, **kw)

    decoder.forward = forward
