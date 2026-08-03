"""Offline unit tests for Fun-ASR-Nano LoRA injection.

The Qwen3 LLM is far too heavy to construct in a unit test, so these tests
drive the real injection method ``FunASRNano._apply_lora_to_llm`` against a
tiny fake LLM that mirrors the Qwen3 parameter layout
(``layers.<i>.self_attn.{q,k,v,o}_proj`` and ``layers.<i>.mlp.{gate,up,down}_proj``).
No model download is needed.

The fake LLM is frozen (``requires_grad=False`` + ``eval()``) *before*
injection, exactly mirroring the ``llm_conf.freeze`` (default true) step that
``FunASRNano.__init__`` runs prior to calling ``_apply_lora_to_llm``.

Coverage requested in review (modelscope/FunASR#3456):
  1. only configured target modules are replaced;
  2. the initial output matches the original linear layers;
  3. base weights stay frozen while ``lora_A``/``lora_B`` remain trainable
     with the documented configuration;
  4. a non-zero adapter survives a state-dict save/load round trip and
     train/eval transitions;
  5. the no-matching-target configuration has an explicit tested outcome.
"""

import os

import pytest
import torch
import torch.nn as nn

from funasr.models.fun_asr_nano.model import FunASRNano
from funasr.models.lora.layers import Linear as LoRALinear

HIDDEN = 8
N_LAYERS = 2

DEFAULT_CONF = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.05,
                "target_modules": ["q_proj", "v_proj"]}

# fp32 non-associativity: a merged forward is one matmul `x @ (W + d).T`, the
# unmerged forward is `x @ W'.T + (x @ A.T @ B.T) * scaling` — different op
# ordering gives ~1e-5 relative differences, well above torch.allclose's default
# 1e-5 rtol. 1e-3 is the fp32-appropriate tolerance and still detects a genuinely
# broken merge (which would differ by the full adapter magnitude, ~O(1-10)).
MERGE_RTOL = 1e-3
MERGE_ATOL = 1e-5


@pytest.fixture(autouse=True)
def _seed_rng():
    # every test must be reproducible: nn.Linear / LoRALinear init and the
    # explicit normal_() / randn() draws all consume the global RNG
    torch.manual_seed(0)


class _FakeQwen3(nn.Module):
    """Minimal Qwen3-layout LLM: N layers x self_attn (q/k/v/o) + mlp (gate/up/down)."""

    def __init__(self, hidden=HIDDEN, n_layers=N_LAYERS):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden)
        layers = []
        for _ in range(n_layers):
            layer = nn.Module()
            attn = nn.Module()
            attn.q_proj = nn.Linear(hidden, hidden)
            attn.k_proj = nn.Linear(hidden, hidden)
            attn.v_proj = nn.Linear(hidden, hidden)
            attn.o_proj = nn.Linear(hidden, hidden)
            mlp = nn.Module()
            mlp.gate_proj = nn.Linear(hidden, hidden)
            mlp.up_proj = nn.Linear(hidden, hidden)
            mlp.down_proj = nn.Linear(hidden, hidden)
            layer.self_attn = attn
            layer.mlp = mlp
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        # mirror the pre-injection freeze applied by llm_conf.freeze (default true)
        self.freeze()

    def freeze(self):
        for p in self.parameters():
            p.requires_grad = False
        self.eval()


def _apply(fake_llm, lora_conf):
    """Run the real injection method without building a full FunASRNano.

    ``__new__`` + ``nn.Module.__init__`` initialize the module machinery while
    skipping ``FunASRNano.__init__`` (which would try to download a real LLM).
    """
    inst = FunASRNano.__new__(FunASRNano)
    nn.Module.__init__(inst)
    inst.llm = fake_llm
    inst._apply_lora_to_llm({"use_lora": True, "lora_conf": lora_conf})


def _lora_layers(fake_llm):
    return [m for m in fake_llm.modules() if isinstance(m, LoRALinear)]


def test_only_configured_target_modules_replaced():
    fake = _FakeQwen3()
    _apply(fake, DEFAULT_CONF)

    for layer in fake.layers:
        # the two configured targets are LoRA adapters
        assert isinstance(layer.self_attn.q_proj, LoRALinear)
        assert isinstance(layer.self_attn.v_proj, LoRALinear)
        # every other linear stays a plain nn.Linear
        assert type(layer.self_attn.k_proj) is nn.Linear
        assert type(layer.self_attn.o_proj) is nn.Linear
        assert type(layer.mlp.gate_proj) is nn.Linear
        assert type(layer.mlp.up_proj) is nn.Linear
        assert type(layer.mlp.down_proj) is nn.Linear

    # exactly 2 targets x N_LAYERS adapters, each carrying lora_A / lora_B
    adapters = _lora_layers(fake)
    assert len(adapters) == 2 * N_LAYERS
    for m in adapters:
        assert hasattr(m, "lora_A") and hasattr(m, "lora_B")


def test_custom_target_modules_restrict_replacement():
    fake = _FakeQwen3()
    _apply(fake, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                  "target_modules": ["o_proj"]})

    for layer in fake.layers:
        assert isinstance(layer.self_attn.o_proj, LoRALinear)
        assert type(layer.self_attn.q_proj) is nn.Linear
        assert type(layer.self_attn.v_proj) is nn.Linear
    assert len(_lora_layers(fake)) == N_LAYERS


def test_initial_output_matches_original_linear():
    fake = _FakeQwen3()
    x = torch.randn(3, HIDDEN)

    with torch.no_grad():
        ref = torch.stack(
            [fake.layers[i].self_attn.q_proj(x) for i in range(N_LAYERS)]
        )

    _apply(fake, DEFAULT_CONF)
    # lora_B is initialized to zero, so the adapter contributes nothing and the
    # shared base weight reproduces the original linear's output bit-for-bit.
    with torch.no_grad():
        after = torch.stack(
            [fake.layers[i].self_attn.q_proj(x) for i in range(N_LAYERS)]
        )
    assert torch.equal(after, ref)


def test_base_frozen_adapters_trainable():
    fake = _FakeQwen3()  # frozen before injection, as llm_conf.freeze does
    _apply(fake, DEFAULT_CONF)

    for layer in fake.layers:
        q = layer.self_attn.q_proj
        # shared pretrained base stays frozen
        assert q.weight.requires_grad is False
        assert q.bias.requires_grad is False
        # the new adapter parameters are trainable
        assert q.lora_A.requires_grad is True
        assert q.lora_B.requires_grad is True


def test_mark_only_lora_as_trainable_keeps_adapters_trainable():
    from funasr.models.lora.utils import mark_only_lora_as_trainable

    fake = _FakeQwen3()
    _apply(fake, DEFAULT_CONF)

    # simulate the documented lora_only=true flow in train.py / train_ds.py:
    # unfreeze the base model first, then mark_only_lora_as_trainable must
    # re-freeze every non-LoRA parameter while lora_A/lora_B stay trainable.
    for p in fake.parameters():
        p.requires_grad = True
    mark_only_lora_as_trainable(fake, bias="none")

    for layer in fake.layers:
        q = layer.self_attn.q_proj
        assert q.weight.requires_grad is False
        assert q.lora_A.requires_grad is True
        assert q.lora_B.requires_grad is True
    # a non-LoRA parameter outside the adapters is frozen by lora_only
    assert fake.embed_tokens.weight.requires_grad is False


def test_adapter_dtype_matches_base_weight_dtype():
    fake = _FakeQwen3().to(torch.bfloat16)  # llm_dtype=bf16 on a real run
    _apply(fake, DEFAULT_CONF)

    for layer in fake.layers:
        q = layer.self_attn.q_proj
        assert q.weight.dtype is torch.bfloat16
        assert q.lora_A.dtype is torch.bfloat16
        assert q.lora_B.dtype is torch.bfloat16


def test_state_dict_round_trip_and_train_eval_transitions(tmp_path):
    lora_conf = {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                 "target_modules": ["q_proj", "v_proj"]}
    fake = _FakeQwen3()
    _apply(fake, lora_conf)

    # force a non-zero adapter
    with torch.no_grad():
        for m in _lora_layers(fake):
            m.lora_A.normal_()
            m.lora_B.normal_(std=0.5)

    x = torch.randn(3, HIDDEN)
    with torch.no_grad():
        train_out = torch.stack(
            [fake.layers[i].self_attn.q_proj(x) for i in range(N_LAYERS)]
        )

    ckpt_path = os.path.join(tmp_path, "llm_lora.pt")
    torch.save(fake.state_dict(), ckpt_path)

    # model.eval() / model.train() transitions never corrupt the adapter: the
    # top-level eval() keeps the adapter in its unmerged state (nn.Module.eval
    # dispatches through LoRALinear.train(False)), dropout is off in eval, and
    # with dropout=0 the outputs are identical.
    with torch.no_grad():
        fake.eval()
        eval_out = torch.stack(
            [fake.layers[i].self_attn.q_proj(x) for i in range(N_LAYERS)]
        )
        fake.train()
        train_out2 = torch.stack(
            [fake.layers[i].self_attn.q_proj(x) for i in range(N_LAYERS)]
        )
    assert torch.allclose(eval_out, train_out2)
    assert torch.equal(train_out, train_out2)

    # The documented merge/unmerge convention on the adapter layer itself:
    # eval() folds W' = W + alpha/r * B @ A into the shared base weight, train()
    # unfolds it again, restoring the exact unmerged output.
    q = fake.layers[0].self_attn.q_proj
    with torch.no_grad():
        q.eval()
        assert q.merged is True
        merged_out = q(x)
        q.train()
        assert q.merged is False
        unmerged_out = q(x)
    # merged and unmerged forwards agree within fp32 tolerance (different op
    # ordering; see MERGE_RTOL). A genuinely broken merge would differ by the
    # full adapter term, not ~1e-5, so the tolerance is still discriminating.
    assert torch.allclose(merged_out, unmerged_out, rtol=MERGE_RTOL, atol=MERGE_ATOL)
    # the merge/unmerge round trip restores the unmerged behavior: the base
    # weight is (w + delta) - delta, which fp32 does not restore bit-exactly
    assert torch.allclose(train_out[0], unmerged_out, rtol=MERGE_RTOL, atol=MERGE_ATOL)

    # a fresh model built with the same lora_conf loads the checkpoint; the
    # non-zero adapter (and its forward behavior) survives the round trip.
    fresh = _FakeQwen3()
    _apply(fresh, lora_conf)
    fresh.load_state_dict(torch.load(ckpt_path))
    with torch.no_grad():
        loaded_out = torch.stack(
            [fresh.layers[i].self_attn.q_proj(x) for i in range(N_LAYERS)]
        )
    assert torch.equal(loaded_out, train_out)


def test_no_matching_target_is_explicit_noop():
    import logging

    fake = _FakeQwen3()
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    handler = _Collect(level=logging.WARNING)
    # model.py logs through module-level logging.warning -> the root logger
    logging.getLogger().addHandler(handler)
    try:
        # must not raise; the configuration is accepted with zero adapters
        _apply(fake, {"r": 4, "lora_alpha": 8, "lora_dropout": 0.0,
                      "target_modules": ["does_not_exist"]})
    finally:
        logging.getLogger().removeHandler(handler)

    assert len(_lora_layers(fake)) == 0
    for layer in fake.layers:
        assert type(layer.self_attn.q_proj) is nn.Linear
        assert type(layer.self_attn.v_proj) is nn.Linear
    # the model still forwards
    with torch.no_grad():
        fake.layers[0].self_attn.q_proj(torch.randn(2, HIDDEN))
    # the outcome is explicit: a warning documents that nothing was replaced
    assert any("no target modules found" in r for r in records)
