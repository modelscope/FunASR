"""llm_conf.sdpa_backends / llm_conf.torch_compile on both copies of FunASRNano (the package
class and the recipe's model.py, imported by path). CPU-only, no weights: tiny stand-ins."""

import importlib.util
import os
import sys
import types

import pytest
import torch
import torch.nn as nn

from funasr.models.fun_asr_nano import model as funasr_model
from funasr.register import tables

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPE_DIR = os.path.join(REPO, "examples", "industrial_data_pretraining", "fun_asr_nano")


class _TinyDecoder(nn.Module):
    """Stands in for the HF decoder stack (``llm.model``); records every eager call."""

    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.eager_calls = []

    def forward(self, input_ids=None, inputs_embeds=None, **kwargs):
        if not torch.compiler.is_compiling():
            cudnn = torch.backends.cuda.cudnn_sdp_enabled()
            self.eager_calls.append({"batch": int(inputs_embeds.shape[0]), "cudnn": cudnn})
        return self.proj(inputs_embeds)


class _TinyLLM(nn.Module):
    def __init__(self, dim=8, vocab=16):
        super().__init__()
        self.model, self.embed = _TinyDecoder(dim), nn.Embedding(vocab, dim)

    def get_input_embeddings(self):
        return self.embed


class _TinyEncoder(nn.Module):
    def __init__(self, input_size=80, **kwargs):
        super().__init__()
        self.lin = nn.Linear(input_size, 4)

    def output_size(self):
        return 4


class _TinyAdaptor(nn.Module):
    def __init__(self, encoder_dim=4, llm_dim=8, **kwargs):
        super().__init__()
        self.lin = nn.Linear(encoder_dim, llm_dim)


def _load_recipe_model():
    previous = tables.model_classes.get("FunASRNano")
    if RECIPE_DIR not in sys.path:
        sys.path.insert(0, RECIPE_DIR)  # the recipe imports ``ctc`` and ``tools.utils`` bare
    path = os.path.join(RECIPE_DIR, "model.py")
    spec = importlib.util.spec_from_file_location("fun_asr_nano_recipe_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # tables.register() calls inspect.getfile on the class
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        sys.modules.pop(spec.name, None)
        pytest.skip(f"recipe model.py not importable here: {exc}")
    finally:
        tables.model_classes["FunASRNano"] = previous  # the package class, registered at import
    return module.FunASRNano


@pytest.fixture(scope="module", params=["funasr", "recipe"])
def model_class(request):
    return funasr_model.FunASRNano if request.param == "funasr" else _load_recipe_model()


@pytest.fixture
def build(monkeypatch, model_class):
    """``build(llm_conf) -> FunASRNano`` on the CPU, with the transformers loaders stubbed."""
    fake = types.ModuleType("transformers")
    fake.AutoConfig = types.SimpleNamespace(from_pretrained=lambda path, **kw: {})
    fake.AutoModelForCausalLM = types.SimpleNamespace(from_config=lambda config, **kw: _TinyLLM())
    monkeypatch.setattr(funasr_model, "AutoConfig", fake.AutoConfig)  # bound at import time
    monkeypatch.setattr(funasr_model, "AutoModelForCausalLM", fake.AutoModelForCausalLM)
    monkeypatch.setitem(sys.modules, "transformers", fake)  # the recipe imports in __init__
    monkeypatch.setitem(tables.encoder_classes, "TinyEnc", _TinyEncoder)
    monkeypatch.setitem(tables.adaptor_classes, "TinyAdp", _TinyAdaptor)
    kw = dict(audio_encoder="TinyEnc", audio_adaptor="TinyAdp", llm="tiny")
    return lambda c: model_class(audio_encoder_conf={}, audio_adaptor_conf={}, llm_conf=c, **kw)


def _run(model, batch):
    device = next(model.llm.parameters()).device
    return model.llm.model.forward(inputs_embeds=torch.zeros(batch, 3, 8, device=device))


def test_default_leaves_sdpa_flag_and_forward_alone(build):
    flag = torch.backends.cuda.cudnn_sdp_enabled()
    model = build({})
    assert torch.backends.cuda.cudnn_sdp_enabled() == flag
    assert "forward" not in model.llm.model.__dict__  # still the class method
    _run(model, 2)
    assert model.llm.model.eager_calls[-1]["cudnn"] == flag


def test_sdpa_backends_disable_cudnn_only_inside_forward(build):
    flag = torch.backends.cuda.cudnn_sdp_enabled()
    model = build({"sdpa_backends": ["flash", "efficient", "math"]})
    assert torch.backends.cuda.cudnn_sdp_enabled() == flag  # construction changed nothing
    _run(model, 2)
    assert model.llm.model.eager_calls[-1]["cudnn"] is False
    assert torch.backends.cuda.cudnn_sdp_enabled() == flag  # restored on return


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_torch_compile_wraps_cuda_llm_and_keeps_batch_one_eager(build):
    model = build({"torch_compile": True}).cuda()  # built on the CPU and moved, as the trainer does
    decoder = model.llm.model
    assert "forward" in decoder.__dict__
    _run(model, 1)
    assert [c["batch"] for c in decoder.eager_calls] == [1]
    out = _run(model, 2)  # compiled: the stand-in does not record under Dynamo
    assert [c["batch"] for c in decoder.eager_calls] == [1]
    torch.testing.assert_close(out, decoder.proj(torch.zeros(2, 3, 8, device=out.device)))


def test_torch_compile_keeps_cpu_decoder_eager(build):
    model = build({"torch_compile": True})
    _run(model, 1)
    _run(model, 4)
    assert [c["batch"] for c in model.llm.model.eager_calls] == [1, 4]
