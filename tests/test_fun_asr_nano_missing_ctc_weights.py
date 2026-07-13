import pytest
import torch

from funasr.models.fun_asr_nano import checkpoint_utils
from funasr.models.fun_asr_nano import model as nano_model
from funasr.models.fun_asr_nano.inference_vllm import FunASRNanoVLLM
from funasr.train_utils.load_pretrained_model import load_pretrained_model


class _HookedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.loaded = torch.nn.Parameter(torch.zeros(2))
        self.mismatched = torch.nn.Parameter(torch.zeros(2))
        self.absent = torch.nn.Parameter(torch.zeros(2))
        self.loaded_checkpoint_keys = None

    def on_pretrained_model_loaded(self, loaded_keys):
        self.loaded_checkpoint_keys = set(loaded_keys)


class _WrappedModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.module = _HookedModel()


def _attach_ctc_modules(instance):
    instance.ctc_decoder = torch.nn.Linear(2, 2)
    instance.ctc = torch.nn.Linear(2, 2)
    instance.ctc_tokenizer = object()
    instance.blank_id = 2


def _complete_ctc_state(instance):
    state = {
        f"ctc_decoder.{key}": value.clone()
        for key, value in instance.ctc_decoder.state_dict().items()
    }
    state.update({f"ctc.{key}": value.clone() for key, value in instance.ctc.state_dict().items()})
    return state


def test_pretrained_loader_reports_only_keys_loaded_from_checkpoint(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {"state_dict": {"loaded": torch.ones(2), "mismatched": torch.ones(1)}},
        checkpoint,
    )
    model = _HookedModel()

    load_pretrained_model(path=str(checkpoint), model=model)

    assert model.loaded_checkpoint_keys == {"loaded"}


def test_pretrained_loader_calls_inner_hook_with_clean_ddp_keys(tmp_path):
    checkpoint = tmp_path / "model.pt"
    torch.save({"state_dict": {"module.loaded": torch.ones(2)}}, checkpoint)
    model = _WrappedModel()

    load_pretrained_model(path=str(checkpoint), model=model)

    assert model.module.loaded_checkpoint_keys == {"loaded"}


def test_checkpoint_state_normalization_unwraps_and_removes_ddp_prefix():
    normalizer = getattr(checkpoint_utils, "normalize_checkpoint_state", None)
    assert callable(normalizer)
    tensor = torch.ones(2)

    state = normalizer({"state_dict": {"model": {"module.weight": tensor}}})

    assert state == {"weight": tensor}


def test_native_model_disables_timestamps_when_ctc_weights_are_missing():
    model = object.__new__(nano_model.FunASRNano)
    torch.nn.Module.__init__(model)
    _attach_ctc_modules(model)

    model.on_pretrained_model_loaded(set())

    assert model.ctc_decoder is None
    assert model.ctc is None
    assert model.ctc_tokenizer is None
    assert model.blank_id is None


def test_native_model_keeps_timestamps_when_ctc_weights_are_complete():
    model = object.__new__(nano_model.FunASRNano)
    torch.nn.Module.__init__(model)
    _attach_ctc_modules(model)
    state = _complete_ctc_state(model)

    model.on_pretrained_model_loaded(state)

    assert model.ctc_decoder is not None
    assert model.ctc is not None
    assert model.ctc_tokenizer is not None
    assert model.blank_id == 2


def test_native_model_disables_timestamps_when_ctc_weights_are_incomplete():
    model = object.__new__(nano_model.FunASRNano)
    torch.nn.Module.__init__(model)
    _attach_ctc_modules(model)
    state = _complete_ctc_state(model)
    state.pop("ctc.bias")

    model.on_pretrained_model_loaded(state)

    assert model.ctc_decoder is None
    assert model.ctc is None
    assert model.ctc_tokenizer is None
    assert model.blank_id is None


def test_vllm_model_disables_timestamps_when_ctc_weights_are_missing():
    model = object.__new__(FunASRNanoVLLM)
    _attach_ctc_modules(model)

    model._load_ctc_weights({})

    assert model.ctc_decoder is None
    assert model.ctc is None
    assert model.ctc_tokenizer is None
    assert model.blank_id is None


def test_vllm_model_keeps_timestamps_when_ctc_weights_are_complete():
    model = object.__new__(FunASRNanoVLLM)
    _attach_ctc_modules(model)
    state = _complete_ctc_state(model)

    model._load_ctc_weights(state)

    assert model.ctc_decoder is not None
    assert model.ctc is not None
    assert model.ctc_tokenizer is not None
    assert model.blank_id == 2


@pytest.mark.parametrize("checkpoint_format", ["ddp", "wrapped"])
def test_vllm_model_keeps_timestamps_for_normalized_checkpoint_state(checkpoint_format):
    model = object.__new__(FunASRNanoVLLM)
    _attach_ctc_modules(model)
    state = _complete_ctc_state(model)
    if checkpoint_format == "ddp":
        state = {f"module.{key}": value for key, value in state.items()}
    else:
        state = {"model_state_dict": state}

    model._load_ctc_weights(state)

    assert model.ctc_decoder is not None
    assert model.ctc is not None
    assert model.ctc_tokenizer is not None
    assert model.blank_id == 2


def test_vllm_model_disables_timestamps_when_ctc_weights_are_incomplete():
    model = object.__new__(FunASRNanoVLLM)
    _attach_ctc_modules(model)
    state = _complete_ctc_state(model)
    state.pop("ctc_decoder.bias")

    model._load_ctc_weights(state)

    assert model.ctc_decoder is None
    assert model.ctc is None
    assert model.ctc_tokenizer is None
    assert model.blank_id is None


@pytest.mark.parametrize("key", ["ctc_decoder.weight", "ctc.bias"])
def test_vllm_model_disables_timestamps_when_ctc_weight_shape_mismatches(key):
    model = object.__new__(FunASRNanoVLLM)
    _attach_ctc_modules(model)
    state = _complete_ctc_state(model)
    state[key] = torch.zeros(1)

    model._load_ctc_weights(state)

    assert model.ctc_decoder is None
    assert model.ctc is None
    assert model.ctc_tokenizer is None
    assert model.blank_id is None
