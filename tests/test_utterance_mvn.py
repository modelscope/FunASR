"""Run with: python -m pytest tests/test_utterance_mvn.py -q (CPU and CUDA)."""

from pathlib import Path

import pytest
import soundfile as sf
import torch
from omegaconf import OmegaConf

from funasr import AutoModel
from funasr.frontends.wav_frontend import WavFrontend
from funasr.models.normalize.utterance_mvn import UtteranceMVN, utterance_mvn
from funasr.models.transformer.model import Transformer
from funasr.tokenizer.char_tokenizer import CharTokenizer
from funasr.utils.misc import prepare_model_dir


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    return request.param


def features(device, dtype=torch.float32):
    generator = torch.Generator().manual_seed(17)
    x = torch.randn(2, 17, 3, generator=generator, dtype=dtype) * 2 + 5
    return x.to(device), torch.tensor([11, 17], device=device)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "norm_means,norm_vars", [(True, True), (True, False), (False, True), (False, False)]
)
def test_formula_and_switches(device, dtype, norm_means, norm_vars):
    x, lengths = features(device, dtype)
    actual, actual_lengths = utterance_mvn(x.clone(), lengths, norm_means, norm_vars)
    assert actual.dtype == dtype
    assert actual.device == x.device
    assert actual_lengths is lengths
    tolerance = {
        torch.float64: 1e-12,
        torch.float32: 2e-6,
        torch.float16: 0.02,
        torch.bfloat16: 0.15,
    }[dtype]
    for i, length in enumerate(lengths.tolist()):
        # Independent oracle: unpadded per-utterance torch.std, evaluated in double.
        valid = x[i, :length].double()
        mean = valid.mean(0)
        expected = valid - mean if norm_means else valid
        if norm_vars:
            expected = expected / valid.std(0, unbiased=False).clamp_min(1e-20)
        torch.testing.assert_close(
            actual[i, :length].double(), expected, rtol=tolerance, atol=tolerance
        )
        if norm_means and norm_vars:
            torch.testing.assert_close(
                actual[i, :length].double().var(0, unbiased=False),
                torch.ones(3, device=device, dtype=torch.float64),
                rtol=tolerance,
                atol=tolerance,
            )
        if norm_means and not norm_vars:
            # Preserve historical mean-only padding.
            torch.testing.assert_close(
                actual[i, length:].double(),
                -mean.expand(17 - length, -1),
                rtol=tolerance,
                atol=tolerance,
            )
        else:
            assert torch.count_nonzero(actual[i, length:]) == 0


def test_batch_length_and_order_invariance(device):
    x, lengths = features(device)
    layer = UtteranceMVN(norm_vars=True)
    batch, _ = layer(x.clone(), lengths)
    single, inferred_lengths = layer(x[:1, :11].clone())
    reverse, _ = layer(x.flip(0), lengths.flip(0))
    torch.testing.assert_close(single[0], batch[0, :11], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(reverse.flip(0), batch, rtol=0, atol=0)
    assert inferred_lengths.tolist() == [11]


@pytest.mark.parametrize(
    "norm_means,norm_vars,expected",
    [
        (True, False, [-2.0, 2.0, -3.0, -3.0]),
        (False, True, [0.5, 2.5, 0.0, 0.0]),
        (False, False, [1.0, 5.0, 0.0, 0.0]),
    ],
)
def test_other_branches_keep_existing_values(device, norm_means, norm_vars, expected):
    x = torch.tensor([[[1.0], [5.0], [0.0], [0.0]]], device=device)
    actual, _ = utterance_mvn(x, torch.tensor([2], device=device), norm_means, norm_vars)
    torch.testing.assert_close(actual.flatten(), x.new_tensor(expected), rtol=0, atol=0)


@pytest.mark.parametrize("padding", [0, 2])
def test_mean_variance_values(device, padding):
    x = torch.tensor([1.0, 5.0] + [0.0] * padding, device=device).view(1, -1, 1)
    actual, _ = UtteranceMVN(norm_vars=True)(x, torch.tensor([2], device=device))
    expected = x.new_tensor([-1.0, 1.0] + [0.0] * padding)
    torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("norm_means", [True, False])
def test_eps_clamps_standard_deviation(device, norm_means):
    x = torch.tensor([[[1.0], [3.0]]], device=device, dtype=torch.float64)
    actual, _ = utterance_mvn(x.clone(), norm_means=norm_means, norm_vars=True, eps=4.0)
    expected = (x - 2.0 if norm_means else x) / 4.0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_corrected_gradients(device):
    x, lengths = features(device, torch.float64)
    x.requires_grad_()
    reference = x.detach().clone().requires_grad_()
    actual, _ = UtteranceMVN(norm_vars=True)(x, lengths)
    weights = torch.linspace(0.1, 1.3, x.numel(), device=device).reshape_as(x)
    (actual * weights).sum().backward()
    loss = 0
    for i, length in enumerate(lengths.tolist()):
        valid = reference[i, :length]
        normalized = (valid - valid.mean(0)) / valid.std(0, unbiased=False)
        loss = loss + (normalized * weights[i, :length]).sum()
    loss.backward()
    assert torch.isfinite(x.grad).all()
    assert torch.count_nonzero(x.grad[0, :11]) > 0
    assert torch.count_nonzero(x.grad[0, 11:]) == 0
    torch.testing.assert_close(x.grad, reference.grad, rtol=1e-9, atol=1e-9)


def model_config(normalize_conf):
    return dict(
        model="Transformer",
        input_size=80,
        vocab_size=9,
        encoder="TransformerEncoder",
        encoder_conf=dict(
            output_size=32,
            attention_heads=2,
            linear_units=64,
            num_blocks=1,
            input_layer="linear",
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            attention_dropout_rate=0.0,
        ),
        decoder="TransformerDecoder",
        decoder_conf=dict(
            attention_heads=2,
            linear_units=64,
            num_blocks=1,
            dropout_rate=0.0,
            positional_dropout_rate=0.0,
            self_attention_dropout_rate=0.0,
            src_attention_dropout_rate=0.0,
        ),
        ctc_weight=0.5,
        normalize="UtteranceMVN",
        normalize_conf=normalize_conf,
        frontend="WavFrontend",
        frontend_conf=dict(fs=16000, dither=0.0),
        tokenizer="CharTokenizer",
        tokenizer_conf=dict(
            token_list=["<blank>", "<s>", "</s>", "你", "好", "语", "音", "识", "<unk>"]
        ),
    )


def test_saved_config_and_checkpoint_round_trip(tmp_path):
    normalize_conf = dict(norm_means=True, norm_vars=True)
    config = model_config(normalize_conf)
    original = Transformer(**config).eval()
    # Use the training config writer and the local AutoModel config/weight loader.
    prepare_model_dir(output_dir=str(tmp_path), **config)
    torch.save({"state_dict": original.state_dict()}, tmp_path / "model.pt")
    saved = OmegaConf.load(tmp_path / "config.yaml")
    assert dict(saved.normalize_conf) == normalize_conf
    restored = AutoModel(
        model=str(tmp_path), device="cpu", ncpu=1, disable_update=True, ignore_init_mismatch=False
    ).model
    assert restored.normalize.state_dict() == {}
    for key, value in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    x = torch.randn(2, 17, 80)
    lengths = torch.tensor([11, 17])
    with torch.no_grad():
        expected, _ = original.encode(x.clone(), lengths)
        actual, _ = restored.encode(x.clone(), lengths)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("norm_vars", [False, True])
def test_real_audio_transformer_inference(device, norm_vars):
    """Real loader/frontend/encoder/CTC; small random model, no downloads or training."""
    audio, fs = sf.read(
        Path(__file__).resolve().parents[1] / "runtime/funasr_api/asr_example.wav", dtype="float32"
    )
    short = audio[: 2 * fs].copy()
    config = model_config(dict(norm_means=True, norm_vars=norm_vars))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(2026)
        model = Transformer(**config).to(device).eval()
    frontend = WavFrontend(**config["frontend_conf"])
    tokenizer = CharTokenizer(**config["tokenizer_conf"])
    captured = {}

    def frontend_hook(module, args, output):
        captured["frontend_device"] = output[0].device.type

    def raw_hook(module, args):
        captured["raw"] = args[0].detach().clone()

    def norm_hook(module, args, output):
        captured["normalized"] = output[0].detach().clone()
        captured["lengths"] = output[1].detach().clone()

    def encoder_hook(module, args, output):
        captured["encoded"] = output[0].detach().clone()

    handles = [
        frontend.register_forward_hook(frontend_hook),
        model.normalize.register_forward_pre_hook(raw_hook),
        model.normalize.register_forward_hook(norm_hook),
        model.encoder.register_forward_hook(encoder_hook),
    ]
    outputs = []
    try:
        for inputs in [[short], [short, audio], [audio, short]]:
            with torch.no_grad():
                results, _ = model.inference(
                    inputs,
                    batch_size=2,
                    frontend=frontend,
                    tokenizer=tokenizer,
                    device=device,
                    fs=fs,
                )
            assert len(results) == len(inputs)
            assert all("text" in result for result in results)
            outputs.append(dict(captured))
    finally:
        for handle in handles:
            handle.remove()
    single, batch, reverse = outputs
    n = int(single["lengths"][0])
    assert batch["lengths"].tolist() == [198, 553]
    assert next(model.parameters()).device.type == device
    assert all(output["frontend_device"] == "cpu" for output in outputs)
    for name in ["raw", "normalized", "encoded"]:
        assert batch[name].device.type == device
        torch.testing.assert_close(single[name][0, :n], batch[name][0, :n], rtol=0, atol=1e-5)
        torch.testing.assert_close(batch[name][0, :n], reverse[name][1, :n], rtol=0, atol=1e-5)
    if norm_vars:
        valid = single["raw"][0, :n].double()
        expected = (valid - valid.mean(0)) / valid.std(0, unbiased=False)
        torch.testing.assert_close(batch["normalized"][0, :n].double(), expected, rtol=0, atol=2e-6)
        torch.testing.assert_close(
            batch["normalized"][0, :n].var(0, unbiased=False),
            torch.ones(80, device=device),
            rtol=0,
            atol=1e-6,
        )
        assert torch.count_nonzero(batch["normalized"][0, n:]) == 0
