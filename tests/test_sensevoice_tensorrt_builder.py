import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = (
    Path(__file__).parents[1]
    / "runtime"
    / "triton_gpu"
    / "scripts"
    / "build_sensevoice_tensorrt.py"
)


def load_builder_module():
    spec = importlib.util.spec_from_file_location("build_sensevoice_tensorrt", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_profile_expands_batch_and_frame_bounds_for_every_input():
    builder = load_builder_module()

    profile = builder.ShapeProfile(
        min_batch=1,
        opt_batch=8,
        max_batch=16,
        min_frames=1,
        opt_frames=512,
        max_frames=4096,
    )

    assert profile.tensor_shapes() == {
        "speech": ((1, 1, 560), (8, 512, 560), (16, 4096, 560)),
        "speech_lengths": ((1,), (8,), (16,)),
        "language": ((1,), (8,), (16,)),
        "textnorm": ((1,), (8,), (16,)),
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_batch": 2, "opt_batch": 1},
        {"opt_batch": 17, "max_batch": 16},
        {"min_frames": 0},
        {"min_frames": 32, "opt_frames": 16},
        {"opt_frames": 4097, "max_frames": 4096},
    ],
)
def test_profile_rejects_invalid_or_non_monotonic_bounds(overrides):
    builder = load_builder_module()
    values = {
        "min_batch": 1,
        "opt_batch": 8,
        "max_batch": 16,
        "min_frames": 1,
        "opt_frames": 512,
        "max_frames": 4096,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match="min <= opt <= max"):
        builder.ShapeProfile(**values)


def test_quantized_onnx_is_rejected_before_tensorrt_parsing():
    builder = load_builder_module()

    with pytest.raises(ValueError, match=r"quantized.*DynamicQuantizeLinear.*281.*quantize=False"):
        builder.reject_unsupported_quantization(
            ["MatMul", "DynamicQuantizeLinear"] * 281 + ["MatMulInteger"] * 281
        )


def test_fp32_onnx_operator_set_passes_quantization_gate():
    builder = load_builder_module()

    builder.reject_unsupported_quantization(["MatMul", "LayerNormalization", "Conv", "Softmax"])


def test_tensor_contract_requires_sensevoice_names_dtypes_and_feature_width():
    builder = load_builder_module()
    valid_inputs = {
        "speech": ("FLOAT", ("batch_size", "feats_length", 560)),
        "speech_lengths": ("INT32", ("batch_size",)),
        "language": ("INT32", ("batch_size",)),
        "textnorm": ("INT32", ("batch_size",)),
    }
    valid_outputs = {
        "ctc_logits": ("FLOAT", ("batch_size", "logits_length", 25055)),
        "encoder_out_lens": ("INT32", ("batch_size",)),
    }

    builder.validate_tensor_contract(valid_inputs, valid_outputs)

    invalid_inputs = dict(valid_inputs)
    invalid_inputs["speech"] = ("FLOAT", ("batch_size", "feats_length", 80))
    with pytest.raises(ValueError, match=r"speech.*560"):
        builder.validate_tensor_contract(invalid_inputs, valid_outputs)


def test_cli_reports_expected_input_errors_without_a_traceback(tmp_path):
    missing = tmp_path / "missing.onnx"
    engine = tmp_path / "model.plan"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(missing), str(engine)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 2
    assert f"error: ONNX model does not exist: {missing}" in result.stderr
    assert "Traceback" not in result.stderr
