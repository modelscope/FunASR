#!/usr/bin/env python3
# Copyright FunASR (https://github.com/modelscope/FunASR). All Rights Reserved.
# MIT License (https://opensource.org/licenses/MIT)

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time


FEATURE_WIDTH = 560
VOCAB_SIZE = 25055
CONTROL_INPUTS = ("speech_lengths", "language", "textnorm")
UNSUPPORTED_QUANTIZED_OPS = frozenset({"DynamicQuantizeLinear", "MatMulInteger"})


class ShapeProfile:
    def __init__(
        self,
        *,
        min_batch,
        opt_batch,
        max_batch,
        min_frames,
        opt_frames,
        max_frames,
    ):
        self.min_batch = min_batch
        self.opt_batch = opt_batch
        self.max_batch = max_batch
        self.min_frames = min_frames
        self.opt_frames = opt_frames
        self.max_frames = max_frames
        self._validate()

    def _validate(self):
        batch = (self.min_batch, self.opt_batch, self.max_batch)
        frames = (self.min_frames, self.opt_frames, self.max_frames)
        if not (0 < batch[0] <= batch[1] <= batch[2]):
            raise ValueError("batch bounds must satisfy 0 < min <= opt <= max")
        if not (0 < frames[0] <= frames[1] <= frames[2]):
            raise ValueError("frame bounds must satisfy 0 < min <= opt <= max")

    def tensor_shapes(self):
        batch = ((self.min_batch,), (self.opt_batch,), (self.max_batch,))
        shapes = {
            "speech": (
                (self.min_batch, self.min_frames, FEATURE_WIDTH),
                (self.opt_batch, self.opt_frames, FEATURE_WIDTH),
                (self.max_batch, self.max_frames, FEATURE_WIDTH),
            )
        }
        shapes.update({name: batch for name in CONTROL_INPUTS})
        return shapes

    def as_dict(self):
        return {
            "batch": [self.min_batch, self.opt_batch, self.max_batch],
            "frames": [self.min_frames, self.opt_frames, self.max_frames],
        }


def reject_unsupported_quantization(op_types):
    counts = Counter(op_types)
    blocked = {name: counts[name] for name in sorted(UNSUPPORTED_QUANTIZED_OPS) if counts[name]}
    if not blocked:
        return
    details = ", ".join(f"{name}={count}" for name, count in blocked.items())
    raise ValueError(
        "The ONNX graph is dynamically quantized and cannot be parsed by the "
        f"supported TensorRT path ({details}). Export the FP32 graph with "
        "model.export(type='onnx', quantize=False), then build FP16 or FP32 "
        "inside TensorRT."
    )


def validate_tensor_contract(inputs, outputs):
    expected_inputs = {"speech", *CONTROL_INPUTS}
    expected_outputs = {"ctc_logits", "encoder_out_lens"}
    if set(inputs) != expected_inputs:
        raise ValueError(
            f"SenseVoice inputs must be {sorted(expected_inputs)}, got {sorted(inputs)}"
        )
    if set(outputs) != expected_outputs:
        raise ValueError(
            f"SenseVoice outputs must be {sorted(expected_outputs)}, got {sorted(outputs)}"
        )

    speech_dtype, speech_shape = inputs["speech"]
    if speech_dtype != "FLOAT" or len(speech_shape) != 3 or speech_shape[-1] != FEATURE_WIDTH:
        raise ValueError(
            f"speech must be FLOAT [batch, frames, {FEATURE_WIDTH}], got "
            f"{speech_dtype} {speech_shape}"
        )
    for name in CONTROL_INPUTS:
        dtype, shape = inputs[name]
        if dtype != "INT32" or len(shape) != 1:
            raise ValueError(f"{name} must be INT32 [batch], got {dtype} {shape}")

    logits_dtype, logits_shape = outputs["ctc_logits"]
    if logits_dtype != "FLOAT" or len(logits_shape) != 3 or logits_shape[-1] != VOCAB_SIZE:
        raise ValueError(
            f"ctc_logits must be FLOAT [batch, frames, {VOCAB_SIZE}], got "
            f"{logits_dtype} {logits_shape}"
        )
    lens_dtype, lens_shape = outputs["encoder_out_lens"]
    if lens_dtype != "INT32" or len(lens_shape) != 1:
        raise ValueError(
            "encoder_out_lens must be INT32 [batch], got " f"{lens_dtype} {lens_shape}"
        )


def _onnx_tensor_specs(values, onnx):
    specs = {}
    for value in values:
        tensor_type = value.type.tensor_type
        dtype = onnx.TensorProto.DataType.Name(tensor_type.elem_type)
        shape = tuple(dim.dim_param or dim.dim_value for dim in tensor_type.shape.dim)
        specs[value.name] = (dtype, shape)
    return specs


def inspect_onnx(onnx_path):
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError("Install the ONNX dependency with `pip install onnx`.") from exc

    onnx.checker.check_model(str(onnx_path))
    model = onnx.load(str(onnx_path), load_external_data=False)
    reject_unsupported_quantization(node.op_type for node in model.graph.node)
    inputs = _onnx_tensor_specs(model.graph.input, onnx)
    outputs = _onnx_tensor_specs(model.graph.output, onnx)
    validate_tensor_contract(inputs, outputs)
    return {
        "ir_version": model.ir_version,
        "opsets": {item.domain or "ai.onnx": item.version for item in model.opset_import},
        "node_count": len(model.graph.node),
        "inputs": inputs,
        "outputs": outputs,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def build_engine(
    onnx_path,
    engine_path,
    profile,
    *,
    precision="fp16",
    workspace_gb=8.0,
    verbose=False,
):
    try:
        import tensorrt as trt
    except ImportError as exc:
        raise RuntimeError(
            "Install TensorRT 10 or newer with `pip install tensorrt-cu12`."
        ) from exc

    severity = trt.Logger.VERBOSE if verbose else trt.Logger.WARNING
    logger = trt.Logger(severity)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT failed to parse {onnx_path}:\n{errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This GPU does not report fast FP16 support.")
        config.set_flag(trt.BuilderFlag.FP16)

    optimization_profile = builder.create_optimization_profile()
    for name, shapes in profile.tensor_shapes().items():
        optimization_profile.set_shape(name, *shapes)
    if not optimization_profile:
        raise RuntimeError("TensorRT rejected the optimization profile.")
    config.add_optimization_profile(optimization_profile)

    started = time.monotonic()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT could not build a serialized engine.")
    _atomic_write(engine_path, serialized)
    return {
        "engine": str(engine_path),
        "engine_bytes": engine_path.stat().st_size,
        "engine_sha256": _sha256(engine_path),
        "onnx": str(onnx_path),
        "onnx_sha256": _sha256(onnx_path),
        "precision": precision,
        "profile": profile.as_dict(),
        "tensorrt_version": trt.__version__,
        "build_seconds": round(time.monotonic() - started, 3),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Build a TensorRT plan for the FunASR SenseVoice encoder."
    )
    parser.add_argument("onnx_model", type=Path)
    parser.add_argument("engine", type=Path)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    parser.add_argument("--workspace-gb", type=float, default=8.0)
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--opt-batch", type=int, default=8)
    parser.add_argument("--max-batch", type=int, default=16)
    parser.add_argument("--min-frames", type=int, default=1)
    parser.add_argument("--opt-frames", type=int, default=512)
    parser.add_argument("--max-frames", type=int, default=4096)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.onnx_model.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {args.onnx_model}")
    if args.engine.exists() and not args.force:
        raise FileExistsError(f"Engine already exists (pass --force): {args.engine}")
    if args.workspace_gb <= 0:
        raise ValueError("--workspace-gb must be greater than zero")

    profile = ShapeProfile(
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        min_frames=args.min_frames,
        opt_frames=args.opt_frames,
        max_frames=args.max_frames,
    )
    graph = inspect_onnx(args.onnx_model)
    print(json.dumps({"onnx_validation": graph}, default=str, sort_keys=True))
    result = build_engine(
        args.onnx_model,
        args.engine,
        profile,
        precision=args.precision,
        workspace_gb=args.workspace_gb,
        verbose=args.verbose,
    )
    print(json.dumps(result, sort_keys=True))


def cli(argv=None):
    try:
        main(argv)
    except (FileNotFoundError, FileExistsError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
