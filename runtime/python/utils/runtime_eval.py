"""Shared command-line boundaries for the standalone runtime evaluation tools."""

import argparse
import importlib
import math
from pathlib import Path
import time


def _boolean(value):
    lowered = value.lower()
    if lowered in {"true", "yes", "t", "y", "1"}:
        return True
    if lowered in {"false", "no", "f", "n", "0"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean")


def _positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _nonnegative(value):
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return number


def parse_args(argv=None, *, gpu=False, cer=False):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--backend", choices=["onnx", "torch"], default="onnx")
    parser.add_argument("--wav_file", required=True, help="WAV SCP: utterance ID and audio path")
    parser.add_argument("--quantize", type=_boolean, default=False)
    parser.add_argument("--intra_op_num_threads", type=_positive, default=1)
    parser.add_argument(
        "--device_id",
        type=_nonnegative if gpu else int,
        default=0 if gpu else -1,
        help="CUDA device index; -1 for CPU",
    )
    parser.add_argument("--batch_size", type=_positive, default=1)
    if cer:
        parser.add_argument("--output_dir", required=True)
    else:
        parser.add_argument("--warmup", type=_nonnegative, default=30)
    args = parser.parse_args(argv)
    if args.device_id < -1:
        parser.error("--device_id must be -1 or a CUDA device index")
    return args


def read_wav_scp(path):
    entries = []
    with open(path, encoding="utf-8") as source:
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            fields = line.strip().split(maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{number}: expected utterance ID and audio path")
            entries.append(tuple(fields))
    if not entries:
        raise ValueError(f"{path}: empty WAV list")
    return entries


def load_model(args):
    backend = importlib.import_module("funasr_" + args.backend)
    model = backend.Paraformer(
        args.model_dir,
        batch_size=args.batch_size,
        device_id=str(args.device_id),
        quantize=args.quantize,
        intra_op_num_threads=args.intra_op_num_threads,
    )
    if args.backend == "onnx" and args.device_id >= 0:
        providers = model.ort_infer.session.get_providers()
        if "CUDAExecutionProvider" not in providers:
            raise RuntimeError("CUDAExecutionProvider is unavailable; refusing a CPU GPU benchmark")
    return model


def _infer(model, paths):
    result = model(paths)
    expected = 1 if isinstance(paths, str) else len(paths)
    if len(result) != expected:
        raise RuntimeError(f"inference returned {len(result)} results for {expected} audio files")
    return result


def transcribe(args):
    entries = read_wav_scp(args.wav_file)
    model = load_model(args)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "text").open("w", encoding="utf-8") as text_file, (output / "token").open(
        "w", encoding="utf-8"
    ) as token_file:
        for utterance, path in entries:
            row = _infer(model, path)[0]
            if isinstance(row["preds"], str):
                text, tokens = row["preds"], row["raw_tokens"]
            else:
                text, tokens = row["preds"]
            if not isinstance(tokens, str):
                tokens = " ".join(tokens)
            text_file.write(f"{utterance} {text}\n")
            token_file.write(f"{utterance} {tokens}\n")


def benchmark(args, *, batched=False):
    entries = read_wav_scp(args.wav_file)
    import librosa

    durations = []
    for _, path in entries:
        audio, sample_rate = librosa.load(path, sr=None)
        duration = len(audio) / sample_rate
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError(f"{path}: audio duration must be positive")
        durations.append(duration)
    model = load_model(args)
    print(f"backend: {args.backend}, device_id: {args.device_id}")

    def synchronize():
        if args.backend == "torch" and args.device_id >= 0:
            import torch

            torch.cuda.synchronize(args.device_id)

    warmup_elapsed = 0.0
    for index in range(args.warmup):
        synchronize()
        start = time.perf_counter()
        result = _infer(model, entries[0][1])
        synchronize()
        elapsed = time.perf_counter() - start
        warmup_elapsed += elapsed
        average = warmup_elapsed / (index + 1)
        print(result)
        print(f"num: 1, time, {elapsed}, avg: {average}, rtf: {average / durations[0]}")

    synchronize()
    start = time.perf_counter()
    if batched:
        _infer(model, [path for _, path in entries])
    else:
        for _, path in entries:
            _infer(model, path)
    synchronize()
    elapsed = time.perf_counter() - start
    duration = sum(durations)
    print(f"total_time_comput_ms: {elapsed * 1000:.6f}")
    print(f"total_time_wav_ms: {duration * 1000:.6f}")
    print(f"total_rtf: {elapsed / duration}")
