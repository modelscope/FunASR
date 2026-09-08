#!/usr/bin/env python3
"""Offline Qwen3-ASR transcription or MOSS HTTP transcription and diarization."""

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path


SAMPLE_RATE = 16000


def _result_value(result, key, default=None):
    if isinstance(result, dict):
        return result.get(key, default)
    return getattr(result, key, default)


def transcribe_chunks(model, chunks, sample_rate, language):
    """Transcribe qwen-asr chunks and restore offsets on the source timeline."""
    segments = []
    for audio, offset_seconds in chunks:
        results = model.transcribe(audio=(audio, sample_rate), language=language)
        if len(results) != 1:
            raise RuntimeError(
                f"Qwen3-ASR must return one result per chunk, received {len(results)}"
            )
        result = results[0]
        segments.append(
            {
                "start_ms": round(offset_seconds * 1000),
                "end_ms": round((offset_seconds + len(audio) / sample_rate) * 1000),
                "text": (_result_value(result, "text", "") or "").strip(),
                "language": _result_value(result, "language"),
            }
        )
    return segments


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline Qwen3-ASR (native vLLM) or MOSS diarization (HTTP)"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--engine", choices=("qwen3", "moss"), default="qwen3")
    qwen = parser.add_argument_group("Qwen3 native vLLM only")
    qwen.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    qwen.add_argument("--language", default=None)
    qwen.add_argument("--chunk-seconds", type=float, default=180.0)
    qwen.add_argument("--max-inference-batch-size", type=int, default=4)
    qwen.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    moss = parser.add_argument_group(
        "MOSS HTTP only (MOSS_VLLM_API_KEY for authentication)"
    )
    moss.add_argument("--vllm-base-url", default=None)
    moss.add_argument("--served-model", default="moss-transcribe-diarize")
    moss.add_argument("--request-timeout", type=float, default=600.0)
    moss.add_argument("--max-completion-tokens", type=int, default=8192)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def _convert_to_mono_wav(audio_path, wav_path):
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(audio_path),
        "-ar",
        str(SAMPLE_RATE),
        "-ac",
        "1",
        "-y",
        str(wav_path),
    ]
    try:
        subprocess.run(command, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg is required to normalize the input audio") from exc


def run_moss(args):
    if not args.vllm_base_url or not args.served_model.strip():
        raise ValueError("MOSS requires --vllm-base-url and a non-empty --served-model")
    if not math.isfinite(args.request_timeout) or args.request_timeout <= 0:
        raise ValueError("request_timeout must be finite and positive")
    if args.max_completion_tokens <= 0:
        raise ValueError("max_completion_tokens must be positive")
    if (
        args.language is not None
        or args.model != "Qwen/Qwen3-ASR-1.7B"
        or args.chunk_seconds != 180.0
        or args.max_inference_batch_size != 4
        or args.gpu_memory_utilization != 0.8
    ):
        raise ValueError("Qwen3 model/language/chunk/GPU options do not apply to MOSS")
    output = args.output or args.audio.with_suffix(".moss-vllm.json")
    if output.resolve() == args.audio.resolve() or (
        output.exists() and output.samefile(args.audio)
    ):
        raise ValueError("output must not overwrite the input audio")

    from funasr import AutoModel

    model = AutoModel(
        model="OpenMOSS-Team/MOSS-Transcribe-Diarize",
        backend="vllm",
        device="cpu",
        vllm_base_url=args.vllm_base_url,
        vllm_model=args.served_model,
        vllm_response_format="diarized_json",
        vllm_timeout=args.request_timeout,
        vllm_api_key=os.environ.get("MOSS_VLLM_API_KEY", "EMPTY"),
        disable_update=True,
        disable_pbar=True,
    )
    # Keep one recording in one request so anonymous speaker labels share a scope.
    results = model.generate(
        str(args.audio), max_completion_tokens=args.max_completion_tokens
    )
    if len(results) != 1:
        raise RuntimeError("MOSS must return one result for the complete recording")
    payload = results[0]
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output, payload


def run(args):
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)
    if args.engine == "moss":
        return run_moss(args)
    if (
        args.vllm_base_url is not None
        or args.served_model != "moss-transcribe-diarize"
        or args.request_timeout != 600.0
        or args.max_completion_tokens != 8192
    ):
        raise ValueError("MOSS HTTP options require --engine moss")
    if args.chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if args.max_inference_batch_size <= 0:
        raise ValueError("max_inference_batch_size must be positive")

    import soundfile as sf
    from qwen_asr import Qwen3ASRModel
    from qwen_asr.inference.utils import split_audio_into_chunks

    with tempfile.TemporaryDirectory(prefix="funasr-qwen3-vllm-") as temp:
        wav_path = Path(temp) / "input.wav"
        _convert_to_mono_wav(args.audio, wav_path)
        audio, sample_rate = sf.read(wav_path, dtype="float32")
        chunks = split_audio_into_chunks(
            audio,
            sample_rate,
            max_chunk_sec=args.chunk_seconds,
        ) or [(audio, 0.0)]
        model = Qwen3ASRModel.LLM(
            model=args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_inference_batch_size=args.max_inference_batch_size,
        )
        segments = transcribe_chunks(model, chunks, sample_rate, args.language)

    detected_language = next(
        (segment["language"] for segment in segments if segment["language"]),
        args.language,
    )
    payload = {
        "text": "".join(segment["text"] for segment in segments),
        "language": detected_language,
        "segments": segments,
    }
    output = args.output or args.audio.with_suffix(".qwen3-vllm.json")
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output, payload


def main():
    args = build_parser().parse_args()
    output, payload = run(args)
    print(payload["text"])
    if args.engine == "moss":
        segments = payload["sentence_info"]
        for segment in segments:
            print(
                f"{segment['start']}..{segment['end']} ms [{segment['spk']}] {segment['text']}"
            )
    else:
        segments = payload["segments"]
    print(f"Wrote {len(segments)} segments to {output}")


if __name__ == "__main__":
    main()
