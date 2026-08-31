#!/usr/bin/env python3
"""Transcribe long audio with Qwen3-ASR's native vLLM backend."""

import argparse
import json
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
                "end_ms": round(
                    (offset_seconds + len(audio) / sample_rate) * 1000
                ),
                "text": (_result_value(result, "text", "") or "").strip(),
                "language": _result_value(result, "language"),
            }
        )
    return segments


def build_parser():
    parser = argparse.ArgumentParser(
        description="Offline long-audio Qwen3-ASR using its native vLLM backend"
    )
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen3-ASR-1.7B")
    parser.add_argument("--language", default=None)
    parser.add_argument("--chunk-seconds", type=float, default=180.0)
    parser.add_argument("--max-inference-batch-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
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


def run(args):
    if args.chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    if args.max_inference_batch_size <= 0:
        raise ValueError("max_inference_batch_size must be positive")
    if not args.audio.is_file():
        raise FileNotFoundError(args.audio)

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
    print(f"Wrote {len(payload['segments'])} segments to {output}")


if __name__ == "__main__":
    main()
