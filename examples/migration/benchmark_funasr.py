#!/usr/bin/env python3
"""Benchmark FunASR on representative audio during ASR migration.

The script intentionally measures only the FunASR side of a migration test. Run
Whisper or a cloud ASR baseline separately, then compare transcripts with your
normal WER/CER or human-review process.
"""

import argparse
import json
import sys
import time
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except Exception as exc:  # pragma: no cover - import message is for users
    print(
        "Failed to import FunASR. Install it with `pip install -U funasr` "
        "or run this script from the repository root.",
        file=sys.stderr,
    )
    raise

AUDIO_EXTENSIONS = (
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".wma",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FunASR over a representative audio set and write migration benchmark results."
    )
    parser.add_argument("--input", "-i", type=Path, required=True, help="Audio file or folder to benchmark.")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("outputs/migration_benchmark"),
        help="Directory for results.jsonl and summary.md.",
    )
    parser.add_argument(
        "--model",
        "-m",
        default="iic/SenseVoiceSmall",
        help="FunASR model name or ModelScope/Hugging Face id.",
    )
    parser.add_argument("--device", "-d", default="cpu", help="Inference device: cpu, cuda, or mps.")
    parser.add_argument("--vad-model", default="fsmn-vad", help="VAD model, or 'none' to disable.")
    parser.add_argument("--spk-model", default="", help="Optional speaker model such as cam++.")
    parser.add_argument("--language", default="auto", help="Language hint passed to model.generate.")
    parser.add_argument("--batch-size", type=int, default=1, help="batch_size passed to model.generate.")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively scan input folders.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=list(AUDIO_EXTENSIONS),
        help="Audio extensions to include when --input is a folder.",
    )
    parser.add_argument(
        "--metadata",
        action="append",
        default=[],
        help="Free-form key=value metadata to copy into the summary, e.g. --metadata baseline=whisper-large-v3.",
    )
    return parser.parse_args()


def normalize_extensions(values: Iterable[str]) -> List[str]:
    return sorted({value.lower() if value.startswith(".") else f".{value.lower()}" for value in values})


def iter_audio_files(path: Path, extensions: List[str], recursive: bool) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Input path does not exist: {path}")
    walker = path.rglob("*") if recursive else path.iterdir()
    return sorted(p for p in walker if p.is_file() and p.suffix.lower() in extensions)


def audio_duration_seconds(path: Path) -> Optional[float]:
    try:
        import soundfile as sf

        info = sf.info(str(path))
        if info.samplerate:
            return float(info.frames) / float(info.samplerate)
    except Exception:
        pass

    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                if rate:
                    return float(frames) / float(rate)
        except Exception:
            return None
    return None


def metadata_dict(items: Iterable[str]) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            parsed[item] = ""
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def extract_text(result: Any) -> str:
    if not result:
        return ""
    first = result[0] if isinstance(result, list) else result
    if isinstance(first, dict):
        text = first.get("text", "")
    else:
        text = str(first)
    try:
        return rich_transcription_postprocess(text)
    except Exception:
        return text


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def markdown_summary(rows: List[Dict[str, Any]], args: argparse.Namespace, model_load_seconds: float) -> str:
    successful = [row for row in rows if not row.get("error")]
    failed = [row for row in rows if row.get("error")]
    known_audio = [row for row in successful if row.get("duration_seconds")]
    total_audio = sum(float(row["duration_seconds"]) for row in known_audio)
    total_elapsed = sum(float(row["elapsed_seconds"]) for row in successful)
    throughput = total_audio / total_elapsed if total_audio and total_elapsed else None
    meta = metadata_dict(args.metadata)

    lines = [
        "# FunASR Migration Benchmark Summary",
        "",
        "## Run configuration",
        "",
        f"- Input: `{args.input}`",
        f"- Model: `{args.model}`",
        f"- Device: `{args.device}`",
        f"- VAD model: `{args.vad_model}`",
        f"- Speaker model: `{args.spk_model or 'none'}`",
        f"- Language: `{args.language}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Model load seconds: `{model_load_seconds:.3f}`",
    ]
    for key, value in meta.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## Aggregate results",
            "",
            f"- Files: `{len(rows)}`",
            f"- Successful: `{len(successful)}`",
            f"- Failed: `{len(failed)}`",
            f"- Known audio seconds: `{total_audio:.3f}`" if known_audio else "- Known audio seconds: `unknown`",
            f"- Inference seconds: `{total_elapsed:.3f}`" if successful else "- Inference seconds: `0.000`",
            f"- Aggregate realtime factor: `{throughput:.3f}x`" if throughput else "- Aggregate realtime factor: `unknown`",
            "",
            "## Per-file results",
            "",
            "| File | Audio sec | Inference sec | RTF | Status | Text preview |",
            "|---|---:|---:|---:|---|---|",
        ]
    )
    for row in rows:
        duration = row.get("duration_seconds")
        elapsed = row.get("elapsed_seconds")
        rtf = row.get("realtime_factor")
        status = "error" if row.get("error") else "ok"
        preview = (row.get("text") or row.get("error") or "").replace("|", "\\|").replace("\n", " ")[:120]
        lines.append(
            "| {file} | {duration} | {elapsed} | {rtf} | {status} | {preview} |".format(
                file=row["input"],
                duration=f"{duration:.3f}" if isinstance(duration, (int, float)) else "",
                elapsed=f"{elapsed:.3f}" if isinstance(elapsed, (int, float)) else "",
                rtf=f"{rtf:.3f}x" if isinstance(rtf, (int, float)) else "",
                status=status,
                preview=preview,
            )
        )
    lines.extend(
        [
            "",
            "## Next comparison steps",
            "",
            "- Run your Whisper or cloud ASR baseline on the same files.",
            "- Compare transcripts with human review or your normal WER/CER workflow.",
            "- Keep model download and warmup time separate from steady-state throughput.",
            "- Share reproducible findings in a FunASR showcase issue when possible.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    extensions = normalize_extensions(args.extensions)
    files = iter_audio_files(args.input, extensions, args.recursive)
    if not files:
        print(f"No audio files found under {args.input} for extensions {extensions}", file=sys.stderr)
        sys.exit(2)

    vad_model = None if args.vad_model.lower() == "none" else args.vad_model
    model_kwargs: Dict[str, Any] = {"model": args.model, "vad_model": vad_model, "device": args.device}
    if args.spk_model:
        model_kwargs["spk_model"] = args.spk_model

    print(f"Loading FunASR model: {args.model} on {args.device}")
    load_start = time.perf_counter()
    model = AutoModel(**model_kwargs)
    model_load_seconds = time.perf_counter() - load_start
    print(f"Model loaded in {model_load_seconds:.3f}s")

    rows: List[Dict[str, Any]] = []
    for index, audio_path in enumerate(files, start=1):
        display = str(audio_path if args.input.is_file() else audio_path.relative_to(args.input))
        print(f"[{index}/{len(files)}] {display}")
        duration = audio_duration_seconds(audio_path)
        start = time.perf_counter()
        row: Dict[str, Any] = {
            "input": display,
            "path": str(audio_path),
            "duration_seconds": duration,
            "model": args.model,
            "device": args.device,
            "language": args.language,
        }
        try:
            result = model.generate(input=str(audio_path), language=args.language, batch_size=args.batch_size)
            elapsed = time.perf_counter() - start
            row["elapsed_seconds"] = elapsed
            row["realtime_factor"] = (duration / elapsed) if duration and elapsed else None
            row["text"] = extract_text(result)
            print(f"  ok: {elapsed:.3f}s" + (f", {row['realtime_factor']:.3f}x" if row["realtime_factor"] else ""))
        except Exception as exc:  # keep benchmarking other files
            elapsed = time.perf_counter() - start
            row["elapsed_seconds"] = elapsed
            row["realtime_factor"] = None
            row["error"] = repr(exc)
            print(f"  error: {exc}", file=sys.stderr)
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "results.jsonl"
    summary_path = args.output_dir / "summary.md"
    write_jsonl(results_path, rows)
    summary_path.write_text(markdown_summary(rows, args, model_load_seconds), encoding="utf-8")
    print(f"\nWrote {results_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
