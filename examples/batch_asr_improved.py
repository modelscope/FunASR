#!/usr/bin/env python3
"""Batch ASR example for FunASR (improved)

Features added:
- argparse for command-line configuration
- optional recursive folder scan
- progress reporting with simple counts
- safer model loading and per-file error handling
- configurable output file and supported extensions
- creates output folder if missing
- configurable VAD model
"""

import argparse
import sys
from pathlib import Path
from typing import List

try:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
except Exception as e:
    print("Error importing funasr. Make sure FunASR is installed (pip install -U funasr) or you're running from the repo root.", file=sys.stderr)
    raise

def find_audio_files(folder: Path, exts: List[str], recursive: bool) -> List[Path]:
    if recursive:
        return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]

def main():
    parser = argparse.ArgumentParser(description="Batch ASR using FunASR AutoModel")
    parser.add_argument("--input-folder", "-i", type=Path, default=Path("examples/audio_samples"),
                        help="Folder with audio files (default: examples/audio_samples)")
    parser.add_argument("--output-file", "-o", type=Path, default=Path("examples/batch_transcriptions.txt"),
                        help="Output text file (default: examples/batch_transcriptions.txt)")
    parser.add_argument("--model", "-m", default="paraformer-zh",
                        help="Model name (default: paraformer-zh). Examples: paraformer-zh, paraformer-en, SenseVoiceSmall")
    parser.add_argument("--device", "-d", default="cpu", help="Device for inference (default: cpu)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Recursively search input folder for audio files")
    parser.add_argument("--extensions", "-e", nargs="+", default=[".wav", ".mp3"], help="Accepted audio extensions (default: .wav .mp3)")
    parser.add_argument("--vad-model", default="fsmn-vad", help="VAD model to use. Set to 'none' to disable. (default: fsmn-vad)")
    args = parser.parse_args()

    if not args.input_folder.exists():
        print(f"Input folder does not exist: {args.input_folder}", file=sys.stderr)
        sys.exit(2)

    exts = [ext.lower() if ext.startswith('.') else f'.{ext.lower()}' for ext in args.extensions]
    files = sorted(find_audio_files(args.input_folder, exts, args.recursive))
    if len(files) == 0:
        print(f"No audio files found in {args.input_folder} with extensions {exts}")
        sys.exit(0)

    vad_arg = args.vad_model if args.vad_model.lower() != 'none' else None
    print(f"Loading model '{args.model}' on device '{args.device}'... (this may take a while)")
    try:
        model = AutoModel(model=args.model, vad_model=vad_arg, device=args.device)
    except Exception as e:
        print("Failed to load model:", e, file=sys.stderr)
        raise

    results = {}
    total = len(files)
    for idx, fpath in enumerate(files, start=1):
        rel_key = str(fpath.relative_to(args.input_folder))
        try:
            print(f"[{idx}/{total}] Processing: {fpath}")
            # `cache` was removed per review because it's not used by generate
            res = model.generate(input=str(fpath), language="auto")
            text = rich_transcription_postprocess(res[0]["text"]) if res and isinstance(res, list) and "text" in res[0] else ""
            results[rel_key] = text
            print(f"  -> {text}")
        except Exception as e:
            print(f"  Error processing {fpath}: {e}", file=sys.stderr)
            results[rel_key] = f"<ERROR: {e}>"

    # Ensure output directory exists
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as outf:
        for fname, transcription in results.items():
            outf.write(f"{fname}: {transcription}\n")

    print(f"\nAll transcriptions saved to {args.output_file}")


if __name__ == "__main__":
    main()
