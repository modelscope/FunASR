"""
FunASR Subtitle Generator

Generate SRT/VTT subtitles from audio/video files.

Usage:
    python generate_subtitle.py input.mp4
    python generate_subtitle.py input.wav --format vtt
    python generate_subtitle.py meeting.mp3 --spk  # with speaker labels
"""

import argparse
import sys
import os
import re


def clean_text(text):
    return re.sub(r'<\|[^|]*\|>', '', text or "").strip()


def format_time_srt(ms):
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms_rem:03d}"


def format_time_vtt(ms):
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    s = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{s:02d}.{ms_rem:03d}"


def timestamp_bounds_ms(result):
    bounds = []
    for key in ("timestamp", "timestamps"):
        for ts in result.get(key, []) or []:
            if isinstance(ts, dict):
                start = ts.get("start_time", ts.get("start"))
                end = ts.get("end_time", ts.get("end"))
                if start is None or end is None:
                    continue
                start_ms = int(float(start) * 1000)
                end_ms = int(float(end) * 1000)
            elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start_ms = int(ts[0])
                end_ms = int(ts[1])
            else:
                continue
            if end_ms > start_ms:
                bounds.append((start_ms, end_ms))
    if not bounds:
        return None
    return min(start for start, _ in bounds), max(end for _, end in bounds)


def main():
    parser = argparse.ArgumentParser(description="Generate subtitles from audio/video using FunASR")
    parser.add_argument("input", help="Audio/video file path")
    parser.add_argument("-o", "--output", help="Output file (default: input.srt)")
    parser.add_argument("--format", choices=["srt", "vtt"], default="srt")
    parser.add_argument("--model", default="iic/SenseVoiceSmall")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--spk", action="store_true", help="Include speaker labels")
    parser.add_argument("--lang", default="auto")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: {args.input} not found")
        sys.exit(1)

    output_path = args.output or f"{os.path.splitext(args.input)[0]}.{args.format}"
    print(f"Input:  {args.input}")
    print(f"Output: {output_path}")

    from funasr import AutoModel

    kwargs = {"model": args.model, "vad_model": "fsmn-vad", "punc_model": "ct-punc",
              "vad_kwargs": {"max_single_segment_time": 30000},
              "device": args.device, "disable_update": True}
    if args.spk:
        kwargs["spk_model"] = "cam++"
    if "Fun-ASR-Nano" in args.model or "Qwen" in args.model:
        kwargs["trust_remote_code"] = True
        kwargs["hub"] = "hf"

    print("Loading model...")
    model = AutoModel(**kwargs)
    print("Transcribing...")
    generate_kwargs = {
        "input": args.input,
        "batch_size": 1,
        "sentence_timestamp": True,
        "output_timestamp": True,
        "return_time_stamps": True,
    }
    if args.lang != "auto":
        generate_kwargs["language"] = args.lang
    result = model.generate(**generate_kwargs)

    segments = []
    result_item = result[0]
    for seg in result_item.get("sentence_info", []) or []:
        text = clean_text(seg.get("sentence") or seg.get("text", ""))
        start = int(seg.get("start", 0) or 0)
        end = int(seg.get("end", 0) or 0)
        if text and end > start:
            segments.append({"start": start, "end": end, "text": text, "spk": seg.get("spk")})

    if not segments:
        text = clean_text(result_item.get("text", ""))
        if text:
            start, end = timestamp_bounds_ms(result_item) or (0, 0)
            segments.append({"start": start, "end": end, "text": text, "spk": None})

    if not segments:
        print("No speech detected.")
        sys.exit(0)

    fmt = format_time_srt if args.format == "srt" else format_time_vtt
    with open(output_path, "w", encoding="utf-8") as f:
        if args.format == "vtt":
            f.write("WEBVTT\n\n")
        for i, seg in enumerate(segments, 1):
            text = f"[Speaker {seg['spk']}] {seg['text']}" if args.spk and seg['spk'] is not None else seg['text']
            if args.format == "srt":
                f.write(f"{i}\n{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n\n")
            else:
                f.write(f"{fmt(seg['start'])} --> {fmt(seg['end'])}\n{text}\n\n")

    print(f"Done! {len(segments)} subtitles → {output_path}")


if __name__ == "__main__":
    main()
