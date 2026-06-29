"""FunASR CLI - Agent-friendly speech recognition from the command line."""

import argparse
import json
import os
import re
import sys
import time

MODEL_CONFIGS = {
    "sensevoice": {"model": "iic/SenseVoiceSmall", "vad_model": "fsmn-vad", "vad_kwargs": {"max_single_segment_time": 30000}},
    "paraformer": {"model": "paraformer-zh", "vad_model": "fsmn-vad", "punc_model": "ct-punc"},
    "paraformer-en": {"model": "paraformer-en", "vad_model": "fsmn-vad"},
    "fun-asr-nano": {"model": "FunAudioLLM/Fun-ASR-Nano-2512", "vad_model": "fsmn-vad"},
}


def clean_text(text):
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def _srt_time(ms):
    s = ms / 1000.0
    h, m, sec = int(s // 3600), int((s % 3600) // 60), int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{int((s % 1) * 1000):03d}"


def format_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{_srt_time(seg.get('start',0))} --> {_srt_time(seg.get('end',0))}", seg.get('text',''), ""]
    return "\n".join(lines)


def format_tsv(segments):
    lines = ["start\tend\ttext"]
    for seg in segments:
        lines.append(f"{seg.get('start',0)/1000:.3f}\t{seg.get('end',0)/1000:.3f}\t{seg.get('text','')}")
    return "\n".join(lines)


def _format_output(text, segments, timestamps, fmt, audio_path, model_name, language, elapsed):
    if fmt == "text":
        return text
    elif fmt == "json":
        obj = {"text": text}
        if segments:
            obj["segments"] = segments
        if timestamps:
            obj["timestamps"] = timestamps
        try:
            import soundfile as sf
            audio_dur = round(sf.info(audio_path).duration, 3)
        except Exception:
            audio_dur = None
        obj.update({"file": os.path.basename(audio_path), "model": model_name, "language": language or "auto", "audio_duration_s": audio_dur, "processing_s": round(elapsed, 3)})
        return json.dumps(obj, ensure_ascii=False, indent=2)
    elif fmt == "srt":
        if segments:
            return format_srt(segments)
        # No per-sentence timestamps: emit one valid cue spanning the whole audio
        # (instead of a bogus 99:59:59 end time).
        try:
            import soundfile as sf
            dur_ms = int(sf.info(audio_path).duration * 1000)
        except Exception:
            dur_ms = 0
        return f"1\n00:00:00,000 --> {_srt_time(dur_ms)}\n{text}\n"
    elif fmt == "tsv":
        return format_tsv(segments) if segments else f"start\tend\ttext\n0.000\t0.000\t{text}"


def _get_version():
    try:
        from funasr import __version__
        return __version__
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(
        prog="funasr",
        description="FunASR - speech recognition CLI. 50+ languages, speaker diarization.",
        epilog="Examples:\n"
               "  funasr audio.wav\n"
               "  funasr audio.wav --model sensevoice -f json\n"
               "  funasr audio.wav -f srt -o ./subs\n"
               "  funasr audio.wav --spk --timestamps\n"
               "  funasr audio.wav --hub hf --model fun-asr-nano\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("audio", nargs="+", help="Audio file(s) to transcribe")
    p.add_argument("--model", "-m", default="sensevoice", choices=list(MODEL_CONFIGS), help="Model (default: sensevoice)")
    p.add_argument("--hub", "-H", default="ms", choices=["ms", "hf"], help="Model hub: ms (ModelScope) or hf (Hugging Face). Default: ms")
    p.add_argument("--language", "-l", default=None, help="Language: zh, en, ja, ko, yue, auto")
    p.add_argument("--device", default=None, help="Device: cuda:0, cpu (default: auto)")
    p.add_argument("--output-format", "-f", default="text", choices=["text", "json", "srt", "tsv"], help="Output format (default: text)")
    p.add_argument("--output-dir", "-o", default=None, help="Write output files to directory")
    p.add_argument("--timestamps", action="store_true", help="Include word-level timestamps")
    p.add_argument("--spk", action="store_true", help="Enable speaker diarization")
    p.add_argument("--hotwords", default=None, help="Comma-separated hotwords")
    p.add_argument("--verbose", "-v", action="store_true", help="Show loading/timing info on stderr")
    p.add_argument("--version", action="version", version=f"%(prog)s {_get_version()}")
    args = p.parse_args()

    if args.verbose:
        print(f"Loading model: {args.model} ...", file=sys.stderr)

    import torch
    from funasr import AutoModel

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    config = MODEL_CONFIGS[args.model].copy()
    config["hub"] = args.hub
    if args.spk and "spk_model" not in config:
        config["spk_model"] = "cam++"
    if "punc_model" not in config and args.model not in ("fun-asr-nano", "sensevoice"):
        config["punc_model"] = "ct-punc"

    t_load = time.time()
    model = AutoModel(device=device, disable_update=True, **config)
    if args.verbose:
        print(f"Model loaded in {time.time() - t_load:.1f}s", file=sys.stderr)

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    for audio_path in args.audio:
        if not os.path.isfile(audio_path):
            print(f"Error: file not found: {audio_path}", file=sys.stderr)
            sys.exit(1)

        if args.verbose:
            print(f"Transcribing: {audio_path}", file=sys.stderr)

        t0 = time.time()
        gen_kw = {"input": audio_path, "batch_size": 1}
        if args.language:
            gen_kw["language"] = args.language
        if args.hotwords:
            gen_kw["hotwords"] = args.hotwords.split(",")

        result = model.generate(**gen_kw)
        elapsed = time.time() - t0

        text = clean_text(result[0].get("text", ""))
        segments = []
        if "sentence_info" in result[0]:
            for seg in result[0]["sentence_info"]:
                s = {"start": seg.get("start", 0), "end": seg.get("end", 0), "text": clean_text(seg.get("sentence") or seg.get("text", ""))}
                if args.spk and "spk" in seg:
                    s["speaker"] = seg["spk"]
                segments.append(s)

        timestamps = result[0].get("timestamps") if args.timestamps else None
        output = _format_output(text, segments, timestamps, args.output_format, audio_path, args.model, args.language, elapsed)

        if args.output_dir:
            ext = {"text": "txt", "json": "json", "srt": "srt", "tsv": "tsv"}[args.output_format]
            out_path = os.path.join(args.output_dir, os.path.splitext(os.path.basename(audio_path))[0] + "." + ext)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output)
            if args.verbose:
                print(f"Written: {out_path}", file=sys.stderr)
        else:
            print(output)

        if args.verbose:
            print(f"Done in {elapsed:.2f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
