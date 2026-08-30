"""FunASR CLI - Agent-friendly speech recognition from the command line."""

import argparse
import json
import os
import re
import sys
import time
import unicodedata

MODEL_CONFIGS = {
    "sensevoice": {"model": "iic/SenseVoiceSmall", "vad_model": "fsmn-vad", "vad_kwargs": {"max_single_segment_time": 30000}},
    "paraformer": {"model": "paraformer-zh", "vad_model": "fsmn-vad", "punc_model": "ct-punc"},
    "paraformer-en": {"model": "paraformer-en", "vad_model": "fsmn-vad"},
    "fun-asr-nano": {"model": "FunAudioLLM/Fun-ASR-Nano-2512", "vad_model": "fsmn-vad"},
}

SUBTITLE_CONTINUATION_PUNCTUATION = (
    "，",
    ",",
    "、",
    "：",
    ":",
    "；",
    ";",
)
SUBTITLE_TRAILING_PUNCTUATION = SUBTITLE_CONTINUATION_PUNCTUATION + (
    "。",
    ".",
    "！",
    "!",
    "？",
    "?",
)


def clean_text(text):
    return re.sub(r"<\|[^|]*\|>", "", text).strip()


def _srt_time(ms):
    ms = max(0, int(round(ms)))
    h = ms // 3600000
    m = (ms % 3600000) // 60000
    sec = (ms % 60000) // 1000
    ms_rem = ms % 1000
    return f"{h:02d}:{m:02d}:{sec:02d},{ms_rem:03d}"


def format_srt(segments):
    lines = []
    for i, seg in enumerate(segments, 1):
        lines += [str(i), f"{_srt_time(seg.get('start',0))} --> {_srt_time(seg.get('end',0))}", seg.get('text',''), ""]
    return "\n".join(lines)


def _subtitle_body_length(text):
    return len(
        str(text).strip().rstrip("".join(SUBTITLE_TRAILING_PUNCTUATION)).strip()
    )


def _join_subtitle_text(left, right):
    left = str(left).rstrip()
    right = str(right).lstrip()
    if (
        left
        and right
        and left[-1].isascii()
        and right[0].isascii()
        and left[-1].isalnum()
        and right[0].isalnum()
    ):
        return f"{left} {right}"
    return left + right


def _subtitle_token_spans(text):
    spans = []
    pending_start = None
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if unicodedata.category(char).startswith("P"):
            if spans:
                spans[-1][1] = index + 1
            elif pending_start is None:
                pending_start = index
            index += 1
            continue

        start = index
        if char.isascii() and (char.isalnum() or char in "_'"):
            index += 1
            while index < len(text):
                char = text[index]
                if not (char.isascii() and (char.isalnum() or char in "_'")):
                    break
                index += 1
        elif _is_supported_subtitle_character(char):
            index += 1
        else:
            return []
        if pending_start is not None:
            start = pending_start
            pending_start = None
        spans.append([start, index])

    if pending_start is not None and spans:
        spans[-1][1] = len(text)
    return spans


def _is_supported_subtitle_character(char):
    codepoint = ord(char)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x323AF
        or 0x3040 <= codepoint <= 0x30FF
        or 0x31F0 <= codepoint <= 0x31FF
        or 0xFF66 <= codepoint <= 0xFF9D
        or 0x1100 <= codepoint <= 0x11FF
        or 0x3130 <= codepoint <= 0x318F
        or 0xAC00 <= codepoint <= 0xD7AF
    )


def _subtitle_word_spans(text, words):
    spans = []
    cursor = 0
    for raw_word in words:
        word = str(raw_word).lstrip("▁").strip()
        if not word:
            return []
        start = text.find(word, cursor)
        if start < 0 or any(
            not (char.isspace() or unicodedata.category(char).startswith("P"))
            for char in text[cursor:start]
        ):
            return []
        if spans:
            spans[-1][1] = start
        elif any(
            not (char.isspace() or unicodedata.category(char).startswith("P"))
            for char in text[:start]
        ):
            return []
        end = start + len(word)
        spans.append([0 if not spans and start else start, end])
        cursor = end

    if any(
        not (char.isspace() or unicodedata.category(char).startswith("P"))
        for char in text[cursor:]
    ):
        return []
    if spans:
        spans[-1][1] = len(text)
    return spans


def _timestamp_pair(item):
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return None
    try:
        start = int(item[0])
        end = int(item[1])
    except (TypeError, ValueError, OverflowError):
        return None
    return [start, end] if end > start else None


def _timestamps_are_ordered(timestamps):
    return bool(timestamps) and all(
        timestamp is not None
        and timestamp[0] >= 0
        and (index == 0 or timestamp[0] >= timestamps[index - 1][1])
        for index, timestamp in enumerate(timestamps)
    )


def _subtitle_break_weights(text, token_spans):
    """Map token-boundary indices to lexical and punctuation preferences."""
    boundary_to_token = {span[1]: index + 1 for index, span in enumerate(token_spans)}
    breaks = {len(token_spans): 0.0}

    try:
        import logging
        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="pkg_resources is deprecated as an API.*",
                category=UserWarning,
            )
            import jieba

        jieba.setLogLevel(logging.ERROR)
        cursor = 0
        for piece in jieba.cut(text, cut_all=False, HMM=True):
            cursor += len(piece)
            if piece.strip() and any(
                char.isalnum() or _is_supported_subtitle_character(char)
                for char in piece
            ):
                token_index = boundary_to_token.get(cursor)
                if token_index is not None:
                    body_length = sum(
                        not (
                            char.isspace()
                            or unicodedata.category(char).startswith("P")
                        )
                        for char in piece
                    )
                    strength = 1.0 if body_length > 1 else 0.2
                    breaks[token_index] = max(
                        breaks.get(token_index, 0.0), strength
                    )
    except (ImportError, RuntimeError, ValueError):
        pass

    for index, span in enumerate(token_spans[:-1], 1):
        boundary = span[1]
        left = text[:boundary].rstrip()
        right = text[boundary:].lstrip()
        strength = 0.0
        if left and left[-1] in ".!?。！？;；":
            strength = 4.0
        elif left and unicodedata.category(left[-1]).startswith("P"):
            strength = 2.0
        elif (
            text[boundary - 1 : boundary].isspace()
            or text[boundary : boundary + 1].isspace()
        ):
            strength = 1.5
        elif left and right and left[-1].isascii() != right[0].isascii():
            strength = 1.0
        if strength:
            breaks[index] = max(breaks.get(index, 0.0), strength)
    return breaks


def _balanced_subtitle_ranges(
    text, token_spans, timestamps, max_duration_ms, max_chars
):
    """Find the fewest valid cues, then optimize their readable boundaries."""
    token_count = len(token_spans)
    if not token_count:
        return []

    total_chars = len(text.strip())
    total_duration = sum(timestamp[1] - timestamp[0] for timestamp in timestamps)
    min_cues = max(1, (total_chars + max_chars - 1) // max_chars)
    break_weights = _subtitle_break_weights(text, token_spans)

    # Dynamic programming avoids the short final fragment produced by a greedy split.
    for cue_count in range(min_cues, token_count + 1):
        target_chars = total_chars / cue_count
        target_duration = total_duration / cue_count
        states = {0: (0.0, [])}
        for cue_index in range(cue_count):
            next_states = {}
            cues_left = cue_count - cue_index - 1
            for token_start, (cost, path) in states.items():
                min_end = token_start + 1
                max_end = token_count - cues_left
                for token_end in range(min_end, max_end + 1):
                    cue_text = text[
                        token_spans[token_start][0] : token_spans[token_end - 1][1]
                    ].strip()
                    cue_duration = timestamps[token_end - 1][1] - timestamps[token_start][0]
                    if len(cue_text) > max_chars or cue_duration > max_duration_ms:
                        if token_end > min_end:
                            break
                        continue

                    break_strength = break_weights.get(token_end)
                    unsafe_break = token_end < token_count and break_strength is None
                    char_error = (len(cue_text) - target_chars) / max(target_chars, 1.0)
                    duration_error = (cue_duration - target_duration) / max(
                        target_duration, 1.0
                    )
                    gap_ms = (
                        timestamps[token_end][0] - timestamps[token_end - 1][1]
                        if token_end < token_count
                        else 0
                    )
                    cue_cost = (
                        (1000.0 if unsafe_break else 0.0)
                        + 8.0 * char_error * char_error
                        + 2.0 * duration_error * duration_error
                        - 2.0 * (break_strength or 0.0)
                        - min(max(gap_ms, 0), 1000) / 1000.0
                    )
                    candidate = (cost + cue_cost, [*path, (token_start, token_end)])
                    previous = next_states.get(token_end)
                    if previous is None or candidate[0] < previous[0]:
                        next_states[token_end] = candidate
            states = next_states
            if not states:
                break

        if token_count in states:
            return states[token_count][1]
    return []


def _sentence_timestamp_words(result):
    sentence_info = result.get("sentence_info", []) or []
    words = result.get("words", []) or []
    raw_timestamps = result.get("timestamp") or result.get("timestamps") or []
    timestamps = [_timestamp_pair(item) for item in raw_timestamps]
    if not words or len(words) != len(timestamps) or not _timestamps_are_ordered(
        timestamps
    ):
        return [None] * len(sentence_info)

    mapped_words = []
    cursor = 0
    for sentence in sentence_info:
        local_timestamps = [
            _timestamp_pair(item)
            for item in (
                sentence.get("timestamp") or sentence.get("timestamps") or []
            )
        ]
        if not _timestamps_are_ordered(local_timestamps):
            mapped_words.append(None)
            continue

        local_cursor = cursor
        selected = []
        for timestamp in local_timestamps:
            while (
                local_cursor < len(timestamps)
                and timestamps[local_cursor] != timestamp
            ):
                local_cursor += 1
            if local_cursor == len(timestamps):
                selected = []
                break
            selected.append(words[local_cursor])
            local_cursor += 1
        if len(selected) == len(local_timestamps):
            mapped_words.append(selected)
            cursor = local_cursor
        else:
            mapped_words.append(None)
    return mapped_words


def _split_subtitle_segment(segment, max_duration_ms, max_chars):
    text = str(segment.get("text", ""))
    start = int(segment.get("start", 0) or 0)
    end = int(segment.get("end", start) or start)
    if not text or (end - start <= max_duration_ms and len(text) <= max_chars):
        return [dict(segment)]

    raw_timestamps = segment.get("timestamp") or segment.get("timestamps") or []
    timestamps = [_timestamp_pair(item) for item in raw_timestamps]
    if not _timestamps_are_ordered(timestamps):
        return [dict(segment)]

    words = segment.get("words") or []
    token_spans = (
        _subtitle_word_spans(text, words) if words else _subtitle_token_spans(text)
    )
    if not timestamps or len(timestamps) != len(token_spans):
        return [dict(segment)]
    for index, span in enumerate(token_spans):
        token_text = text[span[0] : span[1]].strip()
        if (
            timestamps[index][1] - timestamps[index][0] > max_duration_ms
            or len(token_text) > max_chars
        ):
            return [dict(segment)]

    ranges = _balanced_subtitle_ranges(
        text, token_spans, timestamps, max_duration_ms, max_chars
    )
    if not ranges:
        return [dict(segment)]

    cues = []
    for token_start, token_end in ranges:
        cue = dict(segment)
        cue["text"] = text[
            token_spans[token_start][0] : token_spans[token_end - 1][1]
        ].strip()
        cue["start"] = timestamps[token_start][0]
        cue["end"] = timestamps[token_end - 1][1]
        cue["timestamp"] = timestamps[token_start:token_end]
        cue.pop("timestamps", None)
        cue.pop("words", None)
        cues.append(cue)

    return cues


def merge_subtitle_segments(
    segments, max_gap_ms=500, max_duration_ms=8000, max_chars=42
):
    """Group sentence timestamps into bounded, readable subtitle cues."""
    def can_follow(left, right):
        left_text = str(left.get("text", ""))
        right_text = str(right.get("text", ""))
        gap_ms = right.get("start", 0) - left.get("end", 0)
        left_speaker = left.get("speaker", left.get("spk"))
        right_speaker = right.get("speaker", right.get("spk"))
        return (
            left_speaker == right_speaker
            and 0 <= gap_ms <= max_gap_ms
            and (
                left_text.rstrip().endswith(SUBTITLE_CONTINUATION_PUNCTUATION)
                or _subtitle_body_length(left_text) <= 2
                or (
                    gap_ms <= min(max_gap_ms, 100)
                    and _subtitle_body_length(right_text) > 2
                    and right_text.rstrip().endswith(
                        SUBTITLE_CONTINUATION_PUNCTUATION
                    )
                )
            )
        )

    def combine(group):
        cue = dict(group[0])
        cue["end"] = group[-1].get("end", cue.get("end", 0))
        text = str(group[0].get("text", ""))
        for item in group[1:]:
            text = _join_subtitle_text(text, item.get("text", ""))
        cue["text"] = text
        if any(item.get("timestamp") for item in group):
            cue["timestamp"] = [
                timestamp
                for item in group
                for timestamp in item.get("timestamp", [])
            ]
        if len(group) > 1 and any(item.get("words") for item in group):
            if all(
                isinstance(item.get("words"), list)
                and item["words"]
                and len(item["words"]) == len(item.get("timestamp", []))
                for item in group
            ):
                cue["words"] = [word for item in group for word in item["words"]]
            else:
                cue.pop("words", None)
        return cue

    def pack(chain):
        groups = []
        current = [chain[-1]]
        for item in reversed(chain[:-1]):
            candidate = [item, *current]
            combined = combine(candidate)
            duration_ms = combined.get("end", 0) - combined.get("start", 0)
            if (
                duration_ms <= max_duration_ms
                and len(combined.get("text", "")) <= max_chars
            ):
                current = candidate
            else:
                groups.append(current)
                current = [item]
        groups.append(current)
        return [combine(group) for group in reversed(groups)]

    merged = []
    chain = []
    for source in segments:
        for current in _split_subtitle_segment(
            source, max_duration_ms=max_duration_ms, max_chars=max_chars
        ):
            if chain and not can_follow(chain[-1], current):
                merged.extend(pack(chain))
                chain = []
            chain.append(current)
    if chain:
        merged.extend(pack(chain))
    return merged


def format_tsv(segments):
    lines = ["start\tend\ttext"]
    for seg in segments:
        lines.append(f"{seg.get('start',0)/1000:.3f}\t{seg.get('end',0)/1000:.3f}\t{seg.get('text','')}")
    return "\n".join(lines)


def _parse_ms(value, scale=1):
    if value is None:
        return None
    try:
        return int(float(value) * scale)
    except (TypeError, ValueError):
        return None


def _timestamp_bounds_ms(result):
    bounds = []
    for key in ("timestamp", "timestamps"):
        for ts in result.get(key, []) or []:
            if isinstance(ts, dict):
                start = ts.get("start_time", ts.get("start"))
                end = ts.get("end_time", ts.get("end"))
                start_ms = _parse_ms(start, 1000)
                end_ms = _parse_ms(end, 1000)
            elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
                start_ms = _parse_ms(ts[0])
                end_ms = _parse_ms(ts[1])
            else:
                continue
            if start_ms is None or end_ms is None:
                continue
            if end_ms > start_ms:
                bounds.append((start_ms, end_ms))
    if not bounds:
        return None
    return min(start for start, _ in bounds), max(end for _, end in bounds)


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
        # No per-sentence timestamps: emit one valid cue spanning the known
        # timestamp/audio bounds instead of a bogus 99:59:59 end time.
        timestamp_bounds = _timestamp_bounds_ms({"timestamp": timestamps})
        if timestamp_bounds:
            start_ms, end_ms = timestamp_bounds
            return f"1\n{_srt_time(start_ms)} --> {_srt_time(end_ms)}\n{text}\n"
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
    p.add_argument(
        "--subtitle-segment-mode",
        choices=["readable", "sentence"],
        default="readable",
        help="SRT cue grouping: readable (default) or raw model sentence boundaries",
    )
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
    if "punc_model" not in config and args.model != "fun-asr-nano":
        if args.model != "sensevoice" or args.output_format in ("srt", "tsv"):
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
            hotwords = [
                word.strip() for word in args.hotwords.split(",") if word.strip()
            ]
            if args.model == "paraformer":
                gen_kw["hotword"] = " ".join(hotwords)
            else:
                gen_kw["hotwords"] = hotwords

        if args.output_format in ("srt", "tsv"):
            gen_kw.update(
                {
                    "sentence_timestamp": True,
                    "output_timestamp": True,
                    "return_time_stamps": True,
                }
            )

        result = model.generate(**gen_kw)
        elapsed = time.time() - t0

        text = clean_text(result[0].get("text", ""))
        segments = []
        if "sentence_info" in result[0]:
            sentence_words = _sentence_timestamp_words(result[0])
            for index, seg in enumerate(result[0]["sentence_info"]):
                s = {
                    "start": seg.get("start", 0),
                    "end": seg.get("end", 0),
                    "text": clean_text(seg.get("sentence") or seg.get("text", "")),
                    "timestamp": seg.get("timestamp") or seg.get("timestamps"),
                }
                if (
                    args.output_format == "srt"
                    and args.subtitle_segment_mode == "readable"
                    and sentence_words[index]
                ):
                    s["words"] = sentence_words[index]
                if args.spk and "spk" in seg:
                    s["speaker"] = seg["spk"]
                segments.append(s)
        if (
            args.output_format == "srt"
            and args.subtitle_segment_mode == "readable"
        ):
            segments = merge_subtitle_segments(segments)

        timestamps = result[0].get("timestamps") or result[0].get("timestamp")
        if not args.timestamps and args.output_format not in ("srt", "tsv"):
            timestamps = None
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
