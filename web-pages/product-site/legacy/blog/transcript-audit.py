#!/usr/bin/env python3
"""Structural audit for one FunASR MOSS result, not a quality evaluator."""
import argparse
import json
import math
from pathlib import Path


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number in milliseconds")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError(f"{name} must be finite")
    return value


def audit(payload, duration_ms):
    duration = number(duration_ms, "duration_ms")
    if duration <= 0:
        raise ValueError("duration_ms must be positive")
    if not isinstance(payload, dict) or not isinstance(payload.get("sentence_info"), list):
        raise ValueError("expected one result object with sentence_info")
    segments = payload["sentence_info"]
    last_end, previous_start = 0, -1
    speakers, empty, unordered, overlaps = set(), [], [], []
    intervals = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError(f"segment {index} must be an object")
        start = number(segment.get("start"), f"segment {index} start")
        end = number(segment.get("end"), f"segment {index} end")
        if not 0 <= start < end <= duration:
            raise ValueError(f"segment {index} must satisfy 0 <= start < end <= duration_ms")
        speaker, text = segment.get("spk"), segment.get("text")
        if not isinstance(speaker, str) or not speaker.strip() or not isinstance(text, str):
            raise ValueError(f"segment {index} needs a nonempty spk label and string text")
        speakers.add(speaker)
        if not text.strip():
            empty.append(index)
        if start < previous_start:
            unordered.append(index)
        intervals.append((start, end, index))
        previous_start, last_end = start, max(last_end, end)
    # Sweep chronologically, but report indices from the unchanged input.
    frontier = 0
    for start, end, index in sorted(intervals, key=lambda item: (item[0], item[2])):
        if start < frontier:
            overlaps.append(index)
        frontier = max(frontier, end)
    return {
        "segment_count": len(segments),
        "speaker_labels": sorted(speakers),
        "last_end_ms": last_end,
        "tail_not_covered_ms": duration - last_end,
        "empty_text_indices": empty,
        "out_of_order_indices": unordered,
        "overlap_indices": sorted(overlaps),
        "quality_verified": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--duration-ms", type=float, required=True)
    args = parser.parse_args()
    try:
        report = audit(json.loads(args.result.read_text(encoding="utf-8")), args.duration_ms)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
